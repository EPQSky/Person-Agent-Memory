from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from personal_agent_memory.model_client import ModelServiceError, OpenAICompatibleClient


class GraphAdapterError(RuntimeError):
    """The optional graph projection is unavailable or returned invalid data."""


@dataclass(frozen=True, slots=True)
class GraphSourceDocument:
    document_id: str
    path: str
    source_version: str
    content: str


@dataclass(frozen=True, slots=True)
class GraphExpansion:
    content: str
    document_id: str
    path: str
    source_version: str
    hop: int
    graph_object_type: str
    source_anchor: str = ""


class GraphAdapter(Protocol):
    async def rebuild(
        self, library_id: str, documents: tuple[GraphSourceDocument, ...]
    ) -> None: ...

    async def expand(
        self,
        library_id: str,
        seed_document_ids: tuple[str, ...],
        *,
        max_hops: int,
        limit: int,
    ) -> list[GraphExpansion]: ...


class JiuwenMilvusGraphAdapter:
    """Project-owned adapter over Jiuwen's graph store and local Milvus Lite.

    A separate Milvus Lite file is used for every memory library. The adapter never
    treats its contents as authoritative: every row carries a Markdown document ID
    and version which the platform validates again before returning it.
    """

    _PROJECTION_MARKER = "projection.ready"
    _MILVUS_COMPLETE_READ_LIMIT = 16_384

    def __init__(self, root: Path, model_client: OpenAICompatibleClient) -> None:
        self.root = root
        self.model_client = model_client

    async def rebuild(
        self, library_id: str, documents: tuple[GraphSourceDocument, ...]
    ) -> None:
        if self.model_client.graph is None:
            raise GraphAdapterError("graph LLM is not configured")
        extracted: list[tuple[GraphSourceDocument, dict[str, object]]] = []
        for document in documents:
            try:
                payload = await asyncio.to_thread(
                    self.model_client.extract_graph, document.content
                )
            except ModelServiceError as error:
                raise GraphAdapterError("graph LLM extraction failed") from error
            extracted.append((document, payload))
        try:
            await asyncio.to_thread(self._replace_projection, library_id, extracted)
        except GraphAdapterError:
            raise
        except Exception as error:
            raise GraphAdapterError("Milvus Lite graph rebuild failed") from error

    async def expand(
        self,
        library_id: str,
        seed_document_ids: tuple[str, ...],
        *,
        max_hops: int,
        limit: int,
    ) -> list[GraphExpansion]:
        if not seed_document_ids or limit <= 0:
            return []
        bounded_hops = max(1, min(max_hops, 2))
        try:
            return await asyncio.to_thread(
                self._expand_projection,
                library_id,
                set(seed_document_ids),
                bounded_hops,
                limit,
            )
        except Exception as error:
            raise GraphAdapterError("Milvus Lite graph query failed") from error

    def _replace_projection(
        self,
        library_id: str,
        extracted: list[tuple[GraphSourceDocument, dict[str, object]]],
    ) -> None:
        library_root = self.root / library_id
        database = library_root / "graph.db"
        marker = library_root / self._PROJECTION_MARKER
        if not extracted:
            marker.unlink(missing_ok=True)
            database.unlink(missing_ok=True)
            return
        try:
            from jiuwen_memory.common.logging.log_config import (  # type: ignore[import-untyped]
                configure_log_config,
            )

            configure_log_config(
                {
                    "backend": "default",
                    "output": ["console"],
                    "interface_output": ["console"],
                    "performance_output": ["console"],
                }
            )
            from jiuwen_memory.foundation.store.graph import (  # type: ignore[import-untyped]
                Entity,
                Episode,
                Relation,
            )
        except ImportError as error:
            raise GraphAdapterError("JiuwenMemory or Milvus Lite is unavailable") from error

        library_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        marker.unlink(missing_ok=True)

        store = self._open_projection(database)
        try:
            store.rebuild()
            entities, relations, episodes = self._graph_objects(
                library_id, extracted, Entity, Relation, Episode
            )
            asyncio.run(
                self._write_projection(store, entities, relations, episodes)
            )
        finally:
            store.close()
        temporary_marker = marker.with_suffix(".tmp")
        temporary_marker.write_text("ready\n", encoding="ascii")
        temporary_marker.replace(marker)

    @classmethod
    def _graph_objects(
        cls,
        library_id: str,
        extracted: list[tuple[GraphSourceDocument, dict[str, object]]],
        entity_type: Any,
        relation_type: Any,
        episode_type: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        mentions: dict[str, list[tuple[GraphSourceDocument, dict[str, str]]]] = {}
        parsed: list[
            tuple[GraphSourceDocument, list[dict[str, str]], list[dict[str, str]]]
        ] = []
        for document, payload in extracted:
            entities = cls._entities(payload)
            relations = cls._relations(payload)
            parsed.append((document, entities, relations))
            seen_keys: set[str] = set()
            for entity_data in entities:
                key = cls._entity_key(entity_data["name"])
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                mentions.setdefault(key, []).append((document, entity_data))

        graph_entities: dict[str, Any] = {}
        for key in sorted(mentions):
            source_mentions = sorted(
                mentions[key], key=lambda item: (item[0].path, item[0].document_id)
            )
            first = source_mentions[0][1]
            graph_entities[key] = entity_type(
                uuid=cls._object_id("entity", key),
                user_id=library_id,
                name=first["name"],
                content=first["content"],
                metadata={
                    "library_id": library_id,
                    "entity_key": key,
                    "sources": [
                        {
                            "document_id": document.document_id,
                            "path": document.path,
                            "source_version": document.source_version,
                            "content": entity_data["content"],
                        }
                        for document, entity_data in source_mentions
                    ],
                },
                language="en",
                content_embedding=cls._placeholder_vector(),
                name_embedding=cls._placeholder_vector(),
            )

        graph_relations: list[Any] = []
        graph_episodes: list[Any] = []
        for document, entities, relations in parsed:
            document_keys = {
                cls._entity_key(entity_data["name"])
                for entity_data in entities
                if cls._entity_key(entity_data["name"]) in graph_entities
            }
            episode = episode_type(
                uuid=cls._object_id("episode", document.document_id),
                user_id=library_id,
                content=document.content,
                entities=[graph_entities[key].uuid for key in sorted(document_keys)],
                metadata={
                    "library_id": library_id,
                    "document_id": document.document_id,
                    "path": document.path,
                    "source_version": document.source_version,
                },
                language="en",
                content_embedding=cls._placeholder_vector(),
            )
            graph_episodes.append(episode)
            for key in document_keys:
                graph_entities[key].episodes.append(episode.uuid)
            for relation_data in relations:
                lhs_key = cls._entity_key(relation_data["source"])
                rhs_key = cls._entity_key(relation_data["target"])
                if lhs_key not in document_keys or rhs_key not in document_keys:
                    continue
                relation = relation_type(
                    uuid=cls._object_id(
                        "relation",
                        document.document_id,
                        lhs_key,
                        rhs_key,
                        relation_data["content"],
                    ),
                    user_id=library_id,
                    name="related",
                    content=relation_data["content"],
                    lhs=graph_entities[lhs_key],
                    rhs=graph_entities[rhs_key],
                    metadata={
                        "library_id": library_id,
                        "document_id": document.document_id,
                        "path": document.path,
                        "source_version": document.source_version,
                    },
                    language="en",
                    content_embedding=cls._placeholder_vector(),
                ).update_connected_entities()
                graph_relations.append(relation)
        return list(graph_entities.values()), graph_relations, graph_episodes

    @staticmethod
    async def _write_projection(
        store: Any, entities: list[Any], relations: list[Any], episodes: list[Any]
    ) -> None:
        await store.add_entity(entities, flush=False, no_embed=True)
        await store.add_relation(relations, flush=False, no_embed=True)
        await store.add_episode(episodes, flush=False, no_embed=True)
        await store.refresh()

    def _expand_projection(
        self,
        library_id: str,
        seed_documents: set[str],
        max_hops: int,
        limit: int,
    ) -> list[GraphExpansion]:
        library_root = self.root / library_id
        database = library_root / "graph.db"
        marker = library_root / self._PROJECTION_MARKER
        if not database.exists():
            if marker.exists():
                raise GraphAdapterError("graph projection database is missing")
            return []
        if not marker.exists():
            raise GraphAdapterError("graph projection is incomplete")
        store = self._open_projection(database)
        try:
            return asyncio.run(
                self._query_projection(
                    store, library_id, seed_documents, max_hops, limit
                )
            )
        finally:
            store.close()

    @staticmethod
    def _open_projection(database: Path) -> Any:
        try:
            from jiuwen_memory.common.logging.log_config import (
                configure_log_config,
            )

            configure_log_config(
                {
                    "backend": "default",
                    "output": ["console"],
                    "interface_output": ["console"],
                    "performance_output": ["console"],
                }
            )
            from jiuwen_memory.foundation.store.graph import (
                Entity,
                Episode,
                GraphConfig,
                GraphStoreIndexConfig,
                GraphStoreStorageConfig,
                Relation,
            )
            from jiuwen_memory.foundation.store.graph.constants import (  # type: ignore[import-untyped]
                ENTITY_COLLECTION,
                EPISODE_COLLECTION,
                RELATION_COLLECTION,
            )
            from jiuwen_memory.foundation.store.graph.index_field import (  # type: ignore[import-untyped]
                MilvusFLAT,
            )
            from jiuwen_memory.foundation.store.graph.milvus import (  # type: ignore[import-untyped]
                MilvusGraphStore,
            )
            from pymilvus import MilvusClient  # type: ignore[import-untyped]
        except ImportError as error:
            raise GraphAdapterError("JiuwenMemory or Milvus Lite is unavailable") from error

        class MilvusLiteGraphStore(MilvusGraphStore):  # type: ignore[misc]
            """Jiuwen graph store with unsupported Lite database RPCs removed."""

            def __init__(self, config: Any) -> None:
                extras = config.extras.copy()
                self._config = config
                self._embedder = None
                self.alias = extras.setdefault("alias", f"pam-graph-{id(self)}")
                self.client = MilvusClient(
                    uri=config.uri,
                    token=config.token,
                    timeout=config.timeout,
                    **extras,
                )
                self.metric = (
                    config.db_embed_config.distance_metric.replace("dot", "ip")
                    .replace("euclidean", "l2")
                    .upper()
                )
                self.full_text_search_params = MappingProxyType({"metric_type": "BM25"})
                self.dense_search_params = MappingProxyType({"metric_type": self.metric})
                self.field_def = {
                    ENTITY_COLLECTION: [
                        key for key in Entity.model_fields if not key.endswith("_bm25")
                    ],
                    RELATION_COLLECTION: [
                        key for key in Relation.model_fields if not key.endswith("_bm25")
                    ],
                    EPISODE_COLLECTION: [
                        key for key in Episode.model_fields if not key.endswith("_bm25")
                    ],
                }
                self._build_lite_indices()

            def _build_lite_indices(self) -> None:
                from jiuwen_memory.foundation.store.graph.milvus import (
                    generate_milvus_schema,
                )

                original = generate_milvus_schema.icu_analyzer
                original_stopwords = generate_milvus_schema.icu_analyzer_with_stopwords
                standard = {"tokenizer": "standard", "filter": ["lowercase"]}
                try:
                    generate_milvus_schema.icu_analyzer = standard
                    generate_milvus_schema.icu_analyzer_with_stopwords = standard
                    self._build_indices()
                finally:
                    generate_milvus_schema.icu_analyzer = original
                    generate_milvus_schema.icu_analyzer_with_stopwords = original_stopwords

            def rebuild(self) -> None:
                for collection in self.client.list_collections():
                    self.client.drop_collection(collection, timeout=self.config.timeout)
                self._build_lite_indices()

        config = GraphConfig(
            uri=str(database),
            name="default",
            embed_dim=32,
            db_storage_config=GraphStoreStorageConfig(uuid=64, user_id=64),
            db_embed_config=GraphStoreIndexConfig(
                index_type=MilvusFLAT(),
                distance_metric="cosine",
                bm25_analyzer_settings={"tokenizer": "standard", "filter": ["lowercase"]},
            ),
        )
        return MilvusLiteGraphStore(config)

    @classmethod
    async def _query_projection(
        cls,
        store: Any,
        library_id: str,
        seed_documents: set[str],
        max_hops: int,
        limit: int,
    ) -> list[GraphExpansion]:
        from jiuwen_memory.foundation.store.graph.constants import (
            ENTITY_COLLECTION,
            RELATION_COLLECTION,
        )
        from jiuwen_memory.foundation.store.graph.result_ranking import (  # type: ignore[import-untyped]
            WeightedRankConfig,
        )

        if store.is_empty(ENTITY_COLLECTION):
            return []
        entity_count = cls._complete_collection_size(store, ENTITY_COLLECTION)
        rows = await store.query(
            ENTITY_COLLECTION,
            limit=entity_count,
            output_fields=["uuid", "name", "metadata"],
        )
        cls._require_complete_rows(ENTITY_COLLECTION, rows, entity_count)
        entities = {str(row.get("uuid", "")): row for row in rows if row.get("uuid")}
        seed_entities = {
            entity_id
            for entity_id, row in entities.items()
            if any(
                str(source.get("document_id", "")) in seed_documents
                for source in cls._entity_sources(row, library_id)
            )
        }
        if not seed_entities:
            return []
        seed_query = " ".join(
            str(entities[entity_id].get("name", ""))
            for entity_id in sorted(seed_entities)
        )
        searched = await store.search(
            seed_query,
            entity_count,
            ENTITY_COLLECTION,
            WeightedRankConfig(name_dense=1, content_dense=0, content_sparse=0),
            bfs_depth=max_hops,
            bfs_k=entity_count,
            query_embedding=cls._placeholder_vector(),
            output_fields=["uuid", "name", "metadata"],
        )

        raw_candidates = searched.get(ENTITY_COLLECTION, [])
        if not raw_candidates:
            return []
        cls._require_complete_rows(ENTITY_COLLECTION, raw_candidates, entity_count)
        candidate_entities = {
            str(row.get("uuid", ""))
            for row in raw_candidates
            if row.get("uuid") in entities
        }
        if len(candidate_entities) != entity_count:
            raise GraphAdapterError("graph search returned an incomplete entity set")

        distance = {entity_id: 0 for entity_id in seed_entities}
        frontier = set(seed_entities)
        if store.is_empty(RELATION_COLLECTION):
            relations: list[dict[str, object]] = []
        else:
            relation_count = cls._complete_collection_size(store, RELATION_COLLECTION)
            relations = await store.query(
                RELATION_COLLECTION,
                limit=relation_count,
                output_fields=["lhs", "rhs"],
            )
            cls._require_complete_rows(RELATION_COLLECTION, relations, relation_count)
        adjacency: dict[str, set[str]] = {}
        for relation in relations:
            lhs, rhs = relation.get("lhs"), relation.get("rhs")
            if not isinstance(lhs, str) or not isinstance(rhs, str):
                continue
            if lhs not in candidate_entities or rhs not in candidate_entities:
                continue
            adjacency.setdefault(lhs, set()).add(rhs)
            adjacency.setdefault(rhs, set()).add(lhs)
        for depth in range(1, max_hops):
            frontier = {
                neighbor
                for entity_id in frontier
                for neighbor in adjacency.get(entity_id, ())
                if neighbor not in distance
            }
            for entity_id in frontier:
                distance[entity_id] = depth
            if not frontier:
                break

        seen_documents = set(seed_documents)
        results: list[GraphExpansion] = []
        for hop in range(1, max_hops + 1):
            candidates: list[tuple[str, dict[str, str]]] = []
            for entity_id in sorted(
                entity_id for entity_id, depth in distance.items() if depth + 1 == hop
            ):
                row = entities.get(entity_id, {})
                metadata = row.get("metadata")
                anchor = ""
                if isinstance(metadata, dict):
                    anchor = str(metadata.get("entity_key", ""))
                for source in cls._entity_sources(row, library_id):
                    candidates.append((anchor, source))
            for anchor, source in sorted(
                candidates,
                key=lambda item: (item[1]["path"], item[1]["document_id"], item[0]),
            ):
                document_id = source["document_id"]
                if document_id in seen_documents:
                    continue
                seen_documents.add(document_id)
                results.append(
                    GraphExpansion(
                        content=source["content"],
                        document_id=document_id,
                        path=source["path"],
                        source_version=source["source_version"],
                        hop=hop,
                        graph_object_type="entity",
                        source_anchor=anchor,
                    )
                )
                if len(results) >= limit:
                    return results
        return results

    @classmethod
    def _complete_collection_size(cls, store: Any, collection: str) -> int:
        try:
            raw_count = store.client.get_collection_stats(collection).get("row_count")
            count = int(raw_count)
        except (AttributeError, TypeError, ValueError) as error:
            raise GraphAdapterError("graph collection size is unavailable") from error
        if count <= 0:
            raise GraphAdapterError("graph collection size is invalid")
        if count > cls._MILVUS_COMPLETE_READ_LIMIT:
            raise GraphAdapterError(
                f"graph collection exceeds complete read limit: {collection}"
            )
        return count

    @staticmethod
    def _require_complete_rows(
        collection: str, rows: list[dict[str, object]], expected: int
    ) -> None:
        if len(rows) != expected:
            raise GraphAdapterError(
                f"graph collection read was incomplete: {collection}"
            )

    @staticmethod
    def _entity_sources(row: dict[str, object], library_id: str) -> list[dict[str, str]]:
        metadata = row.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("library_id") != library_id:
            return []
        raw_sources = metadata.get("sources")
        if not isinstance(raw_sources, list):
            return []
        sources: list[dict[str, str]] = []
        for raw in raw_sources:
            if not isinstance(raw, dict):
                continue
            document_id = raw.get("document_id")
            path = raw.get("path")
            source_version = raw.get("source_version")
            if not all(
                isinstance(value, str) and value
                for value in (document_id, path, source_version)
            ):
                continue
            sources.append(
                {
                    "document_id": str(document_id),
                    "path": str(path),
                    "source_version": str(source_version),
                    "content": str(raw.get("content", "")),
                }
            )
        return sources

    @staticmethod
    def _entities(payload: dict[str, object]) -> list[dict[str, str]]:
        raw = payload.get("entities", [])
        if not isinstance(raw, list):
            raise GraphAdapterError("graph LLM returned invalid entities")
        result: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                raise GraphAdapterError("graph LLM returned invalid entities")
            name, content = item.get("name"), item.get("content")
            if not isinstance(name, str) or not name.strip():
                raise GraphAdapterError("graph LLM returned an invalid entity name")
            result.append(
                {"name": name.strip()[:500], "content": str(content or name).strip()[:65535]}
            )
        return result

    @staticmethod
    def _relations(payload: dict[str, object]) -> list[dict[str, str]]:
        raw = payload.get("relations", [])
        if not isinstance(raw, list):
            raise GraphAdapterError("graph LLM returned invalid relations")
        result: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                raise GraphAdapterError("graph LLM returned invalid relations")
            source, target, content = item.get("source"), item.get("target"), item.get("content")
            if not all(isinstance(value, str) and value.strip() for value in (source, target)):
                raise GraphAdapterError("graph LLM returned an invalid relation")
            result.append(
                {
                    "source": str(source).strip()[:500],
                    "target": str(target).strip()[:500],
                    "content": str(content or "related").strip()[:65535],
                }
            )
        return result

    @staticmethod
    def _placeholder_vector() -> list[float]:
        return [1.0] + [0.0] * 31

    @staticmethod
    def _entity_key(value: str) -> str:
        normalized = re.sub(r"\s+", " ", value.strip().casefold())
        return normalized[:512]

    @staticmethod
    def _object_id(*parts: str) -> str:
        return hashlib.sha256("\0".join(parts).encode()).hexdigest()
