from __future__ import annotations

import asyncio
import shutil
import threading
import warnings
from pathlib import Path
from typing import Any

import pytest
from pytest import MonkeyPatch

from personal_agent_memory.graph_adapter import (
    GraphAdapterBusyError,
    GraphAdapterError,
    GraphExpansion,
    GraphSourceDocument,
    JiuwenMilvusGraphAdapter,
)
from personal_agent_memory.model_client import ModelEndpoint, OpenAICompatibleClient
from personal_agent_memory.state import PlatformState


class RecordingGraphAdapter:
    def __init__(self) -> None:
        self.documents: dict[str, tuple[GraphSourceDocument, ...]] = {}
        self.expansions: list[GraphExpansion] = []
        self.expand_calls: list[tuple[str, tuple[str, ...], int, int]] = []
        self.fail_search = False
        self.busy_search = False

    async def rebuild(
        self, library_id: str, documents: tuple[GraphSourceDocument, ...]
    ) -> None:
        self.documents[library_id] = documents

    async def expand(
        self,
        library_id: str,
        seed_document_ids: tuple[str, ...],
        *,
        max_hops: int,
        limit: int,
    ) -> list[GraphExpansion]:
        self.expand_calls.append((library_id, seed_document_ids, max_hops, limit))
        if self.busy_search:
            raise GraphAdapterBusyError("deterministic graph mutation")
        if self.fail_search:
            raise GraphAdapterError("deterministic graph outage")
        return [item for item in self.expansions if item.hop <= max_hops][:limit]


class _FakeGraphClient:
    def __init__(self, counts: dict[str, int]) -> None:
        self.counts = counts

    def get_collection_stats(self, collection: str) -> dict[str, int]:
        return {"row_count": self.counts[collection]}


class _FakeGraphStore:
    def __init__(
        self,
        *,
        entities: list[dict[str, object]],
        relations: list[dict[str, object]],
        search_entities: list[dict[str, object]],
        counts: dict[str, int] | None = None,
    ) -> None:
        self.entities = entities
        self.relations = relations
        self.search_entities = search_entities
        normalized_counts = {
            key.upper() if not key.endswith("_COLLECTION") else key: value
            for key, value in (
                counts
                or {
                    "entities": len(entities),
                    "relations": len(relations),
                }
            ).items()
        }
        self.client = _FakeGraphClient(
            {
                "ENTITY_COLLECTION": normalized_counts.get("ENTITIES", 0),
                "RELATION_COLLECTION": normalized_counts.get("RELATIONS", 0),
            }
        )

    def is_empty(self, collection: str) -> bool:
        return self.client.counts[collection] == 0

    async def query(self, collection: str, **kwargs: Any) -> list[dict[str, object]]:
        del kwargs
        return self.entities if collection == "ENTITY_COLLECTION" else self.relations

    async def search(self, *args: Any, **kwargs: Any) -> dict[str, list[dict[str, object]]]:
        del args, kwargs
        return {"ENTITY_COLLECTION": self.search_entities}


async def _wait_for_graph(state: PlatformState, library_id: str) -> None:
    for _ in range(1000):
        status = state.graph_status(library_id)["status"]
        if status in {"ready", "error"}:
            return
        await asyncio.sleep(0.02)
    raise AssertionError("graph projection did not settle")


def test_graph_expansion_is_direct_seeded_source_bounded_and_degrades(tmp_path: Path) -> None:
    async def scenario() -> None:
        library_root = tmp_path / "libraries"
        memory_root = library_root / "project"
        project_root = tmp_path / "project"
        memory_root.mkdir(parents=True)
        project_root.mkdir()
        (memory_root / "seed.md").write_text(
            "# Seed\n\nDirectNeedle belongs to the seed document.\n", encoding="utf-8"
        )
        (memory_root / "one.md").write_text(
            "# One\n\nUnrelated preface.\n\nBeta anchors the one hop source.\n",
            encoding="utf-8",
        )
        (memory_root / "two.md").write_text("# Two\n\nTwo hop source.\n", encoding="utf-8")
        adapter = RecordingGraphAdapter()
        model = OpenAICompatibleClient(
            None,
            None,
            ModelEndpoint("http://127.0.0.1:9", "deterministic-graph", retries=0),
        )
        state = PlatformState(
            tmp_path / "state" / "platform.sqlite3",
            (library_root,),
            model,
            adapter,
        )
        await state.start()
        try:
            library = state.register_library(str(memory_root), "project")
            state.bind_project(str(project_root), library.id)
            await _wait_for_graph(state, library.id)
            documents = {item.path: item for item in adapter.documents[library.id]}
            adapter.expansions = [
                GraphExpansion(
                    "One hop graph content.",
                    documents["one.md"].document_id,
                    "one.md",
                    documents["one.md"].source_version,
                    1,
                    "entity",
                    "beta",
                ),
                GraphExpansion(
                    "Two hop graph content.",
                    documents["two.md"].document_id,
                    "two.md",
                    documents["two.md"].source_version,
                    2,
                    "entity",
                ),
                GraphExpansion("missing", "missing", "missing.md", "missing", 1, "entity"),
            ]

            miss = await state.search_project(str(project_root), "NoSuchDirectHit", 10, 2)
            assert miss["results"] == []
            assert adapter.expand_calls == []

            one_hop = await state.search_project(str(project_root), "DirectNeedle", 10, 1)
            assert [item["path"] for item in one_hop["results"]] == ["seed.md", "one.md"]
            assert one_hop["results"][1]["classification"] == "graph_expansion"
            assert one_hop["results"][1]["graph_hop"] == 1
            assert one_hop["results"][1]["content"] == "Beta anchors the one hop source."
            assert one_hop["results"][1]["heading"] == "One"
            assert one_hop["results"][1]["start_line"] == 5
            assert one_hop["results"][1]["end_line"] == 5
            assert "graph content" not in str(one_hop["results"][1]["content"])

            two_hop = await state.search_project(str(project_root), "DirectNeedle", 10, 2)
            assert [item["path"] for item in two_hop["results"]] == [
                "seed.md",
                "one.md",
                "two.md",
            ]
            assert two_hop["results"][2]["graph_hop"] == 2
            assert max(call[2] for call in adapter.expand_calls) == 2

            adapter.busy_search = True
            busy = await state.search_project(str(project_root), "DirectNeedle", 10, 2)
            assert [item["path"] for item in busy["results"]] == ["seed.md"]
            assert busy["degradation"] == ["graph_unavailable"]
            assert busy["graph_index_status"] == "ready"
            assert state.graph_status(library.id)["status"] == "ready"

            adapter.busy_search = False
            adapter.fail_search = True
            degraded = await state.search_project(str(project_root), "DirectNeedle", 10, 2)
            assert [item["path"] for item in degraded["results"]] == ["seed.md"]
            assert degraded["degradation"] == ["graph_unavailable"]

            adapter.fail_search = False
            (memory_root / "one.md").write_text(
                "# One\n\nChanged authoritative source.\n", encoding="utf-8"
            )
            state.scan_library(library.id)
            stale = await state.search_project(str(project_root), "DirectNeedle", 10, 2)
            assert [item["classification"] for item in stale["results"]] == ["direct"]
            assert stale["graph_index_status"] == "building"
            assert "graph_unavailable" not in stale["degradation"]
        finally:
            await state.close()

    asyncio.run(scenario())


def test_real_adapter_graph_query_fails_fast_while_projection_is_mutating(
    tmp_path: Path,
) -> None:
    adapter = JiuwenMilvusGraphAdapter(
        tmp_path / "graphs", OpenAICompatibleClient(None, None, None)
    )

    acquired = threading.Event()
    release = threading.Event()

    def hold_operation_lock() -> None:
        with adapter._operation_lock:
            acquired.set()
            release.wait(timeout=5)

    holder = threading.Thread(target=hold_operation_lock)
    holder.start()
    assert acquired.wait(timeout=5)
    try:
        with pytest.raises(GraphAdapterBusyError):
            asyncio.run(adapter.expand("library-one", ("seed",), max_hops=1, limit=5))
    finally:
        release.set()
        holder.join(timeout=5)
    assert not holder.is_alive()


def test_real_adapter_releases_milvus_server_after_partial_store_initialization(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    model = OpenAICompatibleClient(
        None,
        None,
        ModelEndpoint("http://127.0.0.1:9", "deterministic-graph", retries=0),
    )
    monkeypatch.setattr(
        model,
        "extract_graph",
        lambda content: {
            "entities": [{"name": "Alpha", "content": content}],
            "relations": [],
        },
    )
    adapter = JiuwenMilvusGraphAdapter(tmp_path / "graphs", model)
    document = GraphSourceDocument("one", "one.md", "v1", "Alpha source")

    from jiuwen_memory.foundation.store.graph.milvus import MilvusGraphStore
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"^pkg_resources is deprecated as an API\.",
            category=UserWarning,
        )
        from milvus_lite.server_manager import server_manager_instance

    attempts = 0

    def fail_index_build(store: object) -> None:
        nonlocal attempts
        del store
        attempts += 1
        raise RuntimeError("injected index build failure")

    monkeypatch.setattr(MilvusGraphStore, "_build_indices", fail_index_build)
    for _ in range(2):
        prepared = asyncio.run(adapter.prepare_rebuild("library-one", (document,)))
        with pytest.raises(RuntimeError, match="injected index build failure"):
            prepared.activate()
        assert server_manager_instance._servers == {}
        assert not list((tmp_path / "graphs").glob(".library-one.replacement-*"))

    assert attempts == 2


def test_projection_write_can_run_from_an_active_event_loop(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    adapter = JiuwenMilvusGraphAdapter(tmp_path / "graphs", object())  # type: ignore[arg-type]
    caller_thread = threading.get_ident()
    write_threads: list[int] = []

    async def record_write(
        store: object,
        entities: list[object],
        relations: list[object],
        episodes: list[object],
    ) -> None:
        del store, entities, relations, episodes
        await asyncio.sleep(0)
        write_threads.append(threading.get_ident())

    monkeypatch.setattr(adapter, "_write_projection", record_write)

    async def scenario() -> None:
        adapter._write_projection_blocking(object(), [], [], [])

    asyncio.run(scenario())

    assert len(write_threads) == 1
    assert write_threads[0] != caller_thread


def test_prepared_rebuild_preserves_official_projection_until_stale_cleanup_succeeds(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    graph_root = tmp_path / "graphs"
    library_id = "library-one"
    projection = graph_root / library_id
    projection.mkdir(parents=True)
    (projection / "graph.db").write_text("old official projection", encoding="utf-8")
    (projection / "projection.ready").write_text("ready\n", encoding="ascii")
    stale_a = graph_root / f".{library_id}.replacement-a"
    stale_b = graph_root / f".{library_id}.replacement-b"
    stale_a.mkdir()
    stale_b.mkdir()
    (stale_a / "stale-a").write_text("disposable", encoding="ascii")
    (stale_b / "stale-b").write_text("retryable", encoding="ascii")
    adapter = JiuwenMilvusGraphAdapter(graph_root, object())  # type: ignore[arg-type]

    def build_replacement(
        temporary_id: str,
        extracted: list[tuple[GraphSourceDocument, dict[str, object]]],
        expected_library_id: str,
    ) -> None:
        del extracted
        assert expected_library_id == library_id
        temporary = graph_root / temporary_id
        temporary.mkdir()
        (temporary / "graph.db").write_text("new official projection", encoding="utf-8")
        (temporary / "projection.ready").write_text("ready\n", encoding="ascii")

    monkeypatch.setattr(adapter, "_replace_projection", build_replacement)
    prepared = asyncio.run(adapter.prepare_rebuild(library_id, ()))
    activated = prepared.activate()
    real_rmtree = shutil.rmtree

    def fail_second_stale(directory: Path, *args: object, **kwargs: object) -> None:
        directory = Path(directory)
        marker = directory / "stale-b"
        if marker.exists():
            marker.unlink()
            raise OSError("injected later stale cleanup failure")
        real_rmtree(directory, *args, **kwargs)

    monkeypatch.setattr("personal_agent_memory.graph_adapter.shutil.rmtree", fail_second_stale)
    with pytest.raises(GraphAdapterError, match="graph projection cleanup failed"):
        activated.commit()
    activated.rollback()

    assert (projection / "graph.db").read_text(encoding="utf-8") == "old official projection"
    assert not stale_a.exists()
    assert (stale_b / "stale-b").read_text(encoding="ascii") == "retryable"
    assert not list(graph_root.glob(f".{library_id}.purge-*"))
    assert not list(graph_root.glob(f".{library_id}.rollback-*"))

    monkeypatch.setattr("personal_agent_memory.graph_adapter.shutil.rmtree", real_rmtree)
    retry = asyncio.run(adapter.prepare_rebuild(library_id, ())).activate()
    retry.commit()
    assert (projection / "graph.db").read_text(encoding="utf-8") == "new official projection"
    assert not list(graph_root.glob(f".{library_id}.replacement-*"))
    assert not list(graph_root.glob(f".{library_id}.purge-*"))


def test_restart_reconciliation_rolls_back_incomplete_staged_rebuild(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    graph_root = tmp_path / "graphs"
    library_id = "library-one"
    cleanup_id = "a" * 32
    projection = graph_root / library_id
    projection.mkdir(parents=True)
    (projection / "graph.db").write_text("old official projection", encoding="utf-8")
    stale = graph_root / f".{library_id}.replacement-stale"
    stale.mkdir()
    (stale / "graph.db").write_text("stale but recoverable", encoding="utf-8")
    adapter = JiuwenMilvusGraphAdapter(graph_root, object())  # type: ignore[arg-type]

    def build_replacement(
        temporary_id: str,
        extracted: list[tuple[GraphSourceDocument, dict[str, object]]],
        expected_library_id: str,
    ) -> None:
        del extracted
        assert expected_library_id == library_id
        temporary = graph_root / temporary_id
        temporary.mkdir()
        (temporary / "graph.db").write_text("uncommitted replacement", encoding="utf-8")

    monkeypatch.setattr(adapter, "_replace_projection", build_replacement)
    staged = adapter.stage_rebuild(library_id, (), cleanup_id=cleanup_id)
    del staged
    assert (projection / "graph.db").read_text(encoding="utf-8") == "uncommitted replacement"

    restarted = JiuwenMilvusGraphAdapter(graph_root, object())  # type: ignore[arg-type]
    restarted.reconcile_staged_purge(library_id, cleanup_id, committed=False)

    assert (projection / "graph.db").read_text(encoding="utf-8") == "old official projection"
    recovered = list(graph_root.glob(f".{library_id}.replacement-recovered-*"))
    assert len(recovered) == 1
    assert (recovered[0] / "graph.db").read_text(encoding="utf-8") == "stale but recoverable"
    assert not list(graph_root.glob(f".{library_id}.purge-{cleanup_id}-*"))


def test_empty_library_projection_is_cleared_by_background_rebuild(tmp_path: Path) -> None:
    async def scenario() -> None:
        library_root = tmp_path / "libraries"
        memory_root = library_root / "project"
        memory_root.mkdir(parents=True)
        note = memory_root / "note.md"
        note.write_text("# Note\n\nEntity: Alpha: Initial source.\n", encoding="utf-8")
        adapter = RecordingGraphAdapter()
        state = PlatformState(
            tmp_path / "state" / "platform.sqlite3",
            (library_root,),
            OpenAICompatibleClient(
                None,
                None,
                ModelEndpoint("http://127.0.0.1:9", "deterministic-graph", retries=0),
            ),
            adapter,
        )
        await state.start()
        try:
            library = state.register_library(str(memory_root), "project")
            await _wait_for_graph(state, library.id)
            assert len(adapter.documents[library.id]) == 1

            note.unlink()
            state.scan_library(library.id)
            assert state.graph_status(library.id)["status"] == "building"
            await _wait_for_graph(state, library.id)

            assert adapter.documents[library.id] == ()
            assert state.graph_status(library.id) == {
                "library_id": library.id,
                "status": "ready",
                "projected_documents": 0,
                "total_documents": 0,
                "last_error": "",
                "updated_at": state.graph_status(library.id)["updated_at"],
            }
        finally:
            await state.close()

    asyncio.run(scenario())


def test_jiuwen_search_results_are_the_only_graph_candidates() -> None:
    seed = {
        "uuid": "alpha",
        "name": "Alpha",
        "metadata": {
            "library_id": "library-one",
            "entity_key": "alpha",
            "sources": [
                {
                    "document_id": "seed",
                    "path": "seed.md",
                    "source_version": "seed-v1",
                    "content": "Seed fact",
                }
            ],
        },
    }
    related = {
        "uuid": "beta",
        "name": "Beta",
        "metadata": {
            "library_id": "library-one",
            "entity_key": "beta",
            "sources": [
                {
                    "document_id": "related",
                    "path": "related.md",
                    "source_version": "related-v1",
                    "content": "Related fact",
                }
            ],
        },
    }
    store = _FakeGraphStore(
        entities=[seed, related],
        relations=[{"lhs": "alpha", "rhs": "beta"}],
        search_entities=[],
    )

    assert (
        asyncio.run(
            JiuwenMilvusGraphAdapter._query_projection(
                store, "library-one", {"seed"}, 2, 10
            )
        )
        == []
    )


@pytest.mark.parametrize(
    ("entities", "counts"),
    [
        ([{"uuid": "alpha"}], {"entities": 2, "relations": 0}),
        (
            [{"uuid": "alpha"}],
            {
                "entities": JiuwenMilvusGraphAdapter._MILVUS_COMPLETE_READ_LIMIT + 1,
                "relations": 0,
            },
        ),
    ],
)
def test_incomplete_graph_reads_fail_explicitly(
    entities: list[dict[str, object]], counts: dict[str, int]
) -> None:
    store = _FakeGraphStore(
        entities=entities,
        relations=[],
        search_entities=entities,
        counts=counts,
    )

    with pytest.raises(GraphAdapterError):
        asyncio.run(
            JiuwenMilvusGraphAdapter._query_projection(
                store, "library-one", {"seed"}, 2, 10
            )
        )


def test_missing_ready_projection_degrades_and_marks_graph_error(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    async def scenario() -> None:
        home = tmp_path / "home"
        library_root = tmp_path / "libraries"
        memory_root = library_root / "project"
        project_root = tmp_path / "project"
        home.mkdir()
        memory_root.mkdir(parents=True)
        project_root.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)
        (memory_root / "seed.md").write_text(
            "# Seed\n\nDirectNeedle belongs to Alpha.\n", encoding="utf-8"
        )
        model = OpenAICompatibleClient(
            None,
            None,
            ModelEndpoint("http://127.0.0.1:9", "deterministic-graph", retries=0),
        )
        monkeypatch.setattr(
            model,
            "extract_graph",
            lambda content: {
                "entities": [{"name": "Alpha", "content": content}],
                "relations": [],
            },
        )
        adapter = JiuwenMilvusGraphAdapter(tmp_path / "graphs", model)
        state = PlatformState(
            tmp_path / "state" / "platform.sqlite3",
            (library_root,),
            model,
            adapter,
        )
        await state.start()
        try:
            library = state.register_library(str(memory_root), "project")
            state.bind_project(str(project_root), library.id)
            await _wait_for_graph(state, library.id)
            database = tmp_path / "graphs" / library.id / "graph.db"
            marker = tmp_path / "graphs" / library.id / adapter._PROJECTION_MARKER
            assert database.is_file()
            assert marker.is_file()
            database.unlink()

            result = await state.search_project(str(project_root), "DirectNeedle", 10, 2)

            assert [item["path"] for item in result["results"]] == ["seed.md"]
            assert result["degradation"] == ["graph_unavailable"]
            assert result["graph_index_status"] == "error"
            assert state.graph_status(library.id)["status"] == "error"
        finally:
            await state.close()

    asyncio.run(scenario())


def test_real_jiuwen_milvus_adapter_projects_and_reads_source_metadata(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    # Milvus Lite 2.5 stores process cache data under HOME. Keep that real backend
    # state inside this test's isolated directory rather than the developer account.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    adapter = JiuwenMilvusGraphAdapter(
        tmp_path / "graphs", OpenAICompatibleClient(None, None, None)
    )
    adapter._replace_projection("never-projected", [])
    assert not (tmp_path / "graphs" / "never-projected" / "graph.db").exists()
    assert asyncio.run(
        adapter.expand("never-projected", ("seed",), max_hops=2, limit=10)
    ) == []
    seed = GraphSourceDocument("seed", "seed.md", "seed-v1", "Seed document")
    related = GraphSourceDocument("related", "related.md", "related-v1", "Related document")
    two_hop = GraphSourceDocument("two", "two.md", "two-v1", "Two hop document")
    adapter._replace_projection(
        "library-one",
        [
            (
                seed,
                {
                    "entities": [
                        {"name": "Alpha", "content": "Alpha in seed"},
                        {"name": "Beta", "content": "Beta in seed"},
                    ],
                    "relations": [
                        {"source": "Alpha", "target": "Beta", "content": "Alpha to Beta"}
                    ],
                },
            ),
            (
                related,
                {
                    "entities": [
                        {"name": "Beta", "content": "Authoritative related fact"},
                        {"name": "Gamma", "content": "Gamma relation anchor"},
                    ],
                    "relations": [
                        {"source": "Beta", "target": "Gamma", "content": "Beta to Gamma"}
                    ],
                },
            ),
            (
                two_hop,
                {
                    "entities": [{"name": "Gamma", "content": "Two hop fact"}],
                    "relations": [],
                },
            ),
        ],
    )

    one_hop = asyncio.run(adapter.expand("library-one", ("seed",), max_hops=1, limit=10))
    two_hops = asyncio.run(adapter.expand("library-one", ("seed",), max_hops=2, limit=10))
    capped = asyncio.run(adapter.expand("library-one", ("seed",), max_hops=99, limit=10))

    assert one_hop == [
        GraphExpansion(
            "Authoritative related fact",
            "related",
            "related.md",
            "related-v1",
            1,
            "entity",
            "beta",
        )
    ]
    assert [(item.path, item.hop) for item in two_hops] == [
        ("related.md", 1),
        ("two.md", 2),
    ]
    assert capped == two_hops
    assert (tmp_path / "graphs" / "library-one" / "graph.db").is_file()
    from milvus_lite.server_manager import server_manager_instance

    assert server_manager_instance._servers == {}

    isolated = GraphSourceDocument(
        "isolated", "isolated.md", "isolated-v1", "Other library document"
    )
    adapter._replace_projection(
        "library-two",
        [
            (
                isolated,
                {
                    "entities": [{"name": "Beta", "content": "Other library Beta"}],
                    "relations": [],
                },
            )
        ],
    )
    assert asyncio.run(
        adapter.expand("library-two", ("seed",), max_hops=2, limit=10)
    ) == []
    assert (tmp_path / "graphs" / "library-two" / "graph.db").is_file()

    related_v2 = GraphSourceDocument("related", "related.md", "related-v2", "Changed source")
    adapter._replace_projection(
        "library-one",
        [
            (
                seed,
                {
                    "entities": [{"name": "Beta", "content": "Beta in seed"}],
                    "relations": [],
                },
            ),
            (
                related_v2,
                {
                    "entities": [{"name": "Beta", "content": "Updated related fact"}],
                    "relations": [],
                },
            ),
        ],
    )

    rebuilt = asyncio.run(adapter.expand("library-one", ("seed",), max_hops=1, limit=10))
    assert [(item.content, item.source_version) for item in rebuilt] == [
        ("Updated related fact", "related-v2")
    ]

    adapter._replace_projection("library-one", [])
    assert asyncio.run(
        adapter.expand("library-one", ("seed",), max_hops=2, limit=10)
    ) == []
    assert not (tmp_path / "logs").exists()
