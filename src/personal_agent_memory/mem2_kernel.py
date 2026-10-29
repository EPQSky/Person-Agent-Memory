"""JiuwenMemory 0.2.0 runtime adapter for the public memory kernel port."""

from __future__ import annotations

import asyncio
import base64
import difflib
import hashlib
import json
import os
import re
import sqlite3
import stat
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, TypedDict, cast

from jiuwen_memory.api import (
    Context,
    DisclosureLevel,
    MemoryRuntime,
    MemoryUnit,
    Modality,
    RequestSecurityContext,
    Scope,
    assemble_runtime,
    legacy_request_context,
)
from jiuwen_memory.common.audit.base import AuditLogger, AuditProducer
from jiuwen_memory.common.embedder.base import Embedder, EmbedderProducer
from jiuwen_memory.common.embedder.embedder_impl.openai_embedder import OpenAIEmbedder
from jiuwen_memory.common.type_def import (
    FilterClause,
    FilterExpr,
    FilterGroup,
    FilterLogic,
    FilterOp,
    memory_key,
)
from jiuwen_memory.common.type_def.memory_codec import dumps
from jiuwen_memory.construction import EvolveMode
from jiuwen_memory.control import (
    DeleteMode,
    DeleteSelector,
    JobStatus,
    MemoryPatch,
    UpdateMode,
)
from jiuwen_memory.storage import FulltextStore, GraphStore, VectorStore
from jiuwen_memory.storage.fulltext import FulltextProducer
from jiuwen_memory.storage.graph import GraphProducer
from jiuwen_memory.storage.kv import load_units
from jiuwen_memory.storage.types import ScoredID, TextQuery
from jiuwen_memory.storage.vector import VectorProducer

from personal_agent_memory.config import MemoryKernelKind
from personal_agent_memory.entity_graph import DETAIL_DEFAULT_LIMIT, EntityGraphStore
from personal_agent_memory.memory_kernel import (
    CWD_SCOPE_PREFIX,
    FORMAL_MEMORY_TYPES,
    LIBRARY_SCOPE_PREFIX,
    AssociateWorktreeRegionRequest,
    AuditMemoryRequest,
    BindProjectRegionRequest,
    CaptureContractError,
    CaptureEventsRequest,
    CaptureMemoryRequest,
    CaptureRoundsRequest,
    CreateMemoryRegionRequest,
    CreateMemoryRequest,
    DeleteMemoryRequest,
    DeleteRegionBindingRequest,
    DocumentInspectRequest,
    DocumentUpdateRequest,
    EvolveMemoryRequest,
    GetMemoryRegionRequest,
    HistoryDiffRequest,
    KernelHealth,
    ListDocumentsRequest,
    ListForgottenRequest,
    ListMemoriesRequest,
    MemoryGovernanceError,
    MemoryIdempotencyConflictError,
    MemoryIdempotencyPendingError,
    MemoryKernelUnavailableError,
    MemoryLifecycleConflictError,
    MemoryRecordNotFoundError,
    MemoryRegionTreeRequest,
    MemoryRevisionConflictError,
    MemoryUpdateRequest,
    MemoryWriteRequest,
    PreviewDocumentDeleteRequest,
    PreviewDocumentUpdateRequest,
    PromotionRunRequest,
    PromotionStatusRequest,
    ResolveRegionRequest,
    RestoreDocumentRequest,
    RestoreForgottenRequest,
    RetentionCleanupRequest,
    RetentionPolicyRequest,
    SearchMemoryRequest,
    UnavailableMemoryKernel,
    UpdateRegionBindingRequest,
    UpdateRetentionPolicyRequest,
    validate_library_id,
)
from personal_agent_memory.model_client import ModelEndpoint, OpenAICompatibleClient
from personal_agent_memory.regions import (
    MAX_SESSION_ID_LENGTH,
    MemoryRegion,
    RegionBindingError,
    RegionError,
    RegionRegistry,
)
from personal_agent_memory.sensitive import inspect_sensitive_text
from personal_agent_memory.storage import (
    MilvusLiteFulltextStore,
    MilvusLiteGraphStore,
    MilvusLiteVectorStore,
    SQLiteAuditLogger,
    SQLiteKVStore,
)

_STORE_TARGET = "personal-agent-memory-local"
_AUDIT_TARGET = "personal-agent-memory-sqlite"
_EMBEDDER_TARGET = "personal-agent-memory-openai"
_GRAPH_PROJECTION_SCHEMA = 2
_GRAPH_PROJECTION_MANIFEST_SCHEMA = 1
_PROMOTION_SCHEMA_VERSION = 2
_PROMOTION_MIGRATION_BATCH_SIZE = 100
_PROMOTION_RECOVERY_BATCH_SIZE = 100
_BROWSER_INDEX_RETRY_BATCH_SIZE = 100
_BROWSER_INDEX_RETRY_MAX = 1_000
_LINEAGE_MIGRATION_BATCH_SIZE = 100
PromotionClassifier = Callable[[str, Mapping[str, object]], Mapping[str, object]]
SemanticExtractor = Callable[[dict[str, object]], list[dict[str, object]]]
EntityGraphExtractor = Callable[
    [dict[str, object]], Mapping[str, object] | list[dict[str, object]]
]
SemanticTask = Callable[[], None]


class SemanticTaskExecutor(Protocol):
    def submit(self, task_key: str, region_id: str, task: SemanticTask) -> bool: ...

    def drain(self) -> None: ...

    def close(self) -> None: ...

    @property
    def worker_count(self) -> int: ...

    @property
    def pending_count(self) -> int: ...


class BoundedSemanticExecutor:
    """Fixed worker pool with a bounded pending queue and per-region limits."""

    def __init__(
        self,
        *,
        max_workers: int = 2,
        max_pending: int = 256,
        max_region_workers: int = 1,
    ) -> None:
        if max_workers < 1:
            raise ValueError("semantic max_workers must be positive")
        if max_pending < 1:
            raise ValueError("semantic max_pending must be positive")
        if max_region_workers < 1 or max_region_workers > max_workers:
            raise ValueError("semantic max_region_workers must be between 1 and max_workers")
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="semantic-distillation",
        )
        self._max_workers = max_workers
        self._max_pending = max_pending
        self._max_region_workers = max_region_workers
        self._condition = threading.Condition()
        self._pending = 0
        self._closed = False
        self._region_semaphores: dict[str, threading.BoundedSemaphore] = {}
        self._on_slot_available: Callable[[], None] | None = None

    def set_slot_callback(self, callback: Callable[[], None]) -> None:
        self._on_slot_available = callback

    @property
    def worker_count(self) -> int:
        return self._max_workers

    @property
    def pending_count(self) -> int:
        with self._condition:
            return self._pending

    def _region_semaphore(self, region_id: str) -> threading.BoundedSemaphore:
        with self._condition:
            semaphore = self._region_semaphores.get(region_id)
            if semaphore is None:
                semaphore = threading.BoundedSemaphore(self._max_region_workers)
                self._region_semaphores[region_id] = semaphore
            return semaphore

    def submit(self, task_key: str, region_id: str, task: SemanticTask) -> bool:
        del task_key
        with self._condition:
            if self._closed or self._pending >= self._max_pending:
                return False
            self._pending += 1
        region_semaphore = self._region_semaphore(region_id)

        def run() -> None:
            try:
                with region_semaphore:
                    task()
            finally:
                with self._condition:
                    self._pending -= 1
                    self._condition.notify_all()
                callback = self._on_slot_available
                if callback is not None:
                    with suppress(Exception):
                        callback()

        try:
            self._pool.submit(run)
        except RuntimeError:
            with self._condition:
                self._pending -= 1
                self._condition.notify_all()
            return False
        return True

    def drain(self) -> None:
        with self._condition:
            while self._pending:
                self._condition.wait()

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
        self._pool.shutdown(wait=True, cancel_futures=False)


class _PromotionCandidate(TypedDict):
    region: MemoryRegion
    scope: Scope
    unit: MemoryUnit
    source_units: tuple[MemoryUnit, ...]
    explicit: bool


class _PromotionDecision(TypedDict):
    category: str
    confidence: float
    explicit: bool
    topic_id: str | None
    claim_subject: str
    claim_value: str
    reason: str
    status: str


class _PromotionOutcome(TypedDict):
    created: bool
    decision: dict[str, object]


class _SemanticValidation(TypedDict):
    content: str
    memory_type: str
    confidence: float
    region_hint: str
    target_region: MemoryRegion
    user_event_id: str
    evidence_snippet: str
    user_occurred_at: str
    claim_subject: str
    claim_value: str
    claims_explicit: bool
    durability: str
    target_fallback_reason: str


_PromotionClassified = tuple[_PromotionCandidate, _PromotionDecision, MemoryRegion]


_LOCAL_ORG = "local"
_LOCAL_USER = "local"
_LOCAL_AGENT = "personal-agent-memory"
_CODEX_AGENT = "codex"
_SEARCH_AGENTS = (_LOCAL_AGENT, _CODEX_AGENT)
_DEFAULT_EMBEDDING_DIM = 1536
_VECTOR_MIN_SIMILARITY = 0.5
_FULLTEXT_CONTENT_LIMIT = 65_535
_EVOLVE_CHUNK_LIMIT = 60_000
_STATE_VERSION = "jiuwen-memory-0.2.0"
_DEFAULT_SESSION_RETENTION_DAYS = 30
_MAX_RECALL_CANDIDATE_POOL = 500
_SEMANTIC_MIN_CONFIDENCE = 0.90
_SEMANTIC_MAX_RESULTS = 20
_SEMANTIC_PROJECT_SCOPE = re.compile(
    r"(?:(?:this|our|my)\s+(?:project|repository|repo|workspace|codebase)\b|"
    r"the\s+(?:current\s+)?(?:project|repository|repo|workspace|codebase)\b|"
    r"the\s+user(?:['’])s\s+(?:project|repository|repo|workspace|codebase)\b|"
    r"(?:^|[.!?。！？]\s+)(?:project|repository|repo|workspace|codebase)\b|"
    r"(?:本项目|当前项目|这个项目|此项目|我的项目|该项目|"
    r"本仓库|当前仓库|我的仓库|该仓库|"
    r"本工作区|当前工作区|用户的项目|用户的仓库|用户的工作区|用户的代码库|"
    r"(?:^|[。！？.!?]\s*)(?:项目|仓库|工作区|代码库)))",
    re.IGNORECASE,
)
_SEMANTIC_USER_SCOPE_PREFIX = re.compile(
    r"^(?:(?:i|we)\s+(?:prefer|like|want|need|require|must)\b|"
    r"the\s+user\s+(?:prefers|likes|wants|needs|requires)\b|"
    r"(?:我|我们)(?:偏好|喜欢|希望|想要|需要|要求)|"
    r"用户(?:偏好|喜欢|希望|需要|要求))",
    re.IGNORECASE,
)
_SEMANTIC_SENSITIVE_CLAIM_FIELD = re.compile(
    r"(?<![\w-])(?:credential|credentials|api[\s_-]*key|access[\s_-]*token|"
    r"client[\s_-]*secret|private[\s_-]*key|password|passwd|passcode|"
    r"secret|token)(?![\w-])",
    re.IGNORECASE,
)
_SEMANTIC_DEICTIC_REFERENCE = re.compile(
    r"上面|上述|前述|刚才|刚刚|上一条|前一条|此前对话|前文|"
    r"如上所述|如前所述|如前文所述|正如前面|"
    r"\b(?:above(?![-\w])|earlier(?![-\w])|just now|previous message|previous turn|"
    r"as discussed|as mentioned|as previously stated|as stated previously|"
    r"as (?:i|we) (?:previously )?(?:said|stated|mentioned|described)|"
    r"like (?:i|we) said(?: (?:before|earlier|previously))?|"
    r"as noted(?: (?:before|earlier|previously|above))?|"
    r"as (?:i|we) (?:agreed|discussed)(?: (?:before|earlier|previously))?|"
    r"per (?:our|my|the) "
    r"(?:prior|previous|earlier|last|latest|recent|most\s+recent) "
    r"(?:discussion|conversation|exchange|message|note)|"
    r"(?:based on|following) (?:our|my|the) "
    r"(?:prior|previous|earlier|last|latest|recent|most\s+recent) "
    r"(?:discussion|conversation|exchange|message|note)|"
    r"(?:as|like) (?:i|we) (?:said|stated|mentioned|described) "
    r"(?:before|earlier|previously|above)|"
    r"as noted (?:before|earlier|previously|above)|"
    r"previously stated|as described (?:before|above|earlier)|"
    r"described (?:before|above|earlier)|previous answer|prior answer|earlier answer|"
    r"former|latter|aforementioned|aforesaid|"
    r"prior\s+(?:option|choice|item|approach|statement|answer|message|turn|claim)|"
    r"(?:previous|first|second|third|last|next)\s+(?:option|one|choice|item|approach|"
    r"statement|answer|message|turn|claim|plan|policy|iteration|release|version|"
    r"attempt|round|step)|"
    r"according to (?:the )?(?:previous|earlier|above) (?:answer|message|turn))\b|"
    r"前者|后者|上一轮|前一轮|上次|此前一轮|依照之前说明|按照之前说明|"
    r"根据之前说明|如之前所述",
    re.IGNORECASE,
)
_SEMANTIC_QUESTION_MARK = re.compile(r"[?？]\s*$")
_SEMANTIC_YES_NO_QUESTION = re.compile(
    r"^(?:is|are|was|were|am|do|does|did|can|could|would|should|will|"
    r"shall|may|might|must|have|has|had)\b",
    re.IGNORECASE,
)
_SEMANTIC_WH_QUESTION = re.compile(
    r"^(?:what|how|why|which|who|whom|whose|when|where)\b"
    r"(?:\s+(?:much|many|long|often|soon|far|old|large|big))?"
    r"\s+(?:is|are|was|were|do|does|did|can|could|would|should|will|"
    r"shall|may|might|must|has|have|had)\b",
    re.IGNORECASE,
)
_SEMANTIC_ZH_WH_CLAUSE = re.compile(
    r"^(?:是否|能否|要不要|是不是|如何|怎么|怎样|为什么|为何|哪个|哪些|谁|"
    r"何时|什么时候|哪里|哪儿|多少|多久)"
)
_SEMANTIC_WH_MODAL_QUESTION = re.compile(
    r"^(?:what|how|why|which|who|whom|whose|when|where)\b.*\b"
    r"(?:should|can|could|would|will|do|does|did|must|may|might)\b",
    re.IGNORECASE,
)
_SEMANTIC_WH_STATEMENT_PREDICATE = re.compile(
    r"^(?:what|how|why|which|who|whom|whose|when|where)\b.*\b"
    r"(?:must|should|can|could|will|would|is|are|was|were)\s+"
    r"(?:be\s+)?(?:documented|stored|supported|required|used|configured|"
    r"retained|logged|enforced|chosen|selected|defined|decided)\b",
    re.IGNORECASE,
)
_SEMANTIC_WH_PASSIVE_QUESTION = re.compile(
    r"^(?:what|which|who|whom|whose|where|when|why|how)\s+"
    r"(?!(?:should|can|could|would|will|do|does|did|must|may|might|is|are|was|were)\b)"
    r"(?P<subject>(?:[a-z][a-z-]*\s+){1,3})(?:is|are|was|were)\s+"
    r"(?:[a-z]+ed|used|stored|supported|required|"
    r"configured|retained|logged|enforced|chosen|selected|defined|decided)\b",
    re.IGNORECASE,
)
_SEMANTIC_WH_CLAUSE_STATEMENT = re.compile(
    r"^(?:"
    r"what\s+(?:caused|causes|matters|mattered|happened|happens|failed|fails|"
    r"changed|changes|broke|breaks)\b|"
    r"(?:what|how|why|which)\s+(?:we|they|i|you)\s+[a-z][a-z-]*\b|"
    r"(?:what|how|why|which)\s+(?:the|our|their|your|my)\s+"
    r"(?:[a-z][a-z-]*\s+){0,2}(?:team|project|organization|company)\s+"
    r"[a-z][a-z-]*\b"
    r").*\b(?:is|are|was|were)\s+(?:be\s+)?[a-z]+ed\b",
    re.IGNORECASE,
)
_SEMANTIC_WH_SHORT_PASSIVE_QUESTION = re.compile(
    r"^(?:what|which|where|when|why|how)\s+"
    r"(?!(?:should|can|could|would|will|do|does|did|must|may|might)\b)"
    r"(?:[a-z][a-z'\-]*\s+){1,64}"
    r"(?:is|are|was|were|have|has|had|isn't|aren't|wasn't|weren't|"
    r"haven't|hasn't|hadn't)\s+"
    r"(?:(?:(?:been|being|not|never)\s+|not\s+yet\s+)){0,3}"
    r"(?:(?:[a-z]+ly)(?:[,\s]+(?:and\s+)?[a-z]+ly){0,5}\s+){0,3}"
    r"(?:[a-z]+ed|stored|created|kept|used|configured|logged|retained|"
    r"supported|required|enforced|selected|defined)\b",
    re.IGNORECASE,
)
_SEMANTIC_ZH_QUESTION_ANY = re.compile(
    r"什么|怎么|怎样|如何|为什么|为何|哪个|哪些|谁|多少|多久|吗|呢"
)
_SEMANTIC_ZH_DECLARATIVE = re.compile(
    r"取决于|说明|记录|需要|必须|应该|支持|使用|采用|通过|用于|已|"
    r"将(?:在|于|会)|会(?:在|被|使用)|由[^。！？?]{0,20}决定|"
    r"尚未|尚无|确定|结论"
)
_SEMANTIC_REQUEST = re.compile(
    r"^(?:帮我|请(?:问|帮我|告诉我)|please\b|help me\b|"
    r"(?:tell|show|give|send|provide|share|walk|guide|take|talk)\s+(?:me|us)\b|"
    r"(?:walk|guide|take|talk)\s+(?:me|us)\s+through\b|"
    r"(?:i(?:['’]d\s+like|\s+(?:want|need|would\s+like))\s+you\s+to)\b|"
    r"can you\b|could you\b|would you\b|would you mind\b)",
    re.IGNORECASE,
)
_SEMANTIC_LEADING_DEMONSTRATIVE = re.compile(
    r"^(?:这|那)(?!(?:个|项|条|份|套|种|类|位|名|些|批|组|篇|款|版|段|次|场|台|家|座|所))"
)
_SEMANTIC_TRANSIENT = re.compile(
    r"临时计划|一次性|下一步|稍后|今天先|暂时先|待办|明天|下周|下个月|"
    r"将来|将要|即将|(?:我们|我|团队|项目组)计划|计划(?:在|于)|准备(?:在|于)?|"
    r"\b(?:temporary plan|for now|next step|one[- ]off|todo|tomorrow|"
    r"next\s+(?:week|month|quarter|sprint|release|year)|will\s+[a-z]+"
    r"(?:\s+[a-z]+){0,3}\s+(?:next\s+year|soon|later|eventually)\b|"
    r"will\s+(?:be\s+)?(?:deploy|ship|"
    r"release|start|finish|run|do|deployed|shipped|released)|going to|"
    r"(?:i|we|they|team|project)\s+(?:plan|plans|planned|planning|intend|intends|"
    r"intended|intending|expect|expects|expected|expecting)\s+to\b|"
    r"(?:i|we|they|team)\s+plan\b|"
    r"in\s+q[1-4]\b|(?:starts?|begins?|launches?|ships?|deploys?|rolls?\s+out|"
    r"takes?\s+place|happens?|occurs?)\s+in\s+"
    r"(?:january|february|march|april|may|june|july|august|september|october|"
    r"november|december|20\d{2})\b|will\s+[a-z]+(?:\s+[a-z]+){0,2}\s+in\s+"
    r"(?:january|february|march|april|may|june|july|august|september|october|"
    r"november|december|20\d{2})\b|"
    r"(?:planned|scheduled|expected|slated|set|targeted|destined|due)\s+"
    r"(?:for|on|in)\s+\w+|"
    r"(?:planned|slated|set|due|targeted|destined)\s+to\s+"
    r"(?:happen|occur|take\s+place|deploy|launch|ship|start|finish|go\s+live)\b|"
    r"(?:as soon as possible|asap)\b|"
    r"(?:planned|scheduled|expected)\s+"
    r"(?:soon|shortly|imminently|later|eventually|tomorrow|today|tonight|"
    r"this\s+(?:morning|afternoon|evening|week|month|quarter)|"
    r"in\s+(?:a\s+few|\d+)\s+(?:minutes?|hours?|days?|weeks?))\b|"
    r"(?:today|tonight|this\s+(?:morning|afternoon|evening|week|month|quarter))\b|"
    r"scheduled (?:for|to)|"
    r"once\s+.+\s+(?:passes|completes|finishes|lands))\b",
    re.IGNORECASE,
)
_SEMANTIC_EN_PRONOUN = re.compile(
    r"\b(?:it|them|they|he|she|him|her)\b",
    re.IGNORECASE,
)
_SEMANTIC_EN_DEMONSTRATIVE = re.compile(
    r"\b(?:this|that|these|those)\b",
    re.IGNORECASE,
)
_SEMANTIC_EN_SUBORDINATOR = re.compile(
    r"\b(?:because|since|when|whenever|while|if|unless|although|though|"
    r"whereas|after|before|once|so\s+that|in\s+case|as\s+soon\s+as|"
    r"which|who|whom|whose|where|that)\b",
    re.IGNORECASE,
)
_SEMANTIC_ZH_PRONOUN = re.compile(
    r"(?<!其)(?:它|他们|她们)|"
    r"(?:依赖|基于|使用|需要|支持|包含|排除|针对|对于)(?:它|他们|她们)"
)
_SEMANTIC_ZH_SUBORDINATOR = re.compile(
    r"因为|由于|如果|若是|若|当|虽然|尽管|即使|一旦|只要|以便|从而"
)
_SEMANTIC_ZH_NOUN_PREFIX = (
    "(?:个|项|条|份|套|种|类|位|名|些|批|组|篇|款|版|段|次|场|台|家|座|所)"
    "(?:项目|服务|系统|模块|接口|数据库|政策|策略|版本|配置|文件|仓库|"
    "用户|请求|事件|回合|记忆|索引|表|库|工具|插件|前端|后端|平台|方案|"
    "流程|规则|约束|设置|选项|功能|特性|资源|主题|区域|字段|参数|时间|"
    "周期|频率|公司|报告|步骤|结论|部署|团队)"
)
_SEMANTIC_AUXILIARIES = frozenset(
    {
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "can",
        "could",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "has",
        "have",
        "had",
    }
)
_SEMANTIC_SUBJECT_CLOSED_CLASS = frozenset(
    {
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "it",
        "they",
        "them",
        "we",
        "i",
        "you",
        "he",
        "she",
        "his",
        "her",
        "its",
        "their",
        "our",
        "my",
        "your",
        "all",
        "both",
        "each",
        "every",
        "some",
        "any",
        "no",
    }
)
_SEMANTIC_DEMONSTRATIVE_BASE_VERBS = frozenset(
    {
        "use", "uses", "have", "has", "do", "does", "provide", "provides",
        "support", "supports", "require", "requires", "need", "needs",
        "contain", "contains", "store", "stores", "run", "runs", "return", "returns",
        "write", "writes", "read", "reads", "allow", "allows", "mean", "means",
        "enable", "enables", "accept", "accepts", "include", "includes",
        "exclude", "excludes", "time", "times", "prefer", "prefers",
        "cache", "caches", "validate", "validates", "encrypt", "encrypts",
        "fetch", "fetches", "process", "processes", "authenticate", "authenticates",
        "forward", "forwards", "serve", "serves", "enforce", "enforces",
        "perform", "performs", "employ", "employs", "handle", "handles",
        "like", "likes", "depend", "depends", "rely", "relies", "cause", "causes",
        "help", "helps", "reduce", "reduces", "fail", "fails", "become", "becomes",
        "deploy", "deploys", "migrate", "migrates",
        "implement", "implements", "create", "creates", "update", "updates",
        "delete", "deletes", "manage", "manages", "keep", "keeps",
        "can", "could", "will", "would", "should", "may", "might", "must",
    }
)
_SEMANTIC_ADVERB_CLOSED_CLASS = frozenset(
    {
        "also",
        "then",
        "now",
        "still",
        "always",
        "never",
        "only",
        "just",
        "even",
        "often",
        "usually",
        "typically",
        "generally",
        "frequently",
        "automatically",
        "currently",
        "already",
        "again",
    }
)
_SEMANTIC_EN_COORDINATION = re.compile(
    r"\b(?:and|or|but|as\s+well\s+as)\b", re.IGNORECASE
)
_SEMANTIC_ZH_COORDINATION = re.compile(
    r"并且|而且|同时|，且|,且|(?<![合吞兼])并(?!且|发|行|列|集|轨|网|购|非)"
)
_SEMANTIC_CLAUSE_BOUNDARY = re.compile(r"[.,;:!?。；！？、]")
_SEMANTIC_EN_PREPOSITION = frozenset(
    {
        "on",
        "in",
        "to",
        "for",
        "with",
        "about",
        "from",
        "into",
        "over",
        "under",
        "around",
        "by",
        "at",
        "per",
        "via",
        "as",
        "of",
    }
)
_SEMANTIC_NON_DURABLE_INTENT = re.compile(
    r"\b(?:considering|evaluating|thinking about|weighing|exploring)\b|"
    r"\b(?:may|might)\s+(?:be|become|use|prefer|need|change|move|adopt|choose|"
    r"migrate|replace|introduce|drop)\b|"
    r"\b(?:need|trying) to (?:choose|decide)\b|"
    r"(?:正在)?(?:考虑|评估|权衡|比较|探索)|需要(?:选择|决定)|"
    r"(?:可能|也许|或许)(?:使用|采用|选择|需要|偏好)|"
    r"尚未(?:选择|决定)",
    re.IGNORECASE,
)
_SEMANTIC_IMPERATIVE = re.compile(
    r"^(?:deploy|run|build|test|apply|restart|migrate|install|update|remove|"
    r"delete|create|configure|enable|disable|use(?!\s+of\b)|prefer|choose|adopt|"
    r"set(?!\s+of\b)|change|add|write(?!\s+(?:access|permission)\b)|"
    r"read(?!\s+(?:access|only)\b)|keep|make|ensure|note|remember|avoid|require|"
    r"containerize|dockerize|provision|virtualize|refactor|optimize|publish|"
    r"upload|download|copy|move|validate|verify|check|monitor|integrate|implement|"
    r"rename|replace|convert|rewrite|switch|send|receive|start|stop|open|close|"
    r"load|store|fetch|post|put|patch|invoke|trigger|schedule|wait|help|give|"
    r"review|analyze|analyse|assess|examine|inspect|audit|evaluate|design|document|"
    r"explain|describe|clarify|summarize|summarise|list|show|tell|"
    r"walk|guide|take|talk|provide|share)\b",
    re.IGNORECASE,
)
_SEMANTIC_FUTURE_ACTION = re.compile(
    r"\bwill\s+[a-z]+\b.*\b(?:at|by|after|before|tomorrow|tonight|"
    r"next\s+(?:week|month|quarter|sprint|release)|midnight|noon)\b",
    re.IGNORECASE,
)
_SEMANTIC_ACTION_WITHOUT_DURABILITY = re.compile(
    r"\b(?:i|we|they|team|service|system|project|user)\s+"
    r"(?:is|are|am|was|were|will|shall)\s+(?:be\s+)?[a-z]+ing\b|"
    r"\b(?:is|are)\s+(?:going to|about to|due to)\b|"
    r"\b(?:on|by|before|after)\s+(?:monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|the weekend)\b|"
    r"正在|即将|将要|计划(?:在|于)|准备(?:在|于)|下(?:周|个月|个季度)",
    re.IGNORECASE,
)
_SEMANTIC_COMPOUND_CLAIM = re.compile(
    r"[,，]\s*(?:and|but)\s+(?:the\s+|this\s+|that\s+|it\s+|they\s+|we\s+|i\s+)?"
    r"(?:requires?|prefers?|uses?|needs?|must|will|should|is|are|was|were)\b|"
    r"\band\s+(?:the\s+|this\s+|that\s+|it\s+|they\s+|we\s+|i\s+)?"
    r"(?:requires?|prefers?|uses?|needs?|must|will|should|is|are|was|were)\b|"
    r"^(?:the\s+)?user\s+(?:prefers?|likes?|requires?|needs?|wants?)\s+"
    r"[^.!?;]+,\s*[^.!?;]+,\s*(?:and|or)\s+[^.!?;]+[.!?]?$|"
    r"^(?:the\s+)?(?:user|i|we)\s+(?:prefers?|likes?|wants?|needs?|requires?)\s+"
    r"[^.!?;]+?\s+(?:and|or|as\s+well\s+as)\s+[^.!?;]+[.!?]?$|"
    r"^用户(?:偏好|喜欢|要求|需要|希望)[^。！？；]+[、，]"
    r"[^。！？；]+(?:和|以及|、)[^。！？；]+[。！？]?$|"
    r"^(?:用户|我)(?:偏好|喜欢|希望|需要|要求)[^。！？；]+?"
    r"(?:和|与|以及)[^。！？；]+[。！？]?$|"
    r"并且|而且|同时|，且|,且|且(?:要求|需要|偏好|使用|采用|支持|禁止|必须|会|将)",
    re.IGNORECASE,
)
_SEMANTIC_COORDINATED_PREDICATE = re.compile(
    r"\b(?:is|are|was|were|seems?|remains?)\s+(?:also\s+)?"
    r"[a-z][a-z-]*(?:\s*,\s*[a-z][a-z-]*)*,?\s+"
    r"(?:and|or|but|yet|as\s+well\s+as)\s+"
    r"(?:also\s+)?[a-z][a-z-]*\b",
    re.IGNORECASE,
)
_SEMANTIC_SUBORDINATING_CLAIM = re.compile(
    r"\b(?:whereas|although|though)\b|"
    r"\bwhile\s+(?:the|a|an|this|that|these|those|it|they|we|i|he|she)\b",
    re.IGNORECASE,
)
_SEMANTIC_COORDINATED_MODIFIER = re.compile(
    r"^(?:the\s+)?(?:user|i|we)\s+(?:prefers?|likes?|wants?|needs?|requires?|uses?)\s+"
    r"[a-z][a-z-]*\s+(?:and|or)\s+[a-z][a-z-]*\s+[a-z][a-z-]*[.!?]?$",
    re.IGNORECASE,
)
_SEMANTIC_NON_SEMANTIC_OUTPUT = re.compile(
    r"隐藏推理|内部推理|工具原始输出|命令输出|标准输出|命令(?:执行|返回|结果)|"
    r"\b(?:hidden reasoning|chain of thought|tool output|stdout|stderr|shell output|"
    r"exit (?:code|status)|the command (?:printed|returned|output|failed|succeeded)|"
    r"traceback|assertionerror|failed\s+tests?/\S+|error\s+\S+\.py|"
    r"(?:passed|failed|skipped)\b[^\n]{0,60}\b(?:tests?|specs?)/\S+|"
    r"(?:tests?|specs?)\b[^\n]{0,40}\b(?:passed|failed))\b|"
    r"测试失败|断言失败",
    re.IGNORECASE,
)
_SESSION_RETENTION_SCHEMA = """CREATE TABLE IF NOT EXISTS session_retention_policy (
    region_id TEXT PRIMARY KEY,
    retention_days INTEGER NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)"""
_CAPTURE_EVOLUTIONS_SCHEMA = """CREATE TABLE IF NOT EXISTS capture_evolutions (
    evolve_key TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    region_id TEXT NOT NULL,
    source_fingerprint TEXT NOT NULL DEFAULT '',
    source_unit_ids TEXT NOT NULL DEFAULT '[]',
    memory_id TEXT,
    status TEXT NOT NULL DEFAULT 'completed'
        CHECK (status IN ('pending', 'completed')),
    last_error TEXT NOT NULL DEFAULT '',
    round_started_at TEXT NOT NULL DEFAULT '',
    round_received_earliest TEXT NOT NULL DEFAULT '',
    round_received_latest TEXT NOT NULL DEFAULT '',
    semantic_status TEXT NOT NULL DEFAULT 'pending',
    semantic_memory_ids TEXT NOT NULL DEFAULT '[]',
    semantic_memory_refs TEXT NOT NULL DEFAULT '[]',
    semantic_diagnostics TEXT NOT NULL DEFAULT '[]',
    semantic_error TEXT NOT NULL DEFAULT '',
    semantic_updated_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT
)"""
_SEMANTIC_PUBLICATION_SCHEMA = """CREATE TABLE IF NOT EXISTS semantic_memory_versions (
    memory_id TEXT NOT NULL,
    region_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    source_fingerprint TEXT NOT NULL,
    source_unit_ids TEXT NOT NULL DEFAULT '[]',
    source_event_ids TEXT NOT NULL DEFAULT '[]',
    source_unit_regions TEXT NOT NULL DEFAULT '[]',
    source_references TEXT NOT NULL DEFAULT '[]',
    source_region_ids TEXT NOT NULL DEFAULT '[]',
    user_event_id TEXT NOT NULL DEFAULT '',
    evidence_snippet TEXT NOT NULL DEFAULT '',
    user_occurred_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL,
    PRIMARY KEY (memory_id, session_id, turn_id)
)"""
_GOVERNANCE_AUDIT_SCHEMA = """CREATE TABLE IF NOT EXISTS governance_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    region_id TEXT NOT NULL,
    unit_id TEXT NOT NULL,
    logical_memory_id TEXT NOT NULL DEFAULT '',
    action TEXT NOT NULL,
    actor TEXT NOT NULL,
    source TEXT NOT NULL,
    operation_id TEXT NOT NULL,
    before_content TEXT NOT NULL DEFAULT '',
    after_content TEXT NOT NULL DEFAULT '',
    detail TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'committed',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)"""
_BROWSER_INDEX_SCHEMA_VERSION = 4
_BROWSER_INDEX_SCHEMA = """CREATE TABLE IF NOT EXISTS memory_browser_index (
    region_id TEXT NOT NULL,
    unit_id TEXT NOT NULL,
    logical_memory_id TEXT NOT NULL DEFAULT '',
    region_name TEXT NOT NULL,
    region_kind TEXT NOT NULL,
    scope_org TEXT NOT NULL,
    scope_space TEXT NOT NULL,
    scope_user TEXT NOT NULL,
    scope_agent TEXT NOT NULL,
    scope_session TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    memory_type_resolution TEXT NOT NULL DEFAULT 'native',
    lifecycle TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    version TEXT NOT NULL,
    supersedes TEXT NOT NULL DEFAULT '',
    source_count INTEGER NOT NULL,
    source_references TEXT NOT NULL DEFAULT '[]',
    provenance TEXT NOT NULL DEFAULT '[]',
    user_metadata TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (region_id, unit_id)
);
CREATE INDEX IF NOT EXISTS memory_browser_index_page
ON memory_browser_index (region_id, updated_at DESC, unit_id DESC);
CREATE INDEX IF NOT EXISTS memory_browser_index_filter
ON memory_browser_index (region_id, lifecycle, memory_type, updated_at DESC, unit_id DESC);
CREATE TABLE IF NOT EXISTS memory_browser_index_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_browser_index_pending (
    token TEXT PRIMARY KEY,
    region_id TEXT NOT NULL,
    scope_org TEXT NOT NULL,
    scope_space TEXT NOT NULL,
    scope_user TEXT NOT NULL,
    scope_agent TEXT NOT NULL,
    scope_session TEXT NOT NULL,
    unit_ids TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS memory_browser_index_pending_created
ON memory_browser_index_pending (created_at, token)"""
_PROMOTION_SCHEMA = """CREATE TABLE IF NOT EXISTS promotion_jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL
        CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    source_region_ids TEXT NOT NULL DEFAULT '[]',
    requested_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    summary TEXT NOT NULL DEFAULT '{}',
    error TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS promotion_records (
    promotion_key TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    source_unit_id TEXT NOT NULL,
    source_region_id TEXT NOT NULL,
    source_region_kind TEXT NOT NULL,
    target_region_id TEXT NOT NULL,
    target_region_kind TEXT NOT NULL,
    target_unit_id TEXT NOT NULL,
    classification TEXT NOT NULL,
    confidence REAL NOT NULL,
    audit_event_id INTEGER,
    evidence_kind TEXT NOT NULL DEFAULT 'explicit',
    source_unit_ids TEXT NOT NULL DEFAULT '[]',
    source_region_ids TEXT NOT NULL DEFAULT '[]',
    source_unit_regions TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS promotion_outbox (
    promotion_key TEXT PRIMARY KEY,
    target_region_id TEXT NOT NULL,
    target_unit_id TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)"""
_MAINTENANCE_SCHEMA = (
    _SESSION_RETENTION_SCHEMA
    + ";\n"
    + _CAPTURE_EVOLUTIONS_SCHEMA
    + ";\n"
    + _GOVERNANCE_AUDIT_SCHEMA
    + ";\n"
    + _SEMANTIC_PUBLICATION_SCHEMA
    + ";\n"
    + _PROMOTION_SCHEMA
    + ";"
)
_STORE_COLLECTOR: ContextVar[list[object] | None] = ContextVar(
    "personal_agent_memory_mem2_store_collector", default=None
)


def _remember_store(store: object) -> None:
    collector = _STORE_COLLECTOR.get()
    if collector is None:
        raise RuntimeError("mem2 storage assembly is not active")
    collector.append(store)


@VectorProducer.register(_STORE_TARGET)  # type: ignore[untyped-decorator]
def _build_vector_store(config: Any) -> VectorStore:
    store = MilvusLiteVectorStore(
        uri=str(config.get("uri")),
        dim=int(config.get("dim", _DEFAULT_EMBEDDING_DIM)),
        collection=str(config.get("collection")),
    )
    _remember_store(store)
    return store


@FulltextProducer.register(_STORE_TARGET)  # type: ignore[untyped-decorator]
def _build_fulltext_store(config: Any) -> FulltextStore:
    store = MilvusLiteFulltextStore(
        uri=str(config.get("uri")),
        collection=str(config.get("collection")),
    )
    _remember_store(store)
    return store


@GraphProducer.register(_STORE_TARGET)  # type: ignore[untyped-decorator]
def _build_graph_store(config: Any) -> GraphStore:
    store = MilvusLiteGraphStore(
        uri=str(config.get("uri")),
        collection=str(config.get("collection")),
        edge_collection=str(config.get("edge_collection")),
    )
    _remember_store(store)
    return store


@AuditProducer.register(_AUDIT_TARGET)  # type: ignore[untyped-decorator]
def _build_audit_logger(config: Any) -> AuditLogger:
    logger = SQLiteAuditLogger(str(config.get("db_path")))
    _remember_store(logger)
    return logger


@EmbedderProducer.register(_EMBEDDER_TARGET)  # type: ignore[untyped-decorator]
def _build_embedder(config: Any) -> Embedder:
    embedder = OpenAIEmbedder(
        model_name=str(config.get("model_name")),
        base_url=str(config.get("base_url")),
        api_key=str(config.get("api_key", "")),
        dimension=int(config.get("dim", _DEFAULT_EMBEDDING_DIM)),
        timeout=float(config.get("timeout")),
        max_retries=int(config.get("max_retries", 0)),
    )
    _remember_store(embedder)
    return embedder


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    for key, value in (overrides or {}).items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True, slots=True)
class _RuntimeBundle:
    runtime: MemoryRuntime
    kv_store: SQLiteKVStore
    stores: tuple[object, ...]
    graph_store: MilvusLiteGraphStore | None


class _SensitiveFilteringEvolver:
    """Filter sensitive units before a graph association job reaches Jiuwen."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    def evolve(self, units: list[MemoryUnit], mode: EvolveMode) -> Any:
        if mode == EvolveMode.ASSOCIATE:
            units = [unit for unit in units if inspect_sensitive_text(unit.content) is None]
        return self._delegate.evolve(units, mode)


class Mem2MemoryKernel(UnavailableMemoryKernel):
    """Persistent JiuwenMemory runtime exposed through ``MemoryKernelPort``."""

    kind: MemoryKernelKind = "mem2"

    def __init__(
        self,
        state_dir: Path,
        *,
        embedding: ModelEndpoint | None = None,
        embedding_dimension: int = _DEFAULT_EMBEDDING_DIM,
        graph: ModelEndpoint | None = None,
        promotion_classifier: PromotionClassifier | None = None,
        semantic_extractor: SemanticExtractor | None = None,
        graph_extractor: EntityGraphExtractor | None = None,
        semantic_executor: SemanticTaskExecutor | None = None,
        config_overrides: Mapping[str, Any] | None = None,
        regions: RegionRegistry | None = None,
    ) -> None:
        if embedding_dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        self._state_dir = state_dir.expanduser().resolve()
        self._embedding = embedding
        self._embedding_dimension = embedding_dimension
        self._promotion_classifier = promotion_classifier
        model_client = OpenAICompatibleClient(None, None, graph) if graph is not None else None
        self._promotion_model = model_client
        self._semantic_extractor = semantic_extractor or (
            model_client.extract_semantic_memories if model_client is not None else None
        )
        self._graph_extractor = graph_extractor or (
            (
                lambda context: model_client.extract_graph(
                    json.dumps(context, ensure_ascii=False, sort_keys=True)
                )
            )
            if model_client is not None
            else None
        )
        self._config_overrides = config_overrides
        self._regions_registry = regions
        self._regions_registry_lock = threading.Lock()
        self._bundle: _RuntimeBundle | None = None
        self._startup_error: str | None = None
        self._state_ready = False
        self._vector_enabled = False
        self._vector_degradation = "embedding_not_configured"
        self._embedding_detail = "embedding endpoint is not configured; vector recall is disabled"
        self._embedding_api_key_cache: str | None = None
        self._graph_errors: dict[str, str] = {}
        self._graph_projection_path = self._state_dir / "mem2" / "graph_projection_errors.json"
        self._graph_projection_manifest_path = (
            self._state_dir / "mem2" / "graph_projection_manifests.json"
        )
        self._graph_projection_lock = threading.RLock()
        self._graph_projection_errors: dict[str, str] = {}
        self._graph_projection_manifests: dict[str, dict[str, str]] = {}
        self._scope_graph_threads: dict[str, threading.Thread] = {}
        self._scope_graph_requested: set[str] = set()
        self._scope_graph_lock = threading.Lock()
        self._lock = threading.RLock()
        self._maintenance_lock = threading.RLock()
        self._browser_index_lock = threading.RLock()
        self._graph_write_lock = threading.RLock()
        self._entity_graph_store = EntityGraphStore(
            self._state_dir / "mem2" / "maintenance.sqlite3"
        )
        self._entity_graph_lock = threading.RLock()
        self._entity_graph_threads: dict[str, threading.Thread] = {}
        self._entity_graph_requested: set[str] = set()
        self._entity_graph_recovery_thread: threading.Thread | None = None
        self._governance_locks: dict[str, threading.RLock] = {}
        self._governance_locks_guard = threading.Lock()
        self._idempotency_locks: dict[tuple[str, str], tuple[threading.Lock, int]] = {}
        self._idempotency_locks_guard = threading.Lock()
        self._idempotency_schema_lock = threading.Lock()
        self._promotion_lock = threading.RLock()
        self._promotion_threads: dict[str, threading.Thread] = {}
        self._promotion_threads_lock = threading.Lock()
        self._semantic_executor = semantic_executor or BoundedSemanticExecutor()
        self._semantic_state_lock = threading.Lock()
        self._semantic_active_versions: dict[str, str] = {}
        self._semantic_requested_versions: dict[str, str] = {}
        self._semantic_deferred_versions: dict[str, str] = {}
        slot_callback = getattr(self._semantic_executor, "set_slot_callback", None)
        if callable(slot_callback):
            slot_callback(self._retry_deferred_semantic_distillations)
        self._semantic_region_locks: dict[str, threading.RLock] = {}
        self._semantic_region_locks_guard = threading.Lock()
        self._promotion_schema_ready = False
        self._browser_index_active_tokens: set[str] = set()

    async def start(self) -> None:
        await asyncio.to_thread(self._start)

    def _start(self) -> None:
        with self._lock:
            if self._bundle is not None:
                return
            self._startup_error = None
            self._state_ready = False
            state_dir = self._prepare_state_dir()
            if state_dir is None:
                return
            self._state_ready = True
            self._prepare_embedding()
            self._graph_projection_errors = self._load_graph_projection_errors()
            self._graph_projection_manifests = self._load_graph_projection_manifests()
            stores: list[object] = []
            kv_store: SQLiteKVStore | None = None
            token = _STORE_COLLECTOR.set(stores)
            try:
                kv_store = SQLiteKVStore(state_dir / "memory.sqlite3")
                runtime = assemble_runtime(
                    kv=kv_store,
                    config=self._runtime_config(state_dir),
                )
            except Exception as error:
                _close_stores(stores)
                if kv_store is not None:
                    _close_quietly(kv_store)
                self._startup_error = f"mem2 runtime startup failed: {error}"
                return
            finally:
                _STORE_COLLECTOR.reset(token)
            graph_store = next(
                (store for store in stores if isinstance(store, MilvusLiteGraphStore)),
                None,
            )
            bundle = _RuntimeBundle(runtime, kv_store, tuple(stores), graph_store)
            self._bundle = bundle
            try:
                self._ensure_browser_index(bundle)
                self._promotion_schema_ready = True
                with closing(self._maintenance_db()):
                    pass
                self._retry_pending_browser_index(bundle, strict=False)
                self._recover_promotion_outbox(bundle)
                self._recover_semantic_distillations()
                self._schedule_entity_graph_recovery()
            except Exception as error:
                self._promotion_schema_ready = False
                self._bundle = None
                _close_quietly(runtime)
                _close_quietly(kv_store)
                _close_stores(list(stores))
                self._startup_error = f"mem2 memory browser startup failed: {error}"

    def _prepare_state_dir(self) -> Path | None:
        state_dir = self._state_dir / "mem2"
        version_path = state_dir / "VERSION"
        flags = os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        try:
            state_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            self._startup_error = f"mem2 state directory is not usable: {error}"
            return None
        try:
            descriptor = os.open(version_path, flags)
        except FileNotFoundError:
            try:
                has_entries = any(state_dir.iterdir())
            except OSError as error:
                self._startup_error = f"mem2 state directory cannot be inspected: {error}"
                return None
            if has_entries:
                self._startup_error = (
                    "mem2 state directory has no VERSION marker; "
                    "reset the state directory explicitly before starting"
                )
                return None
            try:
                descriptor = os.open(
                    version_path,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | os.O_CLOEXEC
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                )
            except OSError as error:
                self._startup_error = f"mem2 state version marker cannot be created: {error}"
                return None
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    stream.write(_STATE_VERSION + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError as error:
                self._startup_error = f"mem2 state version marker cannot be written: {error}"
                return None
            return state_dir
        except OSError as error:
            self._startup_error = f"mem2 state version marker cannot be read safely: {error}"
            return None
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 256:
                self._startup_error = "mem2 state version marker is not a small regular file"
                return None
            contents = os.read(descriptor, 257)
        except OSError as error:
            self._startup_error = f"mem2 state version marker cannot be read: {error}"
            return None
        finally:
            os.close(descriptor)
        try:
            version = contents.decode("utf-8")
        except UnicodeError as error:
            self._startup_error = f"mem2 state version marker is not UTF-8: {error}"
            return None
        if version != _STATE_VERSION + "\n":
            self._startup_error = (
                f"unsupported mem2 state version {version.strip()!r}; "
                f"expected {_STATE_VERSION!r}; reset the state directory explicitly before starting"
            )
            return None
        return state_dir

    async def close(self) -> None:
        await asyncio.to_thread(self._close)

    def _close(self) -> None:
        self._wait_for_semantic_tasks()
        self._wait_for_promotion_threads()
        self._wait_for_scope_graph_updates()
        self._wait_for_entity_graph_updates()
        with self._lock:
            bundle = self._bundle
            self._bundle = None
            if bundle is None:
                return
            _close_quietly(bundle.runtime)
            _close_quietly(bundle.kv_store)
            _close_stores(list(bundle.stores))

    def health(self) -> KernelHealth:
        with self._lock:
            bundle = self._bundle
            startup_error = self._startup_error
            embedding_detail = self._embedding_detail
        if bundle is None:
            detail = startup_error or "mem2 runtime is not started"
            return KernelHealth(kind=self.kind, status="unavailable", detail=detail)
        failures: list[str] = []
        services: tuple[object, ...] = (bundle.kv_store, *bundle.stores)
        seen: set[int] = set()
        for service in services:
            if id(service) in seen:
                continue
            seen.add(id(service))
            health = getattr(service, "health", None)
            if not callable(health):
                continue
            try:
                health()
            except Exception as error:
                failures.append(f"{type(service).__name__}: {error}")
        if not self._vector_enabled:
            failures.insert(0, embedding_detail)
        if failures:
            return KernelHealth(
                kind=self.kind,
                status="degraded",
                detail="; ".join(failures)[:1000],
            )
        return KernelHealth(
            kind=self.kind,
            status="ready",
            detail="JiuwenMemory 0.2.0 runtime",
        )

    def graph_status(self, library_id: str | None = None) -> dict[str, object]:
        with self._lock:
            bundle = self._bundle
        if bundle is None or bundle.graph_store is None:
            return {
                "status": "not_configured",
                "backend": "milvus-lite",
                "library_id": library_id or "",
                "total_documents": 0,
                "projected_documents": 0,
                "last_error": "",
                "updated_at": datetime.now(UTC).isoformat(),
            }
        identifier = validate_library_id(library_id) if library_id is not None else None
        region = self._graph_region_or_none(identifier) if identifier is not None else None
        region_id = region.region_id if region is not None else identifier or ""
        try:
            bundle.graph_store.health()
            counts = {"nodes": 0, "edges": 0}
            store_errors: list[str] = []
            projection_issues: list[str] = []
            projection_errors: list[str] = []
            transient_errors: list[str] = []
            active_documents = 0
            projected_documents = 0
            graph_updates_pending = False
            if identifier is None:
                scopes = self._graph_status_scopes(bundle)
            elif region is None:
                scopes = [_scope_for_normalized_library(identifier)]
            else:
                scopes = self._region_scopes(bundle, region)
            with self._graph_write_lock:
                for scope in scopes:
                    graph_updates_pending = (
                        graph_updates_pending or self._scope_graph_update_pending(scope)
                    )
                    scope_counts = bundle.graph_store.counts(scope)
                    counts["nodes"] += int(scope_counts.get("nodes", 0))
                    counts["edges"] += int(scope_counts.get("edges", 0))
                    expected, actual, mismatch = self._graph_projection_snapshot(
                        bundle,
                        scope,
                    )
                    active_documents += expected
                    projected_documents += actual
                    if mismatch is not None:
                        projection_issues.append(mismatch)
                    store_error = self._graph_store_error(bundle, scope)
                    if store_error is not None:
                        store_errors.append(store_error)
                    projection_error = self._graph_projection_error(scope=scope)
                    if projection_error is not None:
                        projection_errors.append(projection_error)
                    transient_error = self._graph_error_for(scope=scope)
                    if transient_error is not None:
                        transient_errors.append(transient_error)
            if region is not None:
                for projection_error in self._graph_projection_errors_for_region(region):
                    if projection_error not in projection_errors:
                        projection_errors.append(projection_error)
                for transient_error in self._graph_errors_for_region(region):
                    if transient_error not in transient_errors:
                        transient_errors.append(transient_error)
        except Exception as error:
            detail = str(error)
            return {
                "status": "degraded",
                "backend": "milvus-lite",
                "library_id": region_id,
                "detail": detail,
                "total_documents": 0,
                "projected_documents": 0,
                "last_error": detail,
                "updated_at": datetime.now(UTC).isoformat(),
                "nodes": 0,
                "edges": 0,
            }
        if identifier is None:
            projection_error = self._any_graph_projection_error()
            transient_error = self._any_graph_error()
        else:
            projection_error = (
                projection_errors[0]
                if projection_errors
                else self._graph_projection_error(
                    region_id=region.region_id if region is not None else identifier
                )
            )
            transient_error = (
                transient_errors[0]
                if transient_errors
                else self._graph_error_for(
                    region_id=region.region_id if region is not None else identifier
                )
            )
        detail = (
            projection_error
            or (projection_issues[0] if projection_issues else None)
            or (store_errors[0] if store_errors else None)
            or transient_error
            or (
                "graph projection update pending"
                if graph_updates_pending
                else "one-hop graph expansion available"
            )
        )
        degraded = bool(projection_error or projection_issues or store_errors or transient_error)
        return {
            "status": "degraded" if degraded else "ready",
            "backend": "milvus-lite",
            "library_id": region_id,
            "detail": detail,
            "total_documents": active_documents,
            "projected_documents": projected_documents,
            "last_error": "" if not degraded else detail,
            "updated_at": datetime.now(UTC).isoformat(),
            **counts,
        }

    def entity_graph_status(self, library_id: str) -> dict[str, object]:
        region = self._regions.require_region(validate_library_id(library_id))
        if self._graph_extractor is None:
            self._entity_graph_store.not_configured(region.region_id)
        self._wait_for_entity_graph_update(region.region_id)
        return self._entity_graph_store.status(region.region_id)

    async def rebuild_entity_graph(self, library_id: str) -> dict[str, object]:
        return await asyncio.to_thread(self._rebuild_entity_graph_serialized, library_id)

    def _rebuild_entity_graph_serialized(self, library_id: str) -> dict[str, object]:
        region = self._regions.require_region(validate_library_id(library_id))
        with self._governance_region_lock(region.region_id):
            return self._rebuild_entity_graph(region)

    def _rebuild_entity_graph(self, region: MemoryRegion) -> dict[str, object]:
        if self._graph_extractor is None:
            self._entity_graph_store.not_configured(region.region_id)
            return self._entity_graph_store.status(region.region_id)
        bundle = self._require_bundle()
        try:
            memories: list[dict[str, object]] = []
            for scope, unit in self._region_unit_pairs(bundle, region):
                if not _is_formal_semantic_unit(unit):
                    continue
                memories.append(self._entity_graph_memory_payload(bundle, region, scope, unit))
            return self._entity_graph_store.rebuild(
                region.region_id, memories, self._graph_extractor
            )
        except Exception:
            self._entity_graph_store.fail(
                region.region_id, "entity graph extractor returned invalid data"
            )
            return {
                **self._entity_graph_store.status(region.region_id),
                "error": "entity graph projection failed",
            }

    def _project_entity_graph_memory(
        self,
        bundle: _RuntimeBundle,
        region: MemoryRegion,
        scope: Scope,
        unit: MemoryUnit,
    ) -> dict[str, object]:
        if self._graph_extractor is None:
            return {"status": "skipped", "rejected": 0}
        return self._entity_graph_store.project(
            region.region_id,
            self._entity_graph_memory_payload(bundle, region, scope, unit),
            self._graph_extractor,
        )

    def _entity_graph_memory_payload(
        self,
        bundle: _RuntimeBundle,
        region: MemoryRegion,
        scope: Scope,
        unit: MemoryUnit,
    ) -> dict[str, object]:
        evidence = self._browser_evidence(bundle, region, unit)
        source_references = unit.user_metadata.get("source_references")
        if not isinstance(source_references, list):
            source_references = list(unit.provenance)
        return {
            "memory_id": unit.id,
            "content": unit.content,
            "memory_type": _metadata_text(unit, "memory_type"),
            "source_references": list(source_references),
            "evidence": evidence,
            "evidence_snippets": _metadata_strings(unit, "evidence_snippets"),
            "region": {
                "region_id": region.region_id,
                "kind": region.kind,
                "name": region.name,
            },
            "scope": {
                "org": scope.org,
                "space": scope.space,
                "user": scope.user,
                "agent": scope.agent,
                "session": scope.session,
            },
        }

    def _queue_entity_graph_update(self, region_id: str) -> None:
        if self._graph_extractor is None:
            self._entity_graph_store.not_configured(region_id)
            return
        with self._entity_graph_lock:
            self._entity_graph_requested.add(region_id)
            if region_id in self._entity_graph_threads:
                return
            thread = threading.Thread(
                target=self._run_entity_graph_update,
                args=(region_id,),
                name="memory-entity-graph-update",
                daemon=True,
            )
            self._entity_graph_threads[region_id] = thread
        thread.start()

    def _schedule_entity_graph_recovery(self) -> None:
        if self._graph_extractor is None:
            return
        with self._entity_graph_lock:
            if (
                self._entity_graph_recovery_thread is not None
                and self._entity_graph_recovery_thread.is_alive()
            ):
                return
            thread = threading.Thread(
                target=self._recover_entity_graph_projections,
                name="memory-entity-graph-recovery",
                daemon=True,
            )
            self._entity_graph_recovery_thread = thread
        thread.start()

    def _recover_entity_graph_projections(self) -> None:
        try:
            for region in self._regions.list_regions():
                self._queue_entity_graph_update(region.region_id)
        except Exception:
            return
        finally:
            with self._entity_graph_lock:
                if self._entity_graph_recovery_thread is threading.current_thread():
                    self._entity_graph_recovery_thread = None

    def _run_entity_graph_update(self, region_id: str) -> None:
        restart_thread: threading.Thread | None = None
        try:
            region = self._regions.require_region(region_id)
            while True:
                with self._entity_graph_lock:
                    if region_id not in self._entity_graph_requested:
                        return
                    self._entity_graph_requested.discard(region_id)
                with self._governance_region_lock(region_id):
                    self._rebuild_entity_graph(region)
        except Exception:
            self._entity_graph_store.fail(region_id, "entity graph projection failed")
        finally:
            with self._entity_graph_lock:
                if self._entity_graph_threads.get(region_id) is threading.current_thread():
                    self._entity_graph_threads.pop(region_id, None)
                    if region_id in self._entity_graph_requested:
                        restart_thread = threading.Thread(
                            target=self._run_entity_graph_update,
                            args=(region_id,),
                            name="memory-entity-graph-update",
                            daemon=True,
                        )
                        self._entity_graph_threads[region_id] = restart_thread
                    else:
                        self._entity_graph_requested.discard(region_id)
            if restart_thread is not None:
                restart_thread.start()

    def _wait_for_entity_graph_update(
        self, region_id: str, *, timeout: float = 30.0
    ) -> None:
        if timeout < 0:
            raise ValueError("entity graph wait timeout must not be negative")
        with self._entity_graph_lock:
            thread = self._entity_graph_threads.get(region_id)
        if thread is not None:
            thread.join(timeout=timeout)

    def entity_graph(
        self,
        library_id: str,
        *,
        entity_id: str | None = None,
        memory_id: str | None = None,
        entity_type: str | None = None,
        q: str | None = None,
        hops: int = 1,
        limit: int = 200,
        wait_for_update: bool = True,
        wait_timeout: float | None = None,
    ) -> dict[str, object]:
        region = self._regions.require_region(validate_library_id(library_id))
        if wait_for_update:
            self._wait_for_entity_graph_update(
                region.region_id,
                timeout=30.0 if wait_timeout is None else wait_timeout,
            )
        return self._entity_graph_store.graph(
            region.region_id,
            entity_id=entity_id,
            memory_id=memory_id,
            entity_type=entity_type,
            q=q,
            hops=hops,
            limit=limit,
        )

    def entity_graph_memory(
        self,
        library_id: str,
        memory_id: str,
        *,
        wait_for_update: bool = True,
        wait_timeout: float | None = None,
    ) -> dict[str, object]:
        region = self._regions.require_region(validate_library_id(library_id))
        if wait_for_update:
            self._wait_for_entity_graph_update(
                region.region_id,
                timeout=30.0 if wait_timeout is None else wait_timeout,
            )
        return self._entity_graph_store.memory_detail(region.region_id, memory_id)

    def entity_graph_entity(
        self,
        library_id: str,
        entity_id: str,
        *,
        limit: int = DETAIL_DEFAULT_LIMIT,
        offset: int | None = None,
        cursor: str | None = None,
        wait_for_update: bool = True,
        wait_timeout: float | None = None,
    ) -> dict[str, object]:
        region = self._regions.require_region(validate_library_id(library_id))
        if wait_for_update:
            self._wait_for_entity_graph_update(
                region.region_id,
                timeout=30.0 if wait_timeout is None else wait_timeout,
            )
        detail = self._entity_graph_store.entity_detail(
            region.region_id,
            entity_id,
            limit=limit,
            offset=offset,
            cursor=cursor,
        )
        if detail is None:
            raise MemoryRecordNotFoundError("entity not found")
        return detail

    def entity_graph_relation(
        self,
        library_id: str,
        relation_id: str,
        *,
        limit: int = DETAIL_DEFAULT_LIMIT,
        offset: int | None = None,
        cursor: str | None = None,
        wait_for_update: bool = True,
        wait_timeout: float | None = None,
    ) -> dict[str, object]:
        region = self._regions.require_region(validate_library_id(library_id))
        if wait_for_update:
            self._wait_for_entity_graph_update(
                region.region_id,
                timeout=30.0 if wait_timeout is None else wait_timeout,
            )
        detail = self._entity_graph_store.relation_detail(
            region.region_id,
            relation_id,
            limit=limit,
            offset=offset,
            cursor=cursor,
        )
        if detail is None:
            raise MemoryRecordNotFoundError("relation not found")
        return detail

    def _wait_for_entity_graph_updates(self) -> None:
        with self._entity_graph_lock:
            recovery_thread = self._entity_graph_recovery_thread
        if recovery_thread is not None:
            recovery_thread.join(timeout=30.0)
        with self._entity_graph_lock:
            threads = tuple(self._entity_graph_threads.values())
        for thread in threads:
            thread.join(timeout=30.0)

    async def rebuild_graph(self, library_id: str) -> dict[str, object]:
        return await asyncio.to_thread(self._rebuild_graph_serialized, library_id)

    def _rebuild_graph_serialized(self, library_id: str) -> dict[str, object]:
        identifier = validate_library_id(library_id)
        with (
            self._maintenance_lock,
            self._governance_region_lock(identifier),
            self._graph_write_lock,
        ):
            return self._rebuild_graph(identifier)

    def _rebuild_graph(self, library_id: str) -> dict[str, object]:
        bundle = self._require_bundle()
        identifier = validate_library_id(library_id)
        region = self._graph_region_or_none(identifier)
        region_id = region.region_id if region is not None else identifier
        if bundle.graph_store is None:
            entity_graph = (
                self._rebuild_entity_graph(region)
                if region is not None
                else {"status": "not_configured", "region_id": region_id}
            )
            return {
                "library_id": region_id,
                "status": "not_configured",
                "job_id": "",
                "entity_graph": entity_graph,
            }
        job_ids: list[str] = []
        failure: str | None = None
        failure_scope: Scope | None = None
        scopes: list[Scope] = []
        try:
            bundle.graph_store.health()
            scopes = (
                self._region_scopes(bundle, region)
                if region is not None
                else [_scope_for_normalized_library(identifier)]
            )
            for scope in scopes:
                clear_diagnostics = getattr(bundle.graph_store, "clear_diagnostics", None)
                if callable(clear_diagnostics):
                    clear_diagnostics(scope)
                bundle.graph_store.clear(scope)
                job_id = self._associate_graph_scope(
                    bundle,
                    scope,
                    region_id=region.region_id if region is not None else None,
                )
                if job_id is None:
                    failure_scope = scope
                    failure = self._graph_error_for(scope=scope) or "graph association failed"
                    break
                job_ids.append(job_id)
                self._verify_graph_projection(bundle, scope)
        except Exception as error:
            failure = str(error)
        if failure is not None:
            if failure_scope is not None:
                self._set_graph_error(failure, scope=failure_scope)
                self._set_graph_projection_error(failure, scope=failure_scope)
            elif region is None:
                library_scope = _scope_for_normalized_library(identifier)
                self._set_graph_error(failure, scope=library_scope)
                self._set_graph_projection_error(failure, scope=library_scope)
            else:
                self._set_graph_error(failure, region_id=region_id)
                self._set_graph_projection_error(failure, region_id=region_id)
            return {
                "library_id": region_id,
                "status": "degraded",
                "detail": failure,
                "job_id": job_ids[0] if job_ids else "",
                "job_ids": job_ids,
                "entity_graph": self._rebuild_entity_graph(region) if region is not None else {},
            }
        self._clear_graph_error(region_id=region_id)
        if region is not None:
            self._clear_graph_errors_for_region(region)
            self._clear_graph_projection_errors_for_region(region)
        else:
            self._clear_graph_projection_error(scope=_scope_for_normalized_library(identifier))
            self._clear_graph_projection_error(region_id=identifier)
            self._clear_graph_error(region_id=identifier)
        return {
            "library_id": region_id,
            "status": "ready",
            "job_id": job_ids[0] if job_ids else "",
            "job_ids": job_ids,
            "graph": self.graph_status(region_id),
            "entity_graph": self._rebuild_entity_graph(region) if region is not None else {},
        }

    def _verify_graph_projection(self, bundle: _RuntimeBundle, scope: Scope) -> None:
        graph_store = bundle.graph_store
        if graph_store is None:
            return
        store_error = self._graph_store_error(bundle, scope)
        if store_error is not None:
            raise MemoryKernelUnavailableError(store_error)
        _, _, mismatch = self._graph_projection_snapshot(bundle, scope)
        if mismatch is not None:
            raise MemoryKernelUnavailableError(mismatch)

    def _active_graph_units(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
    ) -> list[MemoryUnit]:
        return sorted(
            (
                unit
                for unit in self._units_in_scope(bundle, scope)
                if unit.system_metadata.get("middle") != "true"
                and unit.lifecycle.value == "active"
                and unit.user_metadata.get("restore_pending") is not True
                and inspect_sensitive_text(unit.content) is None
            ),
            key=lambda unit: unit.id,
        )

    def _graph_projection_contents(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        units: list[MemoryUnit],
    ) -> tuple[int, int, str | None]:
        units_by_id = {unit.id: unit for unit in units}
        expected = set(units_by_id)
        graph_store = bundle.graph_store
        node_ids = getattr(graph_store, "node_ids", None)
        if graph_store is None or not callable(node_ids):
            return len(expected), len(expected), None
        try:
            actual = set(node_ids(scope))
        except Exception as error:
            return (
                len(expected),
                0,
                f"graph projection read failed: {error}",
            )
        edge_endpoints = getattr(graph_store, "edge_endpoints", None)
        dangling_edges = 0
        if callable(edge_endpoints):
            try:
                dangling_edges = sum(
                    1
                    for source, target in edge_endpoints(scope)
                    if source not in actual or target not in actual
                )
            except Exception as error:
                return (
                    len(expected),
                    len(actual),
                    f"graph edge projection read failed: {error}",
                )
        stale_nodes = 0
        get_nodes = getattr(graph_store, "get", None)
        if callable(get_nodes):
            try:
                projected_nodes = get_nodes(scope, sorted(expected & actual))
            except Exception as error:
                return (
                    len(expected),
                    len(actual),
                    f"graph node projection read failed: {error}",
                )
            for node in projected_nodes:
                unit = units_by_id.get(node.id)
                if unit is None:
                    continue
                if node.label != unit.tier.value or node.properties.get("content") != unit.content:
                    stale_nodes += 1
        missing = expected - actual
        extra = actual - expected
        if not missing and not extra and dangling_edges == 0 and stale_nodes == 0:
            return len(expected), len(actual), None
        return (
            len(expected),
            len(actual),
            "graph projection mismatch: "
            f"missing={len(missing)} extra={len(extra)} "
            f"dangling_edges={dangling_edges} stale_nodes={stale_nodes}",
        )

    def _graph_projection_snapshot(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
    ) -> tuple[int, int, str | None]:
        update_pending = self._scope_graph_update_pending(scope)
        units = self._active_graph_units(bundle, scope)
        expected_count, actual_count, mismatch = self._graph_projection_contents(
            bundle,
            scope,
            units,
        )
        if mismatch is not None:
            if update_pending:
                return expected_count, actual_count, None
            return expected_count, actual_count, mismatch
        if not units and actual_count == 0:
            return expected_count, actual_count, None
        manifest = self._graph_projection_manifest(scope)
        if manifest is None:
            if update_pending:
                return expected_count, actual_count, None
            return (
                expected_count,
                actual_count,
                "graph projection manifest missing; rebuild required",
            )
        try:
            kv_fingerprint = self._graph_kv_fingerprint(units)
            graph_fingerprint = self._graph_projection_fingerprint(bundle, scope)
        except Exception as error:
            return (
                expected_count,
                actual_count,
                f"graph projection fingerprint read failed: {error}",
            )
        if (
            manifest["kv_fingerprint"] == kv_fingerprint
            and manifest["graph_fingerprint"] == graph_fingerprint
        ):
            return expected_count, actual_count, None
        if update_pending:
            return expected_count, actual_count, None
        edge_mismatch = self._graph_edge_mismatch(bundle, scope, units)
        return (
            expected_count,
            actual_count,
            edge_mismatch or "graph projection changed after verification; rebuild required",
        )

    @staticmethod
    def _graph_properties_fingerprint(properties: Mapping[str, Any]) -> str:
        return json.dumps(
            dict(properties),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    @classmethod
    def _graph_edge_signature(
        cls,
        source: str,
        target: str,
        relation: str,
        properties: Mapping[str, Any],
    ) -> tuple[str, str, str, str]:
        left, right = sorted((str(source), str(target)))
        return (
            left,
            right,
            str(relation),
            cls._graph_properties_fingerprint(properties),
        )

    @classmethod
    def _graph_kv_fingerprint(cls, units: list[MemoryUnit]) -> str:
        payload = [
            {
                "id": unit.id,
                "content": unit.content,
                "tier": unit.tier.value,
            }
            for unit in units
        ]
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    @classmethod
    def _graph_projection_fingerprint(
        cls,
        bundle: _RuntimeBundle,
        scope: Scope,
    ) -> str:
        graph_store = bundle.graph_store
        if graph_store is None:
            return ""
        projection_nodes = getattr(graph_store, "projection_nodes", None)
        projection_edges = getattr(graph_store, "projection_edges", None)
        if not callable(projection_nodes) or not callable(projection_edges):
            return "unavailable"
        nodes = sorted(
            (
                node.id,
                node.label,
                cls._graph_properties_fingerprint(node.properties),
            )
            for node in projection_nodes(scope)
        )
        edges = sorted(
            (
                edge.source,
                edge.target,
                edge.relation,
                cls._graph_properties_fingerprint(edge.properties),
            )
            for edge in projection_edges(scope)
        )
        return hashlib.sha256(
            json.dumps(
                {"nodes": nodes, "edges": edges},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    @classmethod
    def _expected_graph_edge_signatures(
        cls,
        bundle: _RuntimeBundle,
        units: list[MemoryUnit],
    ) -> set[tuple[str, str, str, str]] | None:
        engine = getattr(bundle.runtime.api, "_engine", None)
        evolver = getattr(engine, "_evolver", None)
        associator = getattr(evolver, "_associator", None)
        associate = getattr(associator, "associate", None)
        if not callable(associate):
            return None
        relations = associate(list(units))
        return {
            cls._graph_edge_signature(
                relation.source_id,
                relation.target_id,
                relation.relation,
                getattr(relation, "metadata", {}) or {},
            )
            for relation in relations
        }

    @classmethod
    def _actual_graph_edge_signatures(
        cls,
        bundle: _RuntimeBundle,
        scope: Scope,
    ) -> set[tuple[str, str, str, str]] | None:
        graph_store = bundle.graph_store
        projection_edges = getattr(graph_store, "projection_edges", None)
        if not callable(projection_edges):
            return None
        return {
            cls._graph_edge_signature(
                edge.source,
                edge.target,
                edge.relation,
                edge.properties,
            )
            for edge in projection_edges(scope)
        }

    def _graph_edge_mismatch(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        units: list[MemoryUnit],
    ) -> str | None:
        try:
            expected = self._expected_graph_edge_signatures(bundle, units)
            actual = self._actual_graph_edge_signatures(bundle, scope)
        except Exception as error:
            return f"graph edge projection verification failed: {error}"
        if expected is None or actual is None:
            return None
        missing = expected - actual
        stale = actual - expected
        if not missing and not stale:
            return None
        return (
            "graph edge projection mismatch: "
            f"expected_edges={len(expected)} actual_edges={len(actual)} "
            f"missing_edges={len(missing)} stale_edges={len(stale)}"
        )

    def _record_graph_projection(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
    ) -> None:
        units = self._active_graph_units(bundle, scope)
        _, _, mismatch = self._graph_projection_contents(bundle, scope, units)
        if mismatch is not None:
            raise MemoryKernelUnavailableError(mismatch)
        expected = self._expected_graph_edge_signatures(bundle, units)
        actual = self._actual_graph_edge_signatures(bundle, scope)
        if expected is None or actual is None:
            raise MemoryKernelUnavailableError("graph edge projection verification is unavailable")
        if expected != actual:
            raise MemoryKernelUnavailableError(
                self._graph_edge_mismatch(bundle, scope, units) or "graph edge projection mismatch"
            )
        key = self._graph_projection_key(None, scope)
        manifest = {
            "kv_fingerprint": self._graph_kv_fingerprint(units),
            "graph_fingerprint": self._graph_projection_fingerprint(bundle, scope),
        }
        with self._graph_projection_lock:
            self._graph_projection_manifests[key] = manifest
            self._persist_graph_projection_manifests_locked()

    def _associate_graph_scope(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        *,
        region_id: str | None = None,
    ) -> str | None:
        with self._graph_write_lock:
            return self._associate_graph_scope_locked(
                bundle,
                scope,
                region_id=region_id,
            )

    def _associate_graph_scope_locked(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        *,
        region_id: str | None = None,
    ) -> str | None:
        if bundle.graph_store is None:
            return None
        try:
            active_units = [
                unit
                for unit in self._units_in_scope(bundle, scope)
                if unit.system_metadata.get("middle") != "true"
                and unit.lifecycle.value == "active"
                and unit.user_metadata.get("restore_pending") is not True
            ]
            sensitive_active = any(
                inspect_sensitive_text(unit.content) is not None for unit in active_units
            )
            sensitive_units_present = any(
                inspect_sensitive_text(unit.content) is not None
                for unit in self._units_in_scope(bundle, scope)
            )
            if sensitive_active:
                self._prune_inactive_graph_nodes(
                    bundle,
                    scope,
                    region_id=region_id,
                )
            engine = getattr(bundle.runtime.api, "_engine", None)
            original_evolver = getattr(engine, "_evolver", None)
            if sensitive_units_present and engine is not None and original_evolver is not None:
                engine._evolver = _SensitiveFilteringEvolver(original_evolver)
            security = _security_for(scope)
            try:
                job_id = bundle.runtime.api.evolve(
                    scope,
                    EvolveMode.ASSOCIATE,
                    security=security,
                )
            finally:
                if sensitive_units_present and engine is not None and original_evolver is not None:
                    engine._evolver = original_evolver
            job_error = self._evolve_job_error(bundle, str(job_id))
            if job_error is not None:
                raise MemoryKernelUnavailableError(job_error)
            store_error = self._graph_store_error(bundle, scope)
            if store_error is not None:
                raise MemoryKernelUnavailableError(store_error)
            self._prune_inactive_graph_nodes(
                bundle,
                scope,
                region_id=region_id,
            )
            self._record_graph_projection(bundle, scope)
            self._clear_graph_error(region_id=region_id, scope=scope)
            self._clear_graph_projection_error(scope=scope)
            return str(job_id)
        except Exception as error:
            failure = str(error)
            self._set_graph_error(
                failure,
                region_id=region_id,
                scope=scope,
            )
            self._set_graph_projection_error(failure, scope=scope)
            return None

    @staticmethod
    def _evolve_job_error(bundle: _RuntimeBundle, job_id: str) -> str | None:
        api = bundle.runtime.api
        engine = getattr(api, "_engine", None)
        scheduler = getattr(engine, "_scheduler", None)
        status = getattr(scheduler, "status", None)
        if not callable(status):
            return None
        try:
            job = status(job_id)
        except Exception:
            return None
        if getattr(job, "status", None) != JobStatus.FAILED:
            return None
        detail = getattr(job, "detail", {})
        error = detail.get("error") or detail.get("error_type")
        return f"graph association job failed: {error or 'unknown error'}"

    @staticmethod
    def _graph_store_error(bundle: _RuntimeBundle, scope: Scope) -> str | None:
        graph_store = bundle.graph_store
        if graph_store is None:
            return None
        last_error = getattr(graph_store, "last_error", None)
        if not callable(last_error):
            return None
        try:
            error = last_error(scope)
        except Exception as read_error:
            return f"graph diagnostics unavailable: {read_error}"
        return None if error is None else str(error)

    def _graph_projection_error(
        self,
        *,
        region_id: str | None = None,
        scope: Scope | None = None,
    ) -> str | None:
        key = self._graph_projection_key(region_id, scope)
        if not key:
            return None
        with self._graph_projection_lock:
            return self._graph_projection_errors.get(key)

    def _any_graph_projection_error(self) -> str | None:
        with self._graph_projection_lock:
            return next(iter(self._graph_projection_errors.values()), None)

    @staticmethod
    def _graph_error_key(region_id: str | None, scope: Scope | None) -> str:
        if scope is not None:
            return Mem2MemoryKernel._graph_projection_key(None, scope)
        if region_id is not None:
            return Mem2MemoryKernel._graph_projection_key(region_id, None)
        return ""

    @staticmethod
    def _graph_projection_key(region_id: str | None, scope: Scope | None) -> str:
        if scope is not None:
            payload = json.dumps(
                [scope.org, scope.space, scope.user, scope.agent, scope.session],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            encoded = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")
            return f"@scope/v2:{encoded}"
        if region_id is not None:
            encoded = (
                base64.urlsafe_b64encode(region_id.encode("utf-8")).decode("ascii").rstrip("=")
            )
            return f"@region/v2:{encoded}"
        return ""

    @staticmethod
    def _decode_graph_projection_key(
        key: str,
    ) -> tuple[str | None, Scope | None]:
        prefix = ""
        if key.startswith("@scope/v2:"):
            prefix = "@scope/v2:"
        elif key.startswith("@region/v2:"):
            prefix = "@region/v2:"
        else:
            return None, None
        encoded = key.removeprefix(prefix)
        try:
            padding = "=" * (-len(encoded) % 4)
            decoded = base64.urlsafe_b64decode(encoded + padding).decode("utf-8")
            if prefix == "@region/v2:":
                return decoded, None
            payload = json.loads(decoded)
        except (ValueError, UnicodeError, json.JSONDecodeError):
            return None, None
        if not isinstance(payload, list) or len(payload) != 5:
            return None, None
        return None, Scope(
            org=str(payload[0]),
            space=str(payload[1]),
            user=str(payload[2]),
            agent=str(payload[3]),
            session=str(payload[4]),
        )

    @staticmethod
    def _scope_matches_region(scope: Scope, region: MemoryRegion) -> bool:
        return scope.space == region.space and (
            region.kind != "session" or scope.session == (region.session_id or "")
        )

    def _set_graph_error(
        self,
        error: str,
        *,
        region_id: str | None = None,
        scope: Scope | None = None,
    ) -> None:
        key = self._graph_error_key(region_id, scope)
        with self._graph_projection_lock:
            self._graph_errors[key] = error

    def _clear_graph_error(
        self,
        *,
        region_id: str | None = None,
        scope: Scope | None = None,
    ) -> None:
        key = self._graph_error_key(region_id, scope)
        with self._graph_projection_lock:
            self._graph_errors.pop(key, None)

    def _graph_error_for(
        self,
        *,
        region_id: str | None = None,
        scope: Scope | None = None,
    ) -> str | None:
        key = self._graph_error_key(region_id, scope)
        with self._graph_projection_lock:
            return self._graph_errors.get(key)

    def _any_graph_error(self) -> str | None:
        with self._graph_projection_lock:
            return next(iter(self._graph_errors.values()), None)

    def _load_graph_projection_errors(self) -> dict[str, str]:
        try:
            payload = json.loads(self._graph_projection_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        current_schema = payload.get("schema") == _GRAPH_PROJECTION_SCHEMA
        raw_errors = payload.get("errors") if current_schema else payload
        if not isinstance(raw_errors, dict):
            return {}
        errors: dict[str, str] = {}
        for raw_key, raw_error in raw_errors.items():
            if not str(raw_error):
                continue
            key = str(raw_key)
            if current_schema:
                if not key.startswith(("@scope/v2:", "@region/v2:")):
                    continue
            elif key.startswith("region:"):
                key = self._graph_projection_key(
                    region_id=key.removeprefix("region:"),
                    scope=None,
                )
            elif key.startswith("scope:"):
                continue
            else:
                continue
            errors[key] = str(raw_error)
        return errors

    def _persist_graph_projection_errors_locked(self) -> None:
        self._graph_projection_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._graph_projection_path.with_name(f"{self._graph_projection_path.name}.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema": _GRAPH_PROJECTION_SCHEMA,
                    "errors": self._graph_projection_errors,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._graph_projection_path)

    def _load_graph_projection_manifests(self) -> dict[str, dict[str, str]]:
        try:
            payload = json.loads(self._graph_projection_manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != _GRAPH_PROJECTION_MANIFEST_SCHEMA
            or not isinstance(payload.get("projections"), dict)
        ):
            return {}
        manifests: dict[str, dict[str, str]] = {}
        for raw_key, raw_manifest in payload["projections"].items():
            key = str(raw_key)
            if not key.startswith("@scope/v2:") or not isinstance(raw_manifest, dict):
                continue
            kv_fingerprint = raw_manifest.get("kv_fingerprint")
            graph_fingerprint = raw_manifest.get("graph_fingerprint")
            if not isinstance(kv_fingerprint, str) or not isinstance(
                graph_fingerprint,
                str,
            ):
                continue
            manifests[key] = {
                "kv_fingerprint": kv_fingerprint,
                "graph_fingerprint": graph_fingerprint,
            }
        return manifests

    def _persist_graph_projection_manifests_locked(self) -> None:
        self._graph_projection_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._graph_projection_manifest_path.with_name(
            f"{self._graph_projection_manifest_path.name}.tmp"
        )
        temporary.write_text(
            json.dumps(
                {
                    "schema": _GRAPH_PROJECTION_MANIFEST_SCHEMA,
                    "projections": self._graph_projection_manifests,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._graph_projection_manifest_path)

    def _graph_projection_manifest(
        self,
        scope: Scope,
    ) -> dict[str, str] | None:
        key = self._graph_projection_key(None, scope)
        with self._graph_projection_lock:
            manifest = self._graph_projection_manifests.get(key)
            return None if manifest is None else dict(manifest)

    def _set_graph_projection_error(
        self,
        error: str,
        *,
        region_id: str | None = None,
        scope: Scope | None = None,
    ) -> None:
        key = self._graph_projection_key(region_id, scope)
        if not key:
            return
        with self._graph_projection_lock:
            self._graph_projection_errors[key] = error
            try:
                self._persist_graph_projection_errors_locked()
            except OSError as persist_error:
                self._set_graph_error(
                    f"graph projection status persist failed: {persist_error}",
                    region_id=region_id,
                    scope=scope,
                )

    def _clear_graph_projection_error(
        self,
        *,
        region_id: str | None = None,
        scope: Scope | None = None,
    ) -> None:
        key = self._graph_projection_key(region_id, scope)
        if not key:
            return
        with self._graph_projection_lock:
            if self._graph_projection_errors.pop(key, None) is None:
                return
            try:
                self._persist_graph_projection_errors_locked()
            except OSError as persist_error:
                self._set_graph_error(
                    f"graph projection status persist failed: {persist_error}",
                    region_id=region_id,
                    scope=scope,
                )

    def _clear_graph_projection_errors_for_region(
        self,
        region: MemoryRegion,
    ) -> None:
        region_key = self._graph_projection_key(region.region_id, None)
        with self._graph_projection_lock:
            stale_keys = []
            for key in self._graph_projection_errors:
                region_id, scope = self._decode_graph_projection_key(key)
                if key == region_key or (
                    scope is not None and self._scope_matches_region(scope, region)
                ):
                    stale_keys.append(key)
            if not stale_keys:
                return
            for key in stale_keys:
                self._graph_projection_errors.pop(key, None)
            try:
                self._persist_graph_projection_errors_locked()
            except OSError as persist_error:
                self._set_graph_error(
                    f"graph projection status persist failed: {persist_error}",
                    region_id=region.region_id,
                )

    def _clear_graph_errors_for_region(self, region: MemoryRegion) -> None:
        region_key = self._graph_projection_key(region.region_id, None)
        with self._graph_projection_lock:
            stale_keys = []
            for key in self._graph_errors:
                region_id, scope = self._decode_graph_projection_key(key)
                if key == region_key or (
                    scope is not None and self._scope_matches_region(scope, region)
                ):
                    stale_keys.append(key)
            for key in stale_keys:
                self._graph_errors.pop(key, None)

    def _graph_projection_errors_for_region(
        self,
        region: MemoryRegion,
    ) -> list[str]:
        region_key = self._graph_projection_key(region.region_id, None)
        with self._graph_projection_lock:
            errors = []
            for key, error in self._graph_projection_errors.items():
                region_id, scope = self._decode_graph_projection_key(key)
                if key == region_key or (
                    scope is not None and self._scope_matches_region(scope, region)
                ):
                    errors.append(error)
            return errors

    def _graph_errors_for_region(self, region: MemoryRegion) -> list[str]:
        region_key = self._graph_projection_key(region.region_id, None)
        with self._graph_projection_lock:
            errors = []
            for key, error in self._graph_errors.items():
                region_id, scope = self._decode_graph_projection_key(key)
                if key == region_key or (
                    scope is not None and self._scope_matches_region(scope, region)
                ):
                    errors.append(error)
            return errors

    def _remove_graph_nodes(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        node_ids: list[str],
        region_id: str,
    ) -> None:
        with self._graph_write_lock:
            self._remove_graph_nodes_locked(bundle, scope, node_ids, region_id)

    def _remove_graph_nodes_locked(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        node_ids: list[str],
        region_id: str,
    ) -> None:
        graph_store = bundle.graph_store
        if graph_store is None or not node_ids:
            return
        try:
            graph_store.delete(scope, node_ids=node_ids)
            self._record_graph_projection(bundle, scope)
        except Exception as error:
            failure = f"graph cleanup failed: {error}"
            self._set_graph_error(failure, region_id=region_id)
            self._set_graph_projection_error(failure, scope=scope)

    def _prune_inactive_graph_nodes(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        *,
        region_id: str | None,
    ) -> None:
        graph_store = bundle.graph_store
        if graph_store is None:
            return
        node_ids = getattr(graph_store, "node_ids", None)
        if not callable(node_ids):
            return
        active_ids = {unit.id for unit in self._active_graph_units(bundle, scope)}
        try:
            stale_ids = sorted(set(node_ids(scope)) - active_ids)
            if stale_ids:
                graph_store.delete(scope, node_ids=stale_ids)
        except Exception as error:
            failure = f"graph projection cleanup failed: {error}"
            self._set_graph_error(failure, region_id=region_id)
            self._set_graph_projection_error(failure, scope=scope)
            raise MemoryKernelUnavailableError(failure) from error

    def _reproject_graph_scope(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        *,
        region_id: str,
    ) -> None:
        with self._graph_write_lock:
            self._reproject_graph_scope_locked(
                bundle,
                scope,
                region_id=region_id,
            )

    def _reproject_graph_scope_locked(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        *,
        region_id: str,
    ) -> None:
        graph_store = bundle.graph_store
        if graph_store is None:
            return
        try:
            clear_diagnostics = getattr(graph_store, "clear_diagnostics", None)
            if callable(clear_diagnostics):
                clear_diagnostics(scope)
            graph_store.clear(scope)
            job_id = self._associate_graph_scope(
                bundle,
                scope,
                region_id=region_id,
            )
            if job_id is None:
                return
            self._verify_graph_projection(bundle, scope)
        except Exception as error:
            failure = f"graph reprojection failed: {error}"
            self._set_graph_error(failure, region_id=region_id)
            self._set_graph_projection_error(failure, scope=scope)

    def _region_id_for_scope(self, scope: Scope) -> str | None:
        for region in self._regions.list_regions():
            if region.space != scope.space:
                continue
            if region.kind != "session":
                return region.region_id
            if scope.session and scope.session == region.session_id:
                return region.region_id
            if not scope.session and not region.session_id:
                return region.region_id
        return None

    def _queue_scope_graph_update(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
    ) -> None:
        if bundle.graph_store is None:
            return
        key = self._graph_projection_key(None, scope)
        with self._scope_graph_lock:
            self._scope_graph_requested.add(key)
            if key in self._scope_graph_threads:
                return
            thread = threading.Thread(
                target=self._run_scope_graph_update,
                args=(
                    bundle,
                    scope,
                    key,
                    self._region_id_for_scope(scope),
                ),
                name="memory-scope-graph-update",
                daemon=True,
            )
            self._scope_graph_threads[key] = thread
        thread.start()

    def _mark_scope_graph_pending(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
    ) -> None:
        if bundle.graph_store is None:
            return
        key = self._graph_projection_key(None, scope)
        with self._scope_graph_lock:
            self._scope_graph_requested.add(key)

    def _clear_scope_graph_pending(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
    ) -> None:
        if bundle.graph_store is None:
            return
        key = self._graph_projection_key(None, scope)
        with self._scope_graph_lock:
            if key not in self._scope_graph_threads:
                self._scope_graph_requested.discard(key)

    def _run_scope_graph_update(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        key: str,
        region_id: str | None,
    ) -> None:
        try:
            while True:
                with self._scope_graph_lock:
                    if key not in self._scope_graph_requested:
                        self._scope_graph_threads.pop(key, None)
                        return
                    self._scope_graph_requested.discard(key)
                self._associate_graph_scope(
                    bundle,
                    scope,
                    region_id=region_id,
                )
        except Exception as error:
            with self._scope_graph_lock:
                self._scope_graph_requested.discard(key)
                self._scope_graph_threads.pop(key, None)
            failure = f"graph projection update failed: {error}"
            self._set_graph_error(failure, region_id=region_id, scope=scope)
            self._set_graph_projection_error(failure, scope=scope)

    def _scope_graph_update_pending(self, scope: Scope) -> bool:
        key = self._graph_projection_key(None, scope)
        with self._scope_graph_lock:
            return key in self._scope_graph_requested or key in self._scope_graph_threads

    def _wait_for_scope_graph_updates(self) -> None:
        with self._scope_graph_lock:
            threads = tuple(self._scope_graph_threads.values())
        for thread in threads:
            thread.join(timeout=30.0)

    def _graph_region_or_none(self, library_id: str) -> MemoryRegion | None:
        try:
            return self._regions.require_region(library_id)
        except RegionError:
            return None

    @staticmethod
    def _graph_status_scopes(bundle: _RuntimeBundle) -> list[Scope]:
        scopes = list(bundle.kv_store.scopes())
        graph_scopes = getattr(bundle.graph_store, "graph_scopes", None)
        if callable(graph_scopes):
            for scope in graph_scopes():
                if scope not in scopes:
                    scopes.append(scope)
        return scopes

    def _region_scopes(self, bundle: _RuntimeBundle, region: MemoryRegion) -> list[Scope]:
        scopes = []
        for scope in bundle.kv_store.scopes():
            if scope.space != region.space:
                continue
            if region.kind == "session" and scope.session != (region.session_id or ""):
                continue
            scopes.append(scope)
        graph_scopes = getattr(bundle.graph_store, "graph_scopes", None)
        if callable(graph_scopes):
            for scope in graph_scopes():
                if scope.space != region.space:
                    continue
                if region.kind == "session" and scope.session != (region.session_id or ""):
                    continue
                if scope not in scopes:
                    scopes.append(scope)
        base_session = region.session_id if region.kind == "session" else ""
        for agent in _SEARCH_AGENTS:
            expected = _scope_for_region(
                region,
                session_id=base_session,
                agent=agent,
            )
            if expected not in scopes:
                scopes.append(expected)
        return sorted(
            scopes,
            key=lambda scope: (
                scope.space,
                scope.agent,
                scope.session,
                scope.user,
                scope.org,
            ),
        )

    async def write(self, request: MemoryWriteRequest) -> dict[str, object]:
        return await asyncio.to_thread(self._write, request)

    def _write(self, request: MemoryWriteRequest) -> dict[str, object]:
        if isinstance(request, CreateMemoryRequest):
            if request.region_id is not None:
                region_id = request.region_id.strip()
                if not region_id:
                    raise ValueError("region_id must not be blank")
                self._regions.require_region(region_id)
                normalized = replace(request, region_id=region_id)
                key = (_write_identity(normalized), normalized.idempotency_key)
            else:
                normalized = replace(request, library_id=validate_library_id(request.library_id))
                key = (normalized.library_id, normalized.idempotency_key)
            with self._idempotency_lock(key):
                return self._write_locked(normalized)
        if isinstance(request, CaptureMemoryRequest):
            normalized_capture = replace(
                request,
                session_id=_normalize_session_id(request.session_id),
                topic_id=_normalize_optional_identifier(request.topic_id, "topic_id"),
                occurred_at=_validate_occurred_at(request.occurred_at),
            )
            with self._idempotency_lock(("capture", normalized_capture.event_id)):
                return self._write_locked(normalized_capture)
        return self._write_locked(request)

    @contextmanager
    def _idempotency_lock(self, key: tuple[str, str]) -> Iterator[None]:
        """Serialize check-add-record for one library/key pair."""
        with self._idempotency_locks_guard:
            lock, users = self._idempotency_locks.get(key, (threading.Lock(), 0))
            self._idempotency_locks[key] = (lock, users + 1)
        lock.acquire()
        try:
            yield
        finally:
            lock.release()
            with self._idempotency_locks_guard:
                current_lock, users = self._idempotency_locks[key]
                if users == 1:
                    del self._idempotency_locks[key]
                else:
                    self._idempotency_locks[key] = (current_lock, users - 1)

    def _write_locked(self, request: MemoryWriteRequest) -> dict[str, object]:
        bundle = self._require_bundle()
        if isinstance(request, CreateMemoryRequest):
            if request.region_id is not None:
                region = self._regions.require_region(request.region_id)
                scope = _scope_for_region(region)
                identity = _write_identity(request)
            else:
                scope = _scope_for_normalized_library(request.library_id)
                identity = request.library_id
            request_hash = _write_request_hash(request)
            indexed_region = self._region_for_scope(scope)
            existing_id = self._reserve_idempotency(identity, request.idempotency_key, request_hash)
            if existing_id is not None:
                if indexed_region is not None:
                    self._refresh_browser_index_ids(
                        bundle,
                        indexed_region,
                        scope,
                        [existing_id],
                    )
                    if request.memory_type in FORMAL_MEMORY_TYPES:
                        self._queue_entity_graph_update(indexed_region.region_id)
                return _write_response(request, existing_id, status="existing")
            user_metadata = {
                **dict(request.user_metadata or {}),
                "memory_type": request.memory_type,
                "source_references": list(request.source_references),
                "request_actor": request.actor,
                "idempotency_key": request.idempotency_key,
                "region_id": request.region_id,
                "logical_memory_id": uuid.uuid4().hex,
            }
        elif isinstance(request, CaptureMemoryRequest):
            if request.deadline is not None and time.monotonic() >= request.deadline:
                raise MemoryKernelUnavailableError("capture persistence deadline expired")
            scope, region = self._capture_scope(request)
            identity = f"capture:{request.event_id}"
            request_hash = _capture_request_hash(request, scope)
            existing_id = self._reserve_capture_idempotency(
                bundle,
                scope,
                request,
                identity,
                request_hash,
            )
            if existing_id is not None:
                self._queue_scope_graph_update(bundle, scope)
                self._schedule_semantic_for_capture(
                    bundle,
                    scope,
                    region,
                    request.turn_id,
                )
                return _capture_response(
                    request,
                    existing_id,
                    region,
                    scope,
                    status="accepted",
                    duplicate=True,
                )
            user_metadata = {
                "event_id": request.event_id,
                "capture_event_id": request.event_id,
                "session_id": request.session_id,
                "turn_id": request.turn_id,
                "event_kind": request.event_kind,
                "occurred_at": request.occurred_at,
                "cwd": request.cwd,
                "region_id": region.region_id,
                "region_kind": region.kind,
                "topic_id": request.topic_id,
                "project_id": region.region_id if region.kind == "project" else None,
                "memory_type": "capture",
            }
        else:
            raise TypeError("unsupported memory write request")
        index_token: str | None = None
        if isinstance(request, CreateMemoryRequest) and indexed_region is not None:
            index_token = self._begin_browser_index_pending(indexed_region, scope)
        security = _security_for(scope)
        if isinstance(request, CaptureMemoryRequest):
            self._mark_scope_graph_pending(bundle, scope)
        try:
            units = bundle.runtime.api.add(
                request.content,
                scope,
                Modality.TEXT,
                security=security,
                user_metadata=user_metadata,
            )
        except Exception:
            if index_token is not None:
                self._complete_browser_index_pending(index_token)
            if isinstance(request, CreateMemoryRequest):
                self._release_pending_idempotency(_write_identity(request), request.idempotency_key)
            elif isinstance(request, CaptureMemoryRequest):
                try:
                    self._purge_capture_units(bundle, scope, request.event_id)
                except Exception as cleanup_error:
                    self._clear_scope_graph_pending(bundle, scope)
                    raise MemoryKernelUnavailableError(
                        "capture cleanup is pending after partial persistence"
                    ) from cleanup_error
                self._clear_scope_graph_pending(bundle, scope)
                self._release_pending_idempotency(identity, request.event_id)
            raise
        if not units:
            if index_token is not None:
                self._complete_browser_index_pending(index_token)
            if isinstance(request, CreateMemoryRequest):
                self._release_pending_idempotency(_write_identity(request), request.idempotency_key)
            elif isinstance(request, CaptureMemoryRequest):
                try:
                    self._purge_capture_units(bundle, scope, request.event_id)
                except Exception as cleanup_error:
                    self._clear_scope_graph_pending(bundle, scope)
                    raise MemoryKernelUnavailableError(
                        "capture cleanup is pending after partial persistence"
                    ) from cleanup_error
                self._clear_scope_graph_pending(bundle, scope)
                self._release_pending_idempotency(identity, request.event_id)
            raise MemoryKernelUnavailableError("mem2 runtime did not create a memory unit")
        unit = units[0]
        if isinstance(request, CreateMemoryRequest):
            semantic_write = user_metadata.get("semantic_schema") == "semantic-distillation-v1"
            try:
                if index_token is not None:
                    self._update_browser_index_pending(index_token, [unit.id])
                self._complete_idempotency(
                    _write_identity(request),
                    request.idempotency_key,
                    request_hash,
                    unit.id,
                )
            except Exception as error:
                if semantic_write:
                    if index_token is not None:
                        self._complete_browser_index_pending(index_token)
                    self._release_pending_idempotency(
                        _write_identity(request),
                        request.idempotency_key,
                    )
                elif index_token is not None:
                    # The unit is already persisted, so keep the pending browser
                    # index token with its unit id and the pending idempotency
                    # record. Clearing the in-memory active marker lets a later
                    # retry or a restart rebuild the formal list index without
                    # duplicating the memory.
                    with self._browser_index_lock:
                        self._browser_index_active_tokens.discard(index_token)
                raise MemoryKernelUnavailableError(
                    "memory was created but its idempotency completion is pending"
                ) from error
            if indexed_region is not None:
                self._record_create_audit(indexed_region, unit, request)
            browser_index_sync = "ready"
            if indexed_region is not None and index_token is not None:
                browser_index_sync = self._sync_browser_index_best_effort(
                    bundle,
                    indexed_region,
                    scope,
                    index_token,
                    [unit.id],
                )
            if indexed_region is not None and request.memory_type in FORMAL_MEMORY_TYPES:
                self._queue_entity_graph_update(indexed_region.region_id)
            response = _write_response(request, unit.id, status="created")
            response["browser_index_sync"] = browser_index_sync
            if browser_index_sync == "pending":
                response["warning"] = (
                    "authoritative create committed; browser index sync is pending"
                )
            return response
        try:
            self._complete_idempotency(
                identity,
                request.event_id,
                request_hash,
                unit.id,
            )
        except Exception as error:
            self._queue_scope_graph_update(bundle, scope)
            raise MemoryKernelUnavailableError(
                "capture was stored but its idempotency completion is pending"
            ) from error
        self._schedule_semantic_for_capture(
            bundle,
            scope,
            region,
            request.turn_id,
        )
        self._queue_scope_graph_update(bundle, scope)
        return _capture_response(request, unit.id, region, scope, status="accepted")

    async def search(self, request: SearchMemoryRequest) -> dict[str, object]:
        return await asyncio.to_thread(self._search, request)

    def _search(self, request: SearchMemoryRequest) -> dict[str, object]:
        if request.session_id is not None:
            request = replace(request, session_id=_normalize_session_id(request.session_id))
        if request.topic_id is not None:
            request = replace(
                request,
                topic_id=_normalize_optional_identifier(request.topic_id, "topic_id"),
            )
        bundle = self._require_bundle()
        if request.region_ids is not None:
            return self._search_regions(bundle, request)
        if request.region_id is not None:
            region = self._regions.require_region(request.region_id)
            return self._search_region_session(bundle, request, region, "explicit_region")
        if request.library_id is not None:
            library_id = validate_library_id(request.library_id)
            normalized = replace(request, library_id=library_id)
            return self._search_scope(
                bundle,
                normalized,
                _scope_for_normalized_library(library_id),
            )
        scope, resolved_region = self._scope_for_cwd_with_region(
            request.cwd or "", request.session_id
        )
        if resolved_region is not None:
            return self._search_region_session(bundle, request, resolved_region, "project_cwd")
        if request.topic_id is not None:
            topic_id = request.topic_id.strip()
            if not topic_id:
                raise ValueError("topic_id must not be blank")
            region = self._regions.require_region(topic_id)
            if region.kind != "topic":
                raise ValueError("topic_id must identify a topic memory region")
            return self._search_region_session(bundle, request, region, "topic_session")
        return self._search_non_project(bundle, request, scope)

    def _search_region_session(
        self,
        bundle: _RuntimeBundle,
        request: SearchMemoryRequest,
        region: MemoryRegion,
        scope_kind: str,
    ) -> dict[str, object]:
        targets: list[tuple[Scope, MemoryRegion]] = []
        base_session = region.session_id if region.kind == "session" else ""
        for agent in _SEARCH_AGENTS:
            base_scope = _scope_for_region(
                region,
                session_id=base_session,
                agent=agent,
            )
            if self._scope_has_units(bundle, base_scope):
                targets.append((base_scope, region))
            if request.session_id and region.kind != "session":
                session_scope = _scope_for_region(
                    region,
                    session_id=request.session_id,
                    agent=agent,
                )
                if self._scope_has_units(bundle, session_scope):
                    targets.append((session_scope, region))
        response = self._search_across(
            bundle,
            request,
            targets,
            scope_kind=scope_kind,
            region_ids=[region.region_id],
            regions_by_id={region.region_id: region},
        )
        response["library_id"] = region.region_id
        response["region_id"] = region.region_id
        response["space"] = region.space
        scope = response.get("scope")
        if isinstance(scope, dict):
            scope["space"] = region.space
            scope["library_id"] = region.region_id
            scope["region_id"] = region.region_id
        return response

    def _search_non_project(
        self,
        bundle: _RuntimeBundle,
        request: SearchMemoryRequest,
        synthetic_scope: Scope,
    ) -> dict[str, object]:
        user_region = self._regions.ensure_user_region()
        if request.session_id:
            session_region = self._ensure_session_region(request.session_id)
        else:
            digest = hashlib.sha256(synthetic_scope.space.encode("utf-8")).hexdigest()[:24]
            session_region = MemoryRegion(
                region_id=f"session-{digest}",
                kind="session",
                name="当前会话",
                space="chat",
                session_id="",
                temporary=True,
            )
        targets: list[tuple[Scope, MemoryRegion]] = []
        for agent in _SEARCH_AGENTS:
            user_scope = _scope_for_region(user_region, session_id="", agent=agent)
            if self._scope_has_units(bundle, user_scope):
                targets.append((user_scope, user_region))
            session_scope = _scope_for_region(
                session_region,
                session_id=session_region.session_id,
                agent=agent,
            )
            if self._scope_has_units(bundle, session_scope):
                targets.append((session_scope, session_region))
        return self._search_across(
            bundle,
            request,
            targets,
            scope_kind="user_and_session",
            region_ids=[user_region.region_id, session_region.region_id],
            regions_by_id={
                user_region.region_id: user_region,
                session_region.region_id: session_region,
            },
        )

    def _search_regions(
        self, bundle: _RuntimeBundle, request: SearchMemoryRequest
    ) -> dict[str, object]:
        region_ids = request.region_ids or ()
        regions = [self._regions.require_region(region_id) for region_id in region_ids]
        if any(region.kind == "session" for region in regions):
            raise ValueError(
                "cross-region recall only supports user, topic, and project memory regions"
            )
        targets: list[tuple[Scope, MemoryRegion]] = []
        for region in regions:
            base_session = region.session_id if region.kind == "session" else ""
            for agent in _SEARCH_AGENTS:
                base_scope = _scope_for_region(
                    region,
                    session_id=base_session,
                    agent=agent,
                )
                if self._scope_has_units(bundle, base_scope):
                    targets.append((base_scope, region))
                if request.session_id and region.kind != "session":
                    session_scope = _scope_for_region(
                        region,
                        session_id=request.session_id,
                        agent=agent,
                    )
                    if self._scope_has_units(bundle, session_scope):
                        targets.append((session_scope, region))
        return self._search_across(
            bundle,
            request,
            targets,
            scope_kind="explicit_regions",
            region_ids=list(region_ids),
            regions_by_id={region.region_id: region for region in regions},
        )

    def _search_across(
        self,
        bundle: _RuntimeBundle,
        request: SearchMemoryRequest,
        targets: list[tuple[Scope, MemoryRegion]],
        *,
        scope_kind: str,
        region_ids: list[str],
        regions_by_id: Mapping[str, MemoryRegion],
    ) -> dict[str, object]:
        merged: dict[str, dict[str, object]] = {}
        degradation: list[str] = []
        vector_statuses: set[str] = set()
        graph_statuses: set[str] = set()
        groups: dict[str, dict[str, object]] = {}
        group_vector_statuses: dict[str, set[str]] = {}
        group_graph_statuses: dict[str, set[str]] = {}
        scope_attempts: dict[str, int] = {}
        scope_failures: dict[str, int] = {}
        source_references: dict[str, list[str]] = {}
        retrieval_errors: list[dict[str, object]] = []
        for region_id in region_ids:
            region = regions_by_id[region_id]
            groups[region_id] = {
                "region_id": region_id,
                "kind": region.kind,
                "name": region.name,
                "space": region.space,
                "status": "ok",
                "degraded": False,
                "degradation": [],
                "vector_index_status": "not_configured",
                "graph_index_status": "not_configured",
                "result_ids": [],
                "source_references": [],
            }
            group_vector_statuses[region_id] = set()
            group_graph_statuses[region_id] = set()
            scope_attempts[region_id] = 0
            scope_failures[region_id] = 0
            source_references[region_id] = []
        for scope, region in targets:
            group = groups[region.region_id]
            scope_attempts[region.region_id] += 1
            scoped_request = replace(
                request,
                cwd=None,
                library_id=None,
                region_id=region.region_id,
                region_ids=None,
                topic_id=None,
            )
            try:
                response = self._search_scope(bundle, scoped_request, scope, region=region)
            except Exception as error:
                scope_failures[region.region_id] += 1
                group_degradation = cast(list[object], group["degradation"])
                if "retrieval_unavailable" not in group_degradation:
                    group_degradation.append("retrieval_unavailable")
                if "retrieval_unavailable" not in degradation:
                    degradation.append("retrieval_unavailable")
                group["degraded"] = True
                group["status"] = "degraded"
                classification = _classify_failure(
                    error, embedding_configured=self._vector_enabled
                )
                if classification not in degradation:
                    degradation.append(classification)
                continue
            raw_degradation = response.get("degradation", [])
            if isinstance(raw_degradation, list):
                group_degradation = cast(list[object], group["degradation"])
                for reason in raw_degradation:
                    reason_text = str(reason)
                    if reason_text not in degradation:
                        degradation.append(reason_text)
                    if reason_text not in {
                        "default_recall_candidate_truncated",
                        "fulltext_unavailable",
                        "graph_unavailable",
                        "model_unavailable",
                        "retrieval_unavailable",
                        "vector_unavailable",
                    }:
                        continue
                    if reason_text not in group_degradation:
                        group_degradation.append(reason_text)
                    group["degraded"] = True
            raw_errors = response.get("retrieval_errors", [])
            if isinstance(raw_errors, list):
                for raw_error in raw_errors:
                    if isinstance(raw_error, dict) and raw_error not in retrieval_errors:
                        retrieval_errors.append(raw_error)
            vector_status = str(response.get("vector_index_status", "degraded"))
            graph_status = str(response.get("graph_index_status", "degraded"))
            vector_statuses.add(vector_status)
            graph_statuses.add(graph_status)
            group_vector_statuses[region.region_id].add(vector_status)
            group_graph_statuses[region.region_id].add(graph_status)
            raw_results = response.get("results", [])
            if not isinstance(raw_results, list):
                continue
            for raw_result in raw_results:
                if not isinstance(raw_result, dict):
                    continue
                item = {str(key): value for key, value in raw_result.items()}
                item["region_id"] = region.region_id
                item["space"] = region.space
                item["library_id"] = region.region_id
                references = _result_source_references(item)
                item["source_references"] = references
                for reference in references:
                    if reference not in source_references[region.region_id]:
                        source_references[region.region_id].append(reference)
                item_id = str(
                    item.get("document_id")
                    or item.get("chunk_id")
                    or f"anonymous-{len(merged)}"
                )
                merged_key = f"{region.region_id}\0{item_id}"
                previous = merged.get(merged_key)
                item_score = float(str(item.get("score", 0.0)))
                previous_score = 0.0 if previous is None else float(str(previous.get("score", 0.0)))
                if previous is None or item_score > previous_score:
                    merged[merged_key] = item
        for region_id, group in groups.items():
            failures = scope_failures[region_id]
            attempts = scope_attempts[region_id]
            group_degradation = cast(list[object], group["degradation"])
            if attempts and failures == attempts:
                group["status"] = "unavailable"
                group["degraded"] = True
                if "region_unavailable" not in degradation:
                    degradation.append("region_unavailable")
                marker = f"region_unavailable:{region_id}"
                if marker not in degradation:
                    degradation.append(marker)
            elif failures:
                group["status"] = "degraded"
                group["degraded"] = True
                marker = f"region_degraded:{region_id}"
                if marker not in degradation:
                    degradation.append(marker)
            elif group_degradation:
                group["status"] = "degraded"
                group["degraded"] = True
            group["vector_index_status"] = _combined_index_status(
                group_vector_statuses[region_id]
            )
            group["graph_index_status"] = _combined_index_status(
                group_graph_statuses[region_id]
            )
            group["source_references"] = source_references[region_id]
        ranked = sorted(
            merged.values(),
            key=lambda item: float(str(item.get("score", 0.0))),
            reverse=True,
        )
        results = _truncate_search_results(
            ranked,
            max(1, min(request.limit, 100)),
        )
        if vector_statuses == {"ready"}:
            vector_status = "ready"
        elif "degraded" in vector_statuses:
            vector_status = "degraded"
        else:
            vector_status = "not_configured"
        if graph_statuses == {"ready"}:
            graph_status = "ready"
        elif "degraded" in graph_statuses:
            graph_status = "degraded"
        else:
            graph_status = "not_configured"
        result: dict[str, object] = {
            "status": "bound",
            "scope": {
                "kind": scope_kind,
                "cwd": request.cwd,
                "session_id": request.session_id,
                "topic_id": request.topic_id,
                "region_ids": region_ids,
                "spaces": list(
                    dict.fromkeys(region.space for region in regions_by_id.values())
                ),
            },
            "region_ids": region_ids,
            "query": request.query,
            "degraded": bool(degradation),
            "degradation": degradation,
            "vector_index_status": vector_status,
            "graph_index_status": graph_status,
            "results": results,
        }
        if retrieval_errors:
            result["retrieval_errors"] = retrieval_errors
        if scope_kind == "explicit_regions":
            result["region_groups"] = [groups[region_id] for region_id in region_ids]
        return result

    def _search_scope(
        self,
        bundle: _RuntimeBundle,
        request: SearchMemoryRequest,
        scope: Scope,
        *,
        region: MemoryRegion | None = None,
    ) -> dict[str, object]:
        security = _security_for(scope)
        degradation: list[str] = []
        embedding_configured = self._vector_enabled
        if not embedding_configured:
            degradation.append(self._vector_degradation)
            items, failed, truncated = self._keyword_fallback(bundle, scope, request)
            if failed:
                degradation.append("fulltext_unavailable")
            if truncated:
                degradation.append("default_recall_candidate_truncated")
            items = self._append_graph_expansions(
                bundle, scope, items, request, degradation, region
            )
            graph_status = self._graph_ready_status(bundle, scope, region)
            return _search_response(
                request,
                scope,
                items,
                [],
                degradation,
                vector_status="not_configured",
                graph_status=graph_status,
                region=region,
            )
        try:
            items, reported_errors, truncated = self._search_scope_candidates(
                bundle.runtime.api,
                scope,
                request,
                security,
            )
        except Exception as retrieval_error:
            classification = _classify_failure(
                retrieval_error, embedding_configured=embedding_configured
            )
            if classification not in degradation:
                degradation.append(classification)
            units, failed, truncated = self._keyword_fallback(bundle, scope, request)
            if failed and "fulltext_unavailable" not in degradation:
                degradation.append("fulltext_unavailable")
            if truncated:
                degradation.append("default_recall_candidate_truncated")
            units = self._append_graph_expansions(
                bundle, scope, units, request, degradation, region
            )
            graph_status = self._graph_ready_status(bundle, scope, region)
            return _search_response(
                request,
                scope,
                units,
                [],
                degradation,
                vector_status="degraded",
                graph_status=graph_status,
                region=region,
            )

        if not items and reported_errors:
            recovered, failed, truncated = self._keyword_fallback(bundle, scope, request)
            if recovered:
                items = recovered
            elif failed and "fulltext_unavailable" not in degradation:
                degradation.append("fulltext_unavailable")
            if truncated:
                degradation.append("default_recall_candidate_truncated")
        elif truncated:
            degradation.append("default_recall_candidate_truncated")
        for channel_error in reported_errors:
            channel = getattr(channel_error.channel, "value", str(channel_error.channel))
            failure = _channel_failure(channel, channel_error)
            if failure not in degradation:
                degradation.append(failure)
        vector_failed = any(
            getattr(channel_error.channel, "value", str(channel_error.channel)) == "vector"
            for channel_error in reported_errors
        )
        vector_status = (
            "not_configured"
            if not embedding_configured
            else "degraded"
            if vector_failed
            else "ready"
        )
        items = self._append_graph_expansions(bundle, scope, items, request, degradation, region)
        graph_status = self._graph_ready_status(bundle, scope, region)
        return _search_response(
            request,
            scope,
            items,
            reported_errors,
            degradation,
            vector_status=vector_status,
            graph_status=graph_status,
            region=region,
        )

    @staticmethod
    def _search_scope_candidates(
        api: Any,
        scope: Scope,
        request: SearchMemoryRequest,
        security: RequestSecurityContext,
    ) -> tuple[list[Any], list[Any], bool]:
        limit = max(1, min(request.limit, 100))
        candidate_top_k = min(
            max(limit * 4, 20),
            _MAX_RECALL_CANDIDATE_POOL,
        )
        while True:
            result = api.search(
                request.query,
                Context(scope=scope),
                security=security,
                filters=_default_recall_filter(),
                top_k=candidate_top_k,
                disclosure=DisclosureLevel.ADAPTIVE,
            )
            items = [
                item for item in result.items if _is_default_recall_candidate(item)
            ]
            if (
                len(items) >= limit
                or len(result.items) < candidate_top_k
                or candidate_top_k >= _MAX_RECALL_CANDIDATE_POOL
            ):
                return (
                    items,
                    list(result.errors),
                    (
                        candidate_top_k >= _MAX_RECALL_CANDIDATE_POOL
                        and len(result.items) >= candidate_top_k
                        and len(items) < limit
                    ),
                )
            candidate_top_k = min(
                candidate_top_k * 2,
                _MAX_RECALL_CANDIDATE_POOL,
            )

    def _graph_ready_status(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        region: MemoryRegion | None,
    ) -> str:
        if bundle.graph_store is None:
            return "not_configured"
        with self._graph_write_lock:
            return self._graph_ready_status_locked(bundle, scope, region)

    def _graph_ready_status_locked(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        region: MemoryRegion | None,
    ) -> str:
        graph_store = bundle.graph_store
        assert graph_store is not None
        transient_error = self._graph_error_for(scope=scope)
        if transient_error is None and region is not None:
            transient_error = self._graph_error_for(region_id=region.region_id)
        if transient_error is not None:
            return "degraded"
        projection_error = self._graph_projection_error(scope=scope)
        if projection_error is None and region is not None:
            projection_error = self._graph_projection_error(region_id=region.region_id)
        if (
            projection_error is None
            and region is None
            and scope.space.startswith(LIBRARY_SCOPE_PREFIX)
        ):
            projection_error = self._graph_projection_error(
                region_id=scope.space.removeprefix(LIBRARY_SCOPE_PREFIX)
            )
        if projection_error is not None:
            return "degraded"
        if self._graph_store_error(bundle, scope):
            return "degraded"
        try:
            graph_store.health()
            _, _, mismatch = self._graph_projection_snapshot(bundle, scope)
            if mismatch is not None:
                return "degraded"
        except Exception:
            return "degraded"
        return "ready"

    def _append_graph_expansions(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        items: list[Any],
        request: SearchMemoryRequest,
        degradation: list[str],
        region: MemoryRegion | None,
    ) -> list[Any]:
        graph_store = bundle.graph_store
        if graph_store is None:
            return list(items)
        with self._graph_write_lock:
            return self._append_graph_expansions_locked(
                bundle,
                scope,
                items,
                request,
                degradation,
                region,
            )

    def _append_graph_expansions_locked(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        items: list[Any],
        request: SearchMemoryRequest,
        degradation: list[str],
        region: MemoryRegion | None,
    ) -> list[Any]:
        graph_store = bundle.graph_store
        if graph_store is None:
            return list(items)
        projection_error = self._graph_projection_error(scope=scope)
        if projection_error is None and region is not None:
            projection_error = self._graph_projection_error(region_id=region.region_id)
        if (
            projection_error is None
            and region is None
            and scope.space.startswith(LIBRARY_SCOPE_PREFIX)
        ):
            projection_error = self._graph_projection_error(
                region_id=scope.space.removeprefix(LIBRARY_SCOPE_PREFIX)
            )
        if projection_error is not None:
            self._set_graph_error(
                projection_error,
                region_id=region.region_id if region is not None else None,
                scope=scope,
            )
            if "graph_unavailable" not in degradation:
                degradation.append("graph_unavailable")
            return list(items)
        store_error = self._graph_store_error(bundle, scope)
        if store_error is not None:
            self._set_graph_error(
                store_error,
                region_id=region.region_id if region is not None else None,
                scope=scope,
            )
            if "graph_unavailable" not in degradation:
                degradation.append("graph_unavailable")
            return list(items)
        _, _, projection_mismatch = self._graph_projection_snapshot(bundle, scope)
        if projection_mismatch is not None:
            self._set_graph_error(
                projection_mismatch,
                region_id=region.region_id if region is not None else None,
                scope=scope,
            )
            if "graph_unavailable" not in degradation:
                degradation.append("graph_unavailable")
            return list(items)
        try:
            graph_store.health()
            if not items:
                return list(items)
            direct_ids = {
                str(item.id if isinstance(item, MemoryUnit) else item.unit_id)
                for item in items
                if isinstance(item, MemoryUnit) or hasattr(item, "unit_id")
                if _is_default_recall_item(item)
            }
            if not direct_ids:
                return list(items)
            seeds = graph_store.get(scope, sorted(direct_ids))
            if not seeds:
                return list(items)
            depth = max(1, min(request.max_graph_hops, 2))
            limit = max(1, min(request.limit, 100))
            traversed = graph_store.traverse_many(
                scope,
                [seed.id for seed in seeds],
                depth=depth,
                limit=limit,
            )
            units_by_id = {
                unit.id: unit
                for unit in load_units(
                    bundle.kv_store,
                    scope,
                    [node.id for node, _, _ in traversed],
                )
            }
            expanded: list[dict[str, object]] = []
            for node, hop, seed_id in traversed:
                unit = units_by_id.get(node.id)
                if unit is None:
                    continue
                if unit.lifecycle.value != "active":
                    continue
                if unit.user_metadata.get("restore_pending") is True:
                    continue
                if not _is_default_recall_item(unit):
                    continue
                payload = _result_item(unit, request)
                payload["classification"] = "graph_expansion"
                payload["graph_hop"] = hop
                payload["graph_object_type"] = "node"
                payload["graph_source_id"] = seed_id
                payload["retrieval_sources"] = ["graph"]
                expanded.append(payload)
            self._clear_graph_error(
                region_id=region.region_id if region is not None else None,
                scope=scope,
            )
            return [*items, *expanded]
        except Exception as error:
            self._set_graph_error(
                str(error),
                region_id=region.region_id if region is not None else None,
                scope=scope,
            )
            if "graph_unavailable" not in degradation:
                degradation.append("graph_unavailable")
            return list(items)

    def _scope_for_cwd(self, cwd: str, session_id: str | None) -> Scope:
        scope, _ = self._scope_for_cwd_with_region(cwd, session_id)
        return scope

    def _scope_for_cwd_with_region(
        self, cwd: str, session_id: str | None
    ) -> tuple[Scope, MemoryRegion | None]:
        try:
            resolved = self._regions.resolve(cwd)
        except RegionBindingError:
            resolved = None
        if resolved is None:
            return _scope_for_synthetic_cwd(cwd, session_id), None
        return _scope_for_region(resolved.region, session_id=session_id), resolved.region

    def _keyword_fallback(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        request: SearchMemoryRequest,
    ) -> tuple[list[Any], bool, bool]:
        fulltext = next(
            (
                store
                for store in bundle.stores
                if isinstance(store, MilvusLiteFulltextStore)
                and getattr(store, "_collection", "") == "agent_memory_fulltext"
            ),
            None,
        )
        if fulltext is None:
            return [], True, False
        try:
            limit = max(1, min(request.limit, 100))
            candidate_top_k = min(
                max(limit * 4, 20),
                _MAX_RECALL_CANDIDATE_POOL,
            )
            while True:
                hits: list[ScoredID] = fulltext.search(
                    scope,
                    TextQuery(
                        text=request.query,
                        top_k=candidate_top_k,
                        filters=_default_recall_filter(),
                    ),
                )
                units = list(
                    load_units(bundle.kv_store, scope, [hit.id for hit in hits])
                )
                visible = [
                    unit for unit in units if _is_default_recall_candidate(unit)
                ]
                if (
                    len(visible) >= limit
                    or len(hits) < candidate_top_k
                    or candidate_top_k >= _MAX_RECALL_CANDIDATE_POOL
                ):
                    break
                candidate_top_k = min(
                    candidate_top_k * 2,
                    _MAX_RECALL_CANDIDATE_POOL,
                )
            if not hits:
                return [], False, False
            truncated = (
                candidate_top_k >= _MAX_RECALL_CANDIDATE_POOL
                and len(hits) >= candidate_top_k
                and len(visible) < limit
            )
            return visible, False, truncated
        except Exception:
            return [], True, False

    def list_documents(self, request: ListDocumentsRequest) -> list[dict[str, object]]:
        bundle = self._require_bundle()
        region = self._governance_region(request.library_id)
        payloads = {
            unit.id: _unit_payload(unit, region)
            for _, unit in self._region_unit_pairs(bundle, region)
        }
        return [payloads[key] for key in sorted(payloads)]

    def list_memories(self, request: ListMemoriesRequest) -> dict[str, object]:
        with self._governance_region_lock(request.library_id):
            return self._list_memories_locked(request)

    def _list_memories_locked(self, request: ListMemoriesRequest) -> dict[str, object]:
        bundle = self._require_bundle()
        region = self._governance_region(request.library_id)
        self._retry_pending_browser_index(bundle, region_id=region.region_id)
        where = ["region_id = ?", "lifecycle IN ('active', 'superseded', 'archived')"]
        values: list[object] = [region.region_id]
        if request.memory_type is not None:
            where.append("memory_type = ?")
            values.append(request.memory_type)
        if request.lifecycle is not None:
            where.append("lifecycle = ?")
            values.append(request.lifecycle)
        if request.q is not None:
            where.append("instr(lower(content), lower(?)) > 0")
            values.append(request.q)
        if request.cursor is not None:
            cursor_updated, cursor_id = _decode_memory_cursor(request.cursor)
            where.append("(updated_at < ? OR (updated_at = ? AND unit_id < ?))")
            values.extend(
                [cursor_updated.isoformat(), cursor_updated.isoformat(), cursor_id]
            )
        sql = (
            "SELECT unit_id, summary, memory_type, memory_type_resolution, lifecycle, "
            "updated_at, source_count, version "
            "FROM memory_browser_index WHERE "
            + " AND ".join(where)
            + " ORDER BY updated_at DESC, unit_id DESC LIMIT ?"
        )
        with closing(self._maintenance_db()) as connection:
            rows = connection.execute(sql, (*values, request.limit + 1)).fetchall()
        has_more = len(rows) > request.limit
        rows = rows[: request.limit]
        next_cursor = _encode_memory_cursor_rows(rows[-1]) if has_more and rows else None
        return {
            "items": [
                {
                    "id": str(row[0]),
                    "path": str(row[0]),
                    "region_id": region.region_id,
                    "summary": str(row[1]),
                    "memory_type": str(row[2]),
                    "memory_type_resolution": str(row[3]),
                    "lifecycle": str(row[4]),
                    "updated_at": str(row[5]),
                    "source_count": int(row[6]),
                    "version": str(row[7]),
                }
                for row in rows
            ],
            "next_cursor": next_cursor,
            "has_more": has_more,
            "page_size": request.limit,
        }

    def inspect(self, request: object) -> dict[str, object]:
        if not isinstance(request, DocumentInspectRequest):
            self._raise()
        bundle = self._require_bundle()
        region = self._governance_region(request.library_id)
        scope, unit = self._find_region_unit(bundle, region, request.path)
        payload = _unit_payload(unit, region)
        pairs = self._region_unit_pairs(bundle, region)
        history = [_unit_payload(item, region) for _, item in _version_chain(unit, pairs)]
        payload["history"] = history
        payload["versions"] = history
        try:
            trace = bundle.runtime.api.trace(
                unit.id,
                scope,
                security=_security_for(scope),
            )
        except Exception:
            trace = []
        trace = [item for item in trace if item.id != unit.id]
        if not trace:
            references = unit.user_metadata.get("provenance")
            if not isinstance(references, list):
                references = unit.user_metadata.get("source_unit_ids")
            if isinstance(references, list):
                by_id = {item.id: item for _, item in pairs}
                trace = [
                    by_id[reference]
                    for reference in (str(value) for value in references)
                    if reference in by_id
                ]
        payload["provenance_chain"] = [_unit_payload(item, region) for item in trace]
        return payload

    def inspect_memory(self, request: DocumentInspectRequest) -> dict[str, object]:
        with self._governance_region_lock(request.library_id):
            return self._inspect_memory_locked(request)

    def _inspect_memory_locked(self, request: DocumentInspectRequest) -> dict[str, object]:
        bundle = self._require_bundle()
        region = self._governance_region(request.library_id)
        self._retry_pending_browser_index(bundle, region_id=region.region_id)
        with closing(self._maintenance_db()) as connection:
            row = connection.execute(
                """SELECT scope_org, scope_space, scope_user, scope_agent, scope_session,
                          memory_type, memory_type_resolution, logical_memory_id, lifecycle
                   FROM memory_browser_index WHERE region_id = ? AND unit_id = ?""",
                (region.region_id, request.path.strip()),
            ).fetchone()
        if row is None:
            raise MemoryRecordNotFoundError("memory record not found")
        scope = Scope(
            org=str(row[0]),
            user=str(row[2]),
            agent=str(row[3]),
            session=str(row[4]),
            space=str(row[1]),
        )
        units = load_units(bundle.kv_store, scope, [request.path.strip()])
        if not units:
            raise MemoryRecordNotFoundError("memory record not found")
        unit = units[0]
        logical_memory_id = str(row[7]) or self._resolved_unit_lineage_id(region, unit)
        terminal_lifecycle = self._lineage_terminal_lifecycle(
            region.region_id,
            logical_memory_id,
        )
        if unit.lifecycle.value == "forgotten" or terminal_lifecycle == "forgotten":
            raise MemoryRecordNotFoundError("memory record not found")
        payload = _unit_payload(unit, region)
        payload["memory_type"] = str(row[5])
        payload["memory_type_resolution"] = str(row[6])
        payload["region"] = {
            "region_id": region.region_id,
            "name": region.name,
            "kind": region.kind,
        }
        payload["evidence"] = self._browser_evidence(bundle, region, unit)
        payload["semantic_versions"] = self._semantic_versions_for_memory(
            unit.id,
            region_id=region.region_id,
        )
        payload["provenance_summary"] = self._memory_provenance_summary(
            region,
            payload,
            payload["evidence"],
        )
        history_entry = {
            key: value
            for key, value in payload.items()
            if key not in {"history", "versions", "provenance_chain", "evidence"}
        }
        payload["history"] = [history_entry]
        payload["versions"] = [history_entry]
        payload["provenance_chain"] = payload["evidence"]
        return payload

    async def update(self, request: MemoryUpdateRequest) -> dict[str, object]:
        return await asyncio.to_thread(self._update_document_serialized, request)

    def _update_document_serialized(self, request: MemoryUpdateRequest) -> dict[str, object]:
        if not isinstance(request, DocumentUpdateRequest):
            return self._update_document(request)
        with self._governance_region_lock(request.library_id):
            return self._update_document(request)

    def _update_document(self, request: MemoryUpdateRequest) -> dict[str, object]:
        if not isinstance(request, DocumentUpdateRequest):
            self._raise()
        bundle = self._require_bundle()
        region = self._governance_region(request.library_id)
        self._retry_pending_browser_index(bundle, region_id=region.region_id)
        scope, unit = self._find_region_unit(bundle, region, request.path)
        _require_revision(unit, request.expected_revision)
        _require_active(unit, request.mode)
        mode = UpdateMode.OVERWRITE if request.mode == "overwrite" else UpdateMode.SUPERSEDE
        logical_memory_id = self._resolved_unit_lineage_id(region, unit)
        governance_updated_at = datetime.now(UTC).isoformat()
        index_token = self._begin_browser_index_pending(region, scope)
        try:
            event_id = self._begin_governance_event(
                region.region_id,
                unit.id,
                request.mode,
                request.actor,
                request.source,
                request.operation_id,
                before_content=unit.content,
                after_content=request.content,
                detail={"previous_unit_id": unit.id},
                logical_memory_id=logical_memory_id,
            )
        except Exception:
            self._complete_browser_index_pending(index_token)
            raise
        metadata_patch: dict[str, object] = {
            "governance_actor": request.actor,
            "governance_source": request.source,
            "operation_id": request.operation_id,
            "previous_revision": request.expected_revision,
            "governance_updated_at": governance_updated_at,
            "logical_memory_id": logical_memory_id,
        }
        if request.actor != "semantic-distillation":
            metadata_patch["semantic_owned"] = False
            metadata_patch["semantic_manual_override"] = True
        try:
            updated = bundle.runtime.api.update(
                unit.id,
                scope,
                MemoryPatch(
                    content=request.content,
                    mode=mode,
                    user_metadata=metadata_patch,
                ),
                security=_security_for(scope),
            )
        except Exception:
            self._complete_browser_index_pending(index_token)
            self._finish_governance_event(
                event_id,
                status="failed",
                unit_id=unit.id,
                after_content="",
                detail={"previous_unit_id": unit.id},
            )
            raise
        if request.actor != "semantic-distillation":
            self._clear_manual_override_semantic_metadata(
                bundle,
                scope,
                updated,
                region_id=region.region_id,
            )
        self._persist_unit_logical_memory_id(
            bundle,
            scope,
            unit.id,
            logical_memory_id,
        )
        self._persist_unit_logical_memory_id(
            bundle,
            scope,
            updated.id,
            logical_memory_id,
        )
        updated.user_metadata["logical_memory_id"] = logical_memory_id
        self._finish_governance_event(
            event_id,
            status="committed",
            unit_id=updated.id,
            after_content=updated.content,
            detail={"previous_unit_id": unit.id, "result_unit_id": updated.id},
        )
        browser_index_sync = self._sync_browser_index_best_effort(
            bundle,
            region,
            scope,
            index_token,
            [unit.id, updated.id],
        )
        if bundle.graph_store is not None:
            self._reproject_graph_scope(
                bundle,
                scope,
                region_id=region.region_id,
            )
        self._queue_entity_graph_update(region.region_id)
        payload = _unit_payload(updated, region)
        payload["browser_index_sync"] = browser_index_sync
        if browser_index_sync == "pending":
            payload["warning"] = "authoritative update committed; browser index sync is pending"
        return payload

    def preview_document_update(self, request: PreviewDocumentUpdateRequest) -> dict[str, object]:
        bundle = self._require_bundle()
        region = self._governance_region(request.library_id)
        self._retry_pending_browser_index(bundle, region_id=region.region_id)
        _, unit = self._find_region_unit(bundle, region, request.path)
        _require_revision(unit, request.expected_revision)
        _require_active(unit, "update preview")
        return {
            "path": unit.id,
            "id": unit.id,
            "current_content": unit.content,
            "proposed_content": request.content,
            "source_version": _unit_revision(unit),
            "preview_token": _unit_revision(unit),
            "lifecycle": unit.lifecycle.value,
            "provenance": list(unit.provenance),
            "diff": _content_diff(unit.content, request.content),
        }

    async def preview_document_delete(
        self, request: PreviewDocumentDeleteRequest
    ) -> dict[str, object]:
        return await asyncio.to_thread(self._preview_document_delete, request)

    def _preview_document_delete(self, request: PreviewDocumentDeleteRequest) -> dict[str, object]:
        bundle = self._require_bundle()
        region = self._governance_region(request.library_id)
        self._retry_pending_browser_index(bundle, region_id=region.region_id)
        _, unit = self._find_region_unit(bundle, region, request.path)
        _require_revision(unit, request.expected_revision)
        revision = _unit_revision(unit)
        logical_memory_id = self._resolved_unit_lineage_id(region, unit)
        affected_units = max(
            1,
            len(self._lineage_versions(region.region_id, logical_memory_id)),
        )
        return {
            "path": unit.id,
            "id": unit.id,
            "content": unit.content,
            "expected_source_version": revision,
            "preview_token": revision,
            "lifecycle": unit.lifecycle.value,
            "affected_units": affected_units,
            "provenance": list(unit.provenance),
        }

    async def delete(self, request: DeleteMemoryRequest) -> dict[str, object]:
        return await asyncio.to_thread(self._delete_document_serialized, request)

    def _delete_document_serialized(self, request: DeleteMemoryRequest) -> dict[str, object]:
        with self._governance_region_lock(request.library_id):
            return self._delete_document(request)

    def _delete_document(self, request: DeleteMemoryRequest) -> dict[str, object]:
        bundle = self._require_bundle()
        region = self._governance_region(request.library_id)
        self._retry_pending_browser_index(bundle, region_id=region.region_id)
        scope, unit = self._find_region_unit(bundle, region, request.path)
        revision = _unit_revision(unit)
        _require_revision(unit, request.expected_revision)
        if request.preview_token != revision:
            raise MemoryRevisionConflictError("deletion preview is stale or invalid")
        _require_delete_lifecycle(unit, request.mode)
        logical_memory_id = self._resolved_unit_lineage_id(region, unit)
        versions: list[tuple[str, Scope]] = [(unit.id, scope)]
        if request.mode == "forget":
            versions = self._lineage_versions(region.region_id, logical_memory_id)
            if unit.id not in {unit_id for unit_id, _ in versions}:
                versions.append((unit.id, scope))
        grouped: dict[tuple[str, str, str, str, str], tuple[Scope, list[str]]] = {}
        for unit_id, version_scope in versions:
            key = (
                version_scope.org,
                version_scope.space,
                version_scope.user,
                version_scope.agent,
                version_scope.session,
            )
            grouped.setdefault(key, (version_scope, []))[1].append(unit_id)
        index_tokens = {
            key: self._begin_browser_index_pending(region, version_scope, unit_ids)
            for key, (version_scope, unit_ids) in grouped.items()
        }
        affected_ids = list(dict.fromkeys(unit_id for unit_id, _ in versions))
        try:
            event_id = self._begin_governance_event(
                region.region_id,
                unit.id,
                f"delete:{request.mode}",
                request.actor,
                request.source,
                request.operation_id,
                before_content=unit.content,
                after_content="",
                detail={
                    "mode": request.mode,
                    "affected_unit_ids": affected_ids,
                    "logical_memory_id": logical_memory_id,
                },
                logical_memory_id=logical_memory_id,
            )
        except Exception:
            for token in index_tokens.values():
                self._complete_browser_index_pending(token)
            raise
        deleted_ids: list[str] = []
        try:
            for version_scope, unit_ids in grouped.values():
                deleted = bundle.runtime.api.delete(
                    DeleteSelector(
                        unit_ids=unit_ids,
                        scope=version_scope,
                        mode=DeleteMode(request.mode),
                    ),
                    security=_security_for(version_scope),
                )
                if deleted:
                    deleted_ids.extend(
                        [str(item) for item in deleted]
                        if isinstance(deleted, list)
                        else unit_ids
                    )
        except Exception:
            for token in index_tokens.values():
                self._complete_browser_index_pending(token)
            self._finish_governance_event(
                event_id,
                status="failed",
                unit_id=unit.id,
                after_content="",
                detail={
                    "mode": request.mode,
                    "affected_unit_ids": affected_ids,
                    "logical_memory_id": logical_memory_id,
                },
            )
            raise
        if deleted_ids and request.mode != "downweight":
            self._remove_graph_nodes(
                bundle,
                scope,
                list(dict.fromkeys(deleted_ids)),
                region.region_id,
            )
            self._queue_entity_graph_update(region.region_id)
        detail: dict[str, object] = {
            "deleted": bool(deleted_ids),
            "mode": request.mode,
            "affected_unit_ids": affected_ids,
            "logical_memory_id": logical_memory_id,
        }
        if request.mode in {"forget", "archive", "downweight"} and deleted_ids:
            try:
                now = datetime.now(UTC)
                for version_scope, unit_ids in grouped.values():
                    refreshed = load_units(bundle.kv_store, version_scope, unit_ids)
                    for governed in refreshed:
                        governed.user_metadata["governance_updated_at"] = now.isoformat()
                        if request.mode == "forget":
                            governed.temporal.t_invalid = now
                        bundle.kv_store.update(
                            version_scope,
                            memory_key(governed.id),
                            dumps(governed),
                        )
            except Exception:
                # The deletion already succeeded. Keep it successful and leave
                # a durable audit marker instead of returning a false 500.
                detail["deleted_at_persist_failed"] = True
        self._finish_governance_event(
            event_id,
            status="committed",
            unit_id=unit.id,
            after_content="",
            detail=detail,
        )
        sync_statuses = [
            self._sync_browser_index_best_effort(
                bundle,
                region,
                version_scope,
                index_tokens[key],
                unit_ids,
            )
            for key, (version_scope, unit_ids) in grouped.items()
        ]
        browser_index_sync = "pending" if "pending" in sync_statuses else "ready"
        response = {
            "id": unit.id,
            "path": unit.id,
            "status": request.mode,
            "deleted": "true" if deleted_ids else "false",
            "browser_index_sync": browser_index_sync,
            "affected_units": str(len(affected_ids)),
            "affected_unit_ids": affected_ids,
        }
        if browser_index_sync == "pending":
            response["warning"] = "authoritative delete committed; browser index sync is pending"
        return response

    def list_forgotten(self, request: ListForgottenRequest) -> list[dict[str, object]]:
        bundle = self._require_bundle()
        region = self._governance_region(request.library_id)
        pairs = self._region_unit_pairs(bundle, region)
        forgotten: list[dict[str, object]] = []
        for _, unit in pairs:
            if unit.lifecycle.value != "forgotten":
                continue
            if unit.user_metadata.get("restore_pending") is True:
                continue
            chain = _version_chain(unit, pairs)
            if any(item.lifecycle.value == "active" for _, item in chain):
                continue
            latest_forgotten = _latest_chain_version(
                [item for _, item in chain if item.lifecycle.value == "forgotten"]
            )
            if unit.id != latest_forgotten.id:
                continue
            payload = _unit_payload(unit, region)
            payload["history"] = [_unit_payload(item, region) for _, item in chain]
            forgotten.append(payload)
        return forgotten

    async def restore_forgotten(self, request: RestoreForgottenRequest) -> dict[str, str]:
        return await asyncio.to_thread(self._restore_forgotten_serialized, request)

    def _restore_forgotten_serialized(self, request: RestoreForgottenRequest) -> dict[str, str]:
        with self._governance_region_lock(request.library_id):
            return self._restore_forgotten(request)

    def _restore_forgotten(self, request: RestoreForgottenRequest) -> dict[str, str]:
        bundle = self._require_bundle()
        region = self._governance_region(request.library_id)
        pairs = self._region_unit_pairs(bundle, region)
        _, unit = self._find_region_unit(bundle, region, request.tombstone_id)
        if unit.lifecycle.value != "forgotten":
            raise MemoryGovernanceError("memory is not forgotten")
        chain = _version_chain(unit, pairs)
        chain_ids = {item.id for _, item in chain}
        latest_forgotten = _latest_chain_version(
            [item for _, item in chain if item.lifecycle.value == "forgotten"]
        )
        if unit.id != latest_forgotten.id:
            raise MemoryLifecycleConflictError("restore must use the latest forgotten version")
        old_scope, old = self._find_region_unit(bundle, region, request.commit)
        if old.id not in chain_ids:
            raise MemoryRecordNotFoundError("memory version not found in document history")
        active_versions = [
            item
            for _, item in pairs
            if item.lifecycle.value == "active"
            and (
                item.id in chain_ids
                or item.supersedes in chain_ids
                or str(item.user_metadata.get("restored_from", "")) in chain_ids
                or any(str(reference) in chain_ids for reference in item.provenance)
            )
        ]
        if active_versions:
            raise MemoryLifecycleConflictError("forgotten memory already has an active version")
        logical_memory_id = self._resolved_unit_lineage_id(region, latest_forgotten)
        event_id = self._begin_governance_event(
            region.region_id,
            latest_forgotten.id,
            "restore:forgotten",
            request.actor,
            request.source,
            request.operation_id,
            before_content=unit.content,
            after_content=old.content,
            detail={"restored_from": old.id, "previous_unit_id": latest_forgotten.id},
            logical_memory_id=logical_memory_id,
        )
        try:
            restored = bundle.runtime.api.add(
                old.content,
                old_scope,
                Modality.TEXT,
                security=_security_for(old_scope),
                user_metadata={
                    **dict(old.user_metadata),
                    "operation_id": request.operation_id,
                    "governance_actor": request.actor,
                    "governance_source": request.source,
                    "restored_from": old.id,
                    "provenance": [old.id],
                    "restore_pending": True,
                    "logical_memory_id": logical_memory_id,
                },
            )
        except Exception:
            self._finish_governance_event(
                event_id,
                status="failed",
                unit_id=latest_forgotten.id,
                after_content="",
                detail={
                    "restored_from": old.id,
                    "previous_unit_id": latest_forgotten.id,
                },
            )
            raise
        if not restored:
            self._finish_governance_event(
                event_id,
                status="failed",
                unit_id=latest_forgotten.id,
                after_content="",
                detail={
                    "restored_from": old.id,
                    "previous_unit_id": latest_forgotten.id,
                },
            )
            raise MemoryKernelUnavailableError("forgotten memory restore created no unit")
        restored_unit = restored[0]
        restored_unit.supersedes = latest_forgotten.id
        restored_unit.provenance = [old.id]
        restored_unit.user_metadata.pop("restore_pending", None)
        try:
            bundle.kv_store.update(old_scope, memory_key(restored_unit.id), dumps(restored_unit))
        except Exception:
            try:
                bundle.runtime.api.delete(
                    DeleteSelector(
                        unit_ids=[restored_unit.id],
                        scope=old_scope,
                        mode=DeleteMode.PURGE,
                    ),
                    security=_security_for(old_scope),
                )
            except Exception as purge_error:
                try:
                    bundle.runtime.api.delete(
                        DeleteSelector(
                            unit_ids=[restored_unit.id],
                            scope=old_scope,
                            mode=DeleteMode.FORGET,
                        ),
                        security=_security_for(old_scope),
                    )
                except Exception as forget_error:
                    self._finish_governance_event(
                        event_id,
                        status="pending",
                        unit_id=restored_unit.id,
                        after_content=restored_unit.content,
                        detail={
                            "restored_from": old.id,
                            "previous_unit_id": latest_forgotten.id,
                            "version_link_pending": True,
                            "rollback_error": f"purge={purge_error}; forget={forget_error}",
                        },
                    )
                    raise MemoryKernelUnavailableError(
                        "forgotten memory restore failed and rollback is pending"
                    ) from None
                self._finish_governance_event(
                    event_id,
                    status="failed",
                    unit_id=latest_forgotten.id,
                    after_content="",
                    detail={
                        "restored_from": old.id,
                        "previous_unit_id": latest_forgotten.id,
                        "rollback": "forgotten",
                    },
                )
                raise MemoryKernelUnavailableError(
                    "forgotten memory restore failed; orphan version was forgotten"
                ) from None
            self._finish_governance_event(
                event_id,
                status="failed",
                unit_id=latest_forgotten.id,
                after_content="",
                detail={
                    "restored_from": old.id,
                    "previous_unit_id": latest_forgotten.id,
                    "rollback": "purged",
                },
            )
            raise MemoryKernelUnavailableError(
                "forgotten memory restore could not persist its version link"
            ) from None
        self._refresh_browser_index_ids(
            bundle,
            region,
            old_scope,
            [latest_forgotten.id, restored_unit.id],
        )
        if bundle.graph_store is not None:
            self._reproject_graph_scope(
                bundle,
                old_scope,
                region_id=region.region_id,
            )
        self._queue_entity_graph_update(region.region_id)
        self._finish_governance_event(
            event_id,
            status="committed",
            unit_id=restored_unit.id,
            after_content=restored_unit.content,
            detail={
                "restored_from": old.id,
                "previous_unit_id": latest_forgotten.id,
            },
        )
        payload = _unit_payload(restored_unit, region)
        return {key: str(value) for key, value in payload.items()}

    def history_diff(self, request: HistoryDiffRequest) -> dict[str, object]:
        bundle = self._require_bundle()
        region = self._governance_region(request.library_id)
        _, unit = self._find_region_unit(bundle, region, request.commit)
        pairs = self._region_unit_pairs(bundle, region)
        chain = [item for _, item in _version_chain(unit, pairs)]
        active = [item for item in chain if item.lifecycle.value == "active"]
        current = active[-1] if active else chain[-1]
        return {
            "library_id": region.region_id,
            "commit": unit.id,
            "current": current.id,
            "before": unit.content,
            "after": current.content,
            "diff": _content_diff(unit.content, current.content),
        }

    async def restore_document(self, request: RestoreDocumentRequest) -> dict[str, str]:
        return await asyncio.to_thread(self._restore_document_serialized, request)

    def _restore_document_serialized(self, request: RestoreDocumentRequest) -> dict[str, str]:
        with self._governance_region_lock(request.library_id):
            return self._restore_document(request)

    def _governance_region_lock(self, library_id: str) -> threading.RLock:
        region_id = validate_library_id(library_id)
        with self._governance_locks_guard:
            return self._governance_locks.setdefault(region_id, threading.RLock())

    def _restore_document(self, request: RestoreDocumentRequest) -> dict[str, str]:
        bundle = self._require_bundle()
        region = self._governance_region(request.library_id)
        scope, current = self._find_region_unit(bundle, region, request.path)
        _, old = self._find_region_unit(bundle, region, request.commit)
        pairs = self._region_unit_pairs(bundle, region)
        chain_ids = {item.id for _, item in _version_chain(current, pairs)}
        if old.id not in chain_ids:
            raise MemoryRecordNotFoundError("memory version not found in document history")
        _require_revision(current, request.expected_revision)
        _require_active(current, "history restore")
        logical_memory_id = self._resolved_unit_lineage_id(region, current)
        event_id = self._begin_governance_event(
            region.region_id,
            current.id,
            "restore",
            request.actor,
            request.source,
            request.operation_id,
            before_content=current.content,
            after_content=old.content,
            detail={"restored_from": old.id, "previous_unit_id": current.id},
            logical_memory_id=logical_memory_id,
        )
        try:
            restored = bundle.runtime.api.update(
                current.id,
                scope,
                MemoryPatch(
                    content=old.content,
                    mode=UpdateMode.SUPERSEDE,
                    user_metadata={
                        "operation_id": request.operation_id,
                        "governance_actor": request.actor,
                        "governance_source": request.source,
                        "restored_from": old.id,
                        "logical_memory_id": logical_memory_id,
                    },
                ),
                security=_security_for(scope),
            )
        except Exception:
            self._finish_governance_event(
                event_id,
                status="failed",
                unit_id=current.id,
                after_content="",
                detail={"restored_from": old.id, "previous_unit_id": current.id},
            )
            raise
        self._refresh_browser_index_ids(bundle, region, scope, [current.id, restored.id])
        if bundle.graph_store is not None:
            self._reproject_graph_scope(
                bundle,
                scope,
                region_id=region.region_id,
            )
        self._queue_entity_graph_update(region.region_id)
        self._finish_governance_event(
            event_id,
            status="committed",
            unit_id=restored.id,
            after_content=restored.content,
            detail={"restored_from": old.id, "previous_unit_id": current.id},
        )
        payload = _unit_payload(restored, region)
        return {key: str(value) for key, value in payload.items()}

    def audit(self, request: AuditMemoryRequest) -> list[dict[str, str]]:
        self._require_bundle()
        region: MemoryRegion | None
        try:
            region = self._governance_region(request.library_id)
        except MemoryRecordNotFoundError:
            region = None
        if region is None:
            library_id = validate_library_id(request.library_id)
            scope = _scope_for_normalized_library(library_id)
            target_space = scope.space
        else:
            scope = _scope_for_region(region)
            target_space = region.space
        logical_memory_id = ""
        chain_ids: set[str] = set()
        forgotten_lineages: set[str] = set()
        forgotten_unit_ids: set[str] = set()
        if region is not None:
            (
                logical_memory_id,
                chain_ids,
                forgotten_lineages,
                forgotten_unit_ids,
            ) = self._audit_lineage_context(region.region_id, request.target_id)
        filters = {"target_space": target_space}
        if region is not None and region.kind == "session":
            filters["target_session"] = region.session_id or ""
        audit_logger = SQLiteAuditLogger(self._state_dir / "mem2" / "audit.sqlite3")
        try:
            if request.target_id is not None:
                events_by_id = {}
                for target_id in sorted(chain_ids or {request.target_id}):
                    for event in audit_logger.query_latest(
                        {**filters, "target_id": target_id},
                        limit=max(1, min(request.limit, 1000)),
                    ):
                        events_by_id[event.id] = event
                events = list(events_by_id.values())
            else:
                events = audit_logger.query_latest(
                    filters,
                    limit=max(1, min(request.limit, 1000)),
                )
        finally:
            audit_logger.close()
        payloads = [
            {
                "id": event.id,
                "action": event.action,
                "decision": event.decision,
                "actor": event.actor.user,
                "target": event.target.space,
                "target_id": event.target_id,
                "commit": event.target_id or event.id,
                "subject": event.action,
                "authored_at": (
                    event.occurred_at.isoformat() if event.occurred_at is not None else ""
                ),
                "actor_type": event.actor.user,
                "source": "mem2",
                "occurred_at": (
                    event.occurred_at.isoformat() if event.occurred_at is not None else ""
                ),
            }
            for event in events
            if request.target_id is None
            or not chain_ids
            or event.target_id in chain_ids
        ]
        if region is not None:
            payloads.extend(
                self._governance_audit_events(
                    region.region_id,
                    request.limit,
                    request.target_id,
                    logical_memory_id=logical_memory_id,
                    chain_ids=chain_ids,
                )
            )
        if region is None and not payloads:
            raise MemoryRecordNotFoundError("memory region not found")
        for payload in payloads:
            lineage = str(payload.get("logical_memory_id") or "")
            target = str(payload.get("target_id") or payload.get("commit") or "")
            if lineage in forgotten_lineages or target in forgotten_unit_ids:
                payload["before_content"] = ""
                payload["after_content"] = ""
                payload["content_redacted"] = "true"
        return sorted(payloads, key=_audit_sort_key, reverse=True)[
            : max(1, min(request.limit, 1000))
        ]

    def _audit_lineage_context(
        self,
        region_id: str,
        target_id: str | None,
    ) -> tuple[str, set[str], set[str], set[str]]:
        logical_memory_id = ""
        chain_ids: set[str] = set()
        with closing(self._maintenance_db()) as connection:
            if target_id is not None:
                row = connection.execute(
                    """SELECT logical_memory_id FROM memory_browser_index
                       WHERE region_id = ? AND unit_id = ?""",
                    (region_id, target_id),
                ).fetchone()
                if row is not None:
                    logical_memory_id = str(row[0])
                if not logical_memory_id:
                    row = connection.execute(
                        """SELECT logical_memory_id FROM governance_events
                           WHERE region_id = ? AND unit_id = ? AND logical_memory_id != ''
                           ORDER BY id DESC LIMIT 1""",
                        (region_id, target_id),
                    ).fetchone()
                    if row is not None:
                        logical_memory_id = str(row[0])
                if logical_memory_id:
                    chain_ids = {
                        str(row[0])
                        for row in connection.execute(
                            """SELECT unit_id FROM memory_browser_index
                               WHERE region_id = ? AND logical_memory_id = ?""",
                            (region_id, logical_memory_id),
                        ).fetchall()
                    }
                chain_ids.add(target_id)
            forgotten_lineages = {
                str(row[0])
                for row in connection.execute(
                    """SELECT logical_memory_id FROM memory_browser_index
                       WHERE region_id = ? AND lifecycle = 'forgotten'
                         AND logical_memory_id != ''
                       UNION
                       SELECT logical_memory_id FROM governance_events
                       WHERE region_id = ? AND action = 'delete:forget'
                         AND status = 'committed' AND logical_memory_id != ''""",
                    (region_id, region_id),
                ).fetchall()
            }
            forgotten_unit_ids = {
                str(row[0])
                for row in connection.execute(
                    """SELECT unit_id FROM memory_browser_index
                       WHERE region_id = ? AND lifecycle = 'forgotten'
                       UNION
                       SELECT unit_id FROM governance_events
                       WHERE region_id = ? AND action = 'delete:forget'
                         AND status = 'committed'""",
                    (region_id, region_id),
                ).fetchall()
            }
        if target_id is not None and logical_memory_id:
            forgotten_lineages &= {logical_memory_id}
            forgotten_unit_ids &= chain_ids
        return logical_memory_id, chain_ids, forgotten_lineages, forgotten_unit_ids

    def _begin_governance_event(
        self,
        region_id: str,
        unit_id: str,
        action: str,
        actor: str,
        source: str,
        operation_id: str,
        *,
        before_content: str,
        after_content: str,
        detail: dict[str, object],
        logical_memory_id: str = "",
    ) -> int:
        with closing(self._maintenance_db()) as connection, connection:
            cursor = connection.execute(
                """INSERT INTO governance_events
                   (region_id, unit_id, logical_memory_id, action, actor, source, operation_id,
                    before_content, after_content, detail, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
                (
                    region_id,
                    unit_id,
                    logical_memory_id,
                    action,
                    actor,
                    source,
                    operation_id,
                    before_content,
                    after_content,
                    json.dumps(detail, ensure_ascii=False, sort_keys=True),
                    datetime.now(UTC).isoformat(),
                ),
            )
            if cursor.lastrowid is None:
                raise MemoryKernelUnavailableError("governance audit event was not persisted")
            return cursor.lastrowid

    def _finish_governance_event(
        self,
        event_id: int,
        *,
        status: str,
        unit_id: str,
        after_content: str,
        detail: dict[str, object],
    ) -> None:
        try:
            with closing(self._maintenance_db()) as connection, connection:
                connection.execute(
                    """UPDATE governance_events
                       SET status = ?, unit_id = ?, after_content = ?, detail = ?
                       WHERE id = ?""",
                    (
                        status,
                        unit_id,
                        after_content,
                        json.dumps(detail, ensure_ascii=False, sort_keys=True),
                        event_id,
                    ),
                )
        except (sqlite3.Error, OSError):
            # The pending row already preserves before_content. A completion
            # failure must not turn a committed mutation into a false 500.
            return

    def _governance_audit_events(
        self,
        region_id: str,
        limit: int,
        target_id: str | None = None,
        *,
        logical_memory_id: str = "",
        chain_ids: set[str] | None = None,
    ) -> list[dict[str, str]]:
        where = ["region_id = ?"]
        values: list[object] = [region_id]
        chain = sorted(chain_ids or set())
        if logical_memory_id:
            clauses = ["logical_memory_id = ?"]
            values.append(logical_memory_id)
            if chain:
                clauses.append(f"unit_id IN ({', '.join('?' for _ in chain)})")
                values.extend(chain)
            where.append(f"({' OR '.join(clauses)})")
        elif target_id is not None:
            where.append("unit_id = ?")
            values.append(target_id)
        values.append(max(1, min(limit, 1000)))
        with closing(self._maintenance_db()) as connection:
            rows = connection.execute(
                """SELECT id, unit_id, logical_memory_id, action, actor, source, operation_id,
                          before_content, after_content, detail, status, created_at
                   FROM governance_events
                   WHERE """
                + " AND ".join(where)
                + " ORDER BY id DESC LIMIT ?",
                tuple(values),
            ).fetchall()
        payloads: list[dict[str, str]] = []
        for row in rows:
            occurred = _parse_timestamp(str(row[11]))
            occurred_at = occurred.isoformat() if occurred is not None else str(row[11])
            payloads.append(
                {
                    "id": f"governance-{row[0]}",
                    "action": str(row[3]),
                    "decision": "allow",
                    "actor": str(row[4]),
                    "target": region_id,
                    "target_id": str(row[1]),
                    "logical_memory_id": str(row[2]),
                    "source": str(row[5]),
                    "operation_id": str(row[6]),
                    "before_content": str(row[7]),
                    "after_content": str(row[8]),
                    "detail": str(row[9]),
                    "status": str(row[10]),
                    "commit": str(row[1]),
                    "subject": str(row[3]),
                    "authored_at": occurred_at,
                    "actor_type": str(row[4]),
                    "occurred_at": occurred_at,
                }
            )
        return payloads

    def _governance_region(self, library_id: str) -> MemoryRegion:
        identifier = validate_library_id(library_id)
        try:
            return self._regions.require_region(identifier)
        except RegionError as error:
            raise MemoryRecordNotFoundError(str(error)) from error

    def _region_unit_pairs(
        self, bundle: _RuntimeBundle, region: MemoryRegion
    ) -> list[tuple[Scope, MemoryUnit]]:
        pairs: list[tuple[Scope, MemoryUnit]] = []
        for scope in bundle.kv_store.scopes():
            if scope.space != region.space:
                continue
            if region.kind == "session" and scope.session != (region.session_id or ""):
                continue
            pairs.extend((scope, unit) for unit in self._units_in_scope(bundle, scope))
        return sorted(pairs, key=_unit_sort_key)

    def _find_region_unit(
        self, bundle: _RuntimeBundle, region: MemoryRegion, unit_id: str
    ) -> tuple[Scope, MemoryUnit]:
        normalized = unit_id.strip()
        if not normalized:
            raise MemoryRecordNotFoundError("memory record not found")
        indexed = self._load_browser_index_unit(bundle, region, normalized)
        if indexed is not None:
            return indexed
        for scope, unit in self._region_unit_pairs(bundle, region):
            if unit.id == normalized:
                return scope, unit
        raise MemoryRecordNotFoundError("memory record not found")

    def _load_browser_index_unit(
        self,
        bundle: _RuntimeBundle,
        region: MemoryRegion,
        unit_id: str,
    ) -> tuple[Scope, MemoryUnit] | None:
        with closing(self._maintenance_db()) as connection:
            row = connection.execute(
                """SELECT scope_org, scope_space, scope_user, scope_agent, scope_session
                   FROM memory_browser_index
                   WHERE region_id = ? AND unit_id = ?""",
                (region.region_id, unit_id),
            ).fetchone()
        if row is None:
            return None
        scope = Scope(
            org=str(row[0]),
            space=str(row[1]),
            user=str(row[2]),
            agent=str(row[3]),
            session=str(row[4]),
        )
        units = list(load_units(bundle.kv_store, scope, [unit_id]))
        if not units:
            return None
        return scope, units[0]

    def _lineage_versions(
        self,
        region_id: str,
        logical_memory_id: str,
    ) -> list[tuple[str, Scope]]:
        with closing(self._maintenance_db()) as connection:
            rows = connection.execute(
                """SELECT unit_id, scope_org, scope_space, scope_user,
                          scope_agent, scope_session
                   FROM memory_browser_index
                   WHERE region_id = ? AND logical_memory_id = ?
                   ORDER BY updated_at, unit_id""",
                (region_id, logical_memory_id),
            ).fetchall()
        return [
            (
                str(row[0]),
                Scope(
                    org=str(row[1]),
                    space=str(row[2]),
                    user=str(row[3]),
                    agent=str(row[4]),
                    session=str(row[5]),
                ),
            )
            for row in rows
        ]

    def _resolved_unit_lineage_id(
        self,
        region: MemoryRegion,
        unit: MemoryUnit,
    ) -> str:
        with closing(self._maintenance_db()) as connection:
            row = connection.execute(
                """SELECT logical_memory_id FROM memory_browser_index
                   WHERE region_id = ? AND unit_id = ?""",
                (region.region_id, unit.id),
            ).fetchone()
        if row is not None and str(row[0]):
            return str(row[0])
        return _unit_logical_memory_id(unit)

    def _lineage_terminal_lifecycle(
        self,
        region_id: str,
        logical_memory_id: str,
    ) -> str:
        if not logical_memory_id:
            return ""
        with closing(self._maintenance_db()) as connection:
            row = connection.execute(
                """SELECT lifecycle FROM memory_browser_index
                   WHERE region_id = ? AND logical_memory_id = ?
                     AND unit_id NOT IN (
                         SELECT supersedes FROM memory_browser_index
                         WHERE region_id = ? AND logical_memory_id = ?
                           AND supersedes != ''
                     )
                   ORDER BY updated_at DESC, unit_id DESC LIMIT 1""",
                (region_id, logical_memory_id, region_id, logical_memory_id),
            ).fetchone()
        return str(row[0]) if row is not None else ""

    def _load_units_from_regions(
        self,
        bundle: _RuntimeBundle,
        unit_ids: list[str],
        region_ids: list[str],
    ) -> dict[str, tuple[MemoryUnit, MemoryRegion]]:
        wanted = set(unit_ids)
        found: dict[str, tuple[MemoryUnit, MemoryRegion]] = {}
        for region_id in dict.fromkeys(region_ids):
            try:
                region = self._regions.require_region(region_id)
            except RegionError:
                continue
            remaining = list(wanted - set(found))
            if not remaining:
                break
            for scope in self._region_scopes(bundle, region):
                units = load_units(bundle.kv_store, scope, remaining)
                for unit in units:
                    found.setdefault(unit.id, (unit, region))
                remaining = list(wanted - set(found))
                if not remaining:
                    break
        return found

    def _browser_region_map_for_ids(self, unit_ids: list[str]) -> dict[str, str]:
        ids = list(dict.fromkeys(unit_ids))
        if not ids:
            return {}
        placeholders = ", ".join("?" for _ in ids)
        with closing(self._maintenance_db()) as connection:
            rows = connection.execute(
                f"""SELECT unit_id, region_id FROM memory_browser_index
                    WHERE unit_id IN ({placeholders})""",
                ids,
            ).fetchall()
        return {str(row[0]): str(row[1]) for row in rows}

    def _browser_evidence(
        self,
        bundle: _RuntimeBundle,
        region: MemoryRegion,
        unit: MemoryUnit,
    ) -> list[dict[str, object]]:
        source_ids = _metadata_strings(unit, "source_unit_ids")
        if not source_ids:
            source_ids = _metadata_strings(unit, "provenance")
        source_region_ids = _metadata_strings(unit, "source_region_ids")
        single_region = _metadata_text(unit, "source_region_id")
        if single_region and single_region not in source_region_ids:
            source_region_ids.insert(0, single_region)
        region_by_id = _metadata_unit_region_map(unit)
        if any(source_id not in region_by_id for source_id in source_ids):
            region_by_id.update(self._browser_region_map_for_ids(source_ids))
        ordered_ids = list(dict.fromkeys(source_ids))
        search_region_ids = list(
            dict.fromkeys(
                [
                    *region_by_id.values(),
                    *source_region_ids,
                    region.region_id,
                ]
            )
        )
        found: dict[str, tuple[MemoryUnit, MemoryRegion]] = {}
        while True:
            remaining = [unit_id for unit_id in ordered_ids if unit_id not in found]
            if not remaining:
                break
            loaded = self._load_units_from_regions(
                bundle,
                remaining,
                list(
                    dict.fromkeys(
                        [
                            *[region_by_id.get(unit_id, "") for unit_id in remaining],
                            *search_region_ids,
                        ]
                    )
                ),
            )
            if not loaded:
                break
            added = False
            for source_id, hit in loaded.items():
                if source_id in found:
                    continue
                added = True
                found[source_id] = hit
                region_by_id.setdefault(source_id, hit[1].region_id)
                search_region_ids = list(
                    dict.fromkeys([*search_region_ids, hit[1].region_id])
                )
                child_ids = _metadata_strings(hit[0], "source_unit_ids")
                if not child_ids:
                    child_ids = _metadata_strings(hit[0], "provenance")
                for child_id in child_ids:
                    if child_id not in ordered_ids:
                        ordered_ids.append(child_id)
            if not added:
                break

        payloads: list[dict[str, object]] = []
        for source_id in ordered_ids:
            resolved_hit = found.get(source_id)
            if resolved_hit is None:
                payloads.append(
                    {
                        "id": source_id,
                        "content": "",
                        "lifecycle": "",
                        "created_at": "",
                        "source_references": [],
                        "region_id": region_by_id.get(source_id, ""),
                        "region_kind": "",
                        "session_id": "",
                        "turn_id": "",
                        "event_id": "",
                        "deleted_at": "",
                        "redacted": False,
                        "redaction_reason": "",
                        "resolved": False,
                    }
                )
                continue
            source, source_region = resolved_hit
            source_payload = _unit_payload(source, source_region)
            source_references = source.user_metadata.get("source_references")
            forgotten = source.lifecycle.value == "forgotten"
            sensitive = not forgotten and inspect_sensitive_text(source.content) is not None
            redacted = forgotten or sensitive
            redaction_reason = "forgotten" if forgotten else ("sensitive" if sensitive else "")
            payloads.append(
                {
                    "id": source.id,
                    "content": "" if redacted else source.content,
                    "lifecycle": source.lifecycle.value,
                    "created_at": source_payload.get("created_at", ""),
                    "deleted_at": source_payload.get("deleted_at", ""),
                    "redacted": redacted,
                    "redaction_reason": redaction_reason,
                    "source_references": (
                        []
                        if forgotten
                        else _json_metadata(
                            list(source_references)
                            if isinstance(source_references, list)
                            else list(source.provenance)
                        )
                    ),
                    "region_id": source_region.region_id,
                    "region_kind": source_region.kind,
                    "session_id": _metadata_text(source, "session_id"),
                    "turn_id": _metadata_text(source, "turn_id"),
                    "event_id": (
                        _metadata_text(source, "capture_event_id")
                        or _metadata_text(source, "event_id")
                    ),
                    "resolved": True,
                }
            )
        return payloads

    def _memory_provenance_summary(
        self,
        region: MemoryRegion,
        payload: Mapping[str, object],
        evidence: object,
    ) -> dict[str, object]:
        raw_metadata = payload.get("user_metadata")
        metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
        evidence_items = (
            [item for item in evidence if isinstance(item, Mapping)]
            if isinstance(evidence, list)
            else []
        )
        source_region_ids = _string_list_value(metadata.get("source_region_ids"))
        single_region_id = str(metadata.get("source_region_id") or "").strip()
        if single_region_id and single_region_id not in source_region_ids:
            source_region_ids.insert(0, single_region_id)
        source_region_ids = list(dict.fromkeys(source_region_ids))
        non_target_regions = [
            region_id for region_id in source_region_ids if region_id != region.region_id
        ]
        source_region_ids = non_target_regions or source_region_ids or [region.region_id]
        evidence_kinds = {
            str(item.get("region_id") or ""): str(item.get("region_kind") or "")
            for item in evidence_items
            if str(item.get("region_id") or "")
        }
        source_regions: list[dict[str, str]] = []
        for region_id in source_region_ids:
            try:
                source_region = self._regions.require_region(region_id)
            except RegionError:
                source_regions.append(
                    {
                        "region_id": region_id,
                        "name": "",
                        "kind": evidence_kinds.get(region_id, ""),
                    }
                )
                continue
            source_regions.append(
                {
                    "region_id": source_region.region_id,
                    "name": source_region.name,
                    "kind": source_region.kind,
                }
            )

        source_unit_ids = _string_list_value(metadata.get("source_unit_ids"))
        if not source_unit_ids:
            source_unit_ids = _string_list_value(metadata.get("provenance"))
        source_unit_ids.extend(
            str(item.get("id") or "")
            for item in evidence_items
            if item.get("resolved") is True and str(item.get("id") or "")
        )
        event_ids = _string_list_value(metadata.get("source_event_ids"))
        event_ids.extend(
            str(item.get("event_id") or "")
            for item in evidence_items
            if str(item.get("event_id") or "")
        )
        snippets = _string_list_value(metadata.get("evidence_snippets"))
        if not snippets:
            snippets = _string_list_value(metadata.get("evidence_snippet"))
        promotion = {
            "promoted": metadata.get("promotion_marker") is True,
            "classification": str(metadata.get("promotion_classification") or ""),
            "job_id": str(metadata.get("promotion_job_id") or ""),
            "evidence_kind": str(metadata.get("promotion_evidence_kind") or ""),
        }
        return {
            "target_region": {
                "region_id": region.region_id,
                "name": region.name,
                "kind": region.kind,
            },
            "source_regions": source_regions,
            "evidence_basis": {
                "source_unit_ids": list(dict.fromkeys(source_unit_ids)),
                "event_ids": list(dict.fromkeys(event_ids)),
                "snippets": list(dict.fromkeys(snippets)),
                "references": _string_list_value(metadata.get("source_references")),
            },
            "promotion": promotion,
        }

    def _browser_semantic_type(
        self,
        bundle: _RuntimeBundle,
        unit: MemoryUnit,
    ) -> tuple[str, str] | None:
        memory_type = _metadata_text(unit, "memory_type")
        if memory_type in FORMAL_MEMORY_TYPES:
            resolution = (
                "promotion_source"
                if unit.user_metadata.get("promotion_marker") is True
                else "native"
            )
            return memory_type, resolution
        if memory_type != "promoted" and not _metadata_text(unit, "promotion_key"):
            return None
        source_ids = _metadata_strings(unit, "source_unit_ids")
        if not source_ids:
            source_ids = _metadata_strings(unit, "provenance")
        source_region_ids = _metadata_strings(unit, "source_region_ids")
        single_region = _metadata_text(unit, "source_region_id")
        if single_region and single_region not in source_region_ids:
            source_region_ids.insert(0, single_region)
        region_by_id = _metadata_unit_region_map(unit)
        if any(source_id not in region_by_id for source_id in source_ids):
            region_by_id.update(self._browser_region_map_for_ids(source_ids))
        search_region_ids = list(
            dict.fromkeys([*region_by_id.values(), *source_region_ids])
        )
        if source_ids and search_region_ids:
            found = self._load_units_from_regions(
                bundle,
                source_ids,
                search_region_ids,
            )
            for source_id in source_ids:
                hit = found.get(source_id)
                if hit is None:
                    continue
                source_type = _metadata_text(hit[0], "memory_type")
                if source_type in FORMAL_MEMORY_TYPES:
                    return source_type, "legacy_source"
        return None

    def _browser_index_row(
        self,
        bundle: _RuntimeBundle,
        region: MemoryRegion,
        scope: Scope,
        unit: MemoryUnit,
        *,
        logical_memory_id: str | None = None,
    ) -> tuple[object, ...] | None:
        lifecycle = unit.lifecycle.value
        if lifecycle not in {"active", "superseded", "archived", "forgotten"}:
            return None
        if _is_capture_unit(unit) or _metadata_text(unit, "memory_type") == "evolved":
            return None
        lineage_id = logical_memory_id or _unit_logical_memory_id(unit)
        if lifecycle == "forgotten":
            with closing(self._maintenance_db()) as connection:
                row = connection.execute(
                    """SELECT memory_type, memory_type_resolution, updated_at, version,
                              supersedes
                       FROM memory_browser_index
                       WHERE region_id = ? AND unit_id = ?""",
                    (region.region_id, unit.id),
                ).fetchone()
            return (
                region.region_id,
                unit.id,
                lineage_id,
                region.name,
                region.kind,
                scope.org,
                scope.space,
                scope.user,
                scope.agent,
                scope.session,
                "",
                "",
                str(row[0]) if row is not None else "unknown",
                str(row[1]) if row is not None else "redacted",
                lifecycle,
                str(row[2]) if row is not None else "",
                str(row[3]) if row is not None else "",
                str(row[4]) if row is not None else unit.supersedes,
                0,
                "[]",
                "[]",
                "{}",
            )
        semantic_type = self._browser_semantic_type(bundle, unit)
        if semantic_type is None:
            return None
        memory_type, resolution = semantic_type
        payload = _unit_payload(unit, region)
        return (
            region.region_id,
            unit.id,
            lineage_id,
            region.name,
            region.kind,
            scope.org,
            scope.space,
            scope.user,
            scope.agent,
            scope.session,
            unit.content,
            str(payload.get("summary", "")),
            memory_type,
            resolution,
            unit.lifecycle.value,
            str(payload.get("updated_at", "")),
            str(payload.get("version", "")),
            unit.supersedes,
            int(str(payload.get("source_count", 0))),
            json.dumps(payload.get("source_references", []), ensure_ascii=False),
            json.dumps(payload.get("provenance", []), ensure_ascii=False),
            json.dumps(_json_metadata(dict(unit.user_metadata)), ensure_ascii=False),
        )

    def _index_browser_units(
        self,
        bundle: _RuntimeBundle,
        region: MemoryRegion,
        scope: Scope,
        units: list[MemoryUnit],
    ) -> None:
        lineage_ids = _logical_memory_id_map(units)
        rows = [
            row
            for unit in units
            if (
                row := self._browser_index_row(
                    bundle,
                    region,
                    scope,
                    unit,
                    logical_memory_id=lineage_ids[unit.id],
                )
            )
            is not None
        ]
        missing = [unit.id for unit in units if unit.id not in {str(row[1]) for row in rows}]
        self._write_browser_index_rows(region.region_id, rows, missing)

    def _refresh_browser_index_ids(
        self,
        bundle: _RuntimeBundle,
        region: MemoryRegion,
        scope: Scope,
        unit_ids: list[str],
    ) -> None:
        ids = list(dict.fromkeys(unit_ids))
        units = list(load_units(bundle.kv_store, scope, ids))
        present = {unit.id for unit in units}
        self._index_browser_units(bundle, region, scope, units)
        self._write_browser_index_rows(
            region.region_id,
            [],
            [unit_id for unit_id in ids if unit_id not in present],
        )

    def _sync_browser_index_best_effort(
        self,
        bundle: _RuntimeBundle,
        region: MemoryRegion,
        scope: Scope,
        token: str,
        unit_ids: list[str],
    ) -> str:
        try:
            self._refresh_browser_index_pending(
                bundle,
                region,
                scope,
                token,
                unit_ids,
            )
        except Exception:
            return "pending"
        return "ready"

    def _persist_unit_logical_memory_id(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        unit_id: str,
        logical_memory_id: str,
    ) -> None:
        units = list(load_units(bundle.kv_store, scope, [unit_id]))
        if not units:
            return
        unit = units[0]
        if _metadata_text(unit, "logical_memory_id") == logical_memory_id:
            return
        unit.user_metadata["logical_memory_id"] = logical_memory_id
        bundle.kv_store.update(scope, memory_key(unit.id), dumps(unit))

    def _record_create_audit(
        self,
        region: MemoryRegion,
        unit: MemoryUnit,
        request: CreateMemoryRequest,
    ) -> None:
        event_id = self._begin_governance_event(
            region.region_id,
            unit.id,
            "add",
            request.actor,
            "memory-create",
            f"create:{request.idempotency_key}",
            before_content="",
            after_content=unit.content,
            detail={
                "memory_type": request.memory_type,
                "source_references": list(request.source_references),
            },
            logical_memory_id=_unit_logical_memory_id(unit),
        )
        self._finish_governance_event(
            event_id,
            status="committed",
            unit_id=unit.id,
            after_content=unit.content,
            detail={
                "memory_type": request.memory_type,
                "source_references": list(request.source_references),
            },
        )

    def _begin_browser_index_pending(
        self,
        region: MemoryRegion,
        scope: Scope,
        unit_ids: list[str] | None = None,
    ) -> str:
        token = uuid.uuid4().hex
        with (
            self._browser_index_lock,
            closing(self._maintenance_db()) as connection,
            connection,
        ):
            connection.execute(
                """INSERT INTO memory_browser_index_pending
                       (token, region_id, scope_org, scope_space, scope_user,
                        scope_agent, scope_session, unit_ids, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    token,
                    region.region_id,
                    scope.org,
                    scope.space,
                    scope.user,
                    scope.agent,
                    scope.session,
                    json.dumps(unit_ids or [], ensure_ascii=False),
                    datetime.now(UTC).isoformat(),
                ),
            )
            self._browser_index_active_tokens.add(token)
        return token

    def _update_browser_index_pending(self, token: str, unit_ids: list[str]) -> None:
        with (
            self._browser_index_lock,
            closing(self._maintenance_db()) as connection,
            connection,
        ):
            connection.execute(
                "UPDATE memory_browser_index_pending SET unit_ids = ? WHERE token = ?",
                (json.dumps(list(dict.fromkeys(unit_ids)), ensure_ascii=False), token),
            )

    def _complete_browser_index_pending(self, token: str) -> None:
        with (
            self._browser_index_lock,
            closing(self._maintenance_db()) as connection,
            connection,
        ):
            connection.execute(
                "DELETE FROM memory_browser_index_pending WHERE token = ?",
                (token,),
            )
            self._browser_index_active_tokens.discard(token)

    def _refresh_browser_index_pending(
        self,
        bundle: _RuntimeBundle,
        region: MemoryRegion,
        scope: Scope,
        token: str,
        unit_ids: list[str],
    ) -> None:
        ids = list(dict.fromkeys(unit_ids))
        try:
            self._update_browser_index_pending(token, ids)
            self._refresh_browser_index_ids(bundle, region, scope, ids)
        except Exception:
            with self._browser_index_lock:
                self._browser_index_active_tokens.discard(token)
            raise
        self._complete_browser_index_pending(token)

    def _retry_pending_browser_index(
        self,
        bundle: _RuntimeBundle,
        *,
        region_id: str | None = None,
        batch_size: int = _BROWSER_INDEX_RETRY_BATCH_SIZE,
        total_limit: int = _BROWSER_INDEX_RETRY_MAX,
        strict: bool = True,
    ) -> None:
        where = "WHERE region_id = ?" if region_id is not None else ""
        batch = max(1, batch_size)
        maximum = max(batch, total_limit)
        scanned = 0
        with self._browser_index_lock:
            while scanned < maximum:
                remaining = min(batch, maximum - scanned)
                params: tuple[object, ...] = (
                    (region_id, remaining)
                    if region_id is not None
                    else (remaining,)
                )
                with closing(self._maintenance_db()) as connection:
                    rows = connection.execute(
                        f"""SELECT token, region_id, scope_org, scope_space, scope_user,
                                   scope_agent, scope_session, unit_ids
                            FROM memory_browser_index_pending
                            {where}
                            ORDER BY created_at, token
                            LIMIT ?""",
                        params,
                    ).fetchall()
                if not rows:
                    return
                scanned += len(rows)
                for row in rows:
                    token = str(row[0])
                    if token in self._browser_index_active_tokens:
                        continue
                    try:
                        region = self._regions.require_region(str(row[1]))
                        scope = Scope(
                            org=str(row[2]),
                            space=str(row[3]),
                            user=str(row[4]),
                            agent=str(row[5]),
                            session=str(row[6]),
                        )
                        try:
                            raw_ids = json.loads(str(row[7]))
                        except json.JSONDecodeError:
                            raw_ids = []
                        unit_ids = (
                            [str(value) for value in raw_ids]
                            if isinstance(raw_ids, list)
                            else []
                        )
                        if unit_ids:
                            self._refresh_browser_index_ids(
                                bundle,
                                region,
                                scope,
                                unit_ids,
                            )
                        else:
                            self._rebuild_browser_scope_index(bundle, region, scope)
                    except Exception as error:
                        if strict:
                            raise MemoryKernelUnavailableError(
                                "memory browser index synchronization is pending"
                            ) from error
                        return
                    self._complete_browser_index_pending(token)
            with closing(self._maintenance_db()) as connection:
                pending = connection.execute(
                    f"SELECT 1 FROM memory_browser_index_pending {where} LIMIT 1",
                    (region_id,) if region_id is not None else (),
                ).fetchone()
            if pending is not None and strict:
                raise MemoryKernelUnavailableError(
                    "memory browser index synchronization exceeds the retry limit"
                )

    def _rebuild_browser_scope_index(
        self,
        bundle: _RuntimeBundle,
        region: MemoryRegion,
        scope: Scope,
    ) -> None:
        with closing(self._maintenance_db()) as connection, connection:
            connection.execute(
                """DELETE FROM memory_browser_index
                   WHERE region_id = ? AND scope_org = ? AND scope_space = ?
                     AND scope_user = ? AND scope_agent = ? AND scope_session = ?""",
                (
                    region.region_id,
                    scope.org,
                    scope.space,
                    scope.user,
                    scope.agent,
                    scope.session,
                ),
            )
        self._index_browser_units(
            bundle,
            region,
            scope,
            self._units_in_scope(bundle, scope),
        )

    def _region_for_scope(self, scope: Scope) -> MemoryRegion | None:
        for region in self._regions.list_regions():
            if region.space != scope.space:
                continue
            if region.kind == "session" and scope.session != (region.session_id or ""):
                continue
            return region
        return None

    def _write_browser_index_rows(
        self,
        region_id: str,
        rows: list[tuple[object, ...]],
        missing_ids: list[str],
    ) -> None:
        with closing(self._maintenance_db()) as connection, connection:
            if rows:
                connection.executemany(
                    """INSERT INTO memory_browser_index
                       (region_id, unit_id, logical_memory_id, region_name, region_kind,
                        scope_org, scope_space, scope_user, scope_agent, scope_session,
                        content, summary, memory_type, memory_type_resolution, lifecycle,
                        updated_at, version, supersedes, source_count, source_references,
                        provenance, user_metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(region_id, unit_id) DO UPDATE SET
                         logical_memory_id=excluded.logical_memory_id,
                         region_name=excluded.region_name,
                         region_kind=excluded.region_kind,
                         scope_org=excluded.scope_org,
                         scope_space=excluded.scope_space,
                         scope_user=excluded.scope_user,
                         scope_agent=excluded.scope_agent,
                         scope_session=excluded.scope_session,
                         content=excluded.content,
                         summary=excluded.summary,
                         memory_type=excluded.memory_type,
                         memory_type_resolution=excluded.memory_type_resolution,
                         lifecycle=excluded.lifecycle,
                         updated_at=excluded.updated_at,
                         version=excluded.version,
                         supersedes=excluded.supersedes,
                         source_count=excluded.source_count,
                         source_references=excluded.source_references,
                         provenance=excluded.provenance,
                         user_metadata=excluded.user_metadata""",
                    rows,
                )
            if missing_ids:
                connection.executemany(
                    "DELETE FROM memory_browser_index WHERE region_id = ? AND unit_id = ?",
                    [(region_id, unit_id) for unit_id in missing_ids],
                )

    def _ensure_browser_index(self, bundle: _RuntimeBundle) -> None:
        with closing(self._maintenance_db()) as connection:
            row = connection.execute(
                "SELECT value FROM memory_browser_index_meta WHERE key = 'schema_version'"
            ).fetchone()
        if row is None or str(row[0]) != str(_BROWSER_INDEX_SCHEMA_VERSION):
            self._rebuild_browser_index(bundle)
            with closing(self._maintenance_db()) as connection, connection:
                connection.execute(
                    """INSERT INTO memory_browser_index_meta (key, value)
                       VALUES ('schema_version', ?)
                       ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                    (str(_BROWSER_INDEX_SCHEMA_VERSION),),
                )
        self._backfill_governance_lineage()

    def _backfill_governance_lineage(
        self,
        batch_size: int = _LINEAGE_MIGRATION_BATCH_SIZE,
    ) -> list[int]:
        if batch_size <= 0:
            raise ValueError("lineage migration batch size must be positive")
        batches: list[int] = []
        cursor = 0
        while True:
            with closing(self._maintenance_db()) as connection, connection:
                rows = connection.execute(
                    """SELECT id, unit_id FROM governance_events
                       WHERE logical_memory_id = '' AND id > ?
                       ORDER BY id LIMIT ?""",
                    (cursor, batch_size),
                ).fetchall()
                if not rows:
                    return batches
                unit_ids = [str(row[1]) for row in rows]
                placeholders = ", ".join("?" for _ in unit_ids)
                lineages = {
                    str(row[0]): str(row[1]) or str(row[0])
                    for row in connection.execute(
                        f"""SELECT unit_id, logical_memory_id FROM memory_browser_index
                            WHERE unit_id IN ({placeholders})""",
                        unit_ids,
                    ).fetchall()
                }
                connection.executemany(
                    """UPDATE governance_events SET logical_memory_id = ?
                       WHERE id = ?""",
                    [
                        (lineages.get(str(row[1]), str(row[1])), int(row[0]))
                        for row in rows
                    ],
                )
            cursor = int(rows[-1][0])
            batches.append(len(rows))

    def _rebuild_browser_index(self, bundle: _RuntimeBundle) -> None:
        regions_by_key = {
            (region.space, region.session_id or ""): region
            for region in self._regions.list_regions()
        }
        regions_by_space = {
            region.space: region
            for region in self._regions.list_regions()
            if region.kind != "session"
        }
        with closing(self._maintenance_db()) as connection:
            connection.execute("DELETE FROM memory_browser_index")
            connection.commit()
            batch: list[tuple[object, ...]] = []
            for scope in bundle.kv_store.scopes():
                region = regions_by_key.get((scope.space, scope.session))
                if region is None:
                    region = regions_by_space.get(scope.space)
                if region is None:
                    continue
                units = self._units_in_scope(bundle, scope)
                lineage_ids = _logical_memory_id_map(units)
                for unit in units:
                    row = self._browser_index_row(
                        bundle,
                        region,
                        scope,
                        unit,
                        logical_memory_id=lineage_ids[unit.id],
                    )
                    if row is None:
                        continue
                    batch.append(row)
                    if len(batch) >= 500:
                        connection.executemany(
                            """INSERT INTO memory_browser_index
                               (region_id, unit_id, logical_memory_id, region_name, region_kind,
                                scope_org, scope_space, scope_user, scope_agent, scope_session,
                                content, summary, memory_type, memory_type_resolution, lifecycle,
                                updated_at, version, supersedes, source_count, source_references,
                                provenance, user_metadata)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                       ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            batch,
                        )
                        connection.commit()
                        batch.clear()
            if batch:
                connection.executemany(
                    """INSERT INTO memory_browser_index
                       (region_id, unit_id, logical_memory_id, region_name, region_kind,
                        scope_org, scope_space, scope_user, scope_agent, scope_session,
                        content, summary, memory_type, memory_type_resolution, lifecycle,
                        updated_at, version, supersedes, source_count, source_references,
                        provenance, user_metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                               ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    batch,
                )
                connection.commit()

    async def evolve(self, request: EvolveMemoryRequest) -> dict[str, int]:
        return await asyncio.to_thread(self._evolve, request)

    def _capture_scope(self, request: CaptureMemoryRequest) -> tuple[Scope, MemoryRegion]:
        topic_id = request.topic_id.strip() if request.topic_id is not None else ""
        if request.topic_id is not None and not topic_id:
            raise ValueError("topic_id must not be blank")
        try:
            resolved = self._regions.resolve(request.cwd)
        except RegionBindingError:
            resolved = None
        if resolved is not None:
            region = resolved.region
        elif topic_id:
            region = self._regions.require_region(topic_id)
            if region.kind != "topic":
                raise CaptureContractError("topic_id must identify a topic memory region")
        else:
            region = self._ensure_session_region(request.session_id)
        return (
            _scope_for_region(
                region,
                session_id=request.session_id,
                agent=_CODEX_AGENT,
            ),
            region,
        )

    def _ensure_session_region(self, session_id: str) -> MemoryRegion:
        normalized = _normalize_session_id(session_id)
        for region in self._regions.list_regions():
            if region.kind == "session" and region.session_id == normalized:
                return region
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]
        try:
            return self._regions.create_region(
                "session",
                f"会话 {normalized[:80]}",
                region_id=f"session-{digest}",
                session_id=normalized,
            )
        except RegionError:
            for region in self._regions.list_regions():
                if region.kind == "session" and region.session_id == normalized:
                    return region
            raise

    def _evolve(self, request: EvolveMemoryRequest) -> dict[str, int]:
        with self._maintenance_lock:
            return self._evolve_locked(request)

    def _evolve_reserved_turn(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        region: MemoryRegion,
        target_scope: Scope,
        evolve_key: str,
        source_fingerprint: str,
        source_ids: list[str],
        ordered: list[MemoryUnit],
        turn_id: str,
        status: str,
    ) -> int:
        content = "\n\n".join(
            f"{_event_label(_metadata_text(unit, 'event_kind'))}: {unit.content}"
            for unit in ordered
        )
        chunks = _chunk_evolved_content(content)
        residual = self._evolution_units(bundle, target_scope, evolve_key)
        if status == "pending" and self._evolution_visible(
            bundle,
            target_scope,
            residual,
            len(chunks),
            source_fingerprint,
        ):
            self._complete_evolution(evolve_key, residual[0].id)
            return 0
        if residual:
            self._purge_evolution_units(bundle, target_scope, residual)
        added_units: list[MemoryUnit] = []
        try:
            for part, chunk in enumerate(chunks):
                units_added = bundle.runtime.api.add(
                    chunk,
                    target_scope,
                    Modality.TEXT,
                    security=_security_for(target_scope),
                    user_metadata={
                        "memory_type": "evolved",
                        "evolve_key": evolve_key,
                        "source_fingerprint": source_fingerprint,
                        "evolve_part": part,
                        "evolve_parts": len(chunks),
                        "region_id": region.region_id,
                        "region_kind": region.kind,
                        "session_id": scope.session,
                        "turn_id": turn_id,
                        "source_unit_ids": source_ids,
                        "source_event_ids": [
                            _metadata_text(unit, "event_id") for unit in ordered
                        ],
                        "provenance": source_ids,
                    },
                )
                if not units_added:
                    raise MemoryKernelUnavailableError("evolve did not create a memory unit")
                added_units.append(units_added[0])
            if not self._evolution_visible(
                bundle,
                target_scope,
                added_units,
                len(chunks),
                source_fingerprint,
            ):
                self._purge_evolution_units(bundle, target_scope, added_units)
                raise MemoryKernelUnavailableError(
                    "evolved memory is not visible in the fulltext index"
                )
        except Exception:
            residual = self._evolution_units(bundle, target_scope, evolve_key)
            self._purge_evolution_units(bundle, target_scope, residual)
            raise
        self._complete_evolution(evolve_key, added_units[0].id)
        return 1

    def _evolve_locked(self, request: EvolveMemoryRequest) -> dict[str, int]:
        bundle = self._require_bundle()
        session_id = (
            _normalize_session_id(request.session_id) if request.session_id is not None else None
        )
        self._enqueue_pending_semantic_records(session_id=session_id, retry_failed=True)
        created = 0
        all_scopes = [
            scope
            for scope in bundle.kv_store.scopes()
            if session_id is None or scope.session == session_id
        ]
        scopes = [scope for scope in all_scopes if scope.space not in {"user"}]
        target_scopes: list[Scope] = []
        for scope in scopes:
            units = self._units_in_scope(bundle, scope)
            captured = [
                unit
                for unit in units
                if unit.lifecycle.value == "active" and _is_capture_unit(unit)
            ]
            groups: dict[str, list[MemoryUnit]] = {}
            for unit in captured:
                turn_id = _metadata_text(unit, "turn_id")
                if turn_id:
                    groups.setdefault(turn_id, []).append(unit)
            for turn_id, source_units in groups.items():
                if request.deadline is not None and time.monotonic() >= request.deadline:
                    break
                ordered = sorted(
                    source_units,
                    key=lambda unit: (
                        _event_order(_metadata_text(unit, "event_kind")),
                        _metadata_text(unit, "occurred_at"),
                        unit.id,
                    ),
                )
                if not _capture_turn_is_complete(ordered):
                    continue
                source_ids = [unit.id for unit in ordered]
                started_at, received_earliest, received_latest = _capture_units_timeline(
                    ordered
                )
                region_id = _metadata_text(ordered[0], "region_id")
                if not region_id:
                    continue
                try:
                    region = self._regions.require_region(region_id)
                except RegionError:
                    continue
                if region.space != scope.space:
                    continue
                evolve_key = _stable_evolve_key(scope, region.region_id, turn_id)
                source_fingerprint = _source_fingerprint(source_ids)
                target_scope = _scope_for_region(
                    region,
                    session_id=(region.session_id or "" if region.kind == "session" else ""),
                    agent=scope.agent,
                )
                if target_scope not in target_scopes:
                    target_scopes.append(target_scope)
                status, memory_id = self._reserve_evolution(
                    evolve_key,
                    scope.session,
                    turn_id,
                    region.region_id,
                    source_fingerprint,
                    source_ids,
                    started_at,
                    received_earliest,
                    received_latest,
                )
                self._enqueue_semantic_distillation(evolve_key, retry_failed=True)
                if status == "completed" and memory_id is not None:
                    continue
                try:
                    created += self._evolve_reserved_turn(
                        bundle,
                        scope,
                        region,
                        target_scope,
                        evolve_key,
                        source_fingerprint,
                        source_ids,
                        ordered,
                        turn_id,
                        status,
                    )
                except Exception as error:
                    self._fail_evolution(evolve_key, _evolution_failure_text(error))
                    raise
        self._enqueue_pending_semantic_records(session_id=session_id, retry_failed=True)
        if bundle.graph_store is not None:
            with self._graph_write_lock:
                graph_scopes = list(all_scopes)
                for scope in bundle.kv_store.scopes():
                    if session_id is not None and scope.session != session_id:
                        continue
                    if scope not in graph_scopes:
                        graph_scopes.append(scope)
                for scope in target_scopes:
                    if scope not in graph_scopes:
                        graph_scopes.append(scope)
                for index, scope in enumerate(graph_scopes):
                    if request.deadline is not None and time.monotonic() >= request.deadline:
                        for pending_scope in graph_scopes[index:]:
                            self._queue_scope_graph_update(bundle, pending_scope)
                        break
                    self._associate_graph_scope(
                        bundle,
                        scope,
                        region_id=self._region_id_for_scope(scope),
                    )
        return {"queued": created}

    def _semantic_enabled(self) -> bool:
        return self._semantic_extractor is not None

    def _schedule_semantic_for_capture(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        region: MemoryRegion,
        turn_id: str,
    ) -> None:
        """Queue semantic distillation without putting a model call in the capture budget."""
        if not self._semantic_enabled():
            return
        try:
            units = self._semantic_round_capture_units(
                bundle,
                session_id=scope.session,
                region_id=region.region_id,
                turn_id=turn_id,
            )
            if not _capture_turn_is_complete(units):
                return
            source_ids = [unit.id for unit in units]
            started_at, received_earliest, received_latest = _capture_units_timeline(units)
            evolve_key = _stable_evolve_key(scope, region.region_id, turn_id)
            self._reserve_evolution(
                evolve_key,
                scope.session,
                turn_id,
                region.region_id,
                _source_fingerprint(source_ids),
                source_ids,
                started_at,
                received_earliest,
                received_latest,
            )
            self._enqueue_semantic_distillation(evolve_key, retry_failed=True)
        except Exception as error:
            # Capture remains fail-open. A later consolidate/recovery retries the task.
            with suppress(Exception):
                evolve_key = _stable_evolve_key(scope, region.region_id, turn_id)
                record = self._semantic_record(evolve_key)
                if record is not None:
                    self._set_semantic_failed(
                        evolve_key,
                        _evolution_failure_text(error),
                        expected_fingerprint=str(record.get("source_fingerprint", "")),
                    )

    def _semantic_round_capture_units(
        self,
        bundle: _RuntimeBundle,
        *,
        session_id: str,
        region_id: str,
        turn_id: str,
    ) -> list[MemoryUnit]:
        units: list[MemoryUnit] = []
        for candidate_scope in bundle.kv_store.scopes():
            if candidate_scope.session != session_id:
                continue
            for unit in self._units_in_scope(bundle, candidate_scope):
                if unit.lifecycle.value != "active" or not _is_capture_unit(unit):
                    continue
                if _metadata_text(unit, "session_id") != session_id:
                    continue
                if _metadata_text(unit, "region_id") != region_id:
                    continue
                if _metadata_text(unit, "turn_id") != turn_id:
                    continue
                units.append(unit)
        return sorted(
            units,
            key=lambda unit: (
                _event_order(_metadata_text(unit, "event_kind")),
                _metadata_text(unit, "occurred_at"),
                unit.id,
            ),
        )

    def _enqueue_pending_semantic_records(
        self,
        *,
        session_id: str | None,
        retry_failed: bool,
    ) -> None:
        if not self._semantic_enabled():
            return
        statuses = ["pending", "not_requested"]
        if retry_failed:
            statuses.append("failed")
        placeholders = ", ".join("?" for _ in statuses)
        where = f"semantic_status IN ({placeholders})"
        values: list[object] = list(statuses)
        if session_id is not None:
            where += " AND session_id = ?"
            values.append(session_id)
        with closing(self._maintenance_db()) as connection:
            rows = connection.execute(
                f"SELECT evolve_key FROM capture_evolutions WHERE {where}",
                values,
            ).fetchall()
        for row in rows:
            self._enqueue_semantic_distillation(str(row[0]), retry_failed=retry_failed)

    def _recover_semantic_distillations(self) -> None:
        self._enqueue_pending_semantic_records(session_id=None, retry_failed=True)

    def _retry_deferred_semantic_distillations(self) -> None:
        with self._semantic_state_lock:
            evolve_keys = list(self._semantic_deferred_versions)
            self._semantic_deferred_versions.clear()
        for evolve_key in evolve_keys:
            self._enqueue_semantic_distillation(evolve_key, retry_failed=True)

    def _enqueue_semantic_distillation(
        self,
        evolve_key: str,
        *,
        retry_failed: bool,
    ) -> None:
        if not self._semantic_enabled():
            return
        record = self._semantic_record(evolve_key)
        if record is None:
            return
        status = str(record.get("semantic_status", ""))
        if status in {"completed", "skipped"}:
            return
        if status == "failed" and not retry_failed:
            return
        version = str(record.get("source_fingerprint", ""))
        with self._semantic_state_lock:
            if evolve_key in self._semantic_active_versions:
                self._semantic_requested_versions[evolve_key] = version
                return
            self._semantic_active_versions[evolve_key] = version
            self._semantic_requested_versions[evolve_key] = version
        submission_error: Exception | None
        try:
            submitted = self._semantic_executor.submit(
                evolve_key,
                str(record["region_id"]),
                lambda: self._run_semantic_distillation(evolve_key),
            )
        except Exception as error:
            submitted = False
            submission_error = error
        else:
            submission_error = None
        if submitted:
            return
        with self._semantic_state_lock:
            self._semantic_active_versions.pop(evolve_key, None)
            self._semantic_requested_versions.pop(evolve_key, None)
            if submission_error is None:
                self._semantic_deferred_versions[evolve_key] = version
        with suppress(Exception):
            self._retire_stale_semantic_versions(
                evolve_key,
                session_id=str(record["session_id"]),
                turn_id=str(record["turn_id"]),
                expected_fingerprint=version,
            )
        with suppress(Exception):
            detail = (
                "semantic task queue is full"
                if submission_error is None
                else _evolution_failure_text(submission_error)
            )
            self._set_semantic_failed(evolve_key, detail, expected_fingerprint=version)

    def _wait_for_semantic_tasks(self) -> None:
        self._semantic_executor.drain()
        self._semantic_executor.close()

    def _run_semantic_distillation(self, evolve_key: str) -> None:
        record = self._semantic_record(evolve_key)
        expected_fingerprint = (
            "" if record is None else str(record.get("source_fingerprint", ""))
        )
        try:
            self._distill_semantic_turn(
                evolve_key,
                expected_fingerprint=expected_fingerprint,
            )
        except Exception as error:
            if record is not None:
                with suppress(Exception):
                    self._cleanup_semantic_version(
                        session_id=str(record["session_id"]),
                        turn_id=str(record["turn_id"]),
                        source_fingerprint=expected_fingerprint,
                    )
            with suppress(Exception):
                if self._semantic_version_is_current(evolve_key, expected_fingerprint):
                    self._set_semantic_failed(
                        evolve_key,
                        _evolution_failure_text(error),
                        expected_fingerprint=expected_fingerprint,
                    )
        finally:
            with self._semantic_state_lock:
                active = self._semantic_active_versions.pop(evolve_key, None)
                requested = self._semantic_requested_versions.pop(evolve_key, None)
            latest_record = self._semantic_record(evolve_key)
            latest_fingerprint = (
                "" if latest_record is None else str(latest_record.get("source_fingerprint", ""))
            )
            should_retry = bool(
                latest_fingerprint
                and (
                    latest_fingerprint != expected_fingerprint
                    or (
                        requested is not None
                        and active is not None
                        and requested != active
                    )
                )
            )
        if should_retry:
            self._enqueue_semantic_distillation(evolve_key, retry_failed=True)

    def _distill_semantic_turn(
        self,
        evolve_key: str,
        *,
        expected_fingerprint: str,
    ) -> None:
        bundle = self._require_bundle()
        record = self._semantic_record(evolve_key)
        if record is None:
            return
        if (
            record.get("semantic_status") in {"completed", "skipped"}
            and str(record.get("source_fingerprint", "")) == expected_fingerprint
        ):
            return
        session_id = str(record["session_id"])
        turn_id = str(record["turn_id"])
        region_id = str(record["region_id"])
        units = self._semantic_round_capture_units(
            bundle,
            session_id=session_id,
            region_id=region_id,
            turn_id=turn_id,
        )
        if not _capture_turn_is_complete(units):
            # An archived or not-yet-complete turn stays pending rather than inventing facts.
            return
        source_ids = [unit.id for unit in units]
        unit_fingerprint = _source_fingerprint(source_ids)
        if unit_fingerprint != expected_fingerprint:
            return
        started_at, received_earliest, received_latest = _capture_units_timeline(units)
        status, _ = self._reserve_evolution(
            evolve_key,
            session_id,
            turn_id,
            region_id,
            unit_fingerprint,
            source_ids,
            started_at,
            received_earliest,
            received_latest,
        )
        del status
        record = self._semantic_record(evolve_key)
        if record is None:
            return
        if str(record.get("source_fingerprint", "")) != expected_fingerprint:
            return
        self._set_semantic_pending(evolve_key, expected_fingerprint=expected_fingerprint)
        self._cleanup_semantic_version(
            session_id=session_id,
            turn_id=turn_id,
            source_fingerprint=expected_fingerprint,
        )
        # A superseding assistant version makes older published versions stale
        # before the model runs. This keeps failure paths from exposing a
        # conclusion that no longer belongs to the current turn fingerprint.
        self._retire_stale_semantic_versions(
            evolve_key,
            session_id=session_id,
            turn_id=turn_id,
            expected_fingerprint=expected_fingerprint,
        )
        source_region = self._regions.require_region(region_id)
        region_hints = [
            {"hint": "source", "region_id": source_region.region_id},
            {"hint": "user", "region_id": "user"},
            *[
                {"hint": region.region_id, "region_id": region.region_id, "name": region.name}
                for region in self._regions.list_regions()
                if region.kind == "topic"
            ],
        ]
        user_units = [
            unit for unit in units if _metadata_text(unit, "event_kind") == "user"
        ]
        assistant_units = [
            unit for unit in units if _metadata_text(unit, "event_kind") == "assistant"
        ]
        context: dict[str, object] = {
            "schema": "semantic-distillation-v1",
            "session_id": session_id,
            "turn_id": turn_id,
            "source_region": {
                "region_id": source_region.region_id,
                "kind": source_region.kind,
                "name": source_region.name,
            },
            "region_hints": region_hints,
            "messages": [
                {
                    "role": _metadata_text(unit, "event_kind"),
                    "event_id": _metadata_text(unit, "event_id"),
                    "content": unit.content,
                    "occurred_at": _metadata_text(unit, "occurred_at"),
                    "final_assistant": unit.id == assistant_units[-1].id,
                }
                for unit in units
            ],
        }
        extractor = self._semantic_extractor
        if extractor is None:
            self._set_semantic_failed(
                evolve_key,
                "semantic distillation is not configured",
                expected_fingerprint=expected_fingerprint,
            )
            return
        results = extractor(context)
        if not isinstance(results, list):
            raise MemoryKernelUnavailableError("semantic extractor returned a non-list result")
        if len(results) > _SEMANTIC_MAX_RESULTS:
            raise MemoryKernelUnavailableError("semantic extractor returned too many results")
        if not self._semantic_source_is_current(
            bundle,
            evolve_key,
            expected_fingerprint=expected_fingerprint,
            expected_unit_ids=source_ids,
        ):
            return
        self._retire_stale_semantic_versions(
            evolve_key,
            session_id=session_id,
            turn_id=turn_id,
            expected_fingerprint=expected_fingerprint,
        )
        if not self._semantic_source_is_current(
            bundle,
            evolve_key,
            expected_fingerprint=expected_fingerprint,
            expected_unit_ids=source_ids,
        ):
            return
        if not results:
            self._set_semantic_skipped(
                evolve_key,
                [{"status": "skipped", "reason": "model returned no semantic memories"}],
                expected_fingerprint=expected_fingerprint,
            )
            return

        memory_ids: list[str] = []
        memory_refs: list[dict[str, object]] = []
        diagnostics: list[dict[str, object]] = []
        source_event_ids = [_metadata_text(unit, "event_id") for unit in units]
        source_unit_regions = [
            {"unit_id": unit.id, "region_id": region_id} for unit in units
        ]
        for index, raw_result in enumerate(results):
            validated, reason = self._validate_semantic_result(
                raw_result,
                source_region=source_region,
                user_units=user_units,
            )
            if validated is None:
                if "sensitive-content" in reason:
                    self._record_semantic_sensitive_audit(
                        raw_result=raw_result,
                        reason=reason,
                        source_region=source_region,
                        source_units=units,
                        source_fingerprint=expected_fingerprint,
                    )
                diagnostics.append(
                    {
                        "index": index,
                        "status": "skipped",
                        "reason": reason,
                    }
                )
                continue
            target_region = validated["target_region"]
            memory_type = validated["memory_type"]
            content = validated["content"]
            subject = validated["claim_subject"]
            value = validated["claim_value"]
            evidence_snippet = validated["evidence_snippet"]
            user_event_id = validated["user_event_id"]
            semantic_key = _semantic_key(
                region_id=region_id,
                session_id=session_id,
                turn_id=turn_id,
                user_event_id=user_event_id,
                evidence_snippet=evidence_snippet,
                memory_type=memory_type,
                target_region_id=target_region.region_id,
                content=content,
                claim_subject=subject,
                claim_value=value,
            )
            if not self._semantic_source_is_current(
                bundle,
                evolve_key,
                expected_fingerprint=expected_fingerprint,
                expected_unit_ids=source_ids,
            ):
                return
            with self._semantic_region_lock(target_region.region_id):
                if not self._semantic_source_is_current(
                    bundle,
                    evolve_key,
                    expected_fingerprint=expected_fingerprint,
                    expected_unit_ids=source_ids,
                ):
                    return
                existing = self._find_semantic_memory(
                    bundle,
                    target_region,
                    semantic_key=semantic_key,
                    content=content,
                    memory_type=memory_type,
                    claim_subject=subject,
                    claim_value=value,
                    claims_explicit=validated["claims_explicit"],
                )
                if existing is not None:
                    memory_id = existing.id
                    self._merge_semantic_sources(
                        bundle,
                        target_region,
                        existing,
                        semantic_key=semantic_key,
                        session_id=session_id,
                        turn_id=turn_id,
                        source_region=source_region,
                        source_ids=source_ids,
                        source_event_ids=source_event_ids,
                        source_unit_regions=source_unit_regions,
                        source_fingerprint=expected_fingerprint,
                        user_event_id=user_event_id,
                        evidence_snippet=evidence_snippet,
                        user_occurred_at=validated["user_occurred_at"],
                    )
                    memory_ids.append(memory_id)
                    memory_refs.append(
                        {
                            "memory_id": memory_id,
                            "region_id": target_region.region_id,
                            "user_event_id": user_event_id,
                            "evidence_snippet": evidence_snippet,
                            "source_fingerprint": expected_fingerprint,
                            "reused": True,
                        }
                    )
                    diagnostics.append(
                        {
                            "index": index,
                            "status": "reused",
                            "memory_id": memory_id,
                            "reason": "equivalent semantic memory already exists",
                            **(
                                {
                                    "target_fallback_reason": validated[
                                        "target_fallback_reason"
                                    ]
                                }
                                if validated["target_fallback_reason"]
                                else {}
                            ),
                        }
                    )
                    continue
                conflict = self._semantic_conflict(
                    bundle,
                    target_region,
                    subject=subject,
                    value=value,
                )
                if conflict is not None:
                    conflicting_memory, existing_value = conflict
                    self._record_semantic_conflict_audit(
                        source_region=source_region,
                        source_units=units,
                        source_fingerprint=expected_fingerprint,
                        target_region=target_region,
                        conflicting_memory=conflicting_memory,
                        subject=subject,
                        value=value,
                        existing_value=existing_value,
                        user_event_id=user_event_id,
                    )
                    diagnostics.append(
                        {
                            "index": index,
                            "status": "skipped",
                            "reason": (
                                "semantic claim conflicts with memory "
                                f"{conflicting_memory.id}"
                            ),
                            **(
                                {
                                    "target_fallback_reason": validated[
                                        "target_fallback_reason"
                                    ]
                                }
                                if validated["target_fallback_reason"]
                                else {}
                            ),
                        }
                    )
                    continue
                request = CreateMemoryRequest(
                    library_id=target_region.region_id,
                    region_id=target_region.region_id,
                    content=content,
                    memory_type=memory_type,
                    source_references=tuple(
                        [
                            *(
                                f"event:{event_id}"
                                for event_id in source_event_ids
                                if event_id
                            ),
                            f"session:{session_id}",
                            f"turn:{turn_id}",
                        ]
                    ),
                    actor="semantic-distillation",
                    idempotency_key=f"semantic:{semantic_key}:{expected_fingerprint}",
                    user_metadata={
                        "semantic_schema": "semantic-distillation-v1",
                        "semantic_owned": True,
                        "semantic_key": semantic_key,
                        "semantic_source_fingerprint": expected_fingerprint,
                        "session_id": session_id,
                        "turn_id": turn_id,
                        "source_region_id": region_id,
                        "source_region_kind": source_region.kind,
                        "source_region_ids": [region_id],
                        "source_unit_ids": source_ids,
                        "source_event_ids": source_event_ids,
                        "source_unit_regions": json.dumps(
                            source_unit_regions,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        "provenance": source_ids,
                        "confidence": validated["confidence"],
                        "region_hint": validated["region_hint"],
                        "target_region_id": target_region.region_id,
                        "target_region_kind": target_region.kind,
                        "user_event_id": user_event_id,
                        "evidence_snippet": evidence_snippet,
                        "user_occurred_at": validated["user_occurred_at"],
                        "user_event_ids": [user_event_id],
                        "evidence_snippets": [evidence_snippet],
                        "claim_subject": subject,
                        "claim_value": value,
                        **(
                            {"target_fallback_reason": validated["target_fallback_reason"]}
                            if validated["target_fallback_reason"]
                            else {}
                        ),
                    },
                )
                response = self._write(request)
                memory_id = str(response.get("memory_id") or response.get("id") or "")
                if not memory_id:
                    raise MemoryKernelUnavailableError(
                        "semantic memory write returned no memory identifier"
                    )
                self._record_semantic_version(
                    memory_id,
                    region_id=target_region.region_id,
                    session_id=session_id,
                    turn_id=turn_id,
                    source_fingerprint=expected_fingerprint,
                    source_unit_ids=source_ids,
                    source_event_ids=source_event_ids,
                    source_unit_regions=source_unit_regions,
                    source_references=[
                        *(
                            f"event:{event_id}"
                            for event_id in source_event_ids
                            if event_id
                        ),
                        f"session:{session_id}",
                        f"turn:{turn_id}",
                    ],
                    source_region_ids=[region_id, target_region.region_id],
                    user_event_id=user_event_id,
                    evidence_snippet=evidence_snippet,
                    user_occurred_at=validated["user_occurred_at"],
                )
            memory_ids.append(memory_id)
            memory_refs.append(
                {
                    "memory_id": memory_id,
                    "region_id": target_region.region_id,
                    "user_event_id": user_event_id,
                    "evidence_snippet": evidence_snippet,
                    "source_fingerprint": expected_fingerprint,
                    "reused": False,
                }
            )
            diagnostics.append(
                {
                    "index": index,
                    "status": "published",
                    "memory_id": memory_id,
                    **(
                        {"target_fallback_reason": validated["target_fallback_reason"]}
                        if validated["target_fallback_reason"]
                        else {}
                    ),
                }
            )
        if not self._semantic_source_is_current(
            bundle,
            evolve_key,
            expected_fingerprint=expected_fingerprint,
            expected_unit_ids=source_ids,
        ):
            return
        if memory_ids:
            self._set_semantic_completed(
                evolve_key,
                memory_ids,
                memory_refs,
                diagnostics,
                expected_fingerprint=expected_fingerprint,
            )
        else:
            self._set_semantic_skipped(
                evolve_key,
                diagnostics,
                expected_fingerprint=expected_fingerprint,
            )

    def _validate_semantic_result(
        self,
        raw_result: object,
        *,
        source_region: MemoryRegion,
        user_units: list[MemoryUnit],
    ) -> tuple[_SemanticValidation | None, str]:
        if not isinstance(raw_result, Mapping):
            return None, "semantic result must be an object"
        content = str(raw_result.get("content") or raw_result.get("body") or "").strip()
        if not content:
            return None, "semantic content is missing"
        if len(content) > 65_535:
            return None, "semantic content exceeds the memory limit"
        if not _semantic_content_is_atomic(content):
            return None, "semantic content contains more than one atomic claim"
        memory_type = str(raw_result.get("memory_type") or "").strip()
        if memory_type not in FORMAL_MEMORY_TYPES:
            return None, "memory_type is not an allowed formal semantic type"
        confidence = raw_result.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not _SEMANTIC_MIN_CONFIDENCE <= float(confidence) <= 1.0
        ):
            return None, "confidence must be between 0.90 and 1.0"
        region_hint = str(raw_result.get("region_hint") or "").strip()
        user_event_id = str(raw_result.get("user_event_id") or "").strip()
        evidence_snippet = str(raw_result.get("evidence_snippet") or "").strip()
        if not user_event_id or not evidence_snippet:
            return None, "user evidence event and snippet are required"
        user_by_event = {
            _metadata_text(unit, "event_id"): unit
            for unit in user_units
            if _metadata_text(unit, "event_id")
        }
        user_unit = user_by_event.get(user_event_id)
        if user_unit is None:
            return None, "user_event_id does not identify a user message in this turn"
        if _semantic_user_event_is_pure_question(user_unit.content):
            return None, "the user event is a question rather than a durable statement"
        if _semantic_user_event_is_pure_request(user_unit.content):
            return None, "the user event is a request rather than a durable statement"
        if _semantic_text_is_question(content) or _semantic_text_is_question(evidence_snippet):
            return None, "semantic content is a question rather than a durable claim"
        if _semantic_text_is_request(content) or _semantic_text_is_request(evidence_snippet):
            return None, "semantic content is a request rather than a durable claim"
        if evidence_snippet not in user_unit.content:
            return None, "evidence_snippet is not an exact excerpt of the user event"
        evidence_clause = _semantic_clause_containing_evidence(user_unit.content, evidence_snippet)
        if _SEMANTIC_NON_DURABLE_INTENT.search(evidence_clause):
            return None, "evidence describes a non-durable intention rather than a durable claim"
        if _semantic_text_is_imperative(evidence_clause):
            return None, "evidence describes an imperative request rather than a durable claim"
        if _semantic_text_is_imperative(content):
            return None, "semantic content is an imperative request"
        if _SEMANTIC_DEICTIC_REFERENCE.search(content):
            return None, "semantic content depends on dialogue references"
        if (
            _semantic_has_unbound_pronoun(content)
            or _SEMANTIC_LEADING_DEMONSTRATIVE.search(content)
        ):
            return None, "semantic content contains an unbound pronoun"
        if (
            _SEMANTIC_TRANSIENT.search(content)
            or _SEMANTIC_TRANSIENT.search(evidence_snippet)
            or _SEMANTIC_NON_SEMANTIC_OUTPUT.search(content)
            or _SEMANTIC_NON_SEMANTIC_OUTPUT.search(evidence_snippet)
            or _SEMANTIC_FUTURE_ACTION.search(content)
            or _SEMANTIC_FUTURE_ACTION.search(evidence_snippet)
        ):
            return None, "semantic content is transient or non-semantic output"
        raw_subject = str(raw_result.get("claim_subject") or "").strip()
        raw_value = str(raw_result.get("claim_value") or "").strip()
        sensitive_fields, _ = _semantic_sensitive_fields(raw_result)
        if inspect_sensitive_text(content) is not None or inspect_sensitive_text(
            evidence_snippet
        ) is not None or inspect_sensitive_text(user_unit.content) is not None or sensitive_fields:
            field_suffix = (
                f" in model field(s): {', '.join(sensitive_fields)}"
                if sensitive_fields
                else ""
            )
            return None, f"semantic content failed the sensitive-content check{field_suffix}"
        durability = str(raw_result.get("durability") or "").strip().casefold()
        durable_values = {
            "durable",
            "durable_statement",
            "stable",
            "stable_fact",
        }
        if durability and durability not in durable_values:
            return None, "semantic result is not marked durable"
        if _SEMANTIC_ACTION_WITHOUT_DURABILITY.search(content):
            return None, "semantic result requires an explicit durable classification"
        if not durability:
            return None, "semantic result requires an explicit durable classification"
        claims_explicit = bool(
            _normalize_promotion_text(raw_subject)
            and _normalize_promotion_claim_value(raw_value)
        )
        if not _semantic_evidence_supports_content(
            content,
            evidence_snippet,
            claim_value=raw_value if claims_explicit else "",
            context=evidence_clause,
        ):
            return None, "evidence_snippet does not support the semantic content"
        project_scoped = bool(_SEMANTIC_PROJECT_SCOPE.search(content))
        if (
            project_scoped
            and memory_type in {"preference", "constraint"}
            and _SEMANTIC_USER_SCOPE_PREFIX.search(content)
        ):
            project_scoped = False
        if project_scoped and region_hint not in {"", "source", source_region.region_id}:
            target_region = source_region
            target_fallback_reason = (
                "project-scoped content was retained in the source region instead of "
                f"the requested region_hint {region_hint!r}"
            )
        else:
            target_region, target_fallback_reason = self._semantic_target_region(
                region_hint,
                source_region,
                memory_type=memory_type,
            )
        claim_subject, claim_value = _semantic_claim_fields(
            raw_subject,
            raw_value,
            content,
        )
        if not claim_subject or not claim_value:
            return None, "semantic claim identity cannot be derived for conflict checks"
        return (
            {
                "content": content,
                "memory_type": memory_type,
                "confidence": float(confidence),
                "region_hint": region_hint,
                "target_region": target_region,
                "user_event_id": user_event_id,
                "evidence_snippet": evidence_snippet,
                "user_occurred_at": _metadata_text(user_unit, "occurred_at"),
                "claim_subject": claim_subject,
                "claim_value": claim_value,
                "claims_explicit": claims_explicit,
                "durability": durability,
                "target_fallback_reason": target_fallback_reason,
            },
            "",
        )

    def _semantic_target_region(
        self,
        region_hint: str,
        source_region: MemoryRegion,
        *,
        memory_type: str,
    ) -> tuple[MemoryRegion, str]:
        if not region_hint:
            return source_region, "region_hint was missing; retained in the source region"
        if region_hint in {"source", source_region.region_id}:
            if memory_type == "preference" and source_region.kind == "topic":
                return (
                    self._regions.ensure_user_region(),
                    "preference content cannot target a topic; redirected to the user region",
                )
            return source_region, ""
        if region_hint == "user":
            if memory_type in {
                "domain_fact",
                "reusable_experience",
                "external_reference",
            }:
                return (
                    source_region,
                    f"{memory_type} content cannot target the user region; "
                    "retained in the source region",
                )
            return self._regions.ensure_user_region(), ""
        try:
            target = self._regions.require_region(region_hint)
        except RegionError:
            return (
                source_region,
                f"region_hint {region_hint!r} did not identify an existing topic; "
                "retained in the source region",
            )
        if target.kind == "topic":
            if memory_type == "preference":
                return (
                    self._regions.ensure_user_region(),
                    "preference content cannot target a topic; redirected to the user region",
                )
            return target, ""
        return (
            source_region,
            f"region_hint {region_hint!r} did not identify an existing topic; "
            "retained in the source region",
        )

    def _record_semantic_sensitive_audit(
        self,
        *,
        raw_result: Mapping[str, object],
        reason: str,
        source_region: MemoryRegion,
        source_units: list[MemoryUnit],
        source_fingerprint: str,
    ) -> None:
        source_unit_ids = [unit.id for unit in source_units]
        source_event_ids = [
            event_id
            for unit in source_units
            if (event_id := _metadata_text(unit, "event_id"))
        ]
        content = str(raw_result.get("content") or raw_result.get("body") or "")
        evidence = str(raw_result.get("evidence_snippet") or "")
        operation_id = "semantic-sensitive:" + hashlib.sha256(
            json.dumps(
                [
                    source_fingerprint,
                    source_region.region_id,
                    str(raw_result.get("user_event_id") or ""),
                    hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    hashlib.sha256(evidence.encode("utf-8")).hexdigest(),
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        categories = sorted(
            {
                category
                for text in (content, evidence, *(unit.content for unit in source_units))
                if (finding := inspect_sensitive_text(text)) is not None
                for category in finding.categories
            }
        )
        sensitive_fields, field_categories = _semantic_sensitive_fields(raw_result)
        categories = sorted({*categories, *field_categories})
        user_unit = next(
            (
                unit
                for unit in source_units
                if _metadata_text(unit, "event_kind") == "user"
            ),
            None,
        )
        audit_unit_id = user_unit.id if user_unit is not None else source_unit_ids[0]
        region_hint = str(raw_result.get("region_hint") or "")
        if "region_hint" in sensitive_fields:
            region_hint = "[redacted]"
        self._record_semantic_audit(
            action="sensitive",
            region_id=source_region.region_id,
            unit_id=audit_unit_id,
            operation_id=operation_id,
            detail={
                "classification": "sensitive",
                "reason": reason,
                "region_hint": region_hint,
                "sensitive_fields": list(sensitive_fields),
                "source_region_id": source_region.region_id,
                "source_region_kind": source_region.kind,
                "source_unit_ids": source_unit_ids,
                "source_event_ids": source_event_ids,
                "sensitive_categories": categories,
            },
        )

    def _record_semantic_conflict_audit(
        self,
        *,
        source_region: MemoryRegion,
        source_units: list[MemoryUnit],
        source_fingerprint: str,
        target_region: MemoryRegion,
        conflicting_memory: MemoryUnit,
        subject: str,
        value: str,
        existing_value: str,
        user_event_id: str,
    ) -> None:
        operation_id = "semantic-conflict:" + hashlib.sha256(
            json.dumps(
                [
                    source_fingerprint,
                    target_region.region_id,
                    conflicting_memory.id,
                    subject,
                    value,
                    user_event_id,
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        safe_subject = subject if inspect_sensitive_text(subject) is None else "[redacted]"
        conflicting_values = [
            item if inspect_sensitive_text(item) is None else "[redacted]"
            for item in (existing_value, value)
        ]
        self._record_semantic_audit(
            action="conflict",
            region_id=target_region.region_id,
            unit_id=conflicting_memory.id,
            operation_id=operation_id,
            detail={
                "classification": "conflict",
                "reason": "semantic claim conflicts with current active memory",
                "source_region_id": source_region.region_id,
                "source_region_kind": source_region.kind,
                "target_region_id": target_region.region_id,
                "target_region_kind": target_region.kind,
                "source_unit_ids": [unit.id for unit in source_units],
                "source_event_ids": [
                    event_id
                    for unit in source_units
                    if (event_id := _metadata_text(unit, "event_id"))
                ],
                "claim_subject": safe_subject,
                "conflicting_values": conflicting_values,
                "conflicting_memory_id": conflicting_memory.id,
            },
        )

    def _record_semantic_audit(
        self,
        *,
        action: str,
        region_id: str,
        unit_id: str,
        operation_id: str,
        detail: Mapping[str, object],
    ) -> None:
        serialized_detail = json.dumps(detail, ensure_ascii=False, sort_keys=True)
        with closing(self._maintenance_db()) as connection, connection:
            lineage_row = connection.execute(
                """SELECT logical_memory_id FROM memory_browser_index
                   WHERE region_id = ? AND unit_id = ?""",
                (region_id, unit_id),
            ).fetchone()
            logical_memory_id = str(lineage_row[0]) if lineage_row is not None else ""
            row = connection.execute(
                """SELECT id FROM governance_events
                   WHERE operation_id = ? AND action = ?
                   ORDER BY id LIMIT 1""",
                (operation_id, action),
            ).fetchone()
            if row is None:
                connection.execute(
                    """INSERT INTO governance_events
                       (region_id, unit_id, logical_memory_id, action, actor, source,
                        operation_id, before_content, after_content, detail, status, created_at)
                       VALUES (?, ?, ?, ?, 'semantic-distillation',
                               'semantic-distillation', ?, '', '', ?, 'committed', ?)""",
                    (
                        region_id,
                        unit_id,
                        logical_memory_id,
                        action,
                        operation_id,
                        serialized_detail,
                        datetime.now(UTC).isoformat(),
                    ),
                )
                return
            connection.execute(
                """UPDATE governance_events
                   SET region_id = ?, unit_id = ?, logical_memory_id = ?, detail = ?,
                       status = 'committed'
                   WHERE id = ?""",
                (region_id, unit_id, logical_memory_id, serialized_detail, int(row[0])),
            )

    @contextmanager
    def _semantic_region_lock(self, region_id: str) -> Iterator[None]:
        with self._semantic_region_locks_guard:
            lock = self._semantic_region_locks.get(region_id)
            if lock is None:
                lock = threading.RLock()
                self._semantic_region_locks[region_id] = lock
        with lock:
            yield

    def _semantic_version_is_current(
        self,
        evolve_key: str,
        expected_fingerprint: str,
    ) -> bool:
        if not expected_fingerprint:
            return False
        try:
            bundle = self._require_bundle()
            record = self._semantic_record(evolve_key)
        except Exception:
            return False
        if record is None or str(record.get("source_fingerprint", "")) != expected_fingerprint:
            return False
        try:
            units = self._semantic_round_capture_units(
                bundle,
                session_id=str(record["session_id"]),
                region_id=str(record["region_id"]),
                turn_id=str(record["turn_id"]),
            )
        except Exception:
            return False
        return _source_fingerprint([unit.id for unit in units]) == expected_fingerprint

    def _semantic_source_is_current(
        self,
        bundle: _RuntimeBundle,
        evolve_key: str,
        *,
        expected_fingerprint: str,
        expected_unit_ids: list[str],
    ) -> bool:
        record = self._semantic_record(evolve_key)
        if record is None or str(record.get("source_fingerprint", "")) != expected_fingerprint:
            return False
        units = self._semantic_round_capture_units(
            bundle,
            session_id=str(record["session_id"]),
            region_id=str(record["region_id"]),
            turn_id=str(record["turn_id"]),
        )
        current_ids = [unit.id for unit in units]
        return (
            current_ids == expected_unit_ids
            and _source_fingerprint(current_ids) == expected_fingerprint
        )

    def _semantic_conflict(
        self,
        bundle: _RuntimeBundle,
        region: MemoryRegion,
        *,
        subject: str,
        value: str,
    ) -> tuple[MemoryUnit, str] | None:
        if not subject or not value:
            return None
        target_scope = _scope_for_region(region)
        normalized_subject = _normalize_promotion_text(subject)
        normalized_value = _normalize_promotion_claim_value(value)
        for unit in self._units_in_scope(bundle, target_scope):
            if unit.lifecycle.value != "active" or _metadata_text(unit, "memory_type") not in (
                FORMAL_MEMORY_TYPES
            ):
                continue
            existing_subject = _metadata_text(unit, "claim_subject") or _metadata_text(
                unit, "semantic_claim_subject"
            )
            existing_value = _metadata_text(unit, "claim_value") or _metadata_text(
                unit, "semantic_claim_value"
            )
            if not existing_subject or not existing_value:
                existing_subject, existing_value = _promotion_claim_signature(unit.content)
            if not existing_subject or not existing_value:
                existing_subject, existing_value = _semantic_claim_fields("", "", unit.content)
            existing_subject = _normalize_promotion_text(existing_subject)
            existing_value = _normalize_promotion_claim_value(existing_value)
            if (
                existing_subject
                and existing_value
                and _promotion_subjects_compatible(existing_subject, normalized_subject)
                and _promotion_values_conflict(
                    normalized_subject,
                    (existing_value, normalized_value),
                )
            ):
                return unit, existing_value
        return None

    def _find_semantic_memory(
        self,
        bundle: _RuntimeBundle,
        region: MemoryRegion,
        *,
        semantic_key: str,
        content: str,
        memory_type: str,
        claim_subject: str,
        claim_value: str,
        claims_explicit: bool,
    ) -> MemoryUnit | None:
        target_scope = _scope_for_region(region)
        normalized_subject = _normalize_promotion_text(claim_subject)
        normalized_value = _normalize_promotion_claim_value(claim_value)
        normalized_content = _normalize_promotion_text(content)
        for unit in self._units_in_scope(bundle, target_scope):
            if unit.lifecycle.value != "active":
                continue
            if unit.user_metadata.get("semantic_manual_override") is True:
                if _metadata_text(unit, "memory_type") != memory_type:
                    continue
                existing_subject, existing_value = _semantic_claim_fields(
                    "",
                    "",
                    unit.content,
                )
                if (
                    _promotion_subjects_compatible(
                        _normalize_promotion_text(existing_subject),
                        normalized_subject,
                    )
                    and _promotion_claim_values_equivalent(
                        _normalize_promotion_claim_value(existing_value),
                        normalized_value,
                    )
                ):
                    return unit
                continue
            if _metadata_text(unit, "semantic_key") == semantic_key:
                return unit
            if (
                _metadata_text(unit, "memory_type") == memory_type
                and _normalize_promotion_text(unit.content) == normalized_content
            ):
                return unit
            if not claims_explicit or not normalized_subject or not normalized_value:
                continue
            existing_subject = _metadata_text(unit, "claim_subject")
            existing_value = _metadata_text(unit, "claim_value")
            if not existing_subject or not existing_value:
                existing_subject, existing_value = _semantic_claim_fields(
                    existing_subject,
                    existing_value,
                    unit.content,
                )
            if (
                _promotion_subjects_compatible(
                    _normalize_promotion_text(existing_subject),
                    normalized_subject,
                )
                and _promotion_claim_values_equivalent(
                    _normalize_promotion_claim_value(existing_value),
                    normalized_value,
                )
            ):
                return unit
        return None

    def _record_semantic_version(
        self,
        memory_id: str,
        *,
        region_id: str,
        session_id: str,
        turn_id: str,
        source_fingerprint: str,
        source_unit_ids: list[str],
        source_event_ids: list[str],
        source_unit_regions: list[dict[str, object]],
        source_references: list[str],
        source_region_ids: list[str],
        user_event_id: str,
        evidence_snippet: str,
        user_occurred_at: str,
    ) -> None:
        with closing(self._maintenance_db()) as connection, connection:
            connection.execute(
                """INSERT INTO semantic_memory_versions
                   (memory_id, region_id, session_id, turn_id, source_fingerprint,
                    source_unit_ids, source_event_ids, source_unit_regions,
                    source_references, source_region_ids, user_event_id,
                    evidence_snippet, user_occurred_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(memory_id, session_id, turn_id) DO UPDATE SET
                     region_id = excluded.region_id,
                     source_fingerprint = excluded.source_fingerprint,
                     source_unit_ids = excluded.source_unit_ids,
                     source_event_ids = excluded.source_event_ids,
                     source_unit_regions = excluded.source_unit_regions,
                     source_references = excluded.source_references,
                     source_region_ids = excluded.source_region_ids,
                     user_event_id = excluded.user_event_id,
                     evidence_snippet = excluded.evidence_snippet,
                     user_occurred_at = excluded.user_occurred_at,
                     updated_at = excluded.updated_at""",
                (
                    memory_id,
                    region_id,
                    session_id,
                    turn_id,
                    source_fingerprint,
                    json.dumps(source_unit_ids, ensure_ascii=False),
                    json.dumps(source_event_ids, ensure_ascii=False),
                    json.dumps(source_unit_regions, ensure_ascii=False),
                    json.dumps(source_references, ensure_ascii=False),
                    json.dumps(source_region_ids, ensure_ascii=False),
                    user_event_id,
                    evidence_snippet,
                    user_occurred_at,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def _semantic_versions_for_memory(
        self,
        memory_id: str,
        *,
        region_id: str = "",
    ) -> list[dict[str, object]]:
        with closing(self._maintenance_db()) as connection:
            memory_ids = [memory_id]
            if region_id:
                lineage_row = connection.execute(
                    """SELECT logical_memory_id FROM memory_browser_index
                       WHERE region_id = ? AND unit_id = ?""",
                    (region_id, memory_id),
                ).fetchone()
                logical_memory_id = str(lineage_row[0]) if lineage_row is not None else ""
                if logical_memory_id:
                    lineage_ids = connection.execute(
                        """SELECT unit_id FROM memory_browser_index
                           WHERE region_id = ? AND logical_memory_id = ?""",
                        (region_id, logical_memory_id),
                    ).fetchall()
                    memory_ids.extend(str(row[0]) for row in lineage_ids)
            memory_ids = list(dict.fromkeys(memory_ids))
            placeholders = ", ".join("?" for _ in memory_ids)
            rows = connection.execute(
                f"""SELECT session_id, turn_id, source_fingerprint, source_unit_ids,
                          source_event_ids, source_unit_regions, source_references,
                          source_region_ids, user_event_id, evidence_snippet,
                          user_occurred_at, region_id
                   FROM semantic_memory_versions WHERE memory_id IN ({placeholders})
                   ORDER BY updated_at, session_id, turn_id""",
                tuple(memory_ids),
            ).fetchall()
        return [
            {
                "session_id": str(row[0]),
                "turn_id": str(row[1]),
                "source_fingerprint": str(row[2]),
                "source_unit_ids": _json_string_list(row[3]),
                "source_event_ids": _json_string_list(row[4]),
                "source_unit_regions": _json_object_list(row[5]),
                "source_references": _json_string_list(row[6]),
                "source_region_ids": _json_string_list(row[7]),
                "user_event_id": str(row[8]),
                "evidence_snippet": str(row[9]),
                "user_occurred_at": str(row[10]),
                "region_id": str(row[11]),
            }
            for row in rows
        ]

    def _rebuild_semantic_metadata(
        self,
        unit: MemoryUnit,
        versions: list[dict[str, object]],
    ) -> None:
        metadata = dict(unit.user_metadata)
        latest = versions[-1]
        source_ids: list[str] = []
        source_event_ids: list[str] = []
        source_unit_regions: list[dict[str, object]] = []
        source_references: list[str] = []
        source_region_ids: list[str] = []
        for version in versions:
            source_ids.extend(_json_string_list(version.get("source_unit_ids")))
            source_event_ids.extend(_json_string_list(version.get("source_event_ids")))
            source_unit_regions.extend(_json_object_list(version.get("source_unit_regions")))
            source_references.extend(_json_string_list(version.get("source_references")))
            source_region_ids.extend(_json_string_list(version.get("source_region_ids")))
        metadata["semantic_source_fingerprint"] = str(
            latest.get("source_fingerprint", "")
        )
        metadata["session_id"] = str(latest.get("session_id", ""))
        metadata["turn_id"] = str(latest.get("turn_id", ""))
        metadata["source_unit_ids"] = list(dict.fromkeys(source_ids))
        metadata["source_event_ids"] = list(dict.fromkeys(source_event_ids))
        metadata["source_unit_regions"] = json.dumps(
            _unique_object_dicts(source_unit_regions),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        metadata["source_references"] = list(dict.fromkeys(source_references))
        metadata["source_region_ids"] = list(dict.fromkeys(source_region_ids))
        metadata["provenance"] = list(metadata["source_unit_ids"])
        metadata["user_event_ids"] = [
            str(version["user_event_id"])
            for version in versions
            if str(version.get("user_event_id", "")).strip()
        ]
        metadata["evidence_snippets"] = [
            str(version["evidence_snippet"])
            for version in versions
            if str(version.get("evidence_snippet", "")).strip()
        ]
        first_ref = next(
            (
                version
                for version in versions
                if str(version.get("user_event_id", "")).strip()
                and str(version.get("evidence_snippet", "")).strip()
            ),
            None,
        )
        if first_ref is not None:
            metadata["user_event_id"] = str(first_ref["user_event_id"])
            metadata["evidence_snippet"] = str(first_ref["evidence_snippet"])
            metadata["user_occurred_at"] = str(first_ref.get("user_occurred_at", ""))
        unit.user_metadata.clear()
        unit.user_metadata.update(metadata)
        unit.provenance = list(metadata["source_unit_ids"])

    def _retire_stale_semantic_versions(
        self,
        evolve_key: str,
        *,
        session_id: str,
        turn_id: str,
        expected_fingerprint: str,
    ) -> None:
        bundle = self._require_bundle()
        with closing(self._maintenance_db()) as connection:
            rows = connection.execute(
                """SELECT DISTINCT memory_id, region_id
                   FROM semantic_memory_versions
                   WHERE session_id = ? AND turn_id = ?
                     AND source_fingerprint != ?""",
                (session_id, turn_id, expected_fingerprint),
            ).fetchall()
        for memory_id, stored_region_id in ((str(row[0]), str(row[1])) for row in rows):
            region = self._regions.require_region(stored_region_id)
            target_scope = _scope_for_region(region)
            with self._semantic_region_lock(region.region_id):
                current = load_units(bundle.kv_store, target_scope, [memory_id])
                if not current or current[0].lifecycle.value != "active":
                    continue
                stale = current[0]
                revision = _unit_revision(stale)
                semantic_owned = _metadata_text(stale, "semantic_owned").casefold() == "true"
                remaining = self._drop_stale_semantic_version(
                    stale,
                    session_id=session_id,
                    turn_id=turn_id,
                )
                if remaining or not semantic_owned:
                    bundle.kv_store.update(
                        target_scope,
                        memory_key(stale.id),
                        dumps(stale),
                    )
                    self._refresh_browser_index_ids(
                        bundle,
                        region,
                        target_scope,
                        [stale.id],
                    )
                    if bundle.graph_store is not None:
                        self._reproject_graph_scope(
                            bundle,
                            target_scope,
                            region_id=region.region_id,
                        )
                    continue
                self._delete_document(
                    DeleteMemoryRequest(
                        library_id=region.region_id,
                        path=stale.id,
                        expected_revision=revision,
                        operation_id=(
                            f"semantic-version:{evolve_key[:16]}:{stale.id[:16]}"
                        ),
                        actor="semantic-distillation",
                        source="semantic-version-replacement",
                        preview_token=revision,
                        mode="forget",
                    )
                )
                with closing(self._maintenance_db()) as connection, connection:
                    connection.execute(
                        "DELETE FROM semantic_memory_versions WHERE memory_id = ?",
                        (stale.id,),
                    )

    def _cleanup_semantic_version(
        self,
        *,
        session_id: str,
        turn_id: str,
        source_fingerprint: str,
    ) -> None:
        bundle = self._require_bundle()
        locations: dict[str, str] = {}
        with closing(self._maintenance_db()) as connection:
            rows = connection.execute(
                """SELECT memory_id, region_id FROM semantic_memory_versions
                   WHERE session_id = ? AND turn_id = ?
                     AND source_fingerprint = ?""",
                (session_id, turn_id, source_fingerprint),
            ).fetchall()
        for row in rows:
            locations[str(row[0])] = str(row[1])
        for region in self._regions.list_regions():
            scope = _scope_for_region(region)
            for unit in self._units_in_scope(bundle, scope):
                if unit.lifecycle.value != "active":
                    continue
                if _metadata_text(unit, "semantic_schema") != "semantic-distillation-v1":
                    continue
                if (
                    _metadata_text(unit, "session_id") != session_id
                    or _metadata_text(unit, "turn_id") != turn_id
                    or _metadata_text(unit, "semantic_source_fingerprint")
                    != source_fingerprint
                ):
                    continue
                locations.setdefault(unit.id, region.region_id)
        cleaned_ids: list[str] = []
        for memory_id, region_id in locations.items():
            try:
                region = self._regions.require_region(region_id)
            except RegionError:
                continue
            scope = _scope_for_region(region)
            with self._semantic_region_lock(region.region_id):
                current = load_units(bundle.kv_store, scope, [memory_id])
                if not current or current[0].lifecycle.value != "active":
                    cleaned_ids.append(memory_id)
                    continue
                unit = current[0]
                revision = _unit_revision(unit)
                semantic_owned = _metadata_text(unit, "semantic_owned").casefold() == "true"
                remaining = self._drop_stale_semantic_version(
                    unit,
                    session_id=session_id,
                    turn_id=turn_id,
                )
                if remaining or not semantic_owned:
                    bundle.kv_store.update(scope, memory_key(unit.id), dumps(unit))
                    self._refresh_browser_index_ids(bundle, region, scope, [unit.id])
                    if bundle.graph_store is not None:
                        self._reproject_graph_scope(
                            bundle,
                            scope,
                            region_id=region.region_id,
                        )
                else:
                    self._delete_document(
                        DeleteMemoryRequest(
                            library_id=region.region_id,
                            path=unit.id,
                            expected_revision=revision,
                            operation_id=(
                                "semantic-partial:"
                                f"{source_fingerprint[:16]}:{unit.id[:16]}"
                            ),
                            actor="semantic-distillation",
                            source="semantic-partial-cleanup",
                            preview_token=revision,
                            mode="forget",
                        )
                    )
                    with closing(self._maintenance_db()) as connection, connection:
                        connection.execute(
                            "DELETE FROM semantic_memory_versions WHERE memory_id = ?",
                            (unit.id,),
                        )
                    cleaned_ids.append(memory_id)
        if cleaned_ids:
            with closing(self._idempotency_db()) as connection, connection:
                connection.executemany(
                    "DELETE FROM memory_write_idempotency WHERE memory_id = ?",
                    [(memory_id,) for memory_id in cleaned_ids],
                )
        with closing(self._idempotency_db()) as connection, connection:
            connection.execute(
                """DELETE FROM memory_write_idempotency
                   WHERE status = 'pending'
                     AND idempotency_key LIKE ?""",
                (f"semantic:%:{source_fingerprint}",),
            )
        self._discard_browser_index_pending_for_units(cleaned_ids)

    def _discard_browser_index_pending_for_units(self, unit_ids: list[str]) -> None:
        wanted = set(unit_ids)
        if not wanted:
            return
        with closing(self._maintenance_db()) as connection:
            rows = connection.execute(
                "SELECT token, unit_ids FROM memory_browser_index_pending"
            ).fetchall()
        tokens: list[str] = []
        for row in rows:
            try:
                raw_ids = json.loads(str(row[1]))
            except json.JSONDecodeError:
                continue
            if not isinstance(raw_ids, list):
                continue
            if wanted.intersection(str(value) for value in raw_ids):
                tokens.append(str(row[0]))
        for token in tokens:
            self._complete_browser_index_pending(token)

    def _drop_stale_semantic_version(
        self,
        unit: MemoryUnit,
        *,
        session_id: str,
        turn_id: str,
    ) -> bool:
        versions = self._semantic_versions_for_memory(unit.id)
        remaining = [
            version
            for version in versions
            if not (
                str(version.get("session_id", "")) == session_id
                and str(version.get("turn_id", "")) == turn_id
            )
        ]
        with closing(self._maintenance_db()) as connection, connection:
            connection.execute(
                """DELETE FROM semantic_memory_versions
                   WHERE memory_id = ? AND session_id = ? AND turn_id = ?""",
                (unit.id, session_id, turn_id),
            )
        if remaining:
            self._rebuild_semantic_metadata(unit, remaining)
            return True
        self._clear_semantic_metadata(unit)
        return False

    @staticmethod
    def _clear_semantic_metadata(unit: MemoryUnit) -> None:
        for key in (
            "semantic_schema",
            "semantic_owned",
            "semantic_key",
            "semantic_source_fingerprint",
        ):
            unit.user_metadata.pop(key, None)

    def _clear_manual_override_semantic_metadata(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        unit: MemoryUnit,
        *,
        region_id: str,
    ) -> None:
        promotion_owned = unit.user_metadata.get("promotion_marker") is True
        source_region_ids = _metadata_strings(unit, "source_region_ids")
        source_region_id = _metadata_text(unit, "source_region_id")
        if source_region_id and source_region_id not in source_region_ids:
            source_region_ids.insert(0, source_region_id)
        cross_region_semantic = (
            _metadata_text(unit, "semantic_schema") == "semantic-distillation-v1"
            and any(source_id != region_id for source_id in source_region_ids)
        )
        preserved_provenance_keys = {
            "source_region_id",
            "source_region_kind",
            "source_region_ids",
            "source_unit_ids",
            "source_event_ids",
            "source_unit_regions",
            "provenance",
        }
        if cross_region_semantic:
            preserved_provenance_keys.update(
                {
                    "session_id",
                    "turn_id",
                    "user_event_id",
                    "user_event_ids",
                    "user_occurred_at",
                    "evidence_snippet",
                    "evidence_snippets",
                }
            )
        for key in (
            "semantic_schema",
            "semantic_owned",
            "semantic_key",
            "semantic_source_fingerprint",
            "claim_subject",
            "claim_value",
            "semantic_claim_subject",
            "semantic_claim_value",
            "session_id",
            "turn_id",
            "source_region_id",
            "source_region_kind",
            "source_region_ids",
            "target_region_id",
            "target_region_kind",
            "source_unit_ids",
            "source_event_ids",
            "source_unit_regions",
            "provenance",
            "confidence",
            "region_hint",
            "user_event_id",
            "user_occurred_at",
            "user_event_ids",
            "evidence_snippet",
            "evidence_snippets",
        ):
            if key in preserved_provenance_keys and (
                promotion_owned or cross_region_semantic
            ):
                continue
            unit.user_metadata.pop(key, None)
        if not promotion_owned and not cross_region_semantic:
            unit.provenance = []
        bundle.kv_store.update(scope, memory_key(unit.id), dumps(unit))
        if not promotion_owned and not cross_region_semantic:
            with closing(self._maintenance_db()) as connection, connection:
                connection.execute(
                    "DELETE FROM semantic_memory_versions WHERE memory_id = ?",
                    (unit.id,),
                )

    def _merge_semantic_sources(
        self,
        bundle: _RuntimeBundle,
        region: MemoryRegion,
        unit: MemoryUnit,
        *,
        semantic_key: str,
        session_id: str,
        turn_id: str,
        source_region: MemoryRegion,
        source_ids: list[str],
        source_event_ids: list[str],
        source_unit_regions: list[dict[str, object]],
        source_fingerprint: str,
        user_event_id: str,
        evidence_snippet: str,
        user_occurred_at: str,
    ) -> None:
        scope = _scope_for_region(region)
        current_references = [
            *(f"event:{event_id}" for event_id in source_event_ids if event_id),
            f"session:{session_id}",
            f"turn:{turn_id}",
        ]
        self._record_semantic_version(
            unit.id,
            region_id=region.region_id,
            session_id=session_id,
            turn_id=turn_id,
            source_fingerprint=source_fingerprint,
            source_unit_ids=source_ids,
            source_event_ids=source_event_ids,
            source_unit_regions=source_unit_regions,
            source_references=current_references,
            source_region_ids=[region.region_id, source_region.region_id],
            user_event_id=user_event_id,
            evidence_snippet=evidence_snippet,
            user_occurred_at=user_occurred_at,
        )
        versions = self._semantic_versions_for_memory(unit.id, region_id=region.region_id)
        metadata = dict(unit.user_metadata)
        metadata["semantic_schema"] = "semantic-distillation-v1"
        metadata["semantic_key"] = semantic_key
        unit.user_metadata.clear()
        unit.user_metadata.update(metadata)
        self._rebuild_semantic_metadata(unit, versions)
        bundle.kv_store.update(scope, memory_key(unit.id), dumps(unit))
        self._refresh_browser_index_ids(bundle, region, scope, [unit.id])
        if bundle.graph_store is not None:
            self._reproject_graph_scope(bundle, scope, region_id=region.region_id)

    def _semantic_record(self, evolve_key: str) -> dict[str, object] | None:
        with closing(self._maintenance_db()) as connection:
            row = connection.execute(
                """SELECT session_id, turn_id, region_id, semantic_status,
                          semantic_memory_ids, semantic_memory_refs,
                          semantic_diagnostics, semantic_error, semantic_updated_at,
                          source_fingerprint
                   FROM capture_evolutions WHERE evolve_key = ?""",
                (evolve_key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "session_id": str(row[0]),
            "turn_id": str(row[1]),
            "region_id": str(row[2]),
            "semantic_status": str(row[3]),
            "semantic_memory_ids": _json_string_list(row[4]),
            "semantic_memory_refs": _json_object_list(row[5]),
            "semantic_diagnostics": _json_object_list(row[6]),
            "semantic_error": str(row[7]),
            "semantic_updated_at": None if row[8] is None else str(row[8]),
            "source_fingerprint": str(row[9]),
        }

    def _set_semantic_pending(
        self,
        evolve_key: str,
        *,
        expected_fingerprint: str,
    ) -> None:
        with closing(self._maintenance_db()) as connection, connection:
            connection.execute(
                """UPDATE capture_evolutions
                   SET semantic_status = 'pending', semantic_error = '',
                       semantic_updated_at = ?
                   WHERE evolve_key = ? AND source_fingerprint = ?
                     AND semantic_status != 'completed'""",
                (datetime.now(UTC).isoformat(), evolve_key, expected_fingerprint),
            )

    def _set_semantic_completed(
        self,
        evolve_key: str,
        memory_ids: list[str],
        memory_refs: list[dict[str, object]],
        diagnostics: list[dict[str, object]],
        *,
        expected_fingerprint: str,
    ) -> None:
        unique_ids = list(dict.fromkeys(memory_ids))
        with closing(self._maintenance_db()) as connection, connection:
            connection.execute(
                """UPDATE capture_evolutions
                   SET semantic_status = 'completed', semantic_memory_ids = ?,
                       semantic_memory_refs = ?, semantic_diagnostics = ?,
                       semantic_error = '', semantic_updated_at = ?
                   WHERE evolve_key = ? AND source_fingerprint = ?
                     AND semantic_status != 'completed'""",
                (
                    json.dumps(unique_ids, ensure_ascii=False),
                    json.dumps(memory_refs, ensure_ascii=False),
                    json.dumps(diagnostics, ensure_ascii=False),
                    datetime.now(UTC).isoformat(),
                    evolve_key,
                    expected_fingerprint,
                ),
            )

    def _set_semantic_skipped(
        self,
        evolve_key: str,
        diagnostics: list[dict[str, object]],
        *,
        expected_fingerprint: str,
    ) -> None:
        with closing(self._maintenance_db()) as connection, connection:
            connection.execute(
                """UPDATE capture_evolutions
                   SET semantic_status = 'skipped', semantic_memory_ids = '[]',
                       semantic_memory_refs = '[]', semantic_diagnostics = ?,
                       semantic_error = '', semantic_updated_at = ?
                   WHERE evolve_key = ? AND source_fingerprint = ?
                     AND semantic_status != 'completed'""",
                (
                    json.dumps(diagnostics, ensure_ascii=False),
                    datetime.now(UTC).isoformat(),
                    evolve_key,
                    expected_fingerprint,
                ),
            )

    def _set_semantic_failed(
        self,
        evolve_key: str,
        error: str,
        *,
        expected_fingerprint: str,
    ) -> None:
        with closing(self._maintenance_db()) as connection, connection:
            connection.execute(
                """UPDATE capture_evolutions
                   SET semantic_status = 'failed', semantic_error = ?,
                       semantic_updated_at = ?
                   WHERE evolve_key = ? AND source_fingerprint = ?
                     AND semantic_status != 'completed'""",
                (
                    error[:1000],
                    datetime.now(UTC).isoformat(),
                    evolve_key,
                    expected_fingerprint,
                ),
            )

    async def start_promotion(self, request: PromotionRunRequest) -> dict[str, object]:
        return await asyncio.to_thread(self._start_promotion, request)

    def _start_promotion(self, request: PromotionRunRequest) -> dict[str, object]:
        source_regions = self._promotion_source_regions(request.source_region_ids)
        job_id = f"promotion-{uuid.uuid4().hex}"
        requested_at = datetime.now(UTC).isoformat()
        with closing(self._maintenance_db()) as connection, connection:
            connection.execute(
                """INSERT INTO promotion_jobs
                   (job_id, status, source_region_ids, requested_at)
                   VALUES (?, 'pending', ?, ?)""",
                (
                    job_id,
                    json.dumps(
                        [region.region_id for region in source_regions],
                        ensure_ascii=False,
                    ),
                    requested_at,
                ),
            )
        thread = threading.Thread(
            target=self._run_promotion_job,
            args=(job_id, tuple(source_regions)),
            name=f"memory-promotion-{job_id}",
            daemon=True,
        )
        with self._promotion_threads_lock:
            self._promotion_threads[job_id] = thread
        thread.start()
        return self._promotion_status_payload(job_id)

    def promotion_status(self, request: PromotionStatusRequest) -> dict[str, object]:
        return self._promotion_status_payload(request.job_id)

    def _promotion_source_regions(
        self,
        source_region_ids: tuple[str, ...] | None,
    ) -> list[MemoryRegion]:
        regions = self._regions.list_regions()
        by_id = {region.region_id: region for region in regions}
        if source_region_ids is None:
            selected = [region for region in regions if region.kind != "user"]
        else:
            selected = []
            for region_id in source_region_ids:
                region = by_id.get(region_id)
                if region is None:
                    raise MemoryRecordNotFoundError(f"source memory region not found: {region_id}")
                if region.kind == "user":
                    raise ValueError("user memory cannot be a promotion source")
                selected.append(region)
        return selected

    def _wait_for_promotion_threads(self) -> None:
        with self._promotion_threads_lock:
            threads = tuple(self._promotion_threads.values())
        for thread in threads:
            thread.join(timeout=30.0)

    def _run_promotion_job(
        self,
        job_id: str,
        source_regions: tuple[MemoryRegion, ...],
    ) -> None:
        try:
            self._update_promotion_job(
                job_id,
                status="running",
                started_at=datetime.now(UTC).isoformat(),
            )
            bundle = self._require_bundle()
            candidates = self._promotion_candidates(bundle, source_regions)
            topics = [region for region in self._regions.list_regions() if region.kind == "topic"]
            decisions: list[dict[str, object]] = []
            promoted = 0
            skipped = 0
            failed = 0
            conflict_groups = 0
            classified: list[_PromotionClassified] = []
            for candidate in candidates:
                try:
                    decision = self._classify_promotion_candidate(candidate, topics)
                except Exception as error:
                    failed += 1
                    decisions.append(
                        {
                            "source_unit_id": candidate["unit"].id,
                            "source_region_id": candidate["region"].region_id,
                            "status": "failed",
                            "reason": str(error),
                        }
                    )
                    continue
                target = self._promotion_target(candidate, decision, topics)
                if decision["category"] == "sensitive":
                    self._record_promotion_skip(
                        job_id,
                        candidate,
                        decision,
                        action="sensitive",
                        reason="sensitive content requires manual review",
                    )
                    skipped += 1
                    decisions.append(
                        self._promotion_skip_payload(
                            candidate,
                            decision,
                            target,
                            status="skipped",
                            reason="sensitive content requires manual review",
                        )
                    )
                    continue
                if decision["category"] == "conflict":
                    conflict_groups += 1
                    if target is not None:
                        self._record_promotion_conflict(
                            job_id,
                            decision["claim_subject"] or candidate["unit"].id,
                            target,
                            (candidate["unit"].content,),
                            [(candidate, decision, target)],
                        )
                    else:
                        self._record_promotion_skip(
                            job_id,
                            candidate,
                            decision,
                            action="conflict",
                            reason="classifier reported a conflict without a promotion target",
                        )
                    skipped += 1
                    decisions.append(
                        self._promotion_skip_payload(
                            candidate,
                            decision,
                            target,
                            status="skipped",
                            reason="classifier reported a conflict",
                        )
                    )
                    continue
                if target is None:
                    self._record_promotion_skip(
                        job_id,
                        candidate,
                        decision,
                        action="promotion_skip",
                        reason=decision["reason"],
                    )
                    skipped += 1
                    decisions.append(
                        self._promotion_skip_payload(
                            candidate,
                            decision,
                            target,
                            status="skipped",
                            reason=decision["reason"],
                        )
                    )
                    continue
                classified.append((candidate, decision, target))

            conflicts = self._promotion_conflict_groups(classified)
            conflict_groups += len(conflicts)
            conflicted_units: set[str] = set()
            for (subject, _target_id, values), items in conflicts.items():
                target = items[0][2]
                self._record_promotion_conflict(
                    job_id,
                    subject,
                    target,
                    values,
                    items,
                )
                for candidate, decision, _ in items:
                    conflicted_units.add(candidate["unit"].id)
                    skipped += 1
                    decisions.append(
                        self._promotion_skip_payload(
                            candidate,
                            decision,
                            target,
                            status="conflict",
                            reason="conflicting sources for the same claim",
                        )
                    )

            groups: dict[tuple[str, ...], list[_PromotionClassified]] = {}
            for item in classified:
                if item[0]["unit"].id in conflicted_units:
                    continue
                subject = _normalize_promotion_text(item[1]["claim_subject"])
                value = _normalize_promotion_claim_value(item[1]["claim_value"])
                if subject and value:
                    key = ("claim", subject, value, item[2].region_id)
                else:
                    key = (
                        "content",
                        _normalize_promotion_text(item[0]["unit"].content),
                        "",
                        item[2].region_id,
                    )
                groups.setdefault(key, []).append(item)

            for items in groups.values():
                source_ids = {
                    source_id
                    for candidate, _, _ in items
                    for source_id in self._promotion_evidence_source_ids(candidate)
                }
                source_region_ids = {
                    candidate["region"].region_id for candidate, _, _ in items
                }
                explicit_ok = any(
                    decision["status"] == "promote" and candidate["explicit"]
                    for candidate, decision, _ in items
                )
                if not explicit_ok and len(source_region_ids) < 2:
                    for candidate, decision, target in items:
                        self._record_promotion_skip(
                            job_id,
                            candidate,
                            decision,
                            action="promotion_skip",
                            reason="insufficient independent evidence",
                        )
                        skipped += 1
                        decisions.append(
                            self._promotion_skip_payload(
                                candidate,
                                decision,
                                target,
                                status="skipped",
                                reason="insufficient independent evidence",
                                source_unit_ids=sorted(source_ids),
                            )
                        )
                    continue
                evidence_kind = "explicit" if explicit_ok else "multi_source"
                representative = min(
                    items,
                    key=lambda item: (item[0]["region"].region_id, item[0]["unit"].id),
                )
                candidate, decision, target = representative
                promotion_decision = cast(
                    _PromotionDecision,
                    {
                        **decision,
                        "status": "promote",
                        "reason": (
                            "explicit high-confidence statement"
                            if explicit_ok
                            else f"repeated by {len(source_region_ids)} independent sources"
                        ),
                    },
                )
                try:
                    outcome = self._apply_promotion(
                        bundle,
                        job_id,
                        [item[0] for item in items],
                        target,
                        promotion_decision,
                        evidence_kind,
                    )
                except Exception as error:
                    failed += 1
                    decisions.append(
                        {
                            "source_unit_id": candidate["unit"].id,
                            "source_region_id": candidate["region"].region_id,
                            "status": "failed",
                            "reason": str(error),
                        }
                    )
                    continue
                outcome_status = str(outcome["decision"].get("status", ""))
                if outcome_status == "conflict":
                    conflict_groups += 1
                    skipped += 1
                elif outcome["created"]:
                    promoted += 1
                else:
                    skipped += 1
                decisions.append(outcome["decision"])
            self._update_promotion_job(
                job_id,
                status="completed",
                completed_at=datetime.now(UTC).isoformat(),
                summary={
                    "candidates": len(candidates),
                    "promoted": promoted,
                    "skipped": skipped,
                    "failed": failed,
                    "conflict_groups": conflict_groups,
                    "evidence_groups": len(groups),
                    "decisions": decisions,
                },
            )
        except Exception as error:
            self._update_promotion_job(
                job_id,
                status="failed",
                completed_at=datetime.now(UTC).isoformat(),
                error=str(error),
            )
        finally:
            with self._promotion_threads_lock:
                self._promotion_threads.pop(job_id, None)

    def _update_promotion_job(
        self,
        job_id: str,
        *,
        status: str,
        started_at: str | None = None,
        completed_at: str | None = None,
        summary: Mapping[str, object] | None = None,
        error: str | None = None,
    ) -> None:
        assignments = ["status = ?"]
        values: list[object] = [status]
        if started_at is not None:
            assignments.append("started_at = ?")
            values.append(started_at)
        if completed_at is not None:
            assignments.append("completed_at = ?")
            values.append(completed_at)
        if summary is not None:
            assignments.append("summary = ?")
            values.append(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        if error is not None:
            assignments.append("error = ?")
            values.append(error)
        values.append(job_id)
        with closing(self._maintenance_db()) as connection, connection:
            connection.execute(
                f"UPDATE promotion_jobs SET {', '.join(assignments)} WHERE job_id = ?",
                tuple(values),
            )

    def _promotion_status_payload(self, job_id: str) -> dict[str, object]:
        with closing(self._maintenance_db()) as connection:
            row = connection.execute(
                """SELECT job_id, status, source_region_ids, requested_at,
                          started_at, completed_at, summary, error
                   FROM promotion_jobs WHERE job_id = ?""",
                (job_id,),
            ).fetchone()
        if row is None:
            raise MemoryRecordNotFoundError("promotion job not found")
        try:
            source_region_ids = json.loads(str(row[2]))
        except json.JSONDecodeError:
            source_region_ids = []
        try:
            summary = json.loads(str(row[6]))
        except json.JSONDecodeError:
            summary = {}
        return {
            "job_id": str(row[0]),
            "status": str(row[1]),
            "source_region_ids": source_region_ids,
            "requested_at": str(row[3]),
            "started_at": row[4],
            "completed_at": row[5],
            "summary": summary,
            "error": str(row[7]),
        }

    @staticmethod
    def _promotion_skip_payload(
        candidate: _PromotionCandidate,
        decision: _PromotionDecision,
        target: MemoryRegion | None,
        *,
        status: str,
        reason: str,
        source_unit_ids: list[str] | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "source_unit_id": candidate["unit"].id,
            "source_region_id": candidate["region"].region_id,
            "classification": decision["category"],
            "confidence": decision["confidence"],
            "status": status,
            "reason": reason,
        }
        if target is not None:
            payload["target_region_id"] = target.region_id
        if source_unit_ids is not None:
            payload["source_unit_ids"] = source_unit_ids
        return payload

    def _promotion_conflict_groups(
        self,
        classified: list[_PromotionClassified],
    ) -> dict[tuple[str, str, tuple[str, ...]], list[_PromotionClassified]]:
        claims: dict[tuple[str, str], dict[str, list[_PromotionClassified]]] = {}
        for item in classified:
            candidate, decision, target = item
            subject = _normalize_promotion_text(decision["claim_subject"])
            value = _normalize_promotion_claim_value(decision["claim_value"])
            if not subject or not value:
                continue
            claims.setdefault((target.region_id, subject), {}).setdefault(value, []).append(item)

        conflicts: dict[tuple[str, str, tuple[str, ...]], list[_PromotionClassified]] = {}
        for (target_id, subject), values in claims.items():
            equivalence_groups: list[tuple[str, list[_PromotionClassified]]] = []
            for value, value_items in values.items():
                equivalent_group = next(
                    (
                        group
                        for group in equivalence_groups
                        if _promotion_claim_values_equivalent(group[0], value)
                    ),
                    None,
                )
                if equivalent_group is None:
                    equivalence_groups.append((value, list(value_items)))
                else:
                    equivalent_group[1].extend(value_items)
            if len(equivalence_groups) < 2:
                continue
            items = [item for _, group in equivalence_groups for item in group]
            source_ids = {
                source_id
                for candidate, _, _ in items
                for source_id in self._promotion_evidence_source_ids(candidate)
            }
            if len(source_ids) < 2:
                continue
            representatives = tuple(group[0] for group in equivalence_groups)
            if not _promotion_values_conflict(subject, representatives):
                continue
            conflicts[(subject, target_id, tuple(sorted(representatives)))] = items
        return conflicts

    def _promotion_evidence_source_ids(self, candidate: _PromotionCandidate) -> tuple[str, ...]:
        user_sources = tuple(
            source
            for source in candidate["source_units"]
            if _metadata_text(source, "event_kind") == "user"
        )
        if user_sources:
            return tuple(dict.fromkeys(str(source.id) for source in user_sources))
        return (str(candidate["unit"].id),)

    def _record_promotion_skip(
        self,
        job_id: str,
        candidate: _PromotionCandidate,
        decision: _PromotionDecision,
        *,
        action: str,
        reason: str,
    ) -> None:
        source_unit_ids = self._promotion_evidence_source_ids(candidate)
        detail = {
            "job_id": job_id,
            "source_unit_ids": list(source_unit_ids),
            "source_region_id": candidate["region"].region_id,
            "source_region_kind": candidate["region"].kind,
            "classification": decision["category"],
            "confidence": decision["confidence"],
            "reason": reason,
        }
        operation_material = [
            action,
            candidate["region"].region_id,
            candidate["unit"].id,
            hashlib.sha256(candidate["unit"].content.encode("utf-8")).hexdigest(),
            reason,
        ]
        operation_id = f"{action}:" + hashlib.sha256(
            json.dumps(operation_material, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        self._record_promotion_audit(
            action=action,
            region_id=candidate["region"].region_id,
            unit_id=candidate["unit"].id,
            operation_id=operation_id,
            job_id=job_id,
            detail=detail,
        )

    def _record_promotion_conflict(
        self,
        job_id: str,
        subject: str,
        target: MemoryRegion,
        values: tuple[str, ...],
        items: list[_PromotionClassified],
    ) -> None:
        source_regions = {
            candidate["region"].region_id: candidate["region"]
            for candidate, _, _ in items
        }
        source_unit_ids = sorted(
            {
                str(source.id)
                for candidate, _, _ in items
                for source in (
                    candidate["source_units"] or (candidate["unit"],)
                )
            }
        )
        detail = {
            "job_id": job_id,
            "claim_subject": subject,
            "conflicting_values": sorted(values),
            "source_unit_ids": source_unit_ids,
            "source_region_ids": sorted(source_regions),
            "target_region_id": target.region_id,
        }
        conflict_key = hashlib.sha256(
            json.dumps(
                [target.region_id, subject, sorted(values)],
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        for region_id, region in sorted(source_regions.items()):
            unit_id = next(
                candidate["unit"].id
                for candidate, _, _ in items
                if candidate["region"].region_id == region_id
            )
            self._record_promotion_audit(
                action="conflict",
                region_id=region.region_id,
                unit_id=unit_id,
                operation_id=f"conflict:{conflict_key}:{region_id}",
                job_id=job_id,
                detail=detail,
            )

    def _record_promotion_audit(
        self,
        *,
        action: str,
        region_id: str,
        unit_id: str,
        operation_id: str,
        job_id: str,
        detail: Mapping[str, object],
    ) -> None:
        serialized_detail = json.dumps(detail, ensure_ascii=False, sort_keys=True)
        with closing(self._maintenance_db()) as connection, connection:
            lineage_row = connection.execute(
                """SELECT logical_memory_id FROM memory_browser_index
                   WHERE region_id = ? AND unit_id = ?""",
                (region_id, unit_id),
            ).fetchone()
            logical_memory_id = str(lineage_row[0]) if lineage_row is not None else ""
            row = connection.execute(
                """SELECT id FROM governance_events
                   WHERE operation_id = ? AND action = ?
                   ORDER BY id LIMIT 1""",
                (operation_id, action),
            ).fetchone()
            if row is None:
                connection.execute(
                    """INSERT INTO governance_events
                       (region_id, unit_id, logical_memory_id, action, actor, source, operation_id,
                        before_content, after_content, detail, status, created_at)
                       VALUES (?, ?, ?, ?, 'platform:auto-promotion', ?,
                               ?, '', '', ?, 'committed', ?)""",
                    (
                        region_id,
                        unit_id,
                        logical_memory_id,
                        action,
                        f"promotion:{job_id}",
                        operation_id,
                        serialized_detail,
                        datetime.now(UTC).isoformat(),
                    ),
                )
                return
            connection.execute(
                """UPDATE governance_events
                   SET region_id = ?, unit_id = ?, logical_memory_id = ?, source = ?, detail = ?,
                       status = 'committed'
                   WHERE id = ?""",
                (
                    region_id,
                    unit_id,
                    logical_memory_id,
                    f"promotion:{job_id}",
                    serialized_detail,
                    int(row[0]),
                ),
            )

    def _promotion_candidates(
        self,
        bundle: _RuntimeBundle,
        source_regions: tuple[MemoryRegion, ...],
    ) -> list[_PromotionCandidate]:
        allowed_types = {
            "preference",
            "constraint",
            "decision",
            "domain_fact",
            "reusable_experience",
            "external_reference",
        }
        candidates: list[_PromotionCandidate] = []
        for region in source_regions:
            for scope in bundle.kv_store.scopes():
                if scope.space != region.space:
                    continue
                if region.kind == "session" and scope.session != (region.session_id or ""):
                    continue
                for unit in self._units_in_scope(bundle, scope):
                    if unit.lifecycle.value != "active":
                        continue
                    memory_type = _metadata_text(unit, "memory_type")
                    if memory_type not in allowed_types:
                        continue
                    source_units = self._promotion_source_units(bundle, scope, unit)
                    explicit = self._promotion_has_explicit_source(unit, source_units)
                    candidate: _PromotionCandidate = {
                        "region": region,
                        "scope": scope,
                        "unit": unit,
                        "source_units": source_units,
                        "explicit": explicit,
                    }
                    candidates.append(candidate)
        return sorted(
            candidates,
            key=lambda item: (
                item["region"].region_id,
                item["unit"].id,
            ),
        )

    def _promotion_source_units(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        unit: MemoryUnit,
    ) -> tuple[MemoryUnit, ...]:
        source_ids = _metadata_strings(unit, "source_unit_ids")
        if not source_ids:
            source_ids = _metadata_strings(unit, "provenance")
        if not source_ids:
            return ()
        return tuple(load_units(bundle.kv_store, scope, list(dict.fromkeys(source_ids))))

    @staticmethod
    def _promotion_group_sources(
        candidates: list[_PromotionCandidate],
    ) -> tuple[
        tuple[MemoryUnit, ...],
        list[str],
        list[str],
        list[str],
        list[dict[str, str]],
        list[str],
        list[dict[str, str]],
    ]:
        source_units: list[MemoryUnit] = []
        seen_units: set[str] = set()
        source_references: list[str] = []
        source_regions: list[dict[str, str]] = []
        source_region_ids: list[str] = []
        source_unit_regions: list[dict[str, str]] = []
        for candidate in candidates:
            units = candidate["source_units"] or (candidate["unit"],)
            for source in units:
                source_unit_regions.append(
                    {"unit_id": source.id, "region_id": candidate["region"].region_id}
                )
                if source.id in seen_units:
                    continue
                seen_units.add(str(source.id))
                source_units.append(source)
            region = candidate["region"]
            if region.region_id not in source_region_ids:
                source_region_ids.append(region.region_id)
                source_regions.append(
                    {"region_id": region.region_id, "region_kind": region.kind}
                )
            source_references.extend(
                _promotion_source_references(
                    candidate["unit"],
                    candidate["source_units"],
                    region,
                )
            )
        source_ids = [str(source.id) for source in source_units]
        source_event_ids = list(
            dict.fromkeys(
                event_id
                for source in source_units
                if (event_id := _metadata_text(source, "event_id"))
            )
        )
        return (
            tuple(source_units),
            source_ids,
            source_event_ids,
            list(dict.fromkeys(source_references)),
            source_regions,
            source_region_ids,
            source_unit_regions,
        )

    @staticmethod
    def _promotion_has_explicit_source(
        unit: MemoryUnit,
        source_units: tuple[MemoryUnit, ...],
    ) -> bool:
        if any(_metadata_text(source, "event_kind") == "user" for source in source_units):
            return True
        if _metadata_text(unit, "event_kind") == "user":
            return True
        if source_units:
            return False
        return bool(
            _metadata_text(unit, "request_actor")
            or unit.user_metadata.get("source_references")
            or unit.provenance
        )

    def _classify_promotion_candidate(
        self,
        candidate: _PromotionCandidate,
        topics: list[MemoryRegion],
    ) -> _PromotionDecision:
        unit = candidate["unit"]
        region = candidate["region"]
        context: dict[str, object] = {
            "source_region_id": region.region_id,
            "source_region_kind": region.kind,
            "memory_type": _metadata_text(unit, "memory_type"),
            "explicit_source": candidate["explicit"],
            "topic_regions": [
                {"topic_id": topic.region_id, "name": topic.name} for topic in topics
            ],
        }
        if self._promotion_classifier is not None:
            raw = self._promotion_classifier(unit.content, context)
        elif self._promotion_model is not None:
            raw = self._promotion_model.classify_promotion(unit.content, context)
        else:
            raw = _heuristic_promotion_decision(unit.content, context)
        return _validated_promotion_decision(
            raw,
            content=unit.content,
            explicit_source=bool(candidate["explicit"]),
            topic_ids={topic.region_id for topic in topics},
        )

    def _promotion_target(
        self,
        candidate: _PromotionCandidate,
        decision: _PromotionDecision,
        topics: list[MemoryRegion],
    ) -> MemoryRegion | None:
        source_region = candidate["region"]
        category = decision["category"]
        target: MemoryRegion | None
        if category == "user":
            target = self._regions.ensure_user_region()
        elif category == "topic":
            topic_id = str(decision.get("topic_id") or "")
            target = next(
                (topic for topic in topics if topic.region_id == topic_id),
                None,
            )
            if target is None:
                return None
        else:
            return None
        if target.region_id == source_region.region_id:
            return None
        return target

    def _apply_promotion(
        self,
        bundle: _RuntimeBundle,
        job_id: str,
        candidates: list[_PromotionCandidate],
        target: MemoryRegion,
        decision: _PromotionDecision,
        evidence_kind: str,
    ) -> _PromotionOutcome:
        if not candidates:
            raise ValueError("promotion requires at least one candidate")
        ordered = sorted(
            candidates,
            key=lambda candidate: (candidate["region"].region_id, candidate["unit"].id),
        )
        candidate = ordered[0]
        unit = candidate["unit"]
        source_region = candidate["region"]
        promotion_key = _promotion_key(unit.content, target.region_id)
        target_scope = _scope_for_region(target)
        (
            source_units,
            source_ids,
            source_event_ids,
            source_references,
            source_regions,
            source_region_ids,
            source_unit_regions,
        ) = self._promotion_group_sources(ordered)
        promotion_memory_type = _metadata_text(unit, "memory_type")
        if promotion_memory_type not in FORMAL_MEMORY_TYPES:
            promotion_memory_type = next(
                (
                    memory_type
                    for source in source_units
                    if (memory_type := _metadata_text(source, "memory_type"))
                    in FORMAL_MEMORY_TYPES
                ),
                "unknown",
            )
        target_metadata: dict[str, object] = {
            "memory_type": promotion_memory_type,
            "promotion_marker": True,
            "promotion_source_type": _metadata_text(unit, "memory_type"),
            "promotion_key": promotion_key,
            "promotion_job_id": job_id,
            "promotion_classification": str(decision["category"]),
            "promotion_confidence": float(decision["confidence"]),
            "promotion_claim_subject": decision["claim_subject"],
            "promotion_claim_value": decision["claim_value"],
            "promotion_claim_content_hash": hashlib.sha256(
                unit.content.encode("utf-8")
            ).hexdigest(),
            "promotion_evidence_kind": evidence_kind,
            "promotion_evidence_count": len(source_ids),
            "source_unit_ids": source_ids,
            "source_event_ids": source_event_ids,
            "source_region_id": source_region.region_id,
            "source_region_kind": source_region.kind,
            "source_region_ids": source_region_ids,
            "source_unit_regions": json.dumps(
                source_unit_regions,
                ensure_ascii=False,
            ),
            "source_region_kinds": [
                region["region_kind"] for region in source_regions
            ],
            "source_references": source_references,
            "provenance": source_ids,
            "region_id": target.region_id,
            "region_kind": target.kind,
            "promoted_at": datetime.now(UTC).isoformat(),
        }
        outbox_payload: dict[str, object] = {
            "job_id": job_id,
            "target": {
                "content": unit.content,
                "memory_type": promotion_memory_type,
                "user_metadata": dict(target_metadata),
            },
            "promotion": {
                "source_unit_id": unit.id,
                "source_region_id": source_region.region_id,
                "source_region_kind": source_region.kind,
                "job_id": job_id,
                "classification": str(decision["category"]),
                "confidence": float(decision["confidence"]),
                "evidence_kind": evidence_kind,
                "source_unit_ids": source_ids,
                "source_event_ids": source_event_ids,
                "source_region_ids": source_region_ids,
                "source_unit_regions": source_unit_regions,
                "source_regions": source_regions,
                "source_references": source_references,
            },
        }
        with (
            self._maintenance_lock,
            self._promotion_lock,
            self._governance_region_lock(target.region_id),
            self._graph_write_lock,
        ):
            claim_target = self._promoted_claim_target(bundle, target_scope, decision)
            if claim_target is not None:
                claim_key = _metadata_text(claim_target, "promotion_key") or promotion_key
                promotion_key = claim_key
                target_metadata["promotion_key"] = promotion_key
                target_payload = outbox_payload.get("target")
                if isinstance(target_payload, dict):
                    outbox_metadata = target_payload.get("user_metadata")
                    if isinstance(outbox_metadata, dict):
                        outbox_metadata["promotion_key"] = promotion_key
                claim_record = self._promotion_record(promotion_key)
                if (
                    claim_record is not None
                    and str(claim_record["target_unit_id"]) != claim_target.id
                ):
                    self._repair_promotion_record_target(promotion_key, claim_target)
            conflict = self._promotion_existing_conflict(
                bundle,
                target_scope,
                decision,
                promotion_key=promotion_key,
            )
            if conflict is not None:
                conflict_unit, existing_value, new_value = conflict
                self._record_promotion_conflict(
                    job_id,
                    decision["claim_subject"],
                    target,
                    (existing_value, new_value),
                    [(item, decision, target) for item in ordered],
                )
                return {
                    "created": False,
                    "decision": {
                        "source_unit_id": unit.id,
                        "source_region_id": source_region.region_id,
                        "source_unit_ids": source_ids,
                        "source_region_ids": source_region_ids,
                        "classification": str(decision["category"]),
                        "confidence": decision["confidence"],
                        "evidence_kind": evidence_kind,
                        "status": "conflict",
                        "target_region_id": target.region_id,
                        "conflict_unit_id": conflict_unit.id,
                        "reason": "promotion conflicts with an existing target claim",
                    },
                }
            record = self._promotion_record(promotion_key)
            if record is not None:
                recorded_units = load_units(
                    bundle.kv_store,
                    target_scope,
                    [str(record["target_unit_id"])],
                )
                target_unit = recorded_units[0] if recorded_units else None
                if target_unit is not None and target_unit.lifecycle.value == "forgotten":
                    return {
                        "created": False,
                        "decision": {
                            "source_unit_id": unit.id,
                            "source_region_id": source_region.region_id,
                            "source_unit_ids": source_ids,
                            "source_region_ids": source_region_ids,
                            "classification": str(decision["category"]),
                            "confidence": decision["confidence"],
                            "evidence_kind": evidence_kind,
                            "status": "skipped",
                            "target_region_id": target.region_id,
                            "target_unit_id": target_unit.id,
                            "reason": "promotion target remains forgotten",
                        },
                    }
                if target_unit is None:
                    self._delete_promotion_record(promotion_key)
                    record = None
                if record is not None:
                    assert target_unit is not None
                    expected_hash = _metadata_text(
                        target_unit,
                        "promotion_claim_content_hash",
                    )
                    current_hash = hashlib.sha256(
                        target_unit.content.encode("utf-8")
                    ).hexdigest()
                    target_changed = (
                        _normalize_promotion_text(target_unit.content)
                        != _normalize_promotion_text(unit.content)
                        or (bool(expected_hash) and expected_hash != current_hash)
                    )
                    manual_override = (
                        target_unit.user_metadata.get("semantic_manual_override") is True
                    )
                    if manual_override:
                        existing_subject, existing_value = _promotion_claim_signature(
                            target_unit.content
                        )
                    else:
                        existing_subject = _metadata_text(
                            target_unit,
                            "promotion_claim_subject",
                        )
                        existing_value = _metadata_text(
                            target_unit,
                            "promotion_claim_value",
                        )
                    existing_subject = _normalize_promotion_text(existing_subject)
                    existing_value = _normalize_promotion_claim_value(existing_value)
                    if not existing_subject or not existing_value:
                        existing_subject, existing_value = _promotion_claim_signature(
                            target_unit.content
                        )
                        existing_subject = _normalize_promotion_text(existing_subject)
                        existing_value = _normalize_promotion_claim_value(existing_value)
                    claim_equivalent = (
                        bool(existing_subject)
                        and bool(existing_value)
                        and _promotion_subjects_compatible(
                            existing_subject,
                            _normalize_promotion_text(decision["claim_subject"]),
                        )
                        and _promotion_claim_values_equivalent(
                            existing_value,
                            decision["claim_value"],
                        )
                    )
                    if target_changed and not claim_equivalent:
                        if (
                            existing_subject
                            and existing_value
                            and _promotion_subjects_compatible(
                                existing_subject,
                                _normalize_promotion_text(decision["claim_subject"]),
                            )
                            and _promotion_values_conflict(
                                existing_subject,
                                (existing_value, decision["claim_value"]),
                            )
                        ):
                            self._record_promotion_conflict(
                                job_id,
                                decision["claim_subject"],
                                target,
                                (existing_value, decision["claim_value"]),
                                [(item, decision, target) for item in ordered],
                            )
                            return {
                                "created": False,
                                "decision": {
                                    "source_unit_id": unit.id,
                                    "source_region_id": source_region.region_id,
                                    "source_unit_ids": source_ids,
                                    "source_region_ids": source_region_ids,
                                    "classification": str(decision["category"]),
                                    "confidence": decision["confidence"],
                                    "evidence_kind": evidence_kind,
                                    "status": "conflict",
                                    "target_region_id": target.region_id,
                                    "conflict_unit_id": target_unit.id,
                                    "reason": "corrected target conflicts with late source",
                                },
                            }
                        self._record_promotion_skip(
                            job_id,
                            candidate,
                            decision,
                            action="promotion_skip",
                            reason="promotion target was corrected after promotion",
                        )
                        return {
                            "created": False,
                            "decision": {
                                "source_unit_id": unit.id,
                                "source_region_id": source_region.region_id,
                                "source_unit_ids": source_ids,
                                "source_region_ids": source_region_ids,
                                "classification": str(decision["category"]),
                                "confidence": decision["confidence"],
                                "evidence_kind": evidence_kind,
                                "status": "skipped",
                                "target_region_id": target.region_id,
                                "target_unit_id": target_unit.id,
                                "reason": "promotion target was corrected after promotion",
                            },
                        }
                    self._begin_promotion_outbox(
                        promotion_key,
                        target,
                        target_unit.id,
                        outbox_payload,
                    )
                    merged_unit_ids, merged_region_ids = self._merge_promotion_sources(
                        bundle,
                        target_scope,
                        target,
                        target_unit,
                        record,
                        source_ids,
                        source_event_ids,
                        source_references,
                        source_regions,
                        source_region_ids,
                        source_unit_regions,
                        evidence_kind,
                    )
                    self._complete_promotion_outbox(promotion_key)
                    return {
                        "created": False,
                        "decision": {
                            "source_unit_id": unit.id,
                            "source_region_id": source_region.region_id,
                            "source_unit_ids": merged_unit_ids,
                            "source_region_ids": merged_region_ids,
                            "classification": str(decision["category"]),
                            "confidence": decision["confidence"],
                            "evidence_kind": evidence_kind,
                            "status": "skipped",
                            "target_region_id": target.region_id,
                            "target_unit_id": target_unit.id,
                            "reason": "promotion already persisted; merged new sources",
                        },
                    }
            existing = self._promoted_target(bundle, target_scope, promotion_key)
            if existing is not None:
                pending_recovery = not self._promotion_audit_exists(promotion_key)
                self._begin_promotion_outbox(
                    promotion_key,
                    target,
                    existing.id,
                    outbox_payload,
                )
                self._index_browser_units(bundle, target, target_scope, [existing])
                self._persist_promotion(
                    existing,
                    target,
                    ordered,
                    source_units,
                    source_references,
                    source_unit_regions,
                    job_id,
                    promotion_key,
                    decision,
                    evidence_kind,
                )
                repaired_record = self._promotion_record(promotion_key)
                if repaired_record is None:
                    raise MemoryKernelUnavailableError(
                        "promotion record repair did not persist"
                    )
                merged_unit_ids, merged_region_ids = self._merge_promotion_sources(
                    bundle,
                    target_scope,
                    target,
                    existing,
                    repaired_record,
                    source_ids,
                    source_event_ids,
                    source_references,
                    source_regions,
                    source_region_ids,
                    source_unit_regions,
                    evidence_kind,
                )
                self._complete_promotion_outbox(promotion_key)
                if bundle.graph_store is not None:
                    self._associate_graph_scope(
                        bundle,
                        target_scope,
                        region_id=target.region_id,
                    )
                return {
                    "created": pending_recovery,
                    "decision": {
                        "source_unit_id": unit.id,
                        "source_region_id": source_region.region_id,
                        "source_unit_ids": merged_unit_ids,
                        "source_region_ids": merged_region_ids,
                        "classification": str(decision["category"]),
                        "confidence": decision["confidence"],
                        "evidence_kind": evidence_kind,
                        "status": "promoted" if pending_recovery else "skipped",
                        "target_region_id": target.region_id,
                        "target_unit_id": existing.id,
                        "reason": (
                            "pending promotion target was recovered and indexed"
                            if pending_recovery
                            else "existing promotion target was repaired without a duplicate"
                        ),
                    },
                }

            self._begin_promotion_outbox(
                promotion_key,
                target,
                "",
                outbox_payload,
            )
            units = bundle.runtime.api.add(
                unit.content,
                target_scope,
                Modality.TEXT,
                security=_security_for(target_scope),
                user_metadata=target_metadata,
            )
            if not units:
                raise MemoryKernelUnavailableError("promotion did not create a memory unit")
            added = units[0]
            added.provenance = source_ids
            bundle.kv_store.update(
                target_scope,
                memory_key(added.id),
                dumps(added),
            )
            self._set_promotion_outbox_target(promotion_key, added.id)
            self._index_browser_units(bundle, target, target_scope, [added])
            self._persist_promotion(
                added,
                target,
                ordered,
                source_units,
                source_references,
                source_unit_regions,
                job_id,
                promotion_key,
                decision,
                evidence_kind,
            )
            self._complete_promotion_outbox(promotion_key)
            if bundle.graph_store is not None:
                self._associate_graph_scope(
                    bundle,
                    target_scope,
                    region_id=target.region_id,
                )
            return {
                "created": True,
                "decision": {
                    "source_unit_id": unit.id,
                    "source_region_id": source_region.region_id,
                    "source_unit_ids": source_ids,
                    "source_region_ids": source_region_ids,
                    "classification": str(decision["category"]),
                    "confidence": decision["confidence"],
                    "evidence_kind": evidence_kind,
                    "status": "promoted",
                    "target_region_id": target.region_id,
                    "target_unit_id": added.id,
                    "reason": str(decision["reason"]),
                },
            }

    def _promotion_audit_exists(self, promotion_key: str) -> bool:
        with closing(self._maintenance_db()) as connection:
            return (
                connection.execute(
                    """SELECT 1 FROM governance_events
                       WHERE operation_id = ? AND action = 'promote'
                         AND status = 'committed'""",
                    (promotion_key,),
                ).fetchone()
                is not None
            )

    def _begin_promotion_outbox(
        self,
        promotion_key: str,
        target: MemoryRegion,
        target_unit_id: str,
        payload: Mapping[str, object],
    ) -> None:
        now = datetime.now(UTC).isoformat()
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with closing(self._maintenance_db()) as connection, connection:
            connection.execute(
                """INSERT INTO promotion_outbox
                   (promotion_key, target_region_id, target_unit_id, payload,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(promotion_key) DO UPDATE SET
                     target_region_id = excluded.target_region_id,
                     target_unit_id = excluded.target_unit_id,
                     payload = excluded.payload,
                     updated_at = excluded.updated_at""",
                (
                    promotion_key,
                    target.region_id,
                    target_unit_id,
                    serialized,
                    now,
                    now,
                ),
            )

    def _set_promotion_outbox_target(self, promotion_key: str, target_unit_id: str) -> None:
        with closing(self._maintenance_db()) as connection, connection:
            connection.execute(
                """UPDATE promotion_outbox SET target_unit_id = ?, updated_at = ?
                   WHERE promotion_key = ?""",
                (target_unit_id, datetime.now(UTC).isoformat(), promotion_key),
            )

    def _complete_promotion_outbox(self, promotion_key: str) -> None:
        with closing(self._maintenance_db()) as connection, connection:
            connection.execute(
                "DELETE FROM promotion_outbox WHERE promotion_key = ?",
                (promotion_key,),
            )

    def _recover_promotion_outbox(self, bundle: _RuntimeBundle) -> None:
        while True:
            with closing(self._maintenance_db()) as connection:
                rows = connection.execute(
                    """SELECT promotion_key, target_region_id, target_unit_id, payload
                       FROM promotion_outbox
                       ORDER BY promotion_key
                       LIMIT ?""",
                    (_PROMOTION_RECOVERY_BATCH_SIZE,),
                ).fetchall()
            if not rows:
                return
            for row in rows:
                self._recover_promotion_outbox_row(bundle, row)

    def _recover_promotion_outbox_row(
        self,
        bundle: _RuntimeBundle,
        row: tuple[object, ...],
    ) -> None:
        promotion_key = str(row[0])
        target_region_id = str(row[1])
        target_unit_id = str(row[2] or "")
        try:
            payload = json.loads(str(row[3]))
        except json.JSONDecodeError as error:
            raise MemoryKernelUnavailableError(
                f"promotion outbox payload is invalid: {promotion_key}"
            ) from error
        if not isinstance(payload, dict):
            raise MemoryKernelUnavailableError(
                f"promotion outbox payload is invalid: {promotion_key}"
            )
        target = self._regions.require_region(target_region_id)
        target_scope = _scope_for_region(target)
        target_unit: MemoryUnit | None = None
        if target_unit_id:
            units = load_units(bundle.kv_store, target_scope, [target_unit_id])
            if units and _metadata_text(units[0], "promotion_key") == promotion_key:
                target_unit = units[0]
        if target_unit is None:
            target_unit = self._recovery_promotion_target(
                bundle,
                target_scope,
                promotion_key,
            )
        if target_unit is not None and target_unit.lifecycle.value == "forgotten":
            promotion = payload.get("promotion")
            if not isinstance(promotion, dict):
                raise MemoryKernelUnavailableError(
                    f"promotion outbox record is invalid: {promotion_key}"
                )
            target_unit = self._merge_recovered_promotion_sources(
                bundle,
                target,
                target_scope,
                target_unit,
                promotion,
            )
            self._write_browser_index_rows(target.region_id, [], [target_unit.id])
            self._persist_recovered_promotion(
                target_unit,
                target,
                promotion_key,
                payload,
            )
            self._complete_promotion_outbox(promotion_key)
            return
        if target_unit is None:
            target_payload = payload.get("target")
            if not isinstance(target_payload, dict):
                raise MemoryKernelUnavailableError(
                    f"promotion outbox target is invalid: {promotion_key}"
                )
            content = str(target_payload.get("content") or "")
            metadata = target_payload.get("user_metadata")
            if not content or not isinstance(metadata, dict):
                raise MemoryKernelUnavailableError(
                    f"promotion outbox target is invalid: {promotion_key}"
                )
            units = bundle.runtime.api.add(
                content,
                target_scope,
                Modality.TEXT,
                security=_security_for(target_scope),
                user_metadata=dict(metadata),
            )
            if not units:
                raise MemoryKernelUnavailableError(
                    "promotion recovery did not create a memory unit"
                )
            target_unit = units[0]
            source_ids = _metadata_strings(target_unit, "source_unit_ids")
            if not source_ids:
                source_ids = _metadata_strings(target_unit, "provenance")
            target_unit.provenance = source_ids
            bundle.kv_store.update(
                target_scope,
                memory_key(target_unit.id),
                dumps(target_unit),
            )
            target_unit_id = target_unit.id
            self._set_promotion_outbox_target(promotion_key, target_unit_id)
        promotion = payload.get("promotion")
        if not isinstance(promotion, dict):
            raise MemoryKernelUnavailableError(
                f"promotion outbox record is invalid: {promotion_key}"
            )
        target_unit = self._merge_recovered_promotion_sources(
            bundle,
            target,
            target_scope,
            target_unit,
            promotion,
        )
        self._index_browser_units(bundle, target, target_scope, [target_unit])
        self._persist_recovered_promotion(
            target_unit,
            target,
            promotion_key,
            payload,
        )
        self._complete_promotion_outbox(promotion_key)

    def _recovery_promotion_target(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        promotion_key: str,
    ) -> MemoryUnit | None:
        matches = [
            unit
            for unit in self._units_in_scope(bundle, scope)
            if _metadata_text(unit, "promotion_key") == promotion_key
        ]
        return next(
            (unit for unit in matches if unit.lifecycle.value == "active"),
            None,
        ) or next(
            (unit for unit in matches if unit.lifecycle.value == "forgotten"),
            None,
        ) or (matches[0] if matches else None)

    def _merge_recovered_promotion_sources(
        self,
        bundle: _RuntimeBundle,
        target: MemoryRegion,
        target_scope: Scope,
        target_unit: MemoryUnit,
        promotion: Mapping[str, object],
    ) -> MemoryUnit:
        existing_ids = _metadata_strings(target_unit, "source_unit_ids")
        if not existing_ids:
            existing_ids = _metadata_strings(target_unit, "provenance")
        raw_ids = promotion.get("source_unit_ids")
        recovered_ids = (
            [str(value) for value in raw_ids if str(value)]
            if isinstance(raw_ids, list)
            else []
        )
        source_ids = list(dict.fromkeys([*existing_ids, *recovered_ids]))

        existing_region_ids = _metadata_strings(target_unit, "source_region_ids")
        single_region = _metadata_text(target_unit, "source_region_id")
        if single_region and single_region not in existing_region_ids:
            existing_region_ids.insert(0, single_region)
        raw_region_ids = promotion.get("source_region_ids")
        recovered_region_ids = (
            [str(value) for value in raw_region_ids if str(value)]
            if isinstance(raw_region_ids, list)
            else []
        )
        source_region_ids = list(
            dict.fromkeys([*existing_region_ids, *recovered_region_ids])
        )

        unit_region_map = _metadata_unit_region_map(target_unit)
        raw_unit_regions = promotion.get("source_unit_regions")
        if isinstance(raw_unit_regions, list):
            for item in raw_unit_regions:
                if not isinstance(item, dict):
                    continue
                unit_id = str(item.get("unit_id") or "").strip()
                region_id = str(item.get("region_id") or "").strip()
                if unit_id and region_id:
                    unit_region_map[unit_id] = region_id

        kind_by_region = dict(
            zip(
                _metadata_strings(target_unit, "source_region_ids"),
                _metadata_strings(target_unit, "source_region_kinds"),
                strict=False,
            )
        )
        raw_regions = promotion.get("source_regions")
        if isinstance(raw_regions, list):
            for item in raw_regions:
                if not isinstance(item, dict):
                    continue
                region_id = str(item.get("region_id") or "").strip()
                region_kind = str(item.get("region_kind") or "").strip()
                if region_id and region_kind:
                    kind_by_region[region_id] = region_kind
        regions_by_id = {region.region_id: region for region in self._regions.list_regions()}
        for region_id in source_region_ids:
            kind_by_region.setdefault(
                region_id,
                regions_by_id[region_id].kind if region_id in regions_by_id else target.kind,
            )

        raw_event_ids = promotion.get("source_event_ids")
        recovered_event_ids = (
            [str(value) for value in raw_event_ids if str(value)]
            if isinstance(raw_event_ids, list)
            else []
        )
        source_event_ids = list(
            dict.fromkeys(
                [
                    *_metadata_strings(target_unit, "source_event_ids"),
                    *recovered_event_ids,
                ]
            )
        )
        raw_references = promotion.get("source_references")
        recovered_references = (
            [str(value) for value in raw_references if str(value)]
            if isinstance(raw_references, list)
            else []
        )
        source_references = list(
            dict.fromkeys(
                [
                    *_metadata_strings(target_unit, "source_references"),
                    *recovered_references,
                ]
            )
        )
        recovered_evidence_kind = str(promotion.get("evidence_kind") or "")
        evidence_kind = (
            "multi_source"
            if (
                recovered_evidence_kind == "multi_source"
                or _metadata_text(target_unit, "promotion_evidence_kind") == "multi_source"
                or len(source_ids) > 1
            )
            else "explicit"
        )
        target_unit.user_metadata.update(
            {
                "source_unit_ids": source_ids,
                "source_event_ids": source_event_ids,
                "source_region_ids": source_region_ids,
                "source_region_kinds": [
                    kind_by_region[region_id] for region_id in source_region_ids
                ],
                "source_unit_regions": json.dumps(
                    [
                        {"unit_id": unit_id, "region_id": region_id}
                        for unit_id, region_id in unit_region_map.items()
                    ],
                    ensure_ascii=False,
                ),
                "source_references": source_references,
                "provenance": source_ids,
                "promotion_evidence_kind": evidence_kind,
                "promotion_evidence_count": len(source_ids),
            }
        )
        target_unit.provenance = source_ids
        bundle.kv_store.update(
            target_scope,
            memory_key(target_unit.id),
            dumps(target_unit),
        )
        return target_unit

    def _persist_recovered_promotion(
        self,
        target_unit: MemoryUnit,
        target: MemoryRegion,
        promotion_key: str,
        payload: Mapping[str, object],
    ) -> int:
        promotion = payload.get("promotion")
        if not isinstance(promotion, dict):
            raise MemoryKernelUnavailableError(
                f"promotion outbox record is invalid: {promotion_key}"
            )
        source_ids = _metadata_strings(target_unit, "source_unit_ids")
        if not source_ids:
            source_ids = _metadata_strings(target_unit, "provenance")
        if not source_ids:
            raw_source_ids = promotion.get("source_unit_ids")
            if isinstance(raw_source_ids, list):
                source_ids = [str(value) for value in raw_source_ids if str(value)]
        source_region_ids = _metadata_strings(target_unit, "source_region_ids")
        if not source_region_ids:
            raw_region_ids = promotion.get("source_region_ids")
            if isinstance(raw_region_ids, list):
                source_region_ids = [str(value) for value in raw_region_ids if str(value)]
        source_unit_regions = [
            {"unit_id": unit_id, "region_id": region_id}
            for unit_id, region_id in _metadata_unit_region_map(target_unit).items()
        ]
        if not source_unit_regions:
            raw_unit_regions = promotion.get("source_unit_regions")
            if isinstance(raw_unit_regions, list):
                source_unit_regions = [
                    {
                        "unit_id": str(item["unit_id"]),
                        "region_id": str(item["region_id"]),
                    }
                    for item in raw_unit_regions
                    if isinstance(item, dict)
                    and item.get("unit_id")
                    and item.get("region_id")
                ]
        source_event_ids = _metadata_strings(target_unit, "source_event_ids")
        if not source_event_ids:
            raw_event_ids = promotion.get("source_event_ids")
            if isinstance(raw_event_ids, list):
                source_event_ids = [str(value) for value in raw_event_ids if str(value)]
        source_references = _metadata_strings(target_unit, "source_references")
        if not source_references:
            raw_references = promotion.get("source_references")
            if isinstance(raw_references, list):
                source_references = [str(value) for value in raw_references if str(value)]
        source_region_kinds = _metadata_strings(target_unit, "source_region_kinds")
        kind_by_region = dict(zip(source_region_ids, source_region_kinds, strict=False))
        regions_by_id = {region.region_id: region for region in self._regions.list_regions()}
        source_regions = [
            {
                "region_id": region_id,
                "region_kind": (
                    kind_by_region.get(region_id)
                    or (
                        regions_by_id[region_id].kind
                        if region_id in regions_by_id
                        else target.kind
                    )
                ),
            }
            for region_id in source_region_ids
        ]
        source_unit_id = str(promotion.get("source_unit_id") or "")
        if not source_unit_id:
            source_unit_id = source_ids[0] if source_ids else target_unit.id
        source_region_id = str(promotion.get("source_region_id") or "")
        if not source_region_id:
            source_region_id = source_region_ids[0] if source_region_ids else target.region_id
        source_region_kind = next(
            (
                item["region_kind"]
                for item in source_regions
                if item["region_id"] == source_region_id
            ),
            target.kind,
        )
        evidence_kind = _metadata_text(target_unit, "promotion_evidence_kind")
        if not evidence_kind:
            evidence_kind = str(promotion.get("evidence_kind") or "explicit")
        confidence = promotion.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            confidence = 0.0
        job_id = str(promotion.get("job_id") or "")
        if not job_id:
            job_id = _metadata_text(target_unit, "promotion_job_id")
        return self._persist_promotion_values(
            target_unit=target_unit,
            target=target,
            source_unit_id=source_unit_id,
            source_region_id=source_region_id,
            source_region_kind=source_region_kind,
            job_id=job_id,
            promotion_key=promotion_key,
            classification=str(promotion.get("classification") or "user"),
            confidence=float(confidence),
            evidence_kind=evidence_kind,
            source_ids=source_ids,
            source_event_ids=source_event_ids,
            source_region_ids=source_region_ids,
            source_unit_regions=source_unit_regions,
            source_regions=source_regions,
            source_references=source_references,
        )

    def _delete_promotion_record(self, promotion_key: str) -> None:
        with closing(self._maintenance_db()) as connection, connection:
            connection.execute(
                "DELETE FROM promotion_records WHERE promotion_key = ?",
                (promotion_key,),
            )

    def _promotion_record(self, promotion_key: str) -> dict[str, object] | None:
        with closing(self._maintenance_db()) as connection:
            row = connection.execute(
                """SELECT target_unit_id, source_unit_ids, source_region_ids,
                          audit_event_id, evidence_kind, source_unit_regions
                   FROM promotion_records WHERE promotion_key = ?""",
                (promotion_key,),
            ).fetchone()
        if row is None:
            return None
        try:
            source_unit_ids = json.loads(str(row[1]))
        except json.JSONDecodeError:
            source_unit_ids = []
        try:
            source_region_ids = json.loads(str(row[2]))
        except json.JSONDecodeError:
            source_region_ids = []
        try:
            source_unit_regions = json.loads(str(row[5]))
        except json.JSONDecodeError:
            source_unit_regions = []
        return {
            "target_unit_id": str(row[0]),
            "source_unit_ids": source_unit_ids if isinstance(source_unit_ids, list) else [],
            "source_region_ids": (
                source_region_ids if isinstance(source_region_ids, list) else []
            ),
            "audit_event_id": int(row[3]) if row[3] is not None else None,
            "evidence_kind": str(row[4]),
            "source_unit_regions": (
                source_unit_regions if isinstance(source_unit_regions, list) else []
            ),
        }

    def _repair_promotion_record_target(
        self,
        promotion_key: str,
        target_unit: MemoryUnit,
    ) -> None:
        with closing(self._maintenance_db()) as connection, connection:
            row = connection.execute(
                """SELECT audit_event_id FROM promotion_records
                   WHERE promotion_key = ?""",
                (promotion_key,),
            ).fetchone()
            if row is None:
                return
            connection.execute(
                """UPDATE promotion_records SET target_unit_id = ?
                   WHERE promotion_key = ?""",
                (target_unit.id, promotion_key),
            )
            if row[0] is not None:
                connection.execute(
                    """UPDATE governance_events
                       SET unit_id = ?, logical_memory_id = ?, after_content = ?,
                           status = 'committed'
                       WHERE id = ? AND action = 'promote'""",
                    (
                        target_unit.id,
                        _unit_logical_memory_id(target_unit),
                        target_unit.content,
                        int(row[0]),
                    ),
                )

    def _promotion_existing_conflict(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        decision: _PromotionDecision,
        *,
        promotion_key: str,
    ) -> tuple[MemoryUnit, str, str] | None:
        subject = _normalize_promotion_text(decision["claim_subject"])
        value = _normalize_promotion_claim_value(decision["claim_value"])
        if not subject or not value:
            return None
        for unit in self._units_in_scope(bundle, scope):
            if unit.lifecycle.value != "active":
                continue
            if _metadata_text(unit, "promotion_key") == promotion_key:
                continue
            content_hash = hashlib.sha256(unit.content.encode("utf-8")).hexdigest()
            if _metadata_text(unit, "promotion_claim_content_hash") == content_hash:
                existing_subject = _normalize_promotion_text(
                    _metadata_text(unit, "promotion_claim_subject")
                )
                existing_value = _normalize_promotion_claim_value(
                    _metadata_text(unit, "promotion_claim_value")
                )
            else:
                existing_subject = ""
                existing_value = ""
            if not existing_subject or not existing_value:
                existing_subject, existing_value = _promotion_claim_signature(unit.content)
                existing_subject = _normalize_promotion_text(existing_subject)
                existing_value = _normalize_promotion_claim_value(existing_value)
            if not existing_subject or not existing_value:
                continue
            if (
                not _promotion_subjects_compatible(existing_subject, subject)
                or existing_value == value
            ):
                continue
            if _promotion_values_conflict(subject, (existing_value, value)):
                return unit, existing_value, value
        return None

    def _merge_promotion_sources(
        self,
        bundle: _RuntimeBundle,
        target_scope: Scope,
        target: MemoryRegion,
        target_unit: MemoryUnit,
        record: Mapping[str, object],
        source_ids: list[str],
        source_event_ids: list[str],
        source_references: list[str],
        source_regions: list[dict[str, str]],
        source_region_ids: list[str],
        source_unit_regions: list[dict[str, str]],
        evidence_kind: str,
    ) -> tuple[list[str], list[str]]:
        existing_unit_ids = _metadata_strings(target_unit, "source_unit_ids")
        if not existing_unit_ids:
            existing_unit_ids = _metadata_strings(target_unit, "provenance")
        existing_region_ids = _metadata_strings(target_unit, "source_region_ids")
        existing_region_kinds = _metadata_strings(target_unit, "source_region_kinds")
        if not existing_region_ids:
            existing_region = _metadata_text(target_unit, "source_region_id")
            existing_kind = _metadata_text(target_unit, "source_region_kind")
            existing_region_ids = [existing_region] if existing_region else []
            existing_region_kinds = [existing_kind] if existing_kind else []
        if not existing_unit_ids:
            record_units = record.get("source_unit_ids")
            if isinstance(record_units, list):
                existing_unit_ids = [str(value) for value in record_units]
        if not existing_region_ids:
            record_regions = record.get("source_region_ids")
            if isinstance(record_regions, list):
                existing_region_ids = [str(value) for value in record_regions]
        existing_unit_regions = _metadata_unit_region_map(target_unit)
        if not existing_unit_regions:
            record_unit_regions = record.get("source_unit_regions")
            if isinstance(record_unit_regions, list):
                existing_unit_regions = {
                    str(item["unit_id"]): str(item["region_id"])
                    for item in record_unit_regions
                    if isinstance(item, dict)
                    and item.get("unit_id")
                    and item.get("region_id")
                }

        merged_unit_ids = list(dict.fromkeys([*existing_unit_ids, *source_ids]))
        merged_unit_regions = dict(existing_unit_regions)
        for item in source_unit_regions:
            unit_id = str(item.get("unit_id") or "").strip()
            region_id = str(item.get("region_id") or "").strip()
            if unit_id and region_id:
                merged_unit_regions[unit_id] = region_id
        merged_region_ids = list(dict.fromkeys([*existing_region_ids, *source_region_ids]))
        kind_by_region = dict(zip(existing_region_ids, existing_region_kinds, strict=False))
        for region in source_regions:
            kind_by_region[str(region["region_id"])] = str(region["region_kind"])
        merged_region_kinds = [
            kind_by_region[region_id]
            for region_id in merged_region_ids
            if region_id in kind_by_region
        ]
        merged_source_regions = [
            {"region_id": region_id, "region_kind": kind_by_region[region_id]}
            for region_id in merged_region_ids
            if region_id in kind_by_region
        ]
        merged_event_ids = list(
            dict.fromkeys(
                [
                    *_metadata_strings(target_unit, "source_event_ids"),
                    *source_event_ids,
                ]
            )
        )
        merged_references = list(
            dict.fromkeys(
                [
                    *_metadata_strings(target_unit, "source_references"),
                    *source_references,
                ]
            )
        )
        merged_evidence_kind = (
            "multi_source"
            if (
                evidence_kind == "multi_source"
                or str(record.get("evidence_kind") or "") == "multi_source"
                or len(merged_unit_ids) > 1
            )
            else "explicit"
        )
        metadata_updates: dict[str, object] = {
            "source_unit_ids": merged_unit_ids,
            "source_event_ids": merged_event_ids,
            "source_region_ids": merged_region_ids,
            "source_region_kinds": merged_region_kinds,
            "source_unit_regions": json.dumps(
                [
                    {"unit_id": unit_id, "region_id": region_id}
                    for unit_id, region_id in merged_unit_regions.items()
                ],
                ensure_ascii=False,
            ),
            "source_references": merged_references,
            "provenance": merged_unit_ids,
            "promotion_evidence_kind": merged_evidence_kind,
            "promotion_evidence_count": len(merged_unit_ids),
        }
        target_unit.user_metadata.update(metadata_updates)
        target_unit.provenance = merged_unit_ids
        bundle.kv_store.update(
            target_scope,
            memory_key(target_unit.id),
            dumps(target_unit),
        )
        self._index_browser_units(bundle, target, target_scope, [target_unit])
        audit_id = record.get("audit_event_id")
        with closing(self._maintenance_db()) as connection, connection:
            connection.execute(
                """UPDATE promotion_records
                   SET source_unit_ids = ?, source_region_ids = ?,
                       source_unit_regions = ?, evidence_kind = ?
                   WHERE promotion_key = ?""",
                (
                    json.dumps(merged_unit_ids, ensure_ascii=False),
                    json.dumps(merged_region_ids, ensure_ascii=False),
                    json.dumps(
                        [
                            {"unit_id": unit_id, "region_id": region_id}
                            for unit_id, region_id in merged_unit_regions.items()
                        ],
                        ensure_ascii=False,
                    ),
                    merged_evidence_kind,
                    _metadata_text(target_unit, "promotion_key"),
                ),
            )
            if isinstance(audit_id, int):
                row = connection.execute(
                    "SELECT detail FROM governance_events WHERE id = ?",
                    (audit_id,),
                ).fetchone()
                detail: dict[str, object] = {}
                if row is not None:
                    try:
                        parsed = json.loads(str(row[0]))
                    except json.JSONDecodeError:
                        parsed = {}
                    if isinstance(parsed, dict):
                        detail = parsed
                detail.update(
                    {
                        "source_unit_ids": merged_unit_ids,
                        "source_region_ids": merged_region_ids,
                        "source_unit_regions": [
                            {"unit_id": unit_id, "region_id": region_id}
                            for unit_id, region_id in merged_unit_regions.items()
                        ],
                        "source_regions": merged_source_regions,
                        "source_references": merged_references,
                        "evidence_kind": merged_evidence_kind,
                    }
                )
                connection.execute(
                    "UPDATE governance_events SET detail = ? WHERE id = ?",
                    (
                        json.dumps(detail, ensure_ascii=False, sort_keys=True),
                        audit_id,
                    ),
                )
        return merged_unit_ids, merged_region_ids

    def _promoted_target(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        promotion_key: str,
    ) -> MemoryUnit | None:
        return next(
            (
                unit
                for unit in self._units_in_scope(bundle, scope)
                if unit.lifecycle.value == "active"
                and _metadata_text(unit, "promotion_key") == promotion_key
            ),
            None,
        )

    def _promoted_claim_target(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        decision: _PromotionDecision,
    ) -> MemoryUnit | None:
        subject = _normalize_promotion_text(decision["claim_subject"])
        value = _normalize_promotion_claim_value(decision["claim_value"])
        if not subject or not value:
            return None
        for unit in sorted(
            self._units_in_scope(bundle, scope),
            key=lambda item: item.id,
        ):
            if (
                unit.lifecycle.value != "active"
                or unit.user_metadata.get("promotion_marker") is not True
                or not _metadata_text(unit, "promotion_key")
            ):
                continue
            if unit.user_metadata.get("semantic_manual_override") is True:
                existing_subject, existing_value = _promotion_claim_signature(unit.content)
            else:
                existing_subject = _metadata_text(unit, "promotion_claim_subject")
                existing_value = _metadata_text(unit, "promotion_claim_value")
            if not existing_subject or not existing_value:
                existing_subject, existing_value = _promotion_claim_signature(unit.content)
            existing_subject = _normalize_promotion_text(existing_subject)
            existing_value = _normalize_promotion_claim_value(existing_value)
            if (
                _promotion_subjects_compatible(existing_subject, subject)
                and _promotion_claim_values_equivalent(existing_value, value)
            ):
                return unit
        return None

    def _persist_promotion(
        self,
        target_unit: MemoryUnit,
        target: MemoryRegion,
        candidates: list[_PromotionCandidate],
        source_units: tuple[MemoryUnit, ...],
        source_references: list[str],
        source_unit_regions: list[dict[str, str]],
        job_id: str,
        promotion_key: str,
        decision: _PromotionDecision,
        evidence_kind: str,
    ) -> int:
        ordered = sorted(
            candidates,
            key=lambda candidate: (candidate["region"].region_id, candidate["unit"].id),
        )
        source_unit = ordered[0]["unit"]
        source_region = ordered[0]["region"]
        source_ids = [str(unit.id) for unit in source_units] or [str(source_unit.id)]
        source_event_ids = list(
            dict.fromkeys(
                event_id
                for unit in source_units
                if (event_id := _metadata_text(unit, "event_id"))
            )
        )
        source_region_ids = list(
            dict.fromkeys(candidate["region"].region_id for candidate in ordered)
        )
        source_regions = [
            {
                "region_id": region_id,
                "region_kind": next(
                    candidate["region"].kind
                    for candidate in ordered
                    if candidate["region"].region_id == region_id
                ),
            }
            for region_id in source_region_ids
        ]
        return self._persist_promotion_values(
            target_unit=target_unit,
            target=target,
            source_unit_id=source_unit.id,
            source_region_id=source_region.region_id,
            source_region_kind=source_region.kind,
            job_id=job_id,
            promotion_key=promotion_key,
            classification=str(decision["category"]),
            confidence=float(decision["confidence"]),
            evidence_kind=evidence_kind,
            source_ids=source_ids,
            source_event_ids=source_event_ids,
            source_region_ids=source_region_ids,
            source_unit_regions=source_unit_regions,
            source_regions=source_regions,
            source_references=source_references,
        )

    def _persist_promotion_values(
        self,
        *,
        target_unit: MemoryUnit,
        target: MemoryRegion,
        source_unit_id: str,
        source_region_id: str,
        source_region_kind: str,
        job_id: str,
        promotion_key: str,
        classification: str,
        confidence: float,
        evidence_kind: str,
        source_ids: list[str],
        source_event_ids: list[str],
        source_region_ids: list[str],
        source_unit_regions: list[dict[str, str]],
        source_regions: list[dict[str, str]],
        source_references: list[str],
    ) -> int:
        detail = {
            "source_unit_id": source_unit_id,
            "source_unit_ids": source_ids,
            "source_event_ids": source_event_ids,
            "source_region_id": source_region_id,
            "source_region_kind": source_region_kind,
            "source_region_ids": source_region_ids,
            "source_unit_regions": source_unit_regions,
            "source_regions": source_regions,
            "source_references": source_references,
            "classification": classification,
            "confidence": confidence,
            "evidence_kind": evidence_kind,
            "job_id": job_id,
        }
        with closing(self._maintenance_db()) as connection, connection:
            row = connection.execute(
                """SELECT id FROM governance_events
                   WHERE operation_id = ? AND action = 'promote'
                   ORDER BY id LIMIT 1""",
                (promotion_key,),
            ).fetchone()
            serialized_detail = json.dumps(detail, ensure_ascii=False, sort_keys=True)
            if row is None:
                cursor = connection.execute(
                    """INSERT INTO governance_events
                       (region_id, unit_id, logical_memory_id, action, actor, source, operation_id,
                        before_content, after_content, detail, status, created_at)
                       VALUES (?, ?, ?, 'promote', 'platform:auto-promotion', ?, ?,
                               '', ?, ?, 'committed', ?)""",
                    (
                        target.region_id,
                        target_unit.id,
                        _unit_logical_memory_id(target_unit),
                        f"promotion:{job_id}",
                        promotion_key,
                        target_unit.content,
                        serialized_detail,
                        datetime.now(UTC).isoformat(),
                    ),
                )
                if cursor.lastrowid is None:
                    raise MemoryKernelUnavailableError(
                        "promotion audit event was not persisted"
                    )
                audit_id = int(cursor.lastrowid)
            else:
                audit_id = int(row[0])
                connection.execute(
                    """UPDATE governance_events
                       SET region_id = ?, unit_id = ?, logical_memory_id = ?,
                           actor = 'platform:auto-promotion',
                           source = ?, before_content = '', after_content = ?,
                           detail = ?, status = 'committed'
                       WHERE id = ?""",
                    (
                        target.region_id,
                        target_unit.id,
                        _unit_logical_memory_id(target_unit),
                        f"promotion:{job_id}",
                        target_unit.content,
                        serialized_detail,
                        audit_id,
                    ),
                )
            connection.execute(
                """INSERT INTO promotion_records
                   (promotion_key, job_id, source_unit_id, source_region_id,
                    source_region_kind, target_region_id, target_region_kind,
                    target_unit_id, classification, confidence, audit_event_id,
                    evidence_kind, source_unit_ids, source_region_ids,
                    source_unit_regions, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(promotion_key) DO UPDATE SET
                     job_id = excluded.job_id,
                     source_unit_id = excluded.source_unit_id,
                     source_region_id = excluded.source_region_id,
                     source_region_kind = excluded.source_region_kind,
                     target_region_id = excluded.target_region_id,
                     target_region_kind = excluded.target_region_kind,
                     target_unit_id = excluded.target_unit_id,
                     classification = excluded.classification,
                     confidence = excluded.confidence,
                     audit_event_id = excluded.audit_event_id,
                     evidence_kind = excluded.evidence_kind,
                     source_unit_ids = excluded.source_unit_ids,
                     source_region_ids = excluded.source_region_ids,
                     source_unit_regions = excluded.source_unit_regions""",
                (
                    promotion_key,
                    job_id,
                    source_unit_id,
                    source_region_id,
                    source_region_kind,
                    target.region_id,
                    target.kind,
                    target_unit.id,
                    classification,
                    confidence,
                    audit_id,
                    evidence_kind,
                    json.dumps(source_ids, ensure_ascii=False),
                    json.dumps(source_region_ids, ensure_ascii=False),
                    json.dumps(source_unit_regions, ensure_ascii=False),
                    datetime.now(UTC).isoformat(),
                ),
            )
        return audit_id

    def get_retention_policy(self, request: RetentionPolicyRequest) -> dict[str, object]:
        region = self._session_region(request.library_id)
        days = self._retention_days(region.region_id)
        return self._retention_payload(region, days)

    async def update_retention_policy(
        self, request: UpdateRetentionPolicyRequest
    ) -> dict[str, object]:
        return await asyncio.to_thread(self._update_retention_policy, request)

    def _update_retention_policy(self, request: UpdateRetentionPolicyRequest) -> dict[str, object]:
        region = self._session_region(request.library_id)
        if request.inbox_days < 0:
            raise ValueError("retention days cannot be negative")
        with closing(self._maintenance_db()) as connection, connection:
            connection.execute(
                """INSERT INTO session_retention_policy (region_id, retention_days)
                   VALUES (?, ?)
                   ON CONFLICT(region_id) DO UPDATE SET
                     retention_days = excluded.retention_days,
                     updated_at = CURRENT_TIMESTAMP""",
                (region.region_id, request.inbox_days),
            )
        return self._retention_payload(region, request.inbox_days)

    async def preview_retention_cleanup(
        self, request: RetentionCleanupRequest
    ) -> dict[str, object]:
        return await asyncio.to_thread(self._preview_retention_cleanup, request)

    def _preview_retention_cleanup(self, request: RetentionCleanupRequest) -> dict[str, object]:
        region = self._session_region(request.library_id)
        plan = self._retention_plan(region)
        return {
            "library_id": region.region_id,
            "region_id": region.region_id,
            "as_of": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "policy": self._retention_payload(region, self._retention_days(region.region_id)),
            "counts": {
                "captures_to_archive": len(plan),
                "captures_archived": 0,
                "evolved_units_protected": self._evolved_count(region),
            },
        }

    async def run_retention_cleanup(self, request: RetentionCleanupRequest) -> dict[str, object]:
        return await asyncio.to_thread(self._run_retention_cleanup, request)

    def _run_retention_cleanup(self, request: RetentionCleanupRequest) -> dict[str, object]:
        region = self._session_region(request.library_id)
        with (
            self._maintenance_lock,
            self._governance_region_lock(region.region_id),
            self._graph_write_lock,
        ):
            return self._run_retention_cleanup_locked(request)

    def _run_retention_cleanup_locked(self, request: RetentionCleanupRequest) -> dict[str, object]:
        bundle = self._require_bundle()
        region = self._session_region(request.library_id)
        plan = self._retention_plan(region)
        self._record_archived_incomplete_turns(bundle, plan)
        for scope, units in _group_units_by_scope(plan):
            bundle.runtime.api.delete(
                DeleteSelector(
                    unit_ids=[unit.id for unit in units],
                    scope=scope,
                    mode=DeleteMode.ARCHIVE,
                ),
                security=_security_for(scope),
            )
            self._remove_graph_nodes(
                bundle,
                scope,
                [unit.id for unit in units],
                region.region_id,
            )
        return {
            "library_id": region.region_id,
            "region_id": region.region_id,
            "as_of": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "policy": self._retention_payload(region, self._retention_days(region.region_id)),
            "counts": {
                "captures_to_archive": len(plan),
                "captures_archived": len(plan),
                "evolved_units_protected": self._evolved_count(region),
            },
        }

    def list_capture_events(self, request: CaptureEventsRequest) -> list[dict[str, object]]:
        bundle = self._require_bundle()
        session_id = (
            _normalize_session_id(request.session_id)
            if request.session_id is not None
            else None
        )
        turn_id = _normalize_optional_identifier(request.turn_id, "turn_id")
        region: MemoryRegion | None = None
        if request.region_id is not None:
            region_id = _normalize_optional_identifier(request.region_id, "region_id")
            assert region_id is not None
            region = self._regions.require_region(region_id)
        events: list[dict[str, object]] = []
        for scope in bundle.kv_store.scopes():
            if region is not None and scope.space != region.space:
                continue
            if (
                region is not None
                and region.kind == "session"
                and scope.session != (region.session_id or "")
            ):
                continue
            for unit in self._units_in_scope(bundle, scope):
                if unit.lifecycle.value != "active":
                    continue
                capture_record = _is_capture_record(unit)
                evolved_diagnostic = _is_evolved_unit(unit)
                if not capture_record and not evolved_diagnostic:
                    continue
                event_region_id = _metadata_text(unit, "region_id") or (
                    region.region_id if region is not None else ""
                )
                if region is not None and event_region_id != region.region_id:
                    continue
                event_session_id = _metadata_text(unit, "session_id") or scope.session
                event_turn_id = _metadata_text(unit, "turn_id")
                if session_id is not None and event_session_id != session_id:
                    continue
                if turn_id is not None and event_turn_id != turn_id:
                    continue
                received = unit.temporal.t_ingest
                payload: dict[str, object] = {
                    "record_type": (
                        "capture_event" if capture_record else "evolved_diagnostic"
                    ),
                    "diagnostic_kind": "" if capture_record else "legacy_evolved",
                    "legacy": (
                        not _is_capture_unit(unit) if capture_record else True
                    ),
                    "event_id": _metadata_text(unit, "event_id"),
                    "session_id": event_session_id,
                    "project_id": _metadata_text(unit, "project_id"),
                    "library_id": event_region_id,
                    "region_id": event_region_id,
                    "space": scope.space,
                    "agent": scope.agent,
                    "turn_id": event_turn_id,
                    "event_kind": _metadata_text(unit, "event_kind"),
                    "content": unit.content,
                    "unit_id": unit.id,
                    "occurred_at": _metadata_text(unit, "occurred_at"),
                    "received_at": (received.isoformat() if received is not None else ""),
                    "consolidated_at": None,
                }
                events.append(payload)
        events.sort(key=_capture_event_sort_key)
        return events

    def list_capture_rounds(self, request: CaptureRoundsRequest) -> list[dict[str, object]]:
        bundle = self._require_bundle()
        events = self.list_capture_events(
            CaptureEventsRequest(
                session_id=request.session_id,
                region_id=request.region_id,
                turn_id=request.turn_id,
            )
        )
        session_filter = (
            _normalize_session_id(request.session_id)
            if request.session_id is not None
            else None
        )
        turn_filter = _normalize_optional_identifier(request.turn_id, "turn_id")
        region_filter: MemoryRegion | None = None
        if request.region_id is not None:
            region_id = _normalize_optional_identifier(request.region_id, "region_id")
            assert region_id is not None
            region_filter = self._regions.require_region(region_id)
        evolution_records = self._capture_evolution_records()
        record_source_ids = self._evolution_record_source_ids(
            bundle,
            evolution_records,
        )
        rounds: dict[tuple[str, str, str], dict[str, object]] = {}
        for event in events:
            key = (
                str(event["session_id"]),
                str(event["turn_id"]),
                str(event["region_id"]),
            )
            current = rounds.get(key)
            if current is None:
                rounds[key] = {
                    "session_id": event["session_id"],
                    "turn_id": event["turn_id"],
                    "library_id": event["region_id"],
                    "region_id": event["region_id"],
                    "space": event["space"],
                    "status": "pending",
                    "distillation_status": "pending_completion",
                    "semantic_status": "not_requested",
                    "semantic_memory_ids": [],
                    "semantic_memory_refs": [],
                    "semantic_diagnostics": [],
                    "semantic_error": "",
                    "semantic_updated_at": None,
                    "candidate_id": None,
                    "memory_id": None,
                    "last_error": "",
                    "failure_reason": "",
                    "started_at": (
                        _normalize_timeline_timestamp(event.get("occurred_at"))
                        or _normalize_timeline_timestamp(event.get("received_at"))
                    ),
                    "updated_at": _normalize_timeline_timestamp(
                        event.get("received_at")
                    ),
                    "received_at": _normalize_timeline_timestamp(
                        event.get("received_at")
                    ),
                    "completed_at": None,
                    "created_at": None,
                    "events": [event],
                }
            else:
                received_at = _normalize_timeline_timestamp(event.get("received_at"))
                current["updated_at"] = max(str(current["updated_at"]), received_at)
                current["received_at"] = current["updated_at"]
                cast(list[dict[str, object]], current["events"]).append(event)
        for record_key, evolution_record in evolution_records.items():
            record_session, record_turn, record_region = record_key
            if session_filter is not None and record_session != session_filter:
                continue
            if turn_filter is not None and record_turn != turn_filter:
                continue
            if region_filter is not None and record_region != region_filter.region_id:
                continue
            if record_key in rounds:
                continue
            try:
                record_memory_region = self._regions.require_region(record_region)
            except RegionError:
                continue
            created_at = _normalize_timeline_timestamp(
                evolution_record.get("created_at")
            )
            completed_at = _normalize_timeline_timestamp(
                evolution_record.get("completed_at")
            )
            started_at, received_at, updated_at = self._evolution_record_timeline(
                evolution_record
            )
            rounds[record_key] = {
                "session_id": record_session,
                "turn_id": record_turn,
                "library_id": record_region,
                "region_id": record_region,
                "space": record_memory_region.space,
                "status": "pending",
                "distillation_status": "pending_completion",
                "semantic_status": evolution_record["semantic_status"],
                "semantic_memory_ids": evolution_record["semantic_memory_ids"],
                "semantic_memory_refs": evolution_record["semantic_memory_refs"],
                "semantic_diagnostics": evolution_record["semantic_diagnostics"],
                "semantic_error": evolution_record["semantic_error"],
                "semantic_updated_at": evolution_record["semantic_updated_at"],
                "candidate_id": None,
                "memory_id": evolution_record["memory_id"],
                "last_error": evolution_record["last_error"],
                "failure_reason": evolution_record["last_error"],
                "started_at": started_at,
                "updated_at": updated_at,
                "received_at": received_at,
                "completed_at": completed_at or None,
                "created_at": created_at or completed_at,
                "events": [],
                "diagnostics": [],
                "user_message": None,
                "assistant_final": None,
            }
        for key, round_payload in rounds.items():
            round_events = cast(list[dict[str, object]], round_payload["events"])
            capture_events = [
                event for event in round_events if event["record_type"] == "capture_event"
            ]
            diagnostics = [
                event for event in round_events if event["record_type"] == "evolved_diagnostic"
            ]
            users = [event for event in capture_events if event["event_kind"] == "user"]
            assistants = [
                event for event in capture_events if event["event_kind"] == "assistant"
            ]
            round_payload["user_message"] = users[0] if users else None
            round_payload["assistant_final"] = assistants[-1] if assistants else None
            round_payload["diagnostics"] = diagnostics
            complete = bool(users and assistants)
            record = evolution_records.get(key)
            last_error = "" if record is None else str(record["last_error"])
            round_fingerprint = _capture_round_fingerprint(capture_events)
            current_unit_ids = [str(event["unit_id"]) for event in capture_events]
            current_unit_id_set = set(current_unit_ids)
            source_unit_ids = [] if record is None else record_source_ids.get(key, [])
            source_unit_id_set = set(source_unit_ids)
            source_known = bool(source_unit_id_set)
            current_subset = source_known and current_unit_id_set <= source_unit_id_set
            source_exact = source_known and current_unit_id_set == source_unit_id_set
            source_matches = bool(
                record is not None
                and (
                    record["source_fingerprint"] == round_fingerprint
                    or source_exact
                )
            )
            if (
                complete
                and record is not None
                and record["status"] == "completed"
                and not record["source_fingerprint"]
                and source_exact
            ):
                evolve_key = str(record["evolve_key"])
                started_at, received_earliest, received_latest = _capture_events_timeline(
                    capture_events
                )
                self._backfill_evolution_source_fingerprint(
                    evolve_key,
                    round_fingerprint,
                    source_unit_ids,
                    started_at,
                    received_earliest,
                    received_latest,
                )
                record["source_fingerprint"] = round_fingerprint
                record["source_unit_ids"] = source_unit_ids
                record["round_started_at"] = started_at
                record["round_received_earliest"] = received_earliest
                record["round_received_latest"] = received_latest
                source_matches = True
            record_completed = bool(
                record is not None
                and record["status"] == "completed"
                and not last_error
            )
            record_failed = bool(record is not None and last_error)
            terminal_matches = bool(
                record is not None
                and (record_completed or record_failed)
                and (source_matches or current_subset)
            )
            if not capture_events:
                if record_completed:
                    distillation_status = "completed"
                    legacy_status = "done"
                elif record_failed:
                    distillation_status = "failed"
                    legacy_status = "failed"
                else:
                    distillation_status = "pending_completion"
                    legacy_status = "pending"
            elif record_completed and terminal_matches:
                distillation_status = "completed"
                legacy_status = "done"
            elif record_failed and terminal_matches:
                distillation_status = "failed"
                legacy_status = "failed"
            elif not complete:
                distillation_status = "pending_completion"
                legacy_status = "pending"
            else:
                distillation_status = "pending_distillation"
                legacy_status = "pending"
            round_payload["status"] = legacy_status
            round_payload["distillation_status"] = distillation_status
            round_payload["last_error"] = last_error
            round_payload["failure_reason"] = last_error
            if record is not None:
                round_payload["candidate_id"] = None
                round_payload["memory_id"] = record["memory_id"]
                round_payload["semantic_status"] = record["semantic_status"]
                round_payload["semantic_memory_ids"] = record["semantic_memory_ids"]
                round_payload["semantic_memory_refs"] = record["semantic_memory_refs"]
                round_payload["semantic_diagnostics"] = record["semantic_diagnostics"]
                round_payload["semantic_error"] = record["semantic_error"]
                round_payload["semantic_updated_at"] = record["semantic_updated_at"]
                record_created_at = _normalize_timeline_timestamp(
                    record["created_at"]
                )
                record_completed_at = _normalize_timeline_timestamp(
                    record["completed_at"]
                )
                round_payload["completed_at"] = record_completed_at or None
                round_payload["created_at"] = record_created_at
                if not capture_events or terminal_matches:
                    started_at, received_at, updated_at = (
                        self._evolution_record_timeline(record)
                    )
                    round_payload["started_at"] = started_at
                    round_payload["received_at"] = received_at
                    round_payload["updated_at"] = updated_at
        return sorted(
            rounds.values(),
            key=_capture_round_sort_key,
        )

    def _capture_evolution_records(self) -> dict[tuple[str, str, str], dict[str, object]]:
        with closing(self._maintenance_db()) as connection:
            rows = connection.execute(
                """SELECT session_id, turn_id, region_id, status, memory_id,
                          source_fingerprint, last_error, created_at, completed_at,
                          evolve_key, source_unit_ids, round_started_at,
                          round_received_earliest, round_received_latest,
                          semantic_status, semantic_memory_ids, semantic_memory_refs,
                          semantic_diagnostics, semantic_error, semantic_updated_at
                   FROM capture_evolutions"""
            ).fetchall()
        return {
            (str(row[0]), str(row[1]), str(row[2])): {
                "status": str(row[3]),
                "memory_id": None if row[4] is None else str(row[4]),
                "source_fingerprint": str(row[5]),
                "last_error": str(row[6]),
                "created_at": None if row[7] is None else str(row[7]),
                "completed_at": None if row[8] is None else str(row[8]),
                "evolve_key": str(row[9]),
                "source_unit_ids": _json_string_list(row[10]),
                "round_started_at": str(row[11]),
                "round_received_earliest": str(row[12]),
                "round_received_latest": str(row[13]),
                "semantic_status": str(row[14]),
                "semantic_memory_ids": _json_string_list(row[15]),
                "semantic_memory_refs": _json_object_list(row[16]),
                "semantic_diagnostics": _json_object_list(row[17]),
                "semantic_error": str(row[18]),
                "semantic_updated_at": None if row[19] is None else str(row[19]),
            }
            for row in rows
        }

    def _evolution_record_source_ids(
        self,
        bundle: _RuntimeBundle,
        records: Mapping[tuple[str, str, str], dict[str, object]],
    ) -> dict[tuple[str, str, str], list[str]]:
        resolved = {
            key: _json_string_list(record.get("source_unit_ids"))
            for key, record in records.items()
        }
        missing_memory_ids = {
            str(record["memory_id"])
            for key, record in records.items()
            if not resolved[key] and record.get("memory_id")
        }
        if not missing_memory_ids:
            return resolved
        found: dict[str, list[str]] = {}
        for scope in bundle.kv_store.scopes():
            for unit in self._units_in_scope(bundle, scope):
                if unit.id not in missing_memory_ids:
                    continue
                source_ids = _metadata_strings(unit, "source_unit_ids")
                if not source_ids:
                    source_ids = [str(item) for item in unit.provenance]
                if source_ids:
                    found[unit.id] = source_ids
        for key, record in records.items():
            memory_id = record.get("memory_id")
            if not resolved[key] and memory_id:
                resolved[key] = found.get(str(memory_id), [])
        return resolved

    @staticmethod
    def _evolution_record_timeline(
        record: Mapping[str, object],
    ) -> tuple[str, str, str]:
        created_at = _normalize_timeline_timestamp(record.get("created_at"))
        completed_at = _normalize_timeline_timestamp(record.get("completed_at"))
        round_started_at = _normalize_timeline_timestamp(
            record.get("round_started_at")
        )
        received_earliest = _normalize_timeline_timestamp(
            record.get("round_received_earliest")
        )
        received_latest = _normalize_timeline_timestamp(
            record.get("round_received_latest")
        )
        started_at = round_started_at or received_earliest or created_at or completed_at
        received_at = received_latest or received_earliest or created_at or completed_at
        updated_at = _latest_timeline_timestamp(received_at, completed_at)
        return started_at, received_at, updated_at

    @staticmethod
    def _scope_has_units(bundle: _RuntimeBundle, scope: Scope) -> bool:
        try:
            return bool(bundle.kv_store.list(scope, limit=1).entries)
        except Exception:
            return True

    def _units_in_scope(self, bundle: _RuntimeBundle, scope: Scope) -> list[MemoryUnit]:
        rows = bundle.kv_store.scan(scope, "/memory/")
        unit_ids = [key.removeprefix("/memory/") for key, _ in rows]
        return list(load_units(bundle.kv_store, scope, unit_ids))

    def _session_region(self, region_id: str) -> MemoryRegion:
        region = self._regions.require_region(region_id)
        if region.kind != "session":
            raise RegionBindingError("retention applies to session memory regions")
        return region

    def _retention_days(self, region_id: str) -> int:
        with closing(self._maintenance_db()) as connection:
            row = connection.execute(
                "SELECT retention_days FROM session_retention_policy WHERE region_id = ?",
                (region_id,),
            ).fetchone()
        return _DEFAULT_SESSION_RETENTION_DAYS if row is None else int(row[0])

    def _retention_payload(self, region: MemoryRegion, days: int) -> dict[str, object]:
        return {
            "library_id": region.region_id,
            "region_id": region.region_id,
            "inbox_days": days,
            "retention_days": days,
        }

    def _retention_plan(self, region: MemoryRegion) -> list[tuple[Scope, MemoryUnit]]:
        bundle = self._require_bundle()
        cutoff = datetime.now(UTC) - timedelta(days=self._retention_days(region.region_id))
        expired: list[tuple[Scope, MemoryUnit]] = []
        for agent in _SEARCH_AGENTS:
            scope = _scope_for_region(region, session_id=region.session_id, agent=agent)
            for unit in self._units_in_scope(bundle, scope):
                if unit.lifecycle.value != "active" or not _is_capture_unit(unit):
                    continue
                occurred_at = _parse_timestamp(_metadata_text(unit, "occurred_at"))
                if occurred_at is not None and occurred_at < cutoff:
                    expired.append((scope, unit))
        return expired

    def _record_archived_incomplete_turns(
        self,
        bundle: _RuntimeBundle,
        plan: list[tuple[Scope, MemoryUnit]],
    ) -> None:
        for scope, archived_units in _group_units_by_scope(plan):
            current_captures = [
                unit
                for unit in self._units_in_scope(bundle, scope)
                if unit.lifecycle.value == "active" and _is_capture_unit(unit)
            ]
            groups: dict[str, list[MemoryUnit]] = {}
            for unit in current_captures:
                turn_id = _metadata_text(unit, "turn_id")
                if turn_id:
                    groups.setdefault(turn_id, []).append(unit)
            archived_turn_ids = {
                _metadata_text(unit, "turn_id") for unit in archived_units
            }
            for turn_id in archived_turn_ids:
                source_units = sorted(
                    groups.get(turn_id, []),
                    key=lambda unit: (
                        _event_order(_metadata_text(unit, "event_kind")),
                        _metadata_text(unit, "occurred_at"),
                        unit.id,
                    ),
                )
                if not source_units or _capture_turn_is_complete(source_units):
                    continue
                region_id = _metadata_text(source_units[0], "region_id")
                if not region_id:
                    continue
                evolve_key = _stable_evolve_key(scope, region_id, turn_id)
                source_unit_ids = [unit.id for unit in source_units]
                source_fingerprint = _source_fingerprint(source_unit_ids)
                started_at, received_earliest, received_latest = _capture_units_timeline(
                    source_units
                )
                self._record_incomplete_evolution(
                    evolve_key,
                    scope.session,
                    turn_id,
                    region_id,
                    source_fingerprint,
                    source_unit_ids,
                    started_at,
                    received_earliest,
                    received_latest,
                )

    def _evolved_count(self, region: MemoryRegion) -> int:
        bundle = self._require_bundle()
        count = 0
        for agent in _SEARCH_AGENTS:
            scope = _scope_for_region(region, session_id=region.session_id, agent=agent)
            count += sum(
                1
                for unit in self._units_in_scope(bundle, scope)
                if unit.lifecycle.value == "active"
                and _metadata_text(unit, "memory_type") == "evolved"
            )
        return count

    def _maintenance_db(self) -> sqlite3.Connection:
        self._require_state_ready()
        path = self._state_dir / "mem2" / "maintenance.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=5.0)
        with self._idempotency_schema_lock:
            self._ensure_maintenance_schema(
                connection,
                migrate_promotions=self._promotion_schema_ready,
            )
        return connection

    @staticmethod
    def _ensure_maintenance_schema(
        connection: sqlite3.Connection,
        *,
        migrate_promotions: bool = True,
    ) -> None:
        connection.executescript(_BROWSER_INDEX_SCHEMA)
        connection.executescript(_SEMANTIC_PUBLICATION_SCHEMA)
        browser_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(memory_browser_index)").fetchall()
        }
        if "logical_memory_id" not in browser_columns:
            with connection:
                connection.execute(
                    "ALTER TABLE memory_browser_index "
                    "ADD COLUMN logical_memory_id TEXT NOT NULL DEFAULT ''"
                )
        connection.execute(
            """CREATE INDEX IF NOT EXISTS memory_browser_index_lineage
               ON memory_browser_index (region_id, logical_memory_id, unit_id)"""
        )
        connection.executescript(_PROMOTION_SCHEMA)
        if migrate_promotions:
            Mem2MemoryKernel._ensure_promotion_schema(connection)
        columns = {
            str(row[1]): row
            for row in connection.execute("PRAGMA table_info(capture_evolutions)").fetchall()
        }
        if not columns:
            connection.executescript(_MAINTENANCE_SCHEMA)
            Mem2MemoryKernel._ensure_governance_schema(connection)
            return
        memory_id_column = columns.get("memory_id")
        if (
            "status" in columns
            and "completed_at" in columns
            and memory_id_column is not None
            and not memory_id_column[3]
        ):
            connection.execute(_SESSION_RETENTION_SCHEMA)
            connection.execute(_GOVERNANCE_AUDIT_SCHEMA)
            Mem2MemoryKernel._ensure_governance_schema(connection)
            if "source_fingerprint" not in columns:
                connection.execute(
                    "ALTER TABLE capture_evolutions "
                    "ADD COLUMN source_fingerprint TEXT NOT NULL DEFAULT ''"
                )
            if "last_error" not in columns:
                connection.execute(
                    "ALTER TABLE capture_evolutions "
                    "ADD COLUMN last_error TEXT NOT NULL DEFAULT ''"
                )
            capture_columns = {
                "source_unit_ids": "TEXT NOT NULL DEFAULT '[]'",
                "round_started_at": "TEXT NOT NULL DEFAULT ''",
                "round_received_earliest": "TEXT NOT NULL DEFAULT ''",
                "round_received_latest": "TEXT NOT NULL DEFAULT ''",
                "semantic_status": "TEXT NOT NULL DEFAULT 'not_requested'",
                "semantic_memory_ids": "TEXT NOT NULL DEFAULT '[]'",
                "semantic_memory_refs": "TEXT NOT NULL DEFAULT '[]'",
                "semantic_diagnostics": "TEXT NOT NULL DEFAULT '[]'",
                "semantic_error": "TEXT NOT NULL DEFAULT ''",
                "semantic_updated_at": "TEXT",
            }
            for column, definition in capture_columns.items():
                if column not in columns:
                    connection.execute(
                        f"ALTER TABLE capture_evolutions ADD COLUMN {column} {definition}"
                    )
            return
        with connection:
            connection.execute("ALTER TABLE capture_evolutions RENAME TO capture_evolutions_v1")
            connection.execute(_SESSION_RETENTION_SCHEMA)
            connection.execute(_CAPTURE_EVOLUTIONS_SCHEMA)
            connection.execute(_GOVERNANCE_AUDIT_SCHEMA)
            Mem2MemoryKernel._ensure_governance_schema(connection)
            connection.execute(
                """INSERT OR REPLACE INTO capture_evolutions
                   (evolve_key, session_id, turn_id, region_id, source_fingerprint, memory_id,
                    status, semantic_status, created_at, completed_at)
                   SELECT evolve_key, session_id, turn_id, region_id, '', memory_id,
                          'completed', 'not_requested', created_at, created_at
                   FROM capture_evolutions_v1"""
            )
            connection.execute("DROP TABLE capture_evolutions_v1")

    @staticmethod
    def _ensure_promotion_schema(
        connection: sqlite3.Connection,
        *,
        batch_size: int = _PROMOTION_MIGRATION_BATCH_SIZE,
    ) -> list[int]:
        if batch_size <= 0:
            raise ValueError("promotion migration batch size must be positive")
        connection.execute(
            """CREATE TABLE IF NOT EXISTS memory_browser_index_meta (
                   key TEXT PRIMARY KEY,
                   value TEXT NOT NULL
               )"""
        )
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(promotion_records)").fetchall()
        }
        additions = {
            "evidence_kind": "TEXT NOT NULL DEFAULT 'explicit'",
            "source_unit_ids": "TEXT NOT NULL DEFAULT '[]'",
            "source_region_ids": "TEXT NOT NULL DEFAULT '[]'",
            "source_unit_regions": "TEXT NOT NULL DEFAULT '[]'",
        }
        missing = [name for name in additions if name not in columns]
        with connection:
            for name in missing:
                connection.execute(
                    f"ALTER TABLE promotion_records ADD COLUMN {name} {additions[name]}"
                )
        marker = connection.execute(
            """SELECT value FROM memory_browser_index_meta
               WHERE key = 'promotion_schema_version'"""
        ).fetchone()
        if not missing and marker is not None and str(marker[0]) == str(
            _PROMOTION_SCHEMA_VERSION
        ):
            return []

        batch_sizes: list[int] = []
        cursor = ""
        while True:
            with connection:
                rows = connection.execute(
                    """SELECT promotion_key, source_unit_id, source_region_id,
                              source_unit_ids, source_region_ids, source_unit_regions
                       FROM promotion_records
                       WHERE promotion_key > ?
                       ORDER BY promotion_key
                       LIMIT ?""",
                    (cursor, batch_size),
                ).fetchall()
                if not rows:
                    break
                parsed: list[tuple[str, str, str, list[str], list[str], dict[str, str]]] = []
                source_ids: set[str] = set()
                for row in rows:
                    promotion_key = str(row[0])
                    primary_unit_id = str(row[1] or "")
                    primary_region_id = str(row[2] or "")
                    try:
                        record_unit_ids = json.loads(str(row[3]))
                    except json.JSONDecodeError:
                        record_unit_ids = []
                    try:
                        record_region_ids = json.loads(str(row[4]))
                    except json.JSONDecodeError:
                        record_region_ids = []
                    try:
                        raw_region_map = json.loads(str(row[5]))
                    except json.JSONDecodeError:
                        raw_region_map = []
                    unit_ids = (
                        [str(value) for value in record_unit_ids if str(value)]
                        if isinstance(record_unit_ids, list) and record_unit_ids
                        else ([primary_unit_id] if primary_unit_id else [])
                    )
                    region_ids = (
                        [str(value) for value in record_region_ids if str(value)]
                        if isinstance(record_region_ids, list) and record_region_ids
                        else ([primary_region_id] if primary_region_id else [])
                    )
                    unit_region_map: dict[str, str] = {}
                    if isinstance(raw_region_map, list):
                        for item in raw_region_map:
                            if not isinstance(item, dict):
                                continue
                            unit_id = str(item.get("unit_id") or "").strip()
                            region_id = str(item.get("region_id") or "").strip()
                            if unit_id and region_id:
                                unit_region_map[unit_id] = region_id
                    source_ids.update(unit_ids)
                    parsed.append(
                        (
                            promotion_key,
                            primary_unit_id,
                            primary_region_id,
                            unit_ids,
                            region_ids,
                            unit_region_map,
                        )
                    )
                indexed_regions: dict[str, str] = {}
                ordered_source_ids = sorted(source_ids)
                for offset in range(0, len(ordered_source_ids), 500):
                    chunk = ordered_source_ids[offset : offset + 500]
                    if not chunk:
                        continue
                    placeholders = ", ".join("?" for _ in chunk)
                    index_rows = connection.execute(
                        f"""SELECT unit_id, region_id FROM memory_browser_index
                            WHERE unit_id IN ({placeholders})
                            ORDER BY unit_id, region_id""",
                        chunk,
                    ).fetchall()
                    for index_row in index_rows:
                        indexed_regions.setdefault(str(index_row[0]), str(index_row[1]))
                for (
                    promotion_key,
                    primary_unit_id,
                    primary_region_id,
                    unit_ids,
                    region_ids,
                    unit_region_map,
                ) in parsed:
                    if primary_unit_id and primary_region_id:
                        unit_region_map.setdefault(primary_unit_id, primary_region_id)
                    for unit_id in unit_ids:
                        indexed = indexed_regions.get(unit_id)
                        if indexed:
                            unit_region_map.setdefault(unit_id, indexed)
                    connection.execute(
                        """UPDATE promotion_records
                           SET source_unit_ids = ?, source_region_ids = ?,
                               source_unit_regions = ?
                           WHERE promotion_key = ?""",
                        (
                            json.dumps(unit_ids, ensure_ascii=False),
                            json.dumps(region_ids, ensure_ascii=False),
                            json.dumps(
                                [
                                    {"unit_id": unit_id, "region_id": region_id}
                                    for unit_id, region_id in unit_region_map.items()
                                ],
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            promotion_key,
                        ),
                    )
            cursor = str(rows[-1][0])
            batch_sizes.append(len(rows))
        with connection:
            connection.execute(
                """INSERT INTO memory_browser_index_meta (key, value)
                   VALUES ('promotion_schema_version', ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                (str(_PROMOTION_SCHEMA_VERSION),),
            )
        return batch_sizes

    @staticmethod
    def _ensure_governance_schema(connection: sqlite3.Connection) -> None:
        columns = {
            str(row[1]): row
            for row in connection.execute("PRAGMA table_info(governance_events)").fetchall()
        }
        if "status" not in columns:
            with connection:
                connection.execute(
                    "ALTER TABLE governance_events "
                    "ADD COLUMN status TEXT NOT NULL DEFAULT 'committed'"
                )
        if "logical_memory_id" not in columns:
            with connection:
                connection.execute(
                    "ALTER TABLE governance_events "
                    "ADD COLUMN logical_memory_id TEXT NOT NULL DEFAULT ''"
                )
        connection.execute(
            """CREATE INDEX IF NOT EXISTS governance_events_lineage
               ON governance_events (region_id, logical_memory_id, unit_id)"""
        )

    def _reserve_evolution(
        self,
        evolve_key: str,
        session_id: str,
        turn_id: str,
        region_id: str,
        source_fingerprint: str,
        source_unit_ids: list[str],
        round_started_at: str,
        round_received_earliest: str,
        round_received_latest: str,
    ) -> tuple[str, str | None]:
        with closing(self._maintenance_db()) as connection, connection:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO capture_evolutions
                   (evolve_key, session_id, turn_id, region_id, source_fingerprint,
                    source_unit_ids, memory_id, status, round_started_at,
                    round_received_earliest, round_received_latest, semantic_status)
                   VALUES (?, ?, ?, ?, ?, ?, NULL, 'pending', ?, ?, ?, ?)""",
                (
                    evolve_key,
                    session_id,
                    turn_id,
                    region_id,
                    source_fingerprint,
                    json.dumps(source_unit_ids, ensure_ascii=False),
                    round_started_at,
                    round_received_earliest,
                    round_received_latest,
                    "pending" if self._semantic_enabled() else "not_requested",
                ),
            )
            if cursor.rowcount == 1:
                return "pending", None
            row = connection.execute(
                """SELECT status, memory_id, source_fingerprint, last_error
                   FROM capture_evolutions
                   WHERE evolve_key = ?""",
                (evolve_key,),
            ).fetchone()
        if row is None:
            raise MemoryKernelUnavailableError("evolve reservation was not persisted")
        source_changed = str(row[2]) != source_fingerprint
        if source_changed or str(row[3]):
            with closing(self._maintenance_db()) as connection, connection:
                if source_changed:
                    connection.execute(
                        """UPDATE capture_evolutions
                           SET source_fingerprint = ?, memory_id = NULL,
                               status = 'pending', last_error = '', completed_at = NULL,
                               source_unit_ids = ?, round_started_at = ?,
                               round_received_earliest = ?, round_received_latest = ?,
                               semantic_status = ?,
                               semantic_memory_ids = '[]',
                               semantic_memory_refs = '[]',
                               semantic_diagnostics = '[]',
                               semantic_error = '', semantic_updated_at = NULL
                           WHERE evolve_key = ?""",
                        (
                            source_fingerprint,
                            json.dumps(source_unit_ids, ensure_ascii=False),
                            round_started_at,
                            round_received_earliest,
                            round_received_latest,
                            "pending" if self._semantic_enabled() else "not_requested",
                            evolve_key,
                        ),
                    )
                else:
                    connection.execute(
                        """UPDATE capture_evolutions
                           SET memory_id = NULL, status = 'pending',
                               last_error = '', completed_at = NULL
                           WHERE evolve_key = ?""",
                        (evolve_key,),
                    )
            return "pending", None
        return str(row[0]), None if row[1] is None else str(row[1])

    def _record_incomplete_evolution(
        self,
        evolve_key: str,
        session_id: str,
        turn_id: str,
        region_id: str,
        source_fingerprint: str,
        source_unit_ids: list[str],
        round_started_at: str,
        round_received_earliest: str,
        round_received_latest: str,
    ) -> None:
        with closing(self._maintenance_db()) as connection, connection:
            connection.execute(
                """INSERT INTO capture_evolutions
                   (evolve_key, session_id, turn_id, region_id, source_fingerprint,
                    source_unit_ids, memory_id, status, last_error, completed_at,
                    round_started_at, round_received_earliest, round_received_latest,
                    semantic_status)
                   VALUES (?, ?, ?, ?, ?, ?, NULL, 'pending', '', NULL, ?, ?, ?, ?)
                   ON CONFLICT(evolve_key) DO UPDATE SET
                     source_fingerprint = excluded.source_fingerprint,
                     source_unit_ids = excluded.source_unit_ids,
                     memory_id = NULL,
                     status = 'pending',
                     last_error = '',
                     completed_at = NULL,
                     round_started_at = excluded.round_started_at,
                     round_received_earliest = excluded.round_received_earliest,
                     round_received_latest = excluded.round_received_latest
                   WHERE capture_evolutions.status = 'pending'
                     AND capture_evolutions.last_error = ''
                     AND capture_evolutions.memory_id IS NULL""",
                (
                    evolve_key,
                    session_id,
                    turn_id,
                    region_id,
                    source_fingerprint,
                    json.dumps(source_unit_ids, ensure_ascii=False),
                    round_started_at,
                    round_received_earliest,
                    round_received_latest,
                    "pending" if self._semantic_enabled() else "not_requested",
                ),
            )

    def _complete_evolution(self, evolve_key: str, memory_id: str) -> None:
        with closing(self._maintenance_db()) as connection, connection:
            connection.execute(
                """UPDATE capture_evolutions
                   SET memory_id = ?, status = 'completed',
                       last_error = '', completed_at = CURRENT_TIMESTAMP
                   WHERE evolve_key = ?""",
                (memory_id, evolve_key),
            )

    def _backfill_evolution_source_fingerprint(
        self,
        evolve_key: str,
        source_fingerprint: str,
        source_unit_ids: list[str],
        round_started_at: str,
        round_received_earliest: str,
        round_received_latest: str,
    ) -> None:
        with closing(self._maintenance_db()) as connection, connection:
            connection.execute(
                """UPDATE capture_evolutions
                   SET source_fingerprint = ?, source_unit_ids = ?,
                       round_started_at = ?, round_received_earliest = ?,
                       round_received_latest = ?
                   WHERE evolve_key = ? AND status = 'completed'
                     AND source_fingerprint = ''""",
                (
                    source_fingerprint,
                    json.dumps(source_unit_ids, ensure_ascii=False),
                    round_started_at,
                    round_received_earliest,
                    round_received_latest,
                    evolve_key,
                ),
            )

    def _fail_evolution(self, evolve_key: str, error: str) -> None:
        with closing(self._maintenance_db()) as connection, connection:
            connection.execute(
                """UPDATE capture_evolutions
                   SET last_error = ?, completed_at = NULL
                   WHERE evolve_key = ? AND status != 'completed'""",
                (error[:1000], evolve_key),
            )

    def _evolution_units(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        evolve_key: str,
    ) -> list[MemoryUnit]:
        return [
            unit
            for unit in self._units_in_scope(bundle, scope)
            if _metadata_text(unit, "evolve_key") == evolve_key
        ]

    def _purge_capture_units(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        event_id: str,
    ) -> None:
        units = [
            unit
            for unit in self._units_in_scope(bundle, scope)
            if _metadata_text(unit, "capture_event_id") == event_id
        ]
        if not units:
            return
        bundle.runtime.api.delete(
            DeleteSelector(
                unit_ids=[str(unit.id) for unit in units],
                scope=scope,
                mode=DeleteMode.PURGE,
            ),
            security=_security_for(scope),
        )
        if any(
            _metadata_text(unit, "capture_event_id") == event_id
            for unit in self._units_in_scope(bundle, scope)
        ):
            raise MemoryKernelUnavailableError("capture residue cleanup did not persist")
        if not self._fulltext_visible(
            bundle,
            scope,
            [str(unit.id) for unit in units],
            expected=False,
        ):
            raise MemoryKernelUnavailableError("capture index cleanup did not persist")

    def _capture_visible(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        units: list[MemoryUnit],
    ) -> bool:
        return len(units) == 1 and self._fulltext_visible(
            bundle,
            scope,
            [str(units[0].id)],
            expected=True,
        )

    def _fulltext_visible(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        unit_ids: list[str],
        *,
        expected: bool,
    ) -> bool:
        fulltext = _fulltext_store(bundle)
        if fulltext is None:
            return False
        try:
            documents = fulltext.get(scope, unit_ids)
        except Exception:
            return False
        visible_ids = {str(document.id) for document in documents}
        return visible_ids == set(unit_ids) if expected else not visible_ids.intersection(unit_ids)

    def _evolution_visible(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        units: list[MemoryUnit],
        expected_parts: int,
        source_fingerprint: str,
    ) -> bool:
        if len(units) != expected_parts:
            return False
        parts = {_metadata_int(unit, "evolve_part") for unit in units}
        if parts != set(range(expected_parts)):
            return False
        if any(_metadata_text(unit, "source_fingerprint") != source_fingerprint for unit in units):
            return False
        return self._fulltext_visible(
            bundle,
            scope,
            [str(unit.id) for unit in units],
            expected=True,
        )

    def _purge_evolution_units(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        units: list[MemoryUnit],
    ) -> None:
        if not units:
            return
        bundle.runtime.api.delete(
            DeleteSelector(
                unit_ids=[str(unit.id) for unit in units],
                scope=scope,
                mode=DeleteMode.PURGE,
            ),
            security=_security_for(scope),
        )
        remaining = [
            unit
            for unit in self._units_in_scope(bundle, scope)
            if _metadata_text(unit, "evolve_key")
            in {_metadata_text(item, "evolve_key") for item in units}
        ]
        if remaining:
            raise MemoryKernelUnavailableError("evolved memory cleanup did not persist")

    def _idempotency_db(self) -> sqlite3.Connection:
        self._require_state_ready()
        path = self._state_dir / "mem2" / "idempotency.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=5.0)
        with self._idempotency_schema_lock:
            self._ensure_idempotency_schema(connection)
        return connection

    @staticmethod
    def _create_idempotency_table(connection: sqlite3.Connection) -> None:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS memory_write_idempotency (
                   library_id TEXT NOT NULL,
                   idempotency_key TEXT NOT NULL,
                   request_hash TEXT NOT NULL,
                   memory_id TEXT,
                   status TEXT NOT NULL
                       CHECK (status IN ('pending', 'completed')),
                   created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                   completed_at TEXT,
                   PRIMARY KEY (library_id, idempotency_key)
               )"""
        )

    def _ensure_idempotency_schema(self, connection: sqlite3.Connection) -> None:
        columns = {
            str(row[1]): row
            for row in connection.execute("PRAGMA table_info(memory_write_idempotency)").fetchall()
        }
        if not columns:
            with connection:
                self._create_idempotency_table(connection)
            return
        memory_id_column = columns.get("memory_id")
        if "status" in columns and memory_id_column is not None and not memory_id_column[3]:
            return
        with connection:
            connection.execute(
                "ALTER TABLE memory_write_idempotency RENAME TO memory_write_idempotency_v1"
            )
            self._create_idempotency_table(connection)
            connection.execute(
                """INSERT OR REPLACE INTO memory_write_idempotency
                   (library_id, idempotency_key, request_hash, memory_id,
                    status, created_at, completed_at)
                   SELECT library_id, idempotency_key, request_hash, memory_id,
                          'completed', created_at, created_at
                   FROM memory_write_idempotency_v1"""
            )
            connection.execute("DROP TABLE memory_write_idempotency_v1")

    def _reserve_idempotency(
        self, library_id: str, idempotency_key: str, request_hash: str
    ) -> str | None:
        with closing(self._idempotency_db()) as connection, connection:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO memory_write_idempotency
                   (library_id, idempotency_key, request_hash, memory_id, status)
                   VALUES (?, ?, ?, NULL, 'pending')""",
                (library_id, idempotency_key, request_hash),
            )
            if cursor.rowcount == 1:
                return None
            row = connection.execute(
                """SELECT request_hash, memory_id, status
                   FROM memory_write_idempotency
                   WHERE library_id = ? AND idempotency_key = ?""",
                (library_id, idempotency_key),
            ).fetchone()
        if row is None:
            raise MemoryKernelUnavailableError("idempotency reservation was not persisted")
        if str(row[0]) != request_hash:
            raise MemoryIdempotencyConflictError(
                "idempotency_key was already used for another memory content"
            )
        if str(row[2]) == "completed" and row[1] is not None:
            return str(row[1])
        if str(row[2]) == "pending":
            raise MemoryIdempotencyPendingError(
                "idempotency_key has an incomplete earlier memory write"
            )
        raise MemoryKernelUnavailableError("completed idempotency record has no memory identifier")

    def _reserve_capture_idempotency(
        self,
        bundle: _RuntimeBundle,
        scope: Scope,
        request: CaptureMemoryRequest,
        identity: str,
        request_hash: str,
    ) -> str | None:
        with closing(self._idempotency_db()) as connection, connection:
            cursor = connection.execute(
                """INSERT OR IGNORE INTO memory_write_idempotency
                   (library_id, idempotency_key, request_hash, memory_id, status)
                   VALUES (?, ?, ?, NULL, 'pending')""",
                (identity, request.event_id, request_hash),
            )
            if cursor.rowcount == 1:
                return None
            row = connection.execute(
                """SELECT request_hash, memory_id, status
                   FROM memory_write_idempotency
                   WHERE library_id = ? AND idempotency_key = ?""",
                (identity, request.event_id),
            ).fetchone()
        if row is None:
            raise MemoryKernelUnavailableError("idempotency reservation was not persisted")
        if str(row[0]) != request_hash:
            raise MemoryIdempotencyConflictError(
                "idempotency_key was already used for another memory content"
            )
        if str(row[2]) == "completed" and row[1] is not None:
            return str(row[1])
        if str(row[2]) != "pending":
            raise MemoryKernelUnavailableError(
                "completed idempotency record has no memory identifier"
            )

        residual = [
            unit
            for unit in self._units_in_scope(bundle, scope)
            if _metadata_text(unit, "capture_event_id") == request.event_id
        ]
        if residual and self._capture_visible(bundle, scope, residual):
            self._complete_idempotency(
                identity,
                request.event_id,
                request_hash,
                residual[0].id,
            )
            return str(residual[0].id)
        if residual:
            self._purge_capture_units(bundle, scope, request.event_id)
        # Keep the reservation pending while this retry rebuilds the capture.
        # Releasing it here allows a timed-out background write and a new retry
        # to interleave and create duplicate units.
        return None

    def _complete_idempotency(
        self,
        library_id: str,
        idempotency_key: str,
        request_hash: str,
        memory_id: str,
    ) -> None:
        with closing(self._idempotency_db()) as connection, connection:
            cursor = connection.execute(
                """UPDATE memory_write_idempotency
                   SET memory_id = ?, status = 'completed',
                       completed_at = CURRENT_TIMESTAMP
                   WHERE library_id = ? AND idempotency_key = ?
                     AND request_hash = ? AND status = 'pending'""",
                (memory_id, library_id, idempotency_key, request_hash),
            )
            if cursor.rowcount != 1:
                raise MemoryKernelUnavailableError(
                    "idempotency completion did not update the pending reservation"
                )

    def _release_pending_idempotency(self, library_id: str, idempotency_key: str) -> None:
        with closing(self._idempotency_db()) as connection, connection:
            connection.execute(
                """DELETE FROM memory_write_idempotency
                   WHERE library_id = ? AND idempotency_key = ?
                     AND status = 'pending'""",
                (library_id, idempotency_key),
            )

    def resolve_region(self, request: ResolveRegionRequest) -> dict[str, str]:
        self._require_state_ready()
        resolved = self._regions.resolve(request.cwd)
        if resolved is None:
            return {"status": "unbound", "cwd": self._regions.canonical_cwd(request.cwd)}
        return {
            "status": "bound",
            "cwd": resolved.cwd,
            "project_id": resolved.region.region_id,
            "binding_id": resolved.binding.binding_id,
            "binding_kind": resolved.binding.kind,
            "project_root": resolved.binding.canonical_root,
            "region_id": resolved.region.region_id,
            "library_id": resolved.region.region_id,
            "space": resolved.region.space,
        }

    def create_region(self, request: CreateMemoryRegionRequest) -> dict[str, object]:
        self._require_state_ready()
        return self._regions.create_region(
            request.kind,
            request.name,
            region_id=request.region_id,
            session_id=request.session_id,
        ).payload()

    def list_regions(self) -> list[dict[str, object]]:
        self._require_state_ready()
        return [region.payload() for region in self._regions.list_regions()]

    def get_region(self, request: GetMemoryRegionRequest) -> dict[str, object]:
        self._require_state_ready()
        return self._regions.require_region(request.region_id).payload()

    def region_tree(self, request: MemoryRegionTreeRequest) -> dict[str, object]:
        self._require_state_ready()
        return self._regions.region_tree(request.session_id)

    def bind_project_region(self, request: BindProjectRegionRequest) -> dict[str, str]:
        self._require_state_ready()
        return self._regions.bind_project(request.project_root, request.region_id).payload()

    def associate_worktree_region(self, request: AssociateWorktreeRegionRequest) -> dict[str, str]:
        self._require_state_ready()
        return self._regions.associate_worktree(
            request.worktree_root, request.main_project_binding_id
        ).payload()

    def list_region_bindings(self) -> list[dict[str, str]]:
        self._require_state_ready()
        return [binding.payload() for binding in self._regions.list_bindings()]

    def update_region_binding(self, request: UpdateRegionBindingRequest) -> dict[str, str]:
        self._require_state_ready()
        return self._regions.update_binding(request.binding_id, request.region_id).payload()

    def delete_region_binding(self, request: DeleteRegionBindingRequest) -> None:
        self._require_state_ready()
        self._regions.delete_binding(request.binding_id)

    def _require_state_ready(self) -> None:
        if not self._state_ready:
            raise MemoryKernelUnavailableError(
                self._startup_error or "mem2 state version has not been validated"
            )

    @property
    def _regions(self) -> RegionRegistry:
        self._require_state_ready()
        registry = self._regions_registry
        if registry is None:
            with self._regions_registry_lock:
                if self._regions_registry is None:
                    self._regions_registry = RegionRegistry(
                        self._state_dir / "mem2" / "regions.sqlite3"
                    )
                registry = self._regions_registry
        return registry

    def _require_bundle(self) -> _RuntimeBundle:
        with self._lock:
            bundle = self._bundle
            startup_error = self._startup_error
        if bundle is None:
            raise MemoryKernelUnavailableError(startup_error or "mem2 runtime is not started")
        return bundle

    def _runtime_config(self, state_dir: Path) -> dict[str, Any]:
        milvus_uri = str(state_dir / "milvus.db")

        def store(collection: str) -> dict[str, object]:
            return {
                "target": _STORE_TARGET,
                "params": {
                    "uri": milvus_uri,
                    "dim": self._embedding_dimension,
                    "collection": collection,
                },
            }

        vector_enabled = self._vector_enabled
        engine_params: dict[str, str] = {
            "index_builder": "default",
            "retriever": "default",
        }
        base: dict[str, Any] = {
            "globals": {
                "schema_enabled": False,
                "vector_enabled": vector_enabled,
                "graph_enabled": False,
                "rerank_enabled": False,
                "embedder_dim": self._embedding_dimension,
            },
            "engine": {
                "default": {
                    "target": "cloud",
                    "params": engine_params,
                }
            },
            "constructor": {
                "default": {
                    "target": "hybrid",
                    "params": {"embedder": "default"} if vector_enabled else {},
                }
            },
            "permission": {"default": {"target": "allow_all"}},
            "recaller": {
                "vector": {
                    "target": "vector",
                    "params": {"min_similarity": _VECTOR_MIN_SIMILARITY},
                }
            },
            "vector_store": {
                "default": store("agent_memory_vectors"),
                "layers_l0": store("agent_memory_vectors_l0"),
                "layers_l1": store("agent_memory_vectors_l1"),
            },
            "fulltext_store": {
                "default": store("agent_memory_fulltext"),
                "layers_l0": store("agent_memory_fulltext_l0"),
                "layers_l1": store("agent_memory_fulltext_l1"),
            },
            "graph_store": {
                "default": {
                    "target": _STORE_TARGET,
                    "params": {
                        "uri": milvus_uri,
                        "collection": "agent_memory_graph_nodes",
                        "edge_collection": "agent_memory_graph_edges",
                    },
                }
            },
            "audit": {
                "default": {
                    "target": _AUDIT_TARGET,
                    "params": {"db_path": str(state_dir / "audit.sqlite3")},
                }
            },
        }
        endpoint = self._embedding
        if vector_enabled and endpoint is not None:
            base["embedder"] = {
                "default": {
                    "target": _EMBEDDER_TARGET,
                    "params": {
                        "model_name": endpoint.model,
                        "base_url": endpoint.base_url,
                        "api_key": self._embedding_api_key(),
                        "dim": self._embedding_dimension,
                        "timeout": endpoint.timeout_seconds,
                        "max_retries": endpoint.retries,
                    },
                }
            }
            base["query_parser"] = {
                "default": {
                    "target": "simple",
                    "params": {"embedder": "default"},
                }
            }
        return _deep_merge(base, self._config_overrides)

    def _prepare_embedding(self) -> None:
        endpoint = self._embedding
        if endpoint is None:
            self._vector_enabled = False
            self._vector_degradation = "embedding_not_configured"
            self._embedding_detail = (
                "embedding endpoint is not configured; vector recall is disabled"
            )
            return
        try:
            embedder = self._new_openai_embedder(self._embedding_dimension)
            vectors = embedder.embed(["embedding dimension probe"])
        except Exception as error:
            self._vector_enabled = False
            self._vector_degradation = "embedding_unavailable"
            self._embedding_detail = (
                "embedding endpoint probe failed; vector recall is disabled: "
                f"{type(error).__name__}"
            )
            return
        actual_dimension = len(vectors[0]) if vectors and vectors[0] else 0
        if actual_dimension != self._embedding_dimension:
            self._vector_enabled = False
            self._vector_degradation = "embedding_dimension_mismatch"
            self._embedding_detail = (
                "embedding dimension mismatch: configured "
                f"{self._embedding_dimension}, actual {actual_dimension}; "
                "vector recall is disabled"
            )
            return
        self._vector_enabled = True
        self._vector_degradation = ""
        self._embedding_detail = "embedding endpoint validated"

    def _new_openai_embedder(self, dimension: int) -> OpenAIEmbedder:
        endpoint = self._embedding
        if endpoint is None:
            raise ValueError("embedding endpoint is not configured")
        return OpenAIEmbedder(
            model_name=endpoint.model,
            base_url=endpoint.base_url,
            api_key=self._embedding_api_key(),
            dimension=dimension,
            timeout=endpoint.timeout_seconds,
            max_retries=endpoint.retries,
        )

    def _embedding_api_key(self) -> str:
        endpoint = self._embedding
        if endpoint is None or endpoint.api_key_file is None:
            return ""
        if self._embedding_api_key_cache is None:
            self._embedding_api_key_cache = endpoint.api_key_file.read_text(
                encoding="utf-8"
            ).strip()
        return self._embedding_api_key_cache


def _write_identity(request: CreateMemoryRequest) -> str:
    if request.region_id is not None:
        return f"region:{request.region_id}"
    return request.library_id


def _normalize_session_id(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("session_id must not be blank")
    if len(normalized) > MAX_SESSION_ID_LENGTH:
        raise ValueError(f"session_id exceeds the {MAX_SESSION_ID_LENGTH}-character scope limit")
    return normalized


def _normalize_optional_identifier(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must not be blank")
    return normalized


def _validate_occurred_at(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("occurred_at must include a valid timezone offset")
        parsed.astimezone(UTC)
    except (ValueError, OverflowError) as error:
        raise ValueError("occurred_at must be a valid ISO-8601 timestamp with timezone") from error
    return value


def _capture_request_hash(request: CaptureMemoryRequest, scope: Scope) -> str:
    payload = {
        "event_id": request.event_id,
        "session_id": request.session_id,
        "turn_id": request.turn_id,
        "event_kind": request.event_kind,
        "content": request.content,
        "occurred_at": request.occurred_at,
        "cwd": request.cwd,
        "topic_id": request.topic_id,
        "space": scope.space,
        "agent": scope.agent,
        "session": scope.session,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _capture_response(
    request: CaptureMemoryRequest,
    memory_id: str,
    region: MemoryRegion,
    scope: Scope,
    *,
    status: str,
    duplicate: bool = False,
) -> dict[str, object]:
    response: dict[str, object] = {
        "id": memory_id,
        "memory_id": memory_id,
        "event_id": request.event_id,
        "status": status,
        "region_id": region.region_id,
        "library_id": region.region_id,
        "space": scope.space,
        "agent": scope.agent,
        "session_id": request.session_id,
    }
    if duplicate:
        response["duplicate"] = True
    return response


def _is_capture_unit(unit: MemoryUnit) -> bool:
    return bool(_metadata_text(unit, "capture_event_id"))


def _is_capture_record(unit: MemoryUnit) -> bool:
    return _is_capture_unit(unit) or _metadata_text(unit, "memory_type") == "capture"


def _is_evolved_unit(unit: MemoryUnit) -> bool:
    return _metadata_text(unit, "memory_type") == "evolved"


def _capture_turn_is_complete(units: list[MemoryUnit]) -> bool:
    event_kinds = {_metadata_text(unit, "event_kind") for unit in units}
    return "user" in event_kinds and "assistant" in event_kinds


def _is_default_recall_item(item: Any) -> bool:
    if isinstance(item, MemoryUnit):
        metadata: object = item.user_metadata
    elif isinstance(item, Mapping):
        metadata = item.get("user_metadata")
    else:
        metadata = getattr(item, "user_metadata", None)
    if not isinstance(metadata, Mapping):
        return True
    memory_type = str(metadata.get("memory_type") or "")
    return memory_type not in {"capture", "evolved"} and not bool(
        metadata.get("capture_event_id")
    )


def _is_default_recall_candidate(item: Any) -> bool:
    if isinstance(item, MemoryUnit) and item.user_metadata.get("restore_pending") is True:
        return False
    return _is_default_recall_item(item)


def _default_recall_filter() -> FilterExpr:
    return FilterGroup(
        FilterLogic.AND,
        [
            FilterClause(
                "user_metadata.memory_type",
                FilterOp.NOT_IN,
                ["capture", "evolved"],
            ),
            FilterClause(
                "user_metadata.default_recall_excluded",
                FilterOp.NOT_IN,
                [True],
            ),
        ],
    )


def _fulltext_store(bundle: _RuntimeBundle) -> MilvusLiteFulltextStore | None:
    return next(
        (
            store
            for store in bundle.stores
            if isinstance(store, MilvusLiteFulltextStore)
            and getattr(store, "_collection", "") == "agent_memory_fulltext"
        ),
        None,
    )


def _stable_evolve_key(scope: Scope, region_id: str, turn_id: str) -> str:
    return hashlib.sha256(
        "\0".join((scope.org, scope.space, scope.session, region_id, turn_id)).encode()
    ).hexdigest()


def _source_fingerprint(source_ids: list[str]) -> str:
    return hashlib.sha256("\0".join(source_ids).encode()).hexdigest()


def _semantic_key(
    *,
    region_id: str,
    session_id: str,
    turn_id: str,
    user_event_id: str,
    evidence_snippet: str,
    memory_type: str,
    target_region_id: str,
    content: str,
    claim_subject: str,
    claim_value: str,
) -> str:
    payload = [
        region_id,
        session_id,
        turn_id,
        user_event_id,
        evidence_snippet,
        memory_type,
        target_region_id,
        _normalize_promotion_text(content),
        _normalize_promotion_text(claim_subject),
        _normalize_promotion_claim_value(claim_value),
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _semantic_claim_fields(subject: str, value: str, content: str) -> tuple[str, str]:
    normalized_subject = _normalize_promotion_text(subject)
    normalized_value = _normalize_promotion_claim_value(value)
    if normalized_subject and normalized_value:
        return normalized_subject, normalized_value
    normalized_content = _normalize_promotion_text(content)
    third_person = re.match(
        r"^the user\s+(prefers?|likes?|uses?|requires?|needs?|wants?)\s+(.+)$",
        normalized_content,
    )
    if third_person:
        derived_value = third_person.group(2).strip().rstrip(".!。")
        category = _preference_category(derived_value)
        derived_subject = f"user:preference:{category}" if category else ""
    else:
        chinese_third_person = re.match(
            r"^用户(?:偏好|喜欢|使用|需要|要求|希望)(.+)$", normalized_content
        )
        if chinese_third_person:
            derived_value = chinese_third_person.group(1).strip().rstrip(".!。")
            category = _preference_category(derived_value)
            derived_subject = f"user:preference:{category}" if category else ""
        else:
            derived_subject, derived_value = _promotion_claim_signature(content)
    subject_value = normalized_subject or _normalize_promotion_text(derived_subject)
    claim_value = normalized_value or _normalize_promotion_claim_value(derived_value)
    if not subject_value or not claim_value:
        return "", ""
    return subject_value, claim_value


def _semantic_sentences(content: str) -> list[str]:
    protected = re.sub(
        r"\b(Inc|Ltd|Co|Corp|Dr|Mr|Mrs|Ms|etc|e\.g|i\.e|U\.S)\.(?=\s|$)",
        lambda match: match.group(1) + "<SEMANTIC_DOT>",
        content,
        flags=re.IGNORECASE,
    )
    return [
        part.replace("<SEMANTIC_DOT>", ".").strip()
        for part in re.split(r"(?<=[。！？])|(?<=[.!?])(?=\s)", protected)
        if part.strip()
    ]


def _semantic_clauses(content: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"(?<=[。！？?!.])\s*|[，,；;]\s*", content)
        if part.strip()
    ]


_SEMANTIC_WH_ACTIVE_LEAD = re.compile(
    r"^(?:what|which|who|whom|whose|why|how|when|where)\s+(?P<body>.+)$",
    re.IGNORECASE,
)


def _semantic_is_wh_active_question(sentence: str) -> bool:
    match = _SEMANTIC_WH_ACTIVE_LEAD.match(sentence.strip())
    if match is None:
        return False
    body = match.group("body")
    if re.search(
        r"\b(?:is|are|was|were|must|should|can|could|will|would|do|does|did|"
        r"has|have|had)\b",
        body,
        re.IGNORECASE,
    ):
        return False
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", body)
    return bool(tokens) and any(_semantic_word_is_predicate(token) for token in tokens)


def _semantic_is_wh_passive_question(sentence: str) -> bool:
    match = _SEMANTIC_WH_PASSIVE_QUESTION.match(sentence)
    if match is None:
        return False
    subject_clause = match.group("subject")
    return re.search(r"\b[a-z]{3,}ed\b", subject_clause.casefold()) is None


def _semantic_sentence_is_question(sentence: str) -> bool:
    stripped = sentence.strip()
    if not stripped:
        return False
    if re.match(r"^May\s+\d{4}\b", stripped, re.IGNORECASE):
        return False
    if _SEMANTIC_QUESTION_MARK.search(stripped):
        return True
    if _SEMANTIC_ZH_WH_CLAUSE.match(stripped):
        # A Chinese wh-clause is a question only when it has no declarative
        # predicate. "是否启用缓存取决于部署环境。" states a fact.
        return _SEMANTIC_ZH_DECLARATIVE.search(stripped) is None
    if _SEMANTIC_WH_CLAUSE_STATEMENT.match(stripped):
        return False
    if _SEMANTIC_WH_SHORT_PASSIVE_QUESTION.match(stripped):
        return True
    if _SEMANTIC_WH_STATEMENT_PREDICATE.match(stripped):
        return False
    if _semantic_is_wh_active_question(stripped):
        return True
    if _semantic_is_wh_passive_question(stripped):
        return True
    return bool(
        _SEMANTIC_YES_NO_QUESTION.search(stripped)
        or _SEMANTIC_WH_QUESTION.search(stripped)
    )


def _semantic_sentence_is_request(sentence: str) -> bool:
    stripped = sentence.strip()
    return bool(stripped and _SEMANTIC_REQUEST.search(stripped))


def _semantic_starts_question(content: str) -> bool:
    stripped = content.strip()
    if not stripped:
        return False
    if re.match(r"^May\s+\d{4}\b", stripped, re.IGNORECASE):
        return False
    if _SEMANTIC_QUESTION_MARK.search(stripped) and re.match(
        r"^(?:what|how|why|which|who|whom|whose|when|where|"
        r"is|are|was|were|am|do|does|did|can|could|would|should|will|"
        r"shall|may|might|must|have|has|had)\b",
        stripped,
        re.IGNORECASE,
    ):
        return True
    if _SEMANTIC_WH_CLAUSE_STATEMENT.match(stripped):
        return False
    if _SEMANTIC_WH_SHORT_PASSIVE_QUESTION.match(stripped):
        return True
    if _SEMANTIC_WH_STATEMENT_PREDICATE.match(stripped):
        return False
    if _SEMANTIC_ZH_WH_CLAUSE.match(stripped):
        return _SEMANTIC_ZH_DECLARATIVE.search(stripped) is None
    if _SEMANTIC_ZH_QUESTION_ANY.search(stripped) and not re.search(r"[，,]", stripped):
        return _SEMANTIC_ZH_DECLARATIVE.search(stripped) is None
    if _SEMANTIC_WH_MODAL_QUESTION.search(stripped):
        return True
    return bool(
        _SEMANTIC_YES_NO_QUESTION.search(stripped)
        or _SEMANTIC_WH_QUESTION.search(stripped)
    )


def _semantic_text_is_question(content: str) -> bool:
    if _semantic_starts_question(content):
        return True
    clauses = _semantic_clauses(content)
    return bool(clauses) and all(_semantic_sentence_is_question(item) for item in clauses)


def _semantic_text_is_request(content: str) -> bool:
    stripped = content.strip()
    if stripped and _SEMANTIC_REQUEST.search(stripped):
        return True
    clauses = _semantic_clauses(content)
    return bool(clauses) and all(_semantic_sentence_is_request(item) for item in clauses)


def _semantic_text_is_imperative(content: str) -> bool:
    stripped = content.strip()
    if not stripped:
        return False
    if _SEMANTIC_IMPERATIVE.match(stripped):
        return True
    return bool(
        re.match(
            r"^(?!(?:the|a|an|this|that|these|those|i|we|you|they|he|she|it|there|"
            r"what|how|why|which|who|whom|whose|when|where)\b)"
            r"[A-Za-z][A-Za-z-]+\s+(?:the|a|an|this|that|these|those)\b",
            stripped,
            re.IGNORECASE,
        )
    )


def _semantic_user_event_is_pure_question(content: str) -> bool:
    sentences = _semantic_sentences(content)
    if len(sentences) > 1:
        return all(_semantic_sentence_is_question(item) for item in sentences)
    if _semantic_starts_question(content):
        return True
    clauses = _semantic_clauses(content)
    return bool(clauses) and all(_semantic_sentence_is_question(item) for item in clauses)


def _semantic_user_event_is_pure_request(content: str) -> bool:
    sentences = _semantic_sentences(content)
    if len(sentences) > 1:
        return all(_semantic_sentence_is_request(item) for item in sentences)
    stripped = content.strip()
    if stripped and _SEMANTIC_REQUEST.search(stripped):
        return True
    clauses = _semantic_clauses(content)
    return bool(clauses) and all(_semantic_sentence_is_request(item) for item in clauses)


def _semantic_word_is_predicate(word: str) -> bool:
    folded = word.casefold().strip("'\u2019")
    if not folded:
        return False
    if folded in _SEMANTIC_AUXILIARIES:
        return True
    if not folded.isalpha() or not folded.islower():
        return False
    if folded.endswith(("ss", "us", "is")):
        return False
    if len(folded) >= 3 and folded.endswith("s"):
        return True
    return len(folded) >= 4 and folded.endswith("ed")


def _semantic_coordination_adds_predicate(sentence: str) -> bool:
    if _SEMANTIC_ZH_COORDINATION.search(sentence):
        return True
    for match in _SEMANTIC_EN_COORDINATION.finditer(sentence):
        tail = _SEMANTIC_CLAUSE_BOUNDARY.split(sentence[match.end() :], maxsplit=1)[0]
        tokens = re.findall(r"[A-Za-z][A-Za-z'\u2019.-]*", tail)
        index = 0
        while index < len(tokens) and tokens[index].casefold() in _SEMANTIC_SUBJECT_CLOSED_CLASS:
            index += 1
        while index < len(tokens) and tokens[index].casefold() in _SEMANTIC_ADVERB_CLOSED_CLASS:
            index += 1
        if index >= len(tokens):
            continue
        candidate = tokens[index]
        if not _semantic_word_is_predicate(candidate):
            continue
        folded = candidate.casefold()
        if folded in _SEMANTIC_AUXILIARIES:
            return True
        following = tokens[index + 1 :]
        if not following:
            # "-s" forms at the end of a clause are usually plural nouns in an
            # object list ("retries and timeouts").
            continue
        if folded in _SEMANTIC_AUXILIARIES:
            return True
        if folded.endswith("ed"):
            return True
        stem = folded[:-1] if folded.endswith("s") else folded
        if (
            folded in _SEMANTIC_DEMONSTRATIVE_BASE_VERBS
            or stem in _SEMANTIC_DEMONSTRATIVE_BASE_VERBS
        ):
            return True
    return False


def _semantic_word_after(text: str, index: int) -> str:
    match = re.match(r"\s*([A-Za-z][A-Za-z'\u2019-]*)", text[index:])
    return "" if match is None else match.group(1)


def _semantic_main_clause(sentence: str) -> str:
    match = _SEMANTIC_EN_SUBORDINATOR.search(sentence)
    main = sentence if match is None else sentence[: match.start()]
    zh_match = _SEMANTIC_ZH_SUBORDINATOR.search(main)
    if zh_match is not None:
        main = main[: zh_match.start()]
    return main


def _semantic_has_unbound_pronoun(content: str) -> bool:
    for sentence in _semantic_sentences(content):
        if re.match(r"^(?:it|they|them)\b", sentence, re.IGNORECASE):
            return True
        if re.match(r"^(?:这|那)", sentence) and not re.match(
            r"^(?:这|那)" + _SEMANTIC_ZH_NOUN_PREFIX, sentence
        ):
            return True
        main = _semantic_main_clause(sentence)
        if _SEMANTIC_EN_PRONOUN.search(main):
            return True
        for match in _SEMANTIC_EN_DEMONSTRATIVE.finditer(main):
            word = _semantic_word_after(main, match.end())
            if not word:
                return True
            folded = word.casefold()
            if folded in _SEMANTIC_AUXILIARIES or folded in _SEMANTIC_SUBJECT_CLOSED_CLASS:
                return True
            if _semantic_word_is_predicate(word):
                rest = re.findall(r"[A-Za-z][A-Za-z'\u2019.-]*", main[match.end() :])
                following = rest[1:] if rest and rest[0].casefold() == folded else []
                demonstrative_noun = bool(
                    folded.endswith("s")
                    and following
                    and (
                        following[0].casefold() in _SEMANTIC_DEMONSTRATIVE_BASE_VERBS
                        or (following[0].endswith("s") and len(following) > 1)
                    )
                )
                if not demonstrative_noun:
                    return True
        if _SEMANTIC_ZH_PRONOUN.search(main):
            return True
    return False


_SEMANTIC_SUPPORT_STOPWORDS = frozenset(
    {
        "the", "a", "an", "this", "that", "these", "those", "user", "users",
        "project", "service", "system", "api", "we", "i", "you", "they", "it",
        "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
        "do", "does", "did", "can", "could", "will", "would", "should", "must",
        "may", "might", "use", "uses", "used", "using", "prefer", "prefers",
        "preferred", "like", "likes", "need", "needs", "required", "requires",
        "for", "with", "from", "into", "and", "or", "but", "to", "of", "in",
        "on", "at", "by", "as", "which", "who", "what",
    }
)


_SEMANTIC_NEGATION = re.compile(
    r"\b(?:not|never|no|cannot|can't|couldn't|didn't|doesn't|don't|"
    r"isn't|aren't|wasn't|weren't|won't|wouldn't|shouldn't|mustn't)\b|"
    r"不(?:使用|需要|偏好|支持|采用|包含|运行|是|有|会|允许|要求|记录|"
    r"部署|迁移|工作|正常)|没有|尚未|未曾|并未|无法|不需要|不用|不偏好",
    re.IGNORECASE,
)
_SEMANTIC_CJK_RUN = re.compile(r"[\u4e00-\u9fff]{2,}")
_SEMANTIC_SUPPORT_ALIASES = {
    "postgres": "postgresql",
    "pg": "postgresql",
    "k8s": "kubernetes",
    "golang": "go",
    "mongo": "mongodb",
    "nodejs": "node",
    "node.js": "node",
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
}


def _normalize_semantic_support_text(text: str) -> str:
    normalized = _normalize_promotion_text(text)
    for alias, canonical in _SEMANTIC_SUPPORT_ALIASES.items():
        normalized = re.sub(
            rf"\b{re.escape(alias)}\b",
            canonical,
            normalized,
            flags=re.IGNORECASE,
        )
    return normalized


def _semantic_clause_containing_evidence(text: str, evidence: str) -> str:
    start = text.find(evidence)
    if start < 0:
        return ""
    end = start + len(evidence)
    boundary = re.compile(r"[.!?;；。！？]")
    previous = max((match.end() for match in boundary.finditer(text, 0, start)), default=0)
    following = boundary.search(text, end)
    return text[previous : following.start() if following else len(text)].strip()


_SEMANTIC_PROPER_NOUN_SKIP = frozenset(
    {
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "i",
        "we",
        "you",
        "they",
        "he",
        "she",
        "it",
        "my",
        "our",
        "your",
        "their",
        "there",
        "may",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "how",
        "why",
    }
)


def _semantic_proper_noun_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9.-]+\b", text):
        word = match.group(0)
        if word.casefold() in _SEMANTIC_PROPER_NOUN_SKIP:
            continue
        terms.update(_semantic_support_terms(word))
    return terms


def _semantic_preposition_terms(text: str) -> dict[str, set[str]]:
    normalized = _normalize_semantic_support_text(text)
    phrases: dict[str, set[str]] = {}
    pattern = re.compile(
        r"\b(on|in|to|for|with|about|from|into|over|under|around|by|at|per|via|as|of|"
        r"during|across|within|between|through|toward|against)"
        r"\s+([a-z0-9][a-z0-9 '\-]*)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(normalized):
        boundary = re.split(
            r"\b(?:and|or|but|because|which|who|that)\b|[,.!?;]",
            match.group(2),
            maxsplit=1,
        )[0]
        terms = _semantic_support_terms(boundary)
        if terms:
            phrases.setdefault(match.group(1).casefold(), set()).update(terms)
    return phrases


def _semantic_support_terms(content: str) -> set[str]:
    normalized = _normalize_semantic_support_text(content)
    terms = {
        word
        for word in re.findall(r"[a-z0-9]+", normalized)
        if len(word) >= 2 and word not in _SEMANTIC_SUPPORT_STOPWORDS
    }
    for run in _SEMANTIC_CJK_RUN.findall(normalized):
        terms.update(run[index : index + 2] for index in range(len(run) - 1))
    return terms


_SEMANTIC_UNSUPPORTED_QUALIFIER_TERMS = frozenset(
    {
        "again",
        "all",
        "almost",
        "also",
        "always",
        "any",
        "currently",
        "each",
        "even",
        "ever",
        "every",
        "exclusively",
        "just",
        "mainly",
        "merely",
        "most",
        "nearly",
        "never",
        "now",
        "once",
        "only",
        "presently",
        "primarily",
        "quite",
        "rarely",
        "rather",
        "simply",
        "solely",
        "sometimes",
        "still",
        "strictly",
        "too",
        "usually",
        "very",
        "yet",
    }
)


_SEMANTIC_SUPPORT_BRIDGE_TERMS = frozenset(
    {
        "happen",
        "happened",
        "happening",
        "happens",
        "matter",
        "mattered",
        "mattering",
        "matters",
        "still",
    }
)

_SEMANTIC_SUPPORT_PARAPHRASE_TERMS = frozenset(
    {
        "choice",
        "deployment",
        "documentation",
        "process",
    }
)


def _semantic_support_term_is_present(term: str, evidence_terms: set[str]) -> bool:
    if term in evidence_terms:
        return True
    if term.endswith("ies") and term[:-3] + "y" in evidence_terms:
        return True
    if term.endswith("es") and term[:-2] in evidence_terms:
        return True
    if term.endswith("d") and term[:-1] in evidence_terms:
        return True
    if term.endswith("s") and term[:-1] in evidence_terms:
        return True
    return term + "s" in evidence_terms or term + "d" in evidence_terms


def _semantic_evidence_supports_content(
    content: str,
    evidence: str,
    *,
    claim_value: str = "",
    context: str = "",
) -> bool:
    normalized_evidence = _normalize_semantic_support_text(evidence)
    terms = _semantic_support_terms(content)
    if not terms:
        compact_content = re.sub(r"[\W_]+", "", _normalize_semantic_support_text(content))
        compact_evidence = re.sub(r"[\W_]+", "", normalized_evidence)
        return bool(compact_content) and compact_content in compact_evidence
    numeric_terms = {term for term in terms if term.isdigit()}
    if any(term not in normalized_evidence for term in numeric_terms):
        return False
    semantic_terms = terms - numeric_terms
    if semantic_terms and not any(term in normalized_evidence for term in semantic_terms):
        return False
    value_terms = _semantic_support_terms(
        re.sub(r"\bwithout\b.*$", "", claim_value, flags=re.IGNORECASE)
    ) & terms
    if any(term not in normalized_evidence for term in value_terms):
        return False
    content_prepositions = _semantic_preposition_terms(content)
    evidence_prepositions = _semantic_preposition_terms(evidence)
    if not set(content_prepositions) <= set(evidence_prepositions):
        return False
    for preposition, phrase_terms in content_prepositions.items():
        evidence_terms = evidence_prepositions.get(preposition, set())
        if not phrase_terms <= evidence_terms:
            return False
    content_terms_without_inference = _semantic_support_terms(
        re.sub(r"\bwithout\b.*$", "", content, flags=re.IGNORECASE)
    )
    evidence_terms = _semantic_support_terms(evidence)
    context_terms = _semantic_support_terms(context or evidence)
    unsupported_terms = {
        term
        for term in content_terms_without_inference
        if term.isascii()
        and not term.isdigit()
        and term not in _SEMANTIC_SUPPORT_PARAPHRASE_TERMS
        and not _semantic_support_term_is_present(term, evidence_terms)
        and not _semantic_support_term_is_present(term, context_terms)
    }
    if any(_semantic_value_dimension(term) for term in unsupported_terms):
        return False
    if any(
        term in _SEMANTIC_UNSUPPORTED_QUALIFIER_TERMS
        or (len(term) > 4 and term.endswith("ly"))
        for term in unsupported_terms
    ):
        return False
    if unsupported_terms:
        return False
    proper_nouns = _semantic_proper_noun_terms(content)
    if any(noun not in normalized_evidence for noun in proper_nouns):
        return False
    content_words = set(re.findall(r"[a-z0-9_-]+", _normalize_semantic_support_text(content)))
    evidence_words = set(re.findall(r"[a-z0-9_-]+", normalized_evidence))
    if any(
        _semantic_values_are_opposites(left, right)
        for left in content_words
        for right in evidence_words
    ):
        return False
    content_negated = bool(_SEMANTIC_NEGATION.search(content))
    evidence_negated = bool(_SEMANTIC_NEGATION.search(evidence))
    return content_negated == evidence_negated


def _semantic_content_is_atomic(content: str) -> bool:
    sentences = _semantic_sentences(content)
    clauses = [
        part.strip()
        for part in re.split(
            r"[;；\n]|(?<!\d):(?![/\\\d])|：|"
            r"(?<!\d)(?<!\d\s)[—–](?!\d)(?!\s\d)|\s[—-]\s",
            content,
        )
        if part.strip()
    ]
    compound = _SEMANTIC_COMPOUND_CLAIM.search(content)
    coordinated_modifier = _SEMANTIC_COORDINATED_MODIFIER.fullmatch(content.strip())
    return (
        len(sentences) <= 1
        and len(clauses) <= 1
        and not (compound and coordinated_modifier is None)
        and not _SEMANTIC_COORDINATED_PREDICATE.search(content)
        and not _SEMANTIC_SUBORDINATING_CLAIM.search(content)
        and not _semantic_coordination_adds_predicate(content)
    )


def _metadata_text(unit: MemoryUnit, key: str) -> str:
    value = unit.user_metadata.get(key)
    return "" if value is None else str(value)


def _metadata_unit_region_map(unit: MemoryUnit) -> dict[str, str]:
    raw = unit.user_metadata.get("source_unit_regions")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if not isinstance(raw, list):
        return {}
    mapping: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        unit_id = str(item.get("unit_id") or "").strip()
        region_id = str(item.get("region_id") or "").strip()
        if unit_id and region_id:
            mapping[unit_id] = region_id
    return mapping


def _metadata_int(unit: MemoryUnit, key: str) -> int:
    value = unit.user_metadata.get(key)
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return -1


def _chunk_evolved_content(content: str) -> list[str]:
    remaining = content
    chunks: list[str] = []
    while remaining:
        if (
            len(remaining) <= _FULLTEXT_CONTENT_LIMIT
            and len(remaining.encode("utf-8")) <= _EVOLVE_CHUNK_LIMIT
        ):
            chunks.append(remaining)
            break
        low, high = 1, len(remaining)
        while low < high:
            middle = (low + high + 1) // 2
            candidate = remaining[:middle]
            if (
                len(candidate) <= _FULLTEXT_CONTENT_LIMIT
                and len(candidate.encode("utf-8")) <= _EVOLVE_CHUNK_LIMIT
            ):
                low = middle
            else:
                high = middle - 1
        boundary = max(
            remaining.rfind("\n\n", 0, low),
            remaining.rfind("\n", 0, low),
            remaining.rfind(" ", 0, low),
        )
        cut = boundary + 1 if boundary > low // 2 else low
        chunks.append(remaining[:cut])
        remaining = remaining[cut:]
    return chunks or [""]


def _group_units_by_scope(
    items: list[tuple[Scope, MemoryUnit]],
) -> list[tuple[Scope, list[MemoryUnit]]]:
    grouped: dict[tuple[str, str, str, str, str], tuple[Scope, list[MemoryUnit]]] = {}
    for scope, unit in items:
        key = (scope.org, scope.space, scope.user, scope.agent, scope.session)
        grouped.setdefault(key, (scope, []))[1].append(unit)
    return list(grouped.values())


def _event_order(kind: str) -> int:
    return 0 if kind == "user" else 1


def _event_label(kind: str) -> str:
    return "用户" if kind == "user" else "助手"


def _capture_event_sort_key(event: Mapping[str, object]) -> tuple[datetime, int, datetime, str]:
    occurred = _parse_timestamp(str(event.get("occurred_at", "")))
    received = _parse_timestamp(str(event.get("received_at", "")))
    sort_time = occurred or received or datetime.min.replace(tzinfo=UTC)
    tie_time = received or occurred or datetime.min.replace(tzinfo=UTC)
    return (
        sort_time,
        _event_order(str(event.get("event_kind", ""))),
        tie_time,
        str(event.get("event_id", "")),
    )


def _capture_round_sort_key(item: Mapping[str, object]) -> tuple[datetime, str, str, str]:
    started = _parse_timestamp(str(item.get("started_at", "")))
    received = _parse_timestamp(str(item.get("received_at", "")))
    sort_time = started or received or datetime.min.replace(tzinfo=UTC)
    return (
        sort_time,
        str(item.get("region_id", "")),
        str(item.get("session_id", "")),
        str(item.get("turn_id", "")),
    )


def _normalize_timeline_timestamp(value: object) -> str:
    parsed = _parse_timestamp("" if value is None else str(value))
    return "" if parsed is None else parsed.isoformat()


def _latest_timeline_timestamp(*values: object) -> str:
    normalized = [
        timestamp
        for value in values
        if (timestamp := _normalize_timeline_timestamp(value))
    ]
    return max(normalized, default="")


def _evolution_failure_text(error: Exception) -> str:
    detail = str(error).strip()
    return f"{type(error).__name__}: {detail}" if detail else type(error).__name__


def _capture_round_fingerprint(events: list[dict[str, object]]) -> str:
    ordered = sorted(
        events,
        key=lambda event: (
            _event_order(str(event.get("event_kind", ""))),
            str(event.get("occurred_at", "")),
            str(event.get("unit_id", "")),
        ),
    )
    return _source_fingerprint([str(event.get("unit_id", "")) for event in ordered])


def _capture_units_timeline(units: list[MemoryUnit]) -> tuple[str, str, str]:
    occurred_at: list[datetime] = []
    received_at: list[datetime] = []
    for unit in units:
        occurred = _parse_timestamp(_metadata_text(unit, "occurred_at"))
        if occurred is not None:
            occurred_at.append(occurred)
        received = (
            unit.temporal.t_ingest
            or unit.temporal.t_valid
            or unit.temporal.t_message
        )
        if received is not None:
            received_at.append(
                received.replace(tzinfo=UTC)
                if received.tzinfo is None
                else received.astimezone(UTC)
            )
    started = min(occurred_at) if occurred_at else min(received_at, default=None)
    received_earliest = min(received_at, default=None)
    received_latest = max(received_at, default=None)
    return (
        "" if started is None else started.isoformat(),
        "" if received_earliest is None else received_earliest.isoformat(),
        "" if received_latest is None else received_latest.isoformat(),
    )


def _capture_events_timeline(
    events: list[dict[str, object]],
) -> tuple[str, str, str]:
    occurred_at = [
        parsed
        for event in events
        if (parsed := _parse_timestamp(str(event.get("occurred_at", "")))) is not None
    ]
    received_at = [
        parsed
        for event in events
        if (parsed := _parse_timestamp(str(event.get("received_at", "")))) is not None
    ]
    started = min(occurred_at) if occurred_at else min(received_at, default=None)
    received_earliest = min(received_at, default=None)
    received_latest = max(received_at, default=None)
    return (
        "" if started is None else started.isoformat(),
        "" if received_earliest is None else received_earliest.isoformat(),
        "" if received_latest is None else received_latest.isoformat(),
    )


def _json_string_list(value: object) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if str(item).strip()]


def _json_object_list(value: object) -> list[dict[str, object]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, (list, tuple)):
        return []
    return [
        {str(key): item_value for key, item_value in item.items()}
        for item in value
        if isinstance(item, Mapping)
    ]


def _unique_object_dicts(values: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    result: list[dict[str, object]] = []
    for value in values:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if encoded in seen:
            continue
        seen.add(encoded)
        result.append(value)
    return result


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    except (ValueError, OverflowError):
        return None


def _audit_sort_key(item: dict[str, str]) -> datetime:
    parsed = _parse_timestamp(item.get("occurred_at", ""))
    return parsed if parsed is not None else datetime.min.replace(tzinfo=UTC)


def _write_request_hash(request: CreateMemoryRequest) -> str:
    payload: dict[str, object] = {
        "library_id": request.library_id,
        "region_id": request.region_id,
        "content": request.content,
        "memory_type": request.memory_type,
        "source_references": list(request.source_references),
        "actor": request.actor,
    }
    if request.user_metadata:
        payload["user_metadata"] = dict(request.user_metadata)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _write_response(
    request: MemoryWriteRequest, memory_id: str, *, status: str
) -> dict[str, object]:
    return {
        "id": memory_id,
        "memory_id": memory_id,
        "status": status,
        "library_id": getattr(request, "library_id", ""),
        "region_id": getattr(request, "region_id", None),
        "content": request.content,
        "memory_type": getattr(request, "memory_type", "capture"),
    }


def _scope_for_normalized_library(library_id: str) -> Scope:
    return Scope(
        org=_LOCAL_ORG,
        space=f"{LIBRARY_SCOPE_PREFIX}{library_id}",
        user=_LOCAL_USER,
        agent=_LOCAL_AGENT,
        session="",
    )


def _space_for_region(region: MemoryRegion) -> str:
    return region.space


def _scope_for_region(
    region: MemoryRegion,
    *,
    session_id: str | None = None,
    agent: str = _LOCAL_AGENT,
) -> Scope:
    return Scope(
        org=_LOCAL_ORG,
        space=region.space,
        user=_LOCAL_USER,
        agent=agent,
        session=(region.session_id or "") if session_id is None else session_id,
    )


def _scope_for_synthetic_cwd(cwd: str, session_id: str | None) -> Scope:
    digest = hashlib.sha256(cwd.encode("utf-8")).hexdigest()[:24]
    return Scope(
        org=_LOCAL_ORG,
        space=f"{CWD_SCOPE_PREFIX}{digest}",
        user=_LOCAL_USER,
        agent=_LOCAL_AGENT,
        session=session_id or "",
    )


def _security_for(scope: Scope) -> RequestSecurityContext:
    return legacy_request_context(
        Scope(
            org=scope.org,
            user=scope.user,
            agent=scope.agent,
            session=scope.session,
        )
    )


def _classify_failure(error: Exception, *, embedding_configured: bool) -> str:
    text = f"{type(error).__module__} {type(error).__name__} {error}".casefold()
    if embedding_configured or "embed" in text or "model" in text or "openai" in text:
        return "model_unavailable"
    if "fulltext" in text or "keyword" in text or "bm25" in text:
        return "fulltext_unavailable"
    if "vector" in text or "milvus" in text:
        return "vector_unavailable"
    return "retrieval_unavailable"


def _channel_failure(channel: str, error: Any) -> str:
    text = f"{getattr(error, 'error_type', '')} {getattr(error, 'message', '')}".casefold()
    if channel == "vector" and ("embed" in text or "model" in text):
        return "model_unavailable"
    if channel == "keyword":
        return "fulltext_unavailable"
    if channel == "vector":
        return "vector_unavailable"
    if channel == "graph":
        return "graph_unavailable"
    return f"{channel}_unavailable"


def _combined_index_status(statuses: set[str]) -> str:
    if "degraded" in statuses:
        return "degraded"
    if "ready" in statuses:
        return "ready"
    return "not_configured"


def _search_response(
    request: SearchMemoryRequest,
    scope: Scope,
    items: list[Any],
    errors: list[Any],
    degradation: list[str],
    *,
    vector_status: str,
    graph_status: str = "not_configured",
    region: MemoryRegion | None = None,
) -> dict[str, object]:
    visible_items = [
        item
        for item in items
        if not (isinstance(item, MemoryUnit) and item.user_metadata.get("restore_pending") is True)
        and _is_default_recall_item(item)
    ]
    results = [_result_payload(item, request) for item in visible_items]
    if region is not None:
        for result in results:
            result["region_id"] = region.region_id
            result["space"] = region.space
    scope_kind = (
        "explicit_region"
        if request.region_id is not None
        else "explicit_library"
        if request.library_id is not None
        else "project_cwd"
    )
    response: dict[str, object] = {
        "status": "bound",
        "scope": {
            "kind": scope_kind,
            "cwd": request.cwd,
            "library_id": request.library_id,
            "region_id": request.region_id,
            "space": scope.space,
        },
        "query": request.query,
        "degraded": bool(degradation),
        "degradation": degradation,
        "vector_index_status": vector_status,
        "graph_index_status": graph_status,
        "results": _truncate_search_results(results, max(1, min(request.limit, 100))),
    }
    if request.library_id is not None:
        response["library_id"] = request.library_id
    if request.region_id is not None:
        response["region_id"] = request.region_id
    if region is not None:
        response["region_id"] = region.region_id
        response["space"] = region.space
    if errors:
        response["retrieval_errors"] = [
            {
                "channel": getattr(error.channel, "value", str(error.channel)),
                "source": error.source,
                "error_type": error.error_type,
                "message": error.message,
            }
            for error in errors
        ]
    return response


def _json_metadata(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_metadata(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_metadata(item) for key, item in value.items()}
    return str(value)


def _unit_revision(unit: MemoryUnit) -> str:
    payload = {
        "id": unit.id,
        "content": unit.content,
        "lifecycle": unit.lifecycle.value,
        "supersedes": unit.supersedes,
        "provenance": list(unit.provenance),
        "system_metadata": _json_metadata(dict(unit.system_metadata)),
        "user_metadata": _json_metadata(dict(unit.user_metadata)),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _unit_payload(unit: MemoryUnit, region: MemoryRegion) -> dict[str, object]:
    user_metadata = dict(unit.user_metadata)
    source_references = user_metadata.get("source_references")
    provenance = list(unit.provenance)
    if not provenance:
        metadata_provenance = user_metadata.get("provenance")
        if not isinstance(metadata_provenance, list):
            metadata_provenance = user_metadata.get("source_unit_ids")
        if isinstance(metadata_provenance, list):
            provenance = [str(value) for value in metadata_provenance]
    if not isinstance(source_references, list):
        source_references = provenance
    normalized_source_references = [str(value) for value in source_references]
    memory_type = _metadata_text(unit, "memory_type") or "unknown"
    revision = _unit_revision(unit)
    ingested = unit.temporal.t_ingest or unit.temporal.t_valid or unit.temporal.t_message
    updated_candidates = [
        value
        for value in (
            unit.temporal.t_ingest,
            _parse_timestamp(str(user_metadata.get("governance_updated_at", ""))),
            (
                unit.temporal.t_invalid
                if unit.lifecycle.value in {"superseded", "forgotten", "archived"}
                else None
            ),
        )
        if value is not None
    ]
    updated = max(updated_candidates) if updated_candidates else None
    deleted = unit.temporal.t_invalid if unit.lifecycle.value == "forgotten" else None
    return {
        "id": unit.id,
        "path": unit.id,
        "document_id": unit.id,
        "content": unit.content,
        "summary": _memory_summary(unit.content),
        "memory_type": memory_type,
        "source_count": len(set(normalized_source_references)),
        "metadata": {
            "system": _json_metadata(dict(unit.system_metadata)),
            "user": _json_metadata(user_metadata),
        },
        "system_metadata": _json_metadata(dict(unit.system_metadata)),
        "user_metadata": _json_metadata(user_metadata),
        "lifecycle": unit.lifecycle.value,
        "tier": unit.tier.value,
        "version": revision,
        "revision": revision,
        "source_version": revision,
        "source": unit.source_ref or "mem2",
        "source_references": _json_metadata(normalized_source_references),
        "provenance": provenance,
        "supersedes": unit.supersedes,
        "region_id": region.region_id,
        "library_id": region.region_id,
        "space": region.space,
        "scope": {
            "org": unit.scope.org,
            "space": unit.scope.space,
            "user": unit.scope.user,
            "agent": unit.scope.agent,
            "session": unit.scope.session,
        },
        "created_at": ingested.isoformat() if ingested is not None else "",
        "updated_at": updated.isoformat() if updated else "",
        "deleted_at": deleted.isoformat() if deleted else "",
        "valid_from": unit.temporal.t_valid.isoformat() if unit.temporal.t_valid else "",
        "valid_until": unit.temporal.t_invalid.isoformat() if unit.temporal.t_invalid else "",
        "active": unit.lifecycle.value == "active",
    }


def _memory_summary(content: str, limit: int = 160) -> str:
    normalized = " ".join(content.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit].rstrip()}…"


def _is_formal_semantic_unit(unit: MemoryUnit) -> bool:
    if _is_capture_unit(unit):
        return False
    memory_type = _metadata_text(unit, "memory_type")
    return memory_type in FORMAL_MEMORY_TYPES and unit.lifecycle.value == "active"


def _memory_sort_key(payload: Mapping[str, object]) -> tuple[datetime, str]:
    updated = _parse_timestamp(str(payload.get("updated_at", "")))
    return (
        updated if updated is not None else datetime.min.replace(tzinfo=UTC),
        str(payload.get("id", "")),
    )


def _memory_list_item(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        "id": str(payload.get("id", "")),
        "path": str(payload.get("path", "")),
        "region_id": str(payload.get("region_id", "")),
        "summary": str(payload.get("summary", "")),
        "memory_type": str(payload.get("memory_type", "")),
        "lifecycle": str(payload.get("lifecycle", "")),
        "updated_at": str(payload.get("updated_at", "")),
        "source_count": int(str(payload.get("source_count", 0))),
        "version": str(payload.get("version", "")),
    }


def _encode_memory_cursor_rows(row: tuple[object, ...]) -> str:
    updated = _parse_timestamp(str(row[5]))
    value = {
        "v": 1,
        "updated_at": updated.isoformat() if updated is not None else "",
        "id": str(row[0]),
    }
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return base64.urlsafe_b64encode(encoded.encode("utf-8")).decode("ascii").rstrip("=")


def _encode_memory_cursor(payload: Mapping[str, object]) -> str:
    updated = _parse_timestamp(str(payload.get("updated_at", "")))
    value = {
        "v": 1,
        "updated_at": updated.isoformat() if updated is not None else "",
        "id": str(payload.get("id", "")),
    }
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return base64.urlsafe_b64encode(encoded.encode("utf-8")).decode("ascii").rstrip("=")


def _decode_memory_cursor(cursor: str) -> tuple[datetime, str]:
    try:
        padding = "=" * (-len(cursor) % 4)
        decoded = base64.urlsafe_b64decode(f"{cursor}{padding}".encode("ascii"))
        value = json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MemoryGovernanceError("invalid memory cursor") from error
    if not isinstance(value, dict) or value.get("v") != 1:
        raise MemoryGovernanceError("invalid memory cursor")
    unit_id = value.get("id")
    updated_at = value.get("updated_at")
    if not isinstance(unit_id, str) or not unit_id or not isinstance(updated_at, str):
        raise MemoryGovernanceError("invalid memory cursor")
    updated = _parse_timestamp(updated_at)
    if updated is None:
        raise MemoryGovernanceError("invalid memory cursor")
    return updated, unit_id


def _unit_sort_key(pair: tuple[Scope, MemoryUnit]) -> tuple[str, str]:
    _, unit = pair
    timestamp = (
        unit.temporal.t_ingest
        or unit.temporal.t_valid
        or unit.temporal.t_message
        or unit.temporal.t_event
    )
    return (timestamp.isoformat() if timestamp is not None else "", unit.id)


def _unit_logical_memory_id(unit: MemoryUnit) -> str:
    return _metadata_text(unit, "logical_memory_id") or unit.id


def _logical_memory_id_map(units: list[MemoryUnit]) -> dict[str, str]:
    """Resolve stable lineage IDs for a bounded set of versions."""
    ids = {unit.id for unit in units}
    parents = {unit_id: unit_id for unit_id in ids}

    def find(unit_id: str) -> str:
        root = unit_id
        while parents[root] != root:
            root = parents[root]
        while parents[unit_id] != unit_id:
            parent = parents[unit_id]
            parents[unit_id] = root
            unit_id = parent
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for unit in units:
        if unit.supersedes and unit.supersedes in ids:
            union(unit.id, unit.supersedes)

    components: dict[str, list[MemoryUnit]] = {}
    for unit in units:
        components.setdefault(find(unit.id), []).append(unit)

    result: dict[str, str] = {}
    for members in components.values():
        explicit = sorted(
            {
                lineage
                for unit in members
                if (lineage := _metadata_text(unit, "logical_memory_id"))
            }
        )
        lineage_id = explicit[0] if explicit else min(unit.id for unit in members)
        for unit in members:
            result[unit.id] = lineage_id
    return result


def _version_chain(
    unit: MemoryUnit, pairs: list[tuple[Scope, MemoryUnit]]
) -> list[tuple[Scope, MemoryUnit]]:
    ids = {unit.id}
    changed = True
    while changed:
        changed = False
        for _, candidate in pairs:
            additions = set()
            if candidate.id in ids and candidate.supersedes:
                additions.add(candidate.supersedes)
            if candidate.supersedes in ids:
                additions.add(candidate.id)
            new_ids = additions - ids
            if new_ids:
                ids.update(new_ids)
                changed = True
    return sorted((pair for pair in pairs if pair[1].id in ids), key=_unit_sort_key)


def _latest_chain_version(units: list[MemoryUnit]) -> MemoryUnit:
    superseded_ids = {unit.supersedes for unit in units if unit.supersedes}
    terminals = [unit for unit in units if unit.id not in superseded_ids]
    return (terminals or units)[-1]


def _require_revision(unit: MemoryUnit, expected: str) -> None:
    if not expected or expected != _unit_revision(unit):
        raise MemoryRevisionConflictError("memory revision conflict")


def _require_active(unit: MemoryUnit, operation: str) -> None:
    if unit.lifecycle.value != "active":
        raise MemoryLifecycleConflictError(
            f"memory lifecycle {unit.lifecycle.value} does not allow {operation}"
        )


def _require_delete_lifecycle(unit: MemoryUnit, mode: str) -> None:
    allowed = {
        "forget": {"active", "archived", "superseded"},
        "archive": {"active", "archived"},
        "downweight": {"active"},
        "purge": {"active", "archived", "superseded", "forgotten"},
    }
    if unit.lifecycle.value not in allowed[mode]:
        raise MemoryLifecycleConflictError(
            f"memory lifecycle {unit.lifecycle.value} does not allow delete mode {mode}"
        )


def _content_diff(before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="before",
            tofile="after",
        )
    )


def _truncate_search_results(
    results: list[dict[str, object]], limit: int
) -> list[dict[str, object]]:
    direct = [item for item in results if item.get("classification") != "graph_expansion"]
    graph = [item for item in results if item.get("classification") == "graph_expansion"]
    if limit <= 1:
        return (direct or graph)[: max(limit, 0)]
    if not graph:
        return direct[:limit]
    graph_slots = min(len(graph), max(1, limit // 5))
    if limit > 1:
        graph_slots = min(graph_slots, limit - 1)
    direct_slots = max(0, limit - graph_slots)
    selected = direct[:direct_slots] + graph[:graph_slots]
    remaining = limit - len(selected)
    if remaining > 0:
        selected.extend(direct[direct_slots:][:remaining])
        remaining = limit - len(selected)
    if remaining > 0:
        selected.extend(graph[graph_slots:][:remaining])
    return selected[:limit]


def _result_payload(item: Any, request: SearchMemoryRequest) -> dict[str, object]:
    if isinstance(item, dict):
        return {str(key): value for key, value in item.items()}
    return _result_item(item, request)


def _result_source_references(item: Any) -> list[str]:
    if isinstance(item, dict):
        metadata = item.get("user_metadata")
        references = item.get("source_references")
        provenance = item.get("provenance")
    else:
        metadata = getattr(item, "user_metadata", None)
        references = None
        provenance = getattr(item, "provenance", None)
    if not isinstance(references, list):
        references = metadata.get("source_references") if isinstance(metadata, dict) else None
    if not isinstance(references, list):
        references = metadata.get("provenance") if isinstance(metadata, dict) else None
    if not isinstance(references, list):
        references = provenance
    if not isinstance(references, (list, tuple)):
        return []
    return list(dict.fromkeys(str(value) for value in references if str(value).strip()))


def _result_item(item: Any, request: SearchMemoryRequest) -> dict[str, object]:
    if isinstance(item, MemoryUnit):
        unit_id = item.id
        content = item.content
        score = 0.0
        system_metadata = dict(item.system_metadata)
        user_metadata = dict(item.user_metadata)
    else:
        unit_id = str(item.unit_id)
        content = str(item.content)
        score = float(item.score)
        system_metadata = dict(item.system_metadata)
        user_metadata = dict(item.user_metadata)
    return {
        "chunk_id": unit_id,
        "document_id": unit_id,
        "content": content,
        "library_id": request.region_id or request.library_id or "",
        "path": f"memory://{unit_id}",
        "heading": None,
        "start_line": 1,
        "end_line": max(1, content.count("\n") + 1),
        "source_version": unit_id,
        "source_type": "mem2",
        "classification": "direct",
        "score": score,
        "retrieval_sources": ["hybrid"],
        "source_references": _result_source_references(item),
        "provenance": list(item.provenance) if isinstance(item, MemoryUnit) else [],
        "system_metadata": system_metadata,
        "user_metadata": user_metadata,
    }


def _metadata_strings(unit: MemoryUnit, key: str) -> list[str]:
    value = unit.user_metadata.get(key)
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if str(item).strip()]


def _string_list_value(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_promotion_text(content: str) -> str:
    return " ".join(content.casefold().split())


def _promotion_key(content: str, target_region_id: str) -> str:
    return hashlib.sha256(
        json.dumps(
            [_normalize_promotion_text(content), target_region_id],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _promotion_source_references(
    unit: MemoryUnit,
    source_units: tuple[MemoryUnit, ...],
    source_region: MemoryRegion,
) -> list[str]:
    references = [*_metadata_strings(unit, "source_references"), f"unit:{unit.id}"]
    for source in source_units:
        references.append(f"unit:{source.id}")
        event_id = _metadata_text(source, "event_id")
        if event_id:
            references.append(f"capture:{source_region.region_id}:{event_id}")
    return list(dict.fromkeys(references))


_PREFERENCE_CATEGORY_WORDS = {
    "tab": "indentation",
    "tabs": "indentation",
    "space": "indentation",
    "spaces": "indentation",
    "制表符": "indentation",
    "空格": "indentation",
    "dark": "theme",
    "light": "theme",
    "深色": "theme",
    "浅色": "theme",
    "concise": "verbosity",
    "detailed": "verbosity",
    "verbose": "verbosity",
    "简洁": "verbosity",
    "详细": "verbosity",
    "large": "size",
    "small": "size",
    "compact": "size",
    "大": "size",
    "小": "size",
    "blue": "color",
    "green": "color",
    "red": "color",
    "yellow": "color",
    "蓝": "color",
    "绿": "color",
    "红": "color",
    "黄": "color",
}


def _preference_category(value: str) -> str:
    folded = value.casefold()
    words = re.findall(r"[a-z0-9_.-]+", folded)
    for word in reversed(words):
        if word in _PREFERENCE_CATEGORY_WORDS:
            return _PREFERENCE_CATEGORY_WORDS[word]
    for term, category in _PREFERENCE_CATEGORY_WORDS.items():
        if term in folded:
            return category
    return words[-1] if words else ""


def _promotion_claim_signature(content: str) -> tuple[str, str]:
    text = _normalize_promotion_text(content)
    match = re.match(r"^(?:i|we)\s+(prefer|like|use|need|want)\s+(.+)$", text)
    if match:
        value = match.group(2).strip().rstrip(".!。")
        subject_noun = _preference_category(value)
        return (
            f"user:{match.group(1)}:{subject_noun}",
            value,
        ) if subject_noun else ("", "")
    match = re.match(r"^我(偏好|喜欢|习惯|通常|希望|一直)(.+)$", text)
    if match:
        value = match.group(2).strip().rstrip(".!。")
        subject_noun = _preference_category(value)
        return (
            f"user:preference:{subject_noun}",
            value,
        ) if subject_noun else ("", "")
    match = re.match(r"^(.+?)\s+(?:is|are|means|requires|uses)\s+(.+)$", text)
    if match:
        return (
            match.group(1).strip(),
            match.group(2).strip().rstrip(".!。"),
        )
    match = re.match(r"^(.+?)(是|为|使用|需要)(.+)$", text)
    if match:
        return (
            match.group(1).strip(),
            match.group(3).strip().rstrip(".!。"),
        )
    return "", ""


_SEMANTIC_VALUE_ANTONYMS = {
    frozenset(("enabled", "disabled")),
    frozenset(("enable", "disable")),
    frozenset(("allow", "deny")),
    frozenset(("allow", "forbid")),
    frozenset(("dark", "light")),
    frozenset(("true", "false")),
    frozenset(("on", "off")),
    frozenset(("yes", "no")),
    frozenset(("required", "optional")),
    frozenset(("internal", "external")),
    frozenset(("private", "public")),
    frozenset(("read-only", "read-write")),
    frozenset(("synchronous", "asynchronous")),
    frozenset(("sync", "async")),
    frozenset(("primary", "secondary")),
    frozenset(("include", "exclude")),
    frozenset(("encrypt", "decrypt")),
    frozenset(("local", "remote")),
}


_SEMANTIC_VALUE_DIMENSIONS: dict[str, str] = {
    "hour": "frequency:hour",
    "hourly": "frequency:hour",
    "every hour": "frequency:hour",
    "per hour": "frequency:hour",
    "每小时": "frequency:hour",
    "day": "frequency:day",
    "daily": "frequency:day",
    "every day": "frequency:day",
    "per day": "frequency:day",
    "每天": "frequency:day",
    "night": "frequency:night",
    "nightly": "frequency:night",
    "每晚": "frequency:night",
    "week": "frequency:week",
    "weekly": "frequency:week",
    "every week": "frequency:week",
    "per week": "frequency:week",
    "每周": "frequency:week",
    "biweek": "frequency:biweek",
    "biweekly": "frequency:biweek",
    "fortnight": "frequency:biweek",
    "fortnightly": "frequency:biweek",
    "every two weeks": "frequency:biweek",
    "每两周": "frequency:biweek",
    "month": "frequency:month",
    "monthly": "frequency:month",
    "every month": "frequency:month",
    "per month": "frequency:month",
    "每月": "frequency:month",
    "quarter": "frequency:quarter",
    "quarterly": "frequency:quarter",
    "every quarter": "frequency:quarter",
    "每季度": "frequency:quarter",
    "half-year": "frequency:halfyear",
    "halfyear": "frequency:halfyear",
    "semi-annual": "frequency:halfyear",
    "semi-annually": "frequency:halfyear",
    "semiannual": "frequency:halfyear",
    "semiannually": "frequency:halfyear",
    "每半年": "frequency:halfyear",
    "year": "frequency:year",
    "yearly": "frequency:year",
    "annual": "frequency:year",
    "annually": "frequency:year",
    "every year": "frequency:year",
    "每年": "frequency:year",
    "monday": "weekday:monday",
    "tuesday": "weekday:tuesday",
    "wednesday": "weekday:wednesday",
    "thursday": "weekday:thursday",
    "friday": "weekday:friday",
    "saturday": "weekday:saturday",
    "sunday": "weekday:sunday",
    "周一": "weekday:monday",
    "周二": "weekday:tuesday",
    "周三": "weekday:wednesday",
    "周四": "weekday:thursday",
    "周五": "weekday:friday",
    "周六": "weekday:saturday",
    "周日": "weekday:sunday",
    "january": "month:january",
    "february": "month:february",
    "march": "month:march",
    "april": "month:april",
    "may": "month:may",
    "june": "month:june",
    "july": "month:july",
    "august": "month:august",
    "september": "month:september",
    "october": "month:october",
    "november": "month:november",
    "december": "month:december",
    "true": "boolean:true",
    "yes": "boolean:true",
    "on": "boolean:true",
    "enabled": "boolean:true",
    "enable": "boolean:true",
    "allowed": "boolean:true",
    "是": "boolean:true",
    "真": "boolean:true",
    "开启": "boolean:true",
    "启用": "boolean:true",
    "false": "boolean:false",
    "no": "boolean:false",
    "off": "boolean:false",
    "disabled": "boolean:false",
    "disable": "boolean:false",
    "forbidden": "boolean:false",
    "否": "boolean:false",
    "假": "boolean:false",
    "关闭": "boolean:false",
    "禁用": "boolean:false",
}


def _semantic_value_dimension(value: str) -> str:
    normalized = _normalize_promotion_claim_value(value)
    if not normalized:
        return ""
    collapsed = " ".join(normalized.split())
    return _SEMANTIC_VALUE_DIMENSIONS.get(collapsed, "")


def _semantic_values_share_dimension(left: str, right: str) -> bool:
    left_dimension = _semantic_value_dimension(left)
    right_dimension = _semantic_value_dimension(right)
    if not left_dimension or not right_dimension:
        return False
    left_kind, _, left_value = left_dimension.partition(":")
    right_kind, _, right_value = right_dimension.partition(":")
    return left_kind == right_kind and left_value != right_value


def _semantic_values_are_opposites(left: str, right: str) -> bool:
    if _semantic_values_share_dimension(left, right):
        return True
    left_words = set(re.findall(r"[a-z0-9_-]+", left.casefold()))
    right_words = set(re.findall(r"[a-z0-9_-]+", right.casefold()))
    for left_word in left_words:
        for right_word in right_words:
            if frozenset((left_word, right_word)) in _SEMANTIC_VALUE_ANTONYMS:
                return True
            for prefix in ("im", "in", "un", "non", "dis"):
                if left_word == prefix + right_word or right_word == prefix + left_word:
                    return True
    return False


def _promotion_values_conflict(subject: str, values: tuple[str, ...]) -> bool:
    normalized = [
        _normalize_promotion_text(value)
        for value in values
        if value.strip()
    ]
    if len(set(normalized)) < 2:
        return False
    negation = re.compile(r"^(?:not|no|never|without)\b|^[非不无未]")
    negated = [bool(negation.search(value)) for value in normalized]
    if any(negated) and not all(negated):
        return True
    if any(
        _semantic_values_are_opposites(left, right)
        for index, left in enumerate(normalized)
        for right in normalized[index + 1 :]
    ):
        return True
    if subject.startswith("user:prefer"):
        return True
    return bool(
        re.search(
            r"\b(?:target|mode|theme|status|state|version|branch|path|port|"
            r"default|setting|count|size|language|database|style|indentation|"
            r"verbosity|color|font|speed|timezone|time\s+zone|owner|assignee|"
            r"email|host|endpoint|url|uri|address|location|region|date|deadline|"
            r"currency|name|"
            r"datastore|cloud|package\s+manager|provider|runtime|tool(?:chain)?|"
            r"engine|framework|platform|"
            r"implementation|environment|primary|current|active|selected|chosen|main|official|"
            r"preferred|canonical|authoritative|assigned|designated|configured)\b",
            subject.casefold(),
        )
    )


def _normalize_promotion_claim_value(value: str) -> str:
    return _normalize_promotion_text(value).strip(" .!。！?？,，;；:")


def _semantic_sensitive_fields(
    raw_result: Mapping[str, object],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    fields = (
        ("claim_subject", str(raw_result.get("claim_subject") or "").strip()),
        ("claim_value", str(raw_result.get("claim_value") or "").strip()),
        ("region_hint", str(raw_result.get("region_hint") or "").strip()),
    )
    sensitive_fields: list[str] = []
    categories: set[str] = set()
    for field_name, field_value in fields:
        if not field_value:
            continue
        finding = inspect_sensitive_text(field_value)
        claim_label = _SEMANTIC_SENSITIVE_CLAIM_FIELD.search(field_value) is not None
        if finding is None and not claim_label:
            continue
        sensitive_fields.append(field_name)
        if finding is not None:
            categories.update(finding.categories)
        if claim_label:
            categories.add("sensitive-claim-field")
    return tuple(sensitive_fields), tuple(sorted(categories))


def _promotion_subjects_compatible(left: str, right: str) -> bool:
    def normalized(subject: str) -> tuple[str, str]:
        value = _normalize_promotion_text(subject)
        match = re.match(r"^user:prefer(?:ence)?(?P<suffix>(?::.*)?)$", value)
        if match is not None:
            return "user:preference", match.group("suffix")
        return value, ""

    left_family, left_suffix = normalized(left)
    right_family, right_suffix = normalized(right)
    return (
        left_family == right_family
        and (not left_suffix or not right_suffix or left_suffix == right_suffix)
    )


def _promotion_claim_values_equivalent(left: str, right: str) -> bool:
    left_value = _normalize_promotion_claim_value(left)
    right_value = _normalize_promotion_claim_value(right)
    if left_value == right_value:
        return True
    fillers = {"mode", "preference", "setting", "style", "theme"}
    left_tokens = set(re.findall(r"[a-z0-9_.-]+", left_value))
    right_tokens = set(re.findall(r"[a-z0-9_.-]+", right_value))
    if not left_tokens or not right_tokens:
        return False
    if left_tokens < right_tokens and right_tokens - left_tokens <= fillers:
        return True
    return right_tokens < left_tokens and left_tokens - right_tokens <= fillers


def _heuristic_promotion_decision(
    content: str,
    context: Mapping[str, object],
) -> dict[str, object]:
    if inspect_sensitive_text(content) is not None:
        return {
            "category": "sensitive",
            "confidence": 1.0,
            "explicit": False,
            "topic_id": None,
            "reason": "sensitive content requires manual review",
        }
    text = _normalize_promotion_text(content)
    personal = bool(
        re.search(
            r"\b(?:i|my|me)\b|\b(?:prefer|always|usually|never)\b|"
            r"我(?:偏好|喜欢|习惯|通常|希望|一直)|我的[^。\n]{0,40}(?:是|为)",
            text,
            flags=re.IGNORECASE,
        )
    )
    project_specific = bool(
        re.search(
            r"\b(?:this|the)\s+(?:project|repo|repository|branch|worktree)\b|"
            r"(?:^|\s)(?:\./|/)[\w./-]+|"
            r"本(?:项目|仓库)|当前(?:项目|分支|工作树)|该(?:项目|仓库)",
            text,
            flags=re.IGNORECASE,
        )
    )
    if personal and not project_specific:
        return {
            "category": "user",
            "confidence": 0.95,
            "explicit": True,
            "topic_id": None,
            "reason": "explicit stable personal fact",
        }
    if project_specific:
        return {
            "category": "project",
            "confidence": 0.95,
            "explicit": True,
            "topic_id": None,
            "reason": "project-specific content remains in its source region",
        }
    topic_matches: list[tuple[int, str]] = []
    raw_topics = context.get("topic_regions")
    if isinstance(raw_topics, list):
        for raw_topic in raw_topics:
            if not isinstance(raw_topic, Mapping):
                continue
            topic_id = str(raw_topic.get("topic_id") or "").strip()
            topic_name = str(raw_topic.get("name") or "").strip()
            candidates = {
                value.casefold() for value in (topic_id, topic_name) if len(value.strip()) >= 2
            }
            matched = [
                value
                for value in candidates
                if re.search(
                    rf"(?<![\w]){re.escape(value)}(?![\w])",
                    text,
                )
            ]
            if matched:
                topic_matches.append((max(len(value) for value in matched), topic_id))
    knowledge_marker = bool(
        re.search(
            r"\b(?:is|are|means|allows|provides|enables|ownership|principle|"
            r"concept|syntax)\b|(?:是指|表示|原理|概念|语法|用于|可以)",
            text,
            flags=re.IGNORECASE,
        )
    )
    if topic_matches and knowledge_marker:
        longest = max(length for length, _ in topic_matches)
        best = [topic_id for length, topic_id in topic_matches if length == longest]
        if len(set(best)) == 1:
            return {
                "category": "topic",
                "confidence": 0.95,
                "explicit": True,
                "topic_id": best[0],
                "reason": "explicit generalizable topic knowledge",
            }
    return {
        "category": "retain",
        "confidence": 0.0,
        "explicit": False,
        "topic_id": None,
        "reason": "no explicit high-confidence promotion target",
    }


def _validated_promotion_decision(
    raw: Mapping[str, object],
    *,
    content: str,
    explicit_source: bool,
    topic_ids: set[str],
) -> _PromotionDecision:
    category = str(raw.get("category", "retain")).strip()
    if category not in {"user", "topic", "project", "conflict", "sensitive", "retain"}:
        raise ValueError("promotion classifier returned an invalid category")
    confidence = raw.get("confidence")
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0 <= float(confidence) <= 1
    ):
        raise ValueError("promotion classifier returned an invalid confidence")
    explicit = raw.get("explicit") is True
    topic_id = str(raw.get("topic_id") or "").strip() or None
    claim_subject = str(raw.get("claim_subject") or raw.get("subject") or "").strip()
    claim_value = str(raw.get("claim_value") or raw.get("value") or "").strip()
    if not claim_subject:
        claim_subject, claim_value = _promotion_claim_signature(content)
    reason = str(raw.get("reason") or "").strip() or "classified by promotion policy"
    if inspect_sensitive_text(content) is not None:
        category = "sensitive"
        confidence = 1.0
        explicit = False
        reason = "sensitive content requires manual review"
    status = "skip"
    if category in {"user", "topic"}:
        if not explicit_source or not explicit:
            reason = "promotion requires an explicit source statement"
        elif float(confidence) < 0.9:
            reason = "promotion confidence is below 0.90"
        elif category == "topic" and topic_id not in topic_ids:
            reason = "promotion topic does not match an existing topic region"
        else:
            status = "promote"
    return {
        "category": category,
        "confidence": float(confidence),
        "explicit": explicit,
        "topic_id": topic_id,
        "claim_subject": claim_subject,
        "claim_value": claim_value,
        "reason": reason,
        "status": status,
    }


def _close_quietly(closeable: object) -> None:
    close = getattr(closeable, "close", None)
    if callable(close):
        with suppress(Exception):
            close()


def _close_stores(stores: list[object]) -> None:
    seen: set[int] = set()
    for store in reversed(stores):
        if id(store) in seen:
            continue
        seen.add(id(store))
        _close_quietly(store)
    stores.clear()
