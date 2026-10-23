# 采用 JiuwenMemory 0.2.0 作为唯一记忆内核

## 状态

已接受（2026-10-23）。

## 背景

- 项目当前固定 `JiuwenMemory==0.1.2`，只使用 `jiuwen_memory.foundation.store.graph` 作为图投影存储。
- mem2.0（0.2.0）是上游内核重写。公开入口变为 `MemoryAPI` 和统一 Storage。旧的 `foundation.store.graph` 已移除。
- 现行自研内核覆盖 Markdown 扫描、分块、混合检索、来源校验、10K token 预算、候选治理、本地 Git 版本和本地图投影。这些能力与 mem2.0 重叠。
- 用户决定：不再维护两套记忆内核，直接以 mem2.0 替换现有内核。

## 决策

1. `JiuwenMemory 0.2.0` 成为唯一记忆内核。它负责记忆写入、存储、索引、检索、生命周期和治理。
2. 保留现有产品外壳：FastAPI 管理台、API Key、Codex Hooks、MCP、项目绑定和安装器。
3. 现有自研记忆内核退场。`markdown_index.py`、`graph_adapter.py`、`context_package.py`，以及 `state.py` 中的记忆索引、候选、版本和图投影逻辑，不再作为主实现。
4. 项目仍在开发阶段，只有单用户测试。本次不做旧版本数据迁移，不保证旧状态兼容。使用全新状态目录，允许清空并重建。
5. `JiuwenMemory` 固定到 `0.2.0`。升级测试覆盖 import、写入、检索、图扩展、MCP/Hook 和安装器。
6. 开发测试使用轻量本地组合：SQLite KV，Milvus Lite 3.x 同时承载稠密向量和 BM25 全文索引。不得把 Docker 作为开发前置条件。
7. 开发期依赖使用最新稳定版。spike 通过后，把实际验证的版本锁定到 `pyproject.toml` 和 `uv.lock`，不使用浮动版本。
8. 取消独立候选审核层。mem2.0 的 `evolve`、生命周期、`audit`、`inspect`、`update` 和 `delete` 是唯一治理路径。后台更正通过管理台调用这些接口完成。
9. 图检索是本次目标，不是可选功能。图后端使用项目自有的 Milvus Lite GraphStore 适配器：节点和边分别存入 Milvus Lite collection，遍历在应用层执行。存储 spike 阶段先关闭图，spike 通过后立即接入。`nano-graphrag` 最新版停留在 0.0.8.2（2024-10-19），依赖较重，本次不引入。

## 后果

- 记忆真源从“Markdown 文件”改为 mem2.0 的存储模型。Markdown 只作为可选导出视图，不提供旧 Markdown 导入器。
- 每个记忆库必须映射到 mem2.0 的 `Scope`。项目、用户和会话的映射必须在后续决策中明确。
- 本地 Git 提交历史不再是记忆版本的主机制。mem2.0 的 `audit`、`inspect`、`trace`、`supersedes` 和生命周期成为主机制。
- Codex 检索契约必须在产品外壳中重新适配。独立候选审核层取消；敏感内容治理依靠 mem2.0 的生命周期、审计和更正操作。
- Milvus Lite 3.x 具备 BM25 全文检索能力，但 mem2.0 当前只提供 Milvus 的 `VectorStore` 适配器，没有 `FulltextStore` 适配器。需要项目侧补一个只实现存储契约的 BM25 适配器。
- 项目当前固定的 `milvus-lite==2.5.1` 是旧一代实现，不具备本决策所需的 BM25 能力。需要升级到 3.2.x；新实现使用不同存储格式，但本项目不做旧数据迁移，因此不构成阻塞。
- 最新版 Milvus Lite 3.x 对 JSON 字段的 `in` 和 `not in` 过滤有限制。mem2.0 的 Milvus 适配器需要把这两类表达式改写为等值条件的 OR/AND 组合。
- `nano-graphrag` 最新版停留在 0.0.8.2（2024-10-19），且依赖较重。本次不使用 `nano-graphrag`；图索引由 Milvus Lite GraphStore 适配器提供。图接入放在存储 spike 之后，但不能以关闭图作为最终交付。
- ADR 0001、0003、0004、0005、0007、0008、0009、0010、0011、0014 与本决策冲突。后续 ADR 必须明确取代关系。在此之前，不得同时引用两套真源假设。
- 旧数据、旧图投影和旧状态目录不迁移。开发阶段允许清空并重建。

## 开发阶段约束

- 当前只有单用户测试，不要求旧版本兼容。
- 不提供旧 Markdown 导入器，也不实现双写。
- 状态目录可以清空，测试数据可以丢弃。
- 不使用 Docker 作为开发前置条件。
- 使用 `milvus-lite 3.2.x`；旧版 `2.5.1` 不满足 BM25 全文检索要求。
- 开发期允许使用最新稳定版；每个 spike 通过后锁定精确版本。
- 图索引使用 Milvus Lite 节点/边 collection；不引入 `nano-graphrag`。

## 相关决策

- 记忆区域、会话记忆、多目标晋升、预算和 Scope 映射见 ADR 0022。

## 替代方案

- 只升级图适配器：被拒绝。0.2.0 不再提供旧的 Milvus 图存储，无法低成本保持现有实现。
- 保留 0.1.2：被拒绝。上游已进入 0.2.0，长期维护和升级测试成本更高。
- 双内核并行：被拒绝。双内核会产生两套真源、两套检索和两套治理。
- 用 Milvus Lite 替代 SQLite KV：被拒绝。`KVStore` 需要任意字节值、前缀扫描、批量读取、TTL 和事务性写入；Milvus 是向量、全文和图索引引擎，不是键值真源。把真源和派生索引放在同一引擎会扩大故障域，并需要自定义 KV 适配器。
