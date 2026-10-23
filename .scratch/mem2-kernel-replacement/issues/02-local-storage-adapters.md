# 02 — 存储适配器扩展

**What to build:** 守护进程可以把记忆单元持久化到 SQLite，并用 Milvus Lite 建立稠密向量索引和 BM25 全文索引；审计事件持久化到 SQLite。整套本地存储不需要 Docker，并且重启后仍然可用。

**Blocked by:** 01 — Prefactor：内核端口和特性开关

**Status:** done

- [x] `JiuwenMemory 0.2.0`、`milvus-lite 3.2.2` 和 `pymilvus 3.0.2` 被锁定到项目和锁文件。
- [x] SQLite KV 适配器支持记忆单元写入、读取、更新、删除、按 scope 隔离和重启保留。
- [x] Milvus Lite 稠密向量适配器支持 CRUD、检索、scope 过滤、元数据过滤和重启保留。
- [x] Milvus Lite BM25 适配器支持 CRUD、检索、Jieba 分词、BM25 评分方向、scope 和元数据过滤、重启保留。
- [x] Milvus Lite 本地客户端需要 `db_name="default"` 时，适配器内部处理，不要求调用方感知。
- [x] JSON 字段的 `in` 和 `not in` 过滤被改写为等值条件的 OR 和 AND 组合。
- [x] 持久化 SQLite 审计存储可以写入和查询审计事件，重启后仍可查询。
- [x] 适配器在关闭时释放本地 Milvus 服务和数据目录锁。
- [x] 存储契约测试覆盖 CRUD、scope 隔离、过滤、评分、重启和关闭释放。
- [x] 本 Ticket 不要求 Docker。
