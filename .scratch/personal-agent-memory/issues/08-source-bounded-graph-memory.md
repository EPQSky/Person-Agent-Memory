# 08 — 有来源约束的 Graph Memory

**What to build:** 将正式权威 Markdown 投影到本地 Graph Memory，并仅从直接 Markdown 命中执行有限关联扩展，使每个图结果都能回到当前有效的来源版本。

**Blocked by:** 05 — 平台编辑与隔离 Git 历史；07 — 向量混合检索与模型降级

**Status:** ready-for-agent

- [ ] 正式 Markdown 及其来源版本可以异步投影为 Entity、Relation 和 Episode，候选记忆与 Inbox 不进入 Graph Memory。
- [ ] 图构建使用本地 Milvus Lite 派生存储，并按记忆库隔离图数据和健康状态。
- [ ] 检索只有在获得直接 Markdown 命中后才执行图扩展，默认一跳。
- [ ] 只有一跳结果不足且预算允许时才执行第二跳，任何配置都不能超过两跳。
- [ ] 每个图结果包含可验证的 Markdown 来源标识和版本；无来源、跨库或版本失配结果被丢弃。
- [ ] Markdown 编辑、删除或恢复后，旧图结果立即因版本检查失效，后台最终更新对应图投影。
- [ ] Graph、LLM 或 Milvus Lite 不可用时返回直接 Markdown 结果并标明降级，不影响平台编辑和全文检索。
- [ ] Web 显示每个记忆库的图构建进度、最近错误和同步状态；Docker 验收覆盖一跳、条件二跳、来源丢失、陈旧版本及服务故障。
