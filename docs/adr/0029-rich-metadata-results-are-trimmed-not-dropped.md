# 富元数据的检索结果只裁剪、不丢弃

上下文包把可用预算分为内容、图谱与元数据三份，元数据份额固定为可用预算的 10%。语义记忆携带完整提炼溯源时，单条元数据即可超过该份额。旧实现因此静默丢弃该结果：记忆已被检索命中，调用方却收到空结果。

检索结果不得因元数据过大而消失。分配器按「完整元数据、去掉重型溯源、只留一条来源、最小身份字段」的顺序尝试，第一个满足份额的方案被采用，并写入 `metadata_truncated: true`，同时在包级 `degradation` 追加 `result_metadata_trimmed`。

检索诊断字段（`entity_graph_status`、`query_subgraph`、`query_subgraphs`、`no_mapped_entities`）不属于记忆内容，但同样占用响应字节。它们只在整包仍满足全部预算约束时附加；否则省略并记录 `diagnostics_omitted`。附加或省略之后都必须重算遥测，跨区包按精确序列化计数更新 `used_tokens`，保证 `used_tokens` 不低报真实负载。
