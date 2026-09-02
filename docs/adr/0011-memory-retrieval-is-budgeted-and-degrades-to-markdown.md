# 记忆检索有硬预算并降级到 Markdown

服务端使用实际 tokenizer 将每个记忆上下文包限制在 10K tokens 内，以混合检索获得 Markdown 直接命中，再执行最多两跳且有来源约束的图扩展。Embedding、Reranker、LLM 或 Graph 失败时逐级退化，最低仍使用关键词或 FTS 返回 Markdown，Codex 可以看到降级状态而不会因增强组件异常失去全部记忆能力。
