# Codex 同时使用 Hooks 与 MCP

Codex 插件使用生命周期 Hooks 自动采集会话并在生成前注入相关记忆，同时通过 MCP 暴露显式的检索、写入、删除和记忆库管理能力。目录监听只同步 Markdown 变化，不承担会话捕获；首期服务运行在本地并使用 API Key 认证，真正运行于 Codex Cloud 的远程任务接入延后处理。
