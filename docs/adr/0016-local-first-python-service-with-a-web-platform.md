# 采用本地优先的 Python 服务与 Web 平台

后端使用 Python 3.11+、FastAPI 和固定版本的 JiuwenMemory，管理界面使用 React 与 TypeScript，Codex Hooks 使用轻量 Node.js 脚本；平台状态、FTS 和任务队列落 SQLite，Markdown 向量检索适配 Jiuwen 文件索引，Graph Memory 使用 Milvus Lite。首期通过 `uv` 或 Python 包直接安装而不以 Docker 为主，以便访问宿主机目录、本地 Git 和 Codex Hooks。
