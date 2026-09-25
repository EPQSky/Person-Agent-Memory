# Personal Agent Memory

面向单个用户和本地 Codex 会话的本地优先记忆系统。

## 安装到用户目录

项目内的 `.venv` 只用于开发和测试，不等于完成用户级安装。要把服务、数据目录、
用户级 systemd 服务和 Codex 插件安装到当前用户目录，请在本项目目录执行：

```bash
chmod +x install.sh uninstall.sh
./install.sh
```

安装器不使用 `sudo`，也不写入系统级服务。默认位置如下：

- 平台状态和 API 密钥：`~/.local/share/personal-agent-memory`
- 记忆库根目录：`~/memory-libraries`
- 安装元数据：`~/.config/personal-agent-memory/install.json`
- 用户级服务：`~/.config/systemd/user/personal-agent-memory.service`
- 用户级可执行文件：`~/.local/bin` 或 `uv` 的工具 bin 目录

安装完成后，服务由当前用户的 `systemd --user` 管理，并监听：
`http://127.0.0.1:7331`。

首次启动会生成权限为 `0600` 的 API 密钥：

```bash
cat ~/.local/share/personal-agent-memory/api-key
```

安装器不会自动打开浏览器，也不会打印 API 密钥。打开
`http://127.0.0.1:7331` 后输入该密钥即可查看受保护的服务状态。

自定义目录：

```bash
./install.sh \
  --state-dir "$HOME/data/personal-agent-memory" \
  --library-root "$HOME/documents/memory"
```

记忆库目录属于用户数据，安装、升级和卸载都不会删除它。卸载命令：

```bash
personal-agent-memory-uninstall
```

如需同时删除平台状态和安装元数据：

```bash
personal-agent-memory-uninstall --purge
```

`--purge` 仍然不会删除默认或自定义的记忆库目录。

## 开发环境运行

开发和测试需要 Python 3.11 或更高版本、Git 和 `uv`：

```bash
uv sync --all-extras
uv run personal-agent-memory serve \
  --state-dir ~/.local/share/personal-agent-memory \
  --library-root ~/memory-libraries
```

开发环境命令不会创建用户级 systemd 服务，也不会自动安装全局 Codex 插件。

服务只允许监听回环地址。可以重复指定 `--library-root` 来允许多个本地目录树。
注册时会索引已有 Markdown 文件，但不会改写源文件。扫描会跳过版本控制、依赖、
构建产物、缓存、虚拟环境和平台状态目录，也会排除符号链接和非普通文件。

无法完整按 UTF-8 读取的 Markdown 文件，或扫描期间发生的临时文件系统错误，会使
本次扫描失败，但不会删除已有索引。可以使用以下接口增量处理新增、修改和删除：

```text
GET/PUT /api/v1/libraries/{library_id}/ignore-rules
POST     /api/v1/libraries/{library_id}/scan
```

## 搜索和 Web 界面

关键词搜索可通过以下接口或 Web 界面使用：

```text
POST /api/v1/search
POST /mcp/search
```

两个 API 都接受 `cwd`、`query` 和可选的 `limit`。服务会通过显式项目绑定解析
`cwd`，只搜索该项目对应的记忆库。未绑定的工作目录会返回 `unbound`，不会返回
搜索内容。

Web 界面支持浏览已索引 Markdown、编辑源文件、渲染隔离预览、查看保存差异，以及
审计和恢复本地历史。对应的受保护 REST 接口位于：

```text
/api/v1/libraries/{library_id}/documents
/api/v1/libraries/{library_id}/document
/api/v1/libraries/{library_id}/history
```

每次编辑或恢复都需要期望的源版本和幂等操作 ID。系统会原子替换文档、重新索引，
并创建一次本地 Git 提交。

默认情况下，每个记忆库使用平台状态目录下的裸 Git 旁车仓库，记忆目录本身只是
工作树，不会创建嵌套 `.git`，也不会配置远程、执行拉取或推送。

## 候选记忆和治理

MCP 客户端可以创建和查看候选记忆：

```text
POST /mcp/candidates
GET  /mcp/candidates
GET  /mcp/candidates/{candidate_id}
```

候选记忆在批准前不会进入文本、向量、图或 Codex 检索。Web 界面负责批准、拒绝和
审计，并记录操作者和原因。批准后会发布带来源信息的 Markdown、刷新搜索并创建
一次幂等 Git 提交；拒绝只保留审计记录。

## 向量、重排和图

嵌入和重排服务是可选的 OpenAI 兼容服务：

```text
--embedding-url
--reranker-url
```

API 密钥只从对应的 `--embedding-api-key-file` 和 `--reranker-api-key-file` 读取，
不会存入记忆库、Git 或响应内容。嵌入或重排服务不可用时，系统会保留全文结果并
返回降级标记。

图投影使用固定版本 `JiuwenMemory==0.1.2`，每个记忆库使用独立的 Milvus Lite 文件。
可通过以下参数配置图抽取模型：

```text
--graph-url
--graph-model
--graph-api-key-file
```

图搜索默认扩展一跳，最多两跳，并且会先执行直接 Markdown 命中。图或模型不可用时，
直接 Markdown 检索仍然可用，并报告 `graph_unavailable`。

搜索接口返回 `memory-context-package/v1` 格式。调用方必须选择项目 `cwd` 或明确的
`library_id`。服务会根据请求的 `target_model` tokenizer 统计完整 JSON 包，并执行
`token_budget` 和 10,000 token 的绝对上限。

召回内容会标记为不可信数据，不携带策略、工具授权或命令语义。

## Codex 插件

用户级安装器会自动注册 `plugins/personal-agent-memory`。Codex 可以发现生命周期
Hook 和经过认证的本机 MCP 连接，不需要向每个项目写入文件。

插件只捕获明确的用户提示词和助手最终消息，不读取 transcript 文件、隐藏推理、原始
工具输出、完整文件内容、子代理轨迹或召回的记忆上下文。服务暂时不可用时，Hook
会使用受限的本地 spool，后续再重放有效记录，不阻塞 Codex。

## 服务管理

```bash
systemctl --user status personal-agent-memory.service
systemctl --user start personal-agent-memory.service
systemctl --user stop personal-agent-memory.service
systemctl --user restart personal-agent-memory.service
journalctl --user -u personal-agent-memory.service --no-pager
```

当前 shell 临时导出 API 密钥：

```bash
export PERSONAL_AGENT_MEMORY_API_KEY="$(cat ~/.local/share/personal-agent-memory/api-key)"
```

## 验证

开发环境测试：

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

原生安装和全局插件验收：

```bash
./scripts/verify-native-install.sh
```

完整 Linux 安装器验收：

```bash
PAM_INSTALL_ACCEPTANCE_DEDICATED_USER=1 \
  ./scripts/verify-linux-installer.sh
```

## 支持范围和限制

原生安装目标是 Linux x86_64，已测试 Ubuntu 22.04 和 Ubuntu 24.04。需要 Python
3.11-3.13、Git、Node.js、Codex CLI、`uv` 和可用的 `systemd --user`。

当前系统只服务单个本地用户，服务仅监听回环地址。Windows、macOS、ARM、远程同步、
多用户协作、公开或局域网服务、Codex Cloud、Git 自动推送拉取、系统级服务以及自动
后台升级不在 MVP 范围内。

用户需要自行备份所有权威 Markdown 记忆库和平台状态目录。Git 远程不会由系统自动
配置或同步。
