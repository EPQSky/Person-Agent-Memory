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
  --library-root "$HOME/documents/memory" \
  --port 17331
```

如遇 `7331` 已被占用，可用 `--port PORT` 指定本机回环端口；端口会写入安装元数据，后续升级默认复用。

记忆库目录属于用户数据，安装、升级和卸载都不会删除它。卸载命令：

```bash
personal-agent-memory-uninstall
```

如需同时删除平台状态和安装元数据：

```bash
personal-agent-memory-uninstall --purge
```

`--purge` 仍然不会删除默认或自定义的记忆库目录。

## 5 分钟上手

下面的流程适用于已经执行过 `./install.sh` 的 Linux 用户。开发环境只需要把文中的
`systemctl --user` 命令替换为你手动运行的 `uv run ... serve` 命令。

### 1. 确认服务和网页

```bash
systemctl --user status personal-agent-memory.service
```

看到 `active (running)` 后，打开：

```text
http://127.0.0.1:7331
```

登录密钥在：

```text
~/.local/share/personal-agent-memory/api-key
```

只在当前 shell 查看密钥并复制到网页输入框：

```bash
cat ~/.local/share/personal-agent-memory/api-key
```

密钥不会显示在服务状态接口中，也不会写入项目目录。

首次在管理台输入平台 API key 并登录后，服务会建立固定有效期为 24 小时的本地会话。
会话通过 HttpOnly Cookie 保存，刷新网页或重启记忆服务不会要求重复输入 API key。
显式退出、会话过期或 API key 轮换后，需要重新登录。

### 2. 配置模型服务

模型服务的 key 文件每个只放一行 API key 原文，不要写 `Bearer`、JSON 或引号：

```text
~/.config/personal-agent-memory/embedding.key
~/.config/personal-agent-memory/reranker.key
~/.config/personal-agent-memory/graph.key
```

设置权限：

```bash
chmod 600 ~/.config/personal-agent-memory/*.key
```

SiliconFlow 配置示例：

- 嵌入模型：`Qwen/Qwen3-Embedding-0.6B`
- 重排序模型：`Qwen/Qwen3-Reranker-0.6B`
- 图谱和候选记忆抽取模型：`Qwen/Qwen3.6-35B-A3B`

编辑用户级服务 drop-in：

```bash
systemctl --user edit --drop-in=model.conf personal-agent-memory.service
```

填入下面内容。请把 `/home/epq` 换成当前用户的 home 目录，并保留已有的
`--state-dir` 和 `--library-root` 参数：

```ini
[Service]
ExecStart=
ExecStart="/home/epq/.local/bin/personal-agent-memory" serve --state-dir "/home/epq/.local/share/personal-agent-memory" --library-root "/home/epq/memory-libraries" --host "127.0.0.1" --port "7331" --embedding-url "https://api.siliconflow.cn" --embedding-model "Qwen/Qwen3-Embedding-0.6B" --embedding-api-key-file "/home/epq/.config/personal-agent-memory/embedding.key" --reranker-url "https://api.siliconflow.cn" --reranker-model "Qwen/Qwen3-Reranker-0.6B" --reranker-api-key-file "/home/epq/.config/personal-agent-memory/reranker.key" --graph-url "https://api.siliconflow.cn" --graph-model "Qwen/Qwen3.6-35B-A3B" --graph-api-key-file "/home/epq/.config/personal-agent-memory/graph.key"
```

当前程序会自动在地址后面请求 `/v1/embeddings`、`/v1/rerank` 和
`/v1/chat/completions`，所以这里填写 `https://api.siliconflow.cn`，不要再重复写
`/v1`。

保存后重载并重启：

```bash
systemctl --user daemon-reload
systemctl --user restart personal-agent-memory.service
systemctl --user is-active personal-agent-memory.service
```

### 3. 准备一个项目记忆库

记忆库目录和项目源代码目录是两个概念：

- **记忆库目录**：存放可被检索的 Markdown 记忆。
- **项目根目录**：Codex 工作的源代码目录，用于决定应该检索哪个记忆库。

推荐为项目创建独立的记忆目录：

```bash
mkdir -p "$HOME/memory-libraries/five-chain-deepagents"
```

可以在其中创建 Markdown 文件，例如：

```text
$HOME/memory-libraries/five-chain-deepagents/decisions.md
```

然后在网页的“记忆库”页面填写：

```text
目录路径：/home/epq/memory-libraries/five-chain-deepagents
记忆库类型：项目记忆库
```

点击“注册记忆库”。普通新目录不要勾选“授权使用此专用记忆仓库”；只有需要复用该
目录已有专用 Git 仓库时才勾选。

注册时系统会扫描已有 Markdown 文件。文本索引会先完成，向量和图谱索引在后台异步
完成；可以在表格的“图索引进度”和“图索引错误”列查看状态。

### 4. 绑定项目

进入网页的“项目绑定”页面：

```text
项目根目录：/home/epq/Workspace/emsoft-team/code/five-chain-deepagents
项目记忆库：选择刚才注册的 five-chain-deepagents 记忆库
```

点击“绑定项目”。之后从这个项目目录发起的搜索和 Codex Hook 才会使用该记忆库。

### 5. 检索记忆

网页的“记忆检索”页面填写：

```text
项目工作目录：/home/epq/Workspace/emsoft-team/code/five-chain-deepagents
关键词：项目使用了什么数据库
```

也可以用 REST API 测试：

```bash
export PERSONAL_AGENT_MEMORY_API_KEY="$(
  cat "$HOME/.local/share/personal-agent-memory/api-key"
)"

curl -sS \
  -H "Authorization: Bearer $PERSONAL_AGENT_MEMORY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"cwd":"/home/epq/Workspace/emsoft-team/code/five-chain-deepagents","query":"项目使用了什么数据库","limit":10}' \
  http://127.0.0.1:7331/api/v1/search
```

### 6. 在 Codex 中使用

用户级安装器会自动注册 Codex 插件，不需要把插件复制到每个项目。使用前确认：

```bash
systemctl --user is-active personal-agent-memory.service
codex plugin list --json
```

只要当前工作目录已经完成项目绑定，Codex 的用户提示和助手最终消息就会由 Hook
捕获，并交给后台整理为候选记忆。候选记忆默认不会直接成为权威记忆，需要在网页的
“候选审核”中批准；批准后才会进入正式检索。

### 7. 路径白名单报错

如果看到：

```text
path is outside allowed roots: /home/epq/memory-libraries
```

说明网页填写的记忆库路径不在服务的 `--library-root` 允许范围内。优先把记忆库放在
`~/memory-libraries` 下面，例如：

```text
/home/epq/memory-libraries/five-chain-deepagents
```

如果确实要直接把项目源代码目录作为记忆库，就必须在服务启动参数中额外加入：

```text
--library-root /home/epq/Workspace/emsoft-team/code/five-chain-deepagents
```

这会明确授权系统扫描该目录中的 Markdown，并可能将其内容发送给已配置的嵌入、重排
和图谱模型。修改后执行：

```bash
systemctl --user daemon-reload
systemctl --user restart personal-agent-memory.service
```

## 常用故障排查

服务状态：

```bash
systemctl --user status personal-agent-memory.service
journalctl --user -u personal-agent-memory.service --no-pager -n 100
```

本地健康检查：

```bash
curl -sS -o /dev/null -w '%{http_code}\n' \
  -H "Authorization: Bearer $(cat "$HOME/.local/share/personal-agent-memory/api-key")" \
  http://127.0.0.1:7331/health/live
```

- `unbound`：当前项目根目录没有完成“项目绑定”。
- `graph_unavailable`：图谱模型 URL、模型名、key 或返回格式有问题；全文检索仍可用。
- `embedding_unavailable`：嵌入服务不可用；文本检索仍可用。
- `path is outside allowed roots`：记忆库目录不在 `--library-root` 白名单内。
- `401` 或 `403`：key 文件为空、权限不合适或 key 无效。

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

明确且低风险的会话内容默认自动晋升，仍会保留候选记录、来源引用、自动操作审计和一次幂等
Git 提交。Web 界面主要处理低置信度、冲突、重复、敏感或策略受限的异常候选；待处理候选在
批准前不会进入文本、向量、图或 Codex 检索。拒绝只保留审计记录。

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
