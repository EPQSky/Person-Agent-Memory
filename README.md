# Personal Agent Memory

面向单个用户和本地 Codex 会话的本地优先记忆系统。系统使用
`JiuwenMemory 0.2.0` 作为唯一记忆内核。

长期记忆保存在平台状态目录的 `mem2/` 子目录。该目录包含 `VERSION` 文件，文件内容为
`jiuwen-memory-0.2.0`。记忆单元保存在 SQLite。稠密向量、全文索引和图节点边保存在
本地 Milvus Lite 数据文件中。审计事件保存在 SQLite。系统不需要 Docker。

## 核心概念

记忆区域分为四类：

- **用户记忆**：保存跨项目的稳定事实和偏好。区域标识为 `user`。
- **专题记忆**：保存技能、课程和领域知识。
- **项目记忆**：保存具体项目的事实。项目根目录通过项目绑定关联到项目记忆区域。
- **会话记忆**：保存非项目、非专题会话的临时内容。每次会话使用独立标识隔离。

管理台把用户记忆显示为区域树根节点。存储层中的用户、专题和项目区域是同级空间。
专题和项目区域不继承用户记忆的所有权或生命周期。

项目会话默认只召回当前项目记忆。专题会话默认只召回当前专题记忆。非项目、非专题会话
默认召回用户记忆和当前会话记忆。跨区域召回必须显式选择区域。

## 安装到用户目录

安装需要 Linux x86_64、Python 3.11-3.13、`uv`、Node.js、Codex CLI、Git 和用户级
`systemd`。Git 用于验证工作树继承绑定。

在项目目录执行：

```bash
chmod +x install.sh uninstall.sh
./install.sh
```

安装器不使用 `sudo`，也不写入系统级服务。默认位置如下：

- 平台状态和 API 密钥：`~/.local/share/personal-agent-memory`
- 记忆状态：`~/.local/share/personal-agent-memory/mem2`
- 安装元数据：`~/.config/personal-agent-memory/install.json`
- 用户级服务：`~/.config/systemd/user/personal-agent-memory.service`
- 用户级可执行文件：`~/.local/bin` 或 `uv` 的工具 bin 目录

服务监听 `http://127.0.0.1:7331`。首次启动会生成权限为 `0600` 的 API 密钥：

```bash
cat ~/.local/share/personal-agent-memory/api-key
```

安装器不会自动打开浏览器，也不会打印 API 密钥。

自定义状态目录和端口：

```bash
./install.sh \
  --state-dir "$HOME/data/personal-agent-memory" \
  --port 17331
```

端口会写入安装元数据。后续升级默认复用该端口。记忆区域在管理台创建。可选的
`--library-root` 只用于保护既有用户内容，不参与 mem2.0 记忆读写。

卸载命令：

```bash
personal-agent-memory-uninstall
```

普通卸载保留平台状态、API 密钥和安装元数据。如需同时删除平台状态和安装元数据：

```bash
personal-agent-memory-uninstall --purge
```

`--purge` 会删除 `mem2/` 中的记忆、图数据和审计记录。执行前必须完成备份。

## 5 分钟上手

以下流程适用于执行过 `./install.sh` 的 Linux 用户。

### 1. 确认服务和网页

```bash
systemctl --user status personal-agent-memory.service
```

确认状态为 `active (running)` 后，打开 `http://127.0.0.1:7331`。输入以下文件中的
API 密钥：

```text
~/.local/share/personal-agent-memory/api-key
```

管理台登录会话固定有效期为 24 小时。会话使用 HttpOnly Cookie。退出、会话过期或
API 密钥轮换后，需要重新登录。

### 2. 配置模型服务

嵌入模型和图抽取模型使用 OpenAI 兼容接口。两个服务均为可选。嵌入服务不可用时，
全文召回仍可使用。图服务不可用时，直接命中仍可使用。

密钥文件每行只放 API 密钥原文：

```text
~/.config/personal-agent-memory/embedding.key
~/.config/personal-agent-memory/graph.key
```

设置权限：

```bash
chmod 600 ~/.config/personal-agent-memory/*.key
```

SiliconFlow 配置示例：

- 嵌入模型：`Qwen/Qwen3-Embedding-0.6B`
- 图抽取模型：`Qwen/Qwen3.6-35B-A3B`

编辑用户级服务配置：

```bash
systemctl --user edit --drop-in=model.conf personal-agent-memory.service
```

写入以下内容。请把 `/home/epq` 替换为当前用户的 home 目录：

```ini
[Service]
ExecStart=
ExecStart="/home/epq/.local/bin/personal-agent-memory" serve \
  --state-dir "/home/epq/.local/share/personal-agent-memory" \
  --host "127.0.0.1" \
  --port "7331" \
  --embedding-url "https://api.siliconflow.cn" \
  --embedding-model "Qwen/Qwen3-Embedding-0.6B" \
  --embedding-dimension "1024" \
  --embedding-api-key-file "/home/epq/.config/personal-agent-memory/embedding.key" \
  --graph-url "https://api.siliconflow.cn" \
  --graph-model "Qwen/Qwen3.6-35B-A3B" \
  --graph-api-key-file "/home/epq/.config/personal-agent-memory/graph.key"
```

程序会在地址后请求 `/v1/embeddings` 和 `/v1/chat/completions`。因此填写
`https://api.siliconflow.cn`，不要重复写 `/v1`。

保存后重载并重启：

```bash
systemctl --user daemon-reload
systemctl --user restart personal-agent-memory.service
systemctl --user is-active personal-agent-memory.service
```

### 3. 创建记忆区域并绑定项目

在管理台打开“记忆区域”页面。创建专题或项目记忆区域。建议为专题和项目指定稳定的
区域标识。用户区域使用固定标识 `user`。

在“项目绑定”页面填写项目根目录，并选择项目记忆区域。工作树需要共享主项目记忆时，
必须显式创建工作树继承绑定。系统不会根据目录名推断绑定。

也可以使用 REST 接口：

```bash
export PERSONAL_AGENT_MEMORY_API_KEY="$(
  cat "$HOME/.local/share/personal-agent-memory/api-key"
)"

curl -sS \
  -H "Authorization: Bearer $PERSONAL_AGENT_MEMORY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"kind":"project","name":"示例项目","region_id":"example-project"}' \
  http://127.0.0.1:7331/api/v1/regions

curl -sS \
  -H "Authorization: Bearer $PERSONAL_AGENT_MEMORY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"project_root":"/home/epq/Workspace/example","region_id":"example-project"}' \
  http://127.0.0.1:7331/api/v1/project-bindings
```

### 4. 检索记忆

默认检索使用项目工作目录。系统解析项目绑定，并只查询当前项目记忆区域：

```bash
curl -sS \
  -H "Authorization: Bearer $PERSONAL_AGENT_MEMORY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"cwd":"/home/epq/Workspace/example","query":"项目使用了什么数据库","limit":10,"token_budget":10000}' \
  http://127.0.0.1:7331/api/v1/search
```

跨区域召回必须显式列出用户、专题或项目区域：

```bash
curl -sS \
  -H "Authorization: Bearer $PERSONAL_AGENT_MEMORY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"region_ids":["user","example-project"],"query":"项目经验和用户偏好","limit":10,"token_budget":10000}' \
  http://127.0.0.1:7331/api/v1/search
```

跨区域响应包含 `region_groups`。每个结果保留来源区域和来源引用。

### 5. 在 Codex 中使用

用户级安装器自动注册 Codex 插件。插件通过 Hook 捕获明确的用户提示词和助手最终回复。
项目会话写入项目记忆区域。绑定专题的学习会话写入专题记忆区域。无项目、无专题的会话
写入会话记忆区域。

服务暂时不可用时，Hook 使用有界的本地暂存区，并在后续运行中重放。Hook 不阻塞
Codex。插件不读取隐藏推理、原始工具输出、完整文件或召回的记忆内容。

MCP 连接使用本机回环地址和 API 密钥。MCP 只公开读取工具：

- `memory_search`：按项目绑定、单个区域或显式区域集合检索记忆。
- `memory_get`：按区域和记忆标识读取一个记忆单元及其版本、生命周期和来源。

MCP 不提供写入、审批或治理工具。REST 接口不提供独立审批队列。记忆治理由已认证的
管理台或受保护的后台接口执行。

## 记忆写入和自动晋升

项目采集默认写入项目记忆区域。专题会话写入专题记忆区域。其他会话写入会话记忆区域。
会话记忆默认保留 30 天。系统可按保留策略归档或遗忘过期内容。已晋升单元不受会话清理
影响。

`evolve` 负责巩固采集内容。自动晋升支持多个目标：

- 稳定个人事实进入用户记忆。
- 可泛化技能或课程知识进入匹配的专题记忆。
- 项目特定事实留在项目记忆。
- 临时内容留在会话记忆。
- 无法确定目标的内容保留在来源区域。

晋升需要明确且高置信度的陈述，或至少两个独立来源的证据。敏感内容、冲突内容和证据
不足的内容不自动晋升。晋升创建带 `provenance` 的新单元。来源单元保留。晋升操作
幂等，并写入审计记录。

## 检索、图和预算

检索同时使用 BM25 全文召回和稠密向量召回。全文索引使用 Jieba 分析器。嵌入服务未配置
时，系统使用全文召回并报告降级状态。

图节点对应真实记忆单元。图边保存在本地 Milvus Lite collection 中。图扩展默认一跳，
最大两跳。图数据不可用时，检索返回直接命中，并报告降级原因。

每次检索默认使用 10,000 token 预算。10,000 token 也是绝对上限。跨区域召回默认分配
如下：

- 用户记忆最多 30%。
- 每个选中专题或项目最多 20%。
- 图扩展最多 20%。
- 来源和结构信息最多 10%。

预算可通过请求参数调整。调用方可以请求更小的总预算。

召回内容标记为不可信数据。召回内容不携带策略、工具授权或命令语义。

## 检查和治理

管理台通过后台 `inspect`、`update`、`delete` 和 `audit` 操作治理记忆。用户可以：

- 按区域查看记忆单元、版本、生命周期和来源。
- 使用 `supersede` 更新并保留旧版本。
- 使用 `overwrite` 原地更新，并把旧内容写入审计。
- 使用 `forget`、`archive`、`downweight` 或 `purge` 删除记忆。
- 恢复旧内容。恢复操作创建新版本，不重写审计历史。
- 查看审计事件和来源链。
- 以只读方式导出记忆。导出不构成写入路径。

审计事件保存在 `mem2/audit.sqlite3`。服务重启后，审计记录仍可查询。

## 服务管理

```bash
systemctl --user status personal-agent-memory.service
systemctl --user start personal-agent-memory.service
systemctl --user stop personal-agent-memory.service
systemctl --user restart personal-agent-memory.service
journalctl --user -u personal-agent-memory.service --no-pager
```

健康检查：

```bash
curl -sS -o /dev/null -w '%{http_code}\n' \
  -H "Authorization: Bearer $(cat "$HOME/.local/share/personal-agent-memory/api-key")" \
  http://127.0.0.1:7331/health/live
```

常见状态：

- `unbound`：当前项目根目录没有项目绑定。
- `degraded`：一个或多个召回通道不可用。其他通道仍返回结果。
- `401` 或 `403`：API 密钥为空、权限不正确或密钥无效。
- `mem2 state version` 错误：状态目录版本不兼容。停止服务并恢复正确备份。

## 开发环境

开发和测试需要 Python 3.11-3.13、`uv`、Node.js 和 Git：

```bash
uv sync --all-extras
uv run personal-agent-memory serve \
  --state-dir ~/.local/share/personal-agent-memory
```

开发命令不会创建用户级服务，也不会自动安装全局 Codex 插件。服务只接受回环地址。

测试和静态检查：

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

原生安装验收：

```bash
./scripts/verify-native-install.sh
```

完整 Linux 安装器验收：

```bash
PAM_INSTALL_ACCEPTANCE_DEDICATED_USER=1 \
  ./scripts/verify-linux-installer.sh
```

## 备份和恢复

`mem2/` 是记忆、图数据和审计记录的唯一持久化位置。备份必须包含整个
`~/.local/share/personal-agent-memory/mem2/` 目录，并保留其中的 `VERSION` 文件。
不要只复制单个 SQLite 文件。

为获得一致快照，先停止服务：

```bash
systemctl --user stop personal-agent-memory.service
umask 077
tar -C "$HOME/.local/share" \
  -czf "$HOME/personal-agent-memory-state.tgz" \
  personal-agent-memory
systemctl --user start personal-agent-memory.service
```

自定义状态目录时，请对 `--state-dir` 指定的目录执行相同操作。

配置和密钥可在单独备份中包含：

```bash
umask 077
tar -C "$HOME/.config" \
  -czf "$HOME/personal-agent-memory-config.tgz" \
  personal-agent-memory
```

备份文件可能包含 API 密钥和模型密钥。请限制备份文件权限，并使用加密存储。

恢复时先停止服务。把备份恢复到原状态目录。确认 `mem2/VERSION` 内容为
`jiuwen-memory-0.2.0`。然后启动服务并执行健康检查。

系统不读取其他版本的状态目录，也不迁移其他版本的数据。需要重置时，先备份，再停止
服务，并由用户显式删除状态目录。

## 支持范围和限制

原生安装目标是 Linux x86_64。已测试 Ubuntu 22.04 和 Ubuntu 24.04。需要
Python 3.11-3.13、`uv`、Node.js、Codex CLI、Git 和用户级 `systemd`。

当前系统只服务单个本地用户。服务只监听回环地址。系统不提供远程同步、多用户协作、
公开或局域网服务、Codex Cloud、系统级服务或自动后台升级。

嵌入和图抽取服务是可选外部服务。请自行保护模型密钥。系统不会把密钥写入记忆、图数据
或审计事件。
