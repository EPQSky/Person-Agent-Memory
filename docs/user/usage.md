# Personal Agent Memory 使用说明

本文面向通过用户级安装器使用 Personal Agent Memory 的本地用户。内容包括安装、模型配置、
管理台、记忆区域、项目绑定、检索、Codex 集成、治理、备份和故障排查。

发行资产、支持矩阵、升级、卸载和发行验收见 [Linux 发行说明](release.md)。

## 1. 适用范围

运行要求：

- Linux x86_64。
- Ubuntu 22.04 或 Ubuntu 24.04。其他主流 x86_64 Linux 发行版属于能力兼容目标。
- Python 3.11-3.13。
- `uv`。
- Node.js。
- Codex CLI。
- Git。Git 用于验证工作树继承绑定。
- 用户级 `systemd`。

服务只监听回环地址，只服务单个本地用户。安装、使用和卸载不需要 `sudo`。安装器不写
系统级服务，也不启用 systemd linger。

## 2. 安装与首次启动

在已审查的发行资产目录执行：

```bash
chmod +x install.sh uninstall.sh
./install.sh
```

安装器执行以下操作：

1. 创建用户级服务。
2. 安装 Python 工具和 Codex 插件。
3. 启动服务。
4. 执行带认证的健康检查。
5. 校验 `mem2/VERSION` 和 MCP 读取工具。

如果 `uv` 缺失，安装器会把 `uv` 安装到用户可执行目录。安装器不会打开浏览器，也不会
打印 API 密钥。

默认位置：

| 内容 | 路径 |
| --- | --- |
| 平台状态 | `~/.local/share/personal-agent-memory` |
| 记忆状态 | `~/.local/share/personal-agent-memory/mem2` |
| 安装元数据 | `~/.config/personal-agent-memory/install.json` |
| 用户级服务 | `~/.config/systemd/user/personal-agent-memory.service` |
| API 密钥 | `~/.local/share/personal-agent-memory/api-key` |

自定义状态目录和端口：

```bash
./install.sh \
  --state-dir "$HOME/data/personal-agent-memory" \
  --port 17331
```

自定义路径会写入安装元数据。后续升级和卸载会复用这些路径。可选的 `--library-root`
只用于保护既有用户内容，不参与 mem2.0 记忆读写。

首次启动需要初始化本地索引。安装器在健康检查阶段超时，不代表安装必然失败。先检查服务：

```bash
systemctl --user status personal-agent-memory.service
journalctl --user -u personal-agent-memory.service -n 100 --no-pager
```

确认服务已经运行后，可以重新执行 `./install.sh` 完成幂等验收。安装器会重新安装依赖并
重启服务。如果服务没有运行，先按日志修复问题。

## 3. 登录管理台

确认服务状态：

```bash
systemctl --user status personal-agent-memory.service
```

状态应为 `active (running)`。然后打开：

```text
http://127.0.0.1:7331
```

输入 API 密钥。密钥文件只允许当前用户读取：

```bash
cat "$HOME/.local/share/personal-agent-memory/api-key"
```

不要在聊天、日志、截图或公开文档中暴露 API 密钥。

管理台登录会话固定有效期为 24 小时，使用 HttpOnly Cookie。退出、会话过期或 API 密钥
轮换后，需要重新登录。

## 4. 配置模型服务

嵌入模型和图抽取模型通过 OpenAI 兼容接口访问。两个服务均为可选。启用模型服务后，
待嵌入或图抽取的文本会发送到你配置的外部服务。请确认服务商的数据处理和保留政策。

- 未配置嵌入服务时，BM25 全文召回仍可使用。
- 未配置图服务时，直接命中仍可使用。
- 模型服务不可用时，检索返回可用结果，并报告降级状态。

### 4.1 写入密钥文件

密钥文件：

```text
~/.config/personal-agent-memory/embedding.key
~/.config/personal-agent-memory/graph.key
```

每个文件只放一行 API 密钥原文。不要写入 `Bearer`、引号、URL、注释或额外说明。

在本机 bash 终端安全写入：

```bash
umask 077
read -rsp "模型服务 API Key: " PAM_KEY; echo
printf '%s\n' "$PAM_KEY" > "$HOME/.config/personal-agent-memory/embedding.key"
printf '%s\n' "$PAM_KEY" > "$HOME/.config/personal-agent-memory/graph.key"
unset PAM_KEY
chmod 600 "$HOME"/.config/personal-agent-memory/*.key
```

嵌入和图抽取使用不同密钥时，分别写入两个文件。

### 4.2 编辑用户级服务配置

使用 drop-in 覆盖主服务文件。不要直接编辑安装器管理的 unit 文件：

```bash
systemctl --user edit --drop-in=model.conf personal-agent-memory.service
```

写入以下内容。把 `<user>` 替换为当前用户名：

```ini
[Service]
ExecStart=
ExecStart="/home/<user>/.local/bin/personal-agent-memory" serve \
  --state-dir "/home/<user>/.local/share/personal-agent-memory" \
  --library-root "/home/<user>/memory-libraries" \
  --host "127.0.0.1" \
  --port "7331" \
  --embedding-url "https://api.siliconflow.cn/v1" \
  --embedding-model "Qwen/Qwen3-Embedding-0.6B" \
  --embedding-dimension "1024" \
  --embedding-api-key-file "/home/<user>/.config/personal-agent-memory/embedding.key" \
  --graph-url "https://api.siliconflow.cn" \
  --graph-model "deepseek-ai/DeepSeek-V3.1-Terminus" \
  --graph-api-key-file "/home/<user>/.config/personal-agent-memory/graph.key"
```

修改 `ExecStart` 前必须先写一行空的 `ExecStart=`。否则 systemd 会追加参数，而不是覆盖
原参数。

### 4.3 地址和维度规则

嵌入和图抽取的 base URL 写法不同：

| 用途 | 参数 | 正确写法 | 原因 |
| --- | --- | --- | --- |
| 嵌入 | `--embedding-url` | `https://api.siliconflow.cn/v1` | 嵌入客户端使用 OpenAI SDK，SDK 自动追加 `/embeddings` |
| 图抽取 | `--graph-url` | `https://api.siliconflow.cn` | 图客户端自行追加 `/v1/chat/completions` |

不要把嵌入地址写成 `https://api.siliconflow.cn`。这会产生
`https://api.siliconflow.cn/embeddings`，服务返回 404。

不要把图地址写成 `https://api.siliconflow.cn/v1`。这会产生
`https://api.siliconflow.cn/v1/v1/chat/completions`。

`--embedding-dimension` 必须等于嵌入模型的实际输出维度。`Qwen/Qwen3-Embedding-0.6B`
使用 `1024`。不要保留默认值 `1536`。

示例模型以服务商当前可用列表为准。可用模型可通过以下接口查询：

```bash
curl -sS \
  -H "Authorization: Bearer $(cat "$HOME/.config/personal-agent-memory/embedding.key")" \
  https://api.siliconflow.cn/v1/models
```

### 4.4 重载并验证

```bash
systemctl --user daemon-reload
systemctl --user restart personal-agent-memory.service
systemctl --user is-active personal-agent-memory.service
```

验证健康状态：

```bash
export PERSONAL_AGENT_MEMORY_API_KEY="$(
  cat "$HOME/.local/share/personal-agent-memory/api-key"
)"
curl -sS \
  -H "Authorization: Bearer $PERSONAL_AGENT_MEMORY_API_KEY" \
  http://127.0.0.1:7331/mcp/health
unset PERSONAL_AGENT_MEMORY_API_KEY
```

正常状态：

```json
{
  "status": "ready",
  "memory_kernel": "mem2",
  "memory_kernel_status": "ready",
  "graph_index_status": "ready"
}
```

`degraded` 表示至少一个召回通道不可用。其他通道仍可返回结果。具体原因见
`memory_kernel_detail` 和 `graph_index_status`。

## 5. 记忆区域与项目绑定

用户区域使用固定标识 `user`。专题和项目区域在管理台创建。建议为专题和项目指定稳定、
可读的区域标识。

在管理台执行：

1. 打开“记忆区域”页面。
2. 创建专题或项目记忆区域。
3. 打开“项目绑定”页面。
4. 填写项目根目录，并选择项目记忆区域。

项目绑定必须显式创建。系统不会根据目录名推断项目。工作树需要共享主项目记忆时，必须
显式创建工作树继承绑定。

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
  -d '{"project_root":"/home/<user>/Workspace/example","region_id":"example-project"}' \
  http://127.0.0.1:7331/api/v1/project-bindings

unset PERSONAL_AGENT_MEMORY_API_KEY
```

创建专题区域时，把 `"kind"` 改为 `"topic"`。

## 6. 检索记忆

管理台提供记忆浏览和检索入口。自动化调用可以使用 REST 或 MCP。

### 6.1 默认项目召回

默认检索使用项目工作目录，只查询当前项目记忆区域：

```bash
export PERSONAL_AGENT_MEMORY_API_KEY="$(
  cat "$HOME/.local/share/personal-agent-memory/api-key"
)"

curl -sS \
  -H "Authorization: Bearer $PERSONAL_AGENT_MEMORY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"cwd":"/home/<user>/Workspace/example","query":"项目使用了什么数据库","limit":10,"token_budget":10000}' \
  http://127.0.0.1:7331/api/v1/search

unset PERSONAL_AGENT_MEMORY_API_KEY
```

### 6.2 显式跨区域召回

跨区域召回必须显式列出用户、专题或项目区域：

```bash
export PERSONAL_AGENT_MEMORY_API_KEY="$(
  cat "$HOME/.local/share/personal-agent-memory/api-key"
)"

curl -sS \
  -H "Authorization: Bearer $PERSONAL_AGENT_MEMORY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"region_ids":["user","example-project"],"query":"项目经验和用户偏好","limit":10,"token_budget":10000}' \
  http://127.0.0.1:7331/api/v1/search

unset PERSONAL_AGENT_MEMORY_API_KEY
```

跨区域响应包含 `region_groups`。每个结果保留来源区域和来源引用。

### 6.3 检索预算和降级

每次检索默认使用 10,000 token 预算。10,000 token 也是绝对上限。跨区域召回默认分配：

- 用户记忆最多 30%。
- 每个选中专题或项目最多 20%。
- 图扩展最多 20%。
- 来源和结构信息最多 10%。

调用方可以请求更小的总预算。图扩展默认一跳，最大两跳。

结果可能包含以下状态：

- `vector_index_status`：向量召回状态。
- `graph_index_status`：图索引状态。
- `degradation`：降级原因。

召回内容标记为不可信数据。召回内容不携带策略、工具授权或命令语义。

## 7. 在 Codex 中使用

用户级安装器自动注册 Codex 插件。当前 Codex 会话没有加载插件时，重启 Codex 或新建
会话。

插件通过 Hook 捕获明确的用户提示词和助手最终回复。会话标识和项目目录用于路由和隔离。
插件不读取隐藏推理、原始工具输出、完整文件或召回的记忆内容。

会话写入规则：

- 项目会话写入项目记忆区域。
- 绑定专题的学习会话写入专题记忆区域。
- 无项目、无专题的会话写入会话记忆区域。

服务暂时不可用时，Hook 使用有界本地暂存区，并在后续运行中重放。Hook 不阻塞 Codex。

采集不等于晋升。采集内容先进入对应记忆区域，只有符合晋升规则的内容才进入正式长期
记忆。

MCP 使用本机回环地址和 API 密钥，只公开以下读取工具：

- `memory_search`：按项目绑定、单个区域或显式区域集合检索记忆。
- `memory_get`：按区域和记忆标识读取一个记忆单元及其版本、生命周期和来源。

MCP 不提供写入、审批或治理工具。REST 接口不提供独立审批队列。记忆治理由已认证的
管理台或受保护的后台接口执行。

## 8. 记忆写入与自动晋升

项目采集默认写入项目记忆区域。专题会话写入专题记忆区域。其他会话写入会话记忆区域。

会话记忆默认保留 30 天。系统可以按保留策略归档或遗忘过期内容。已晋升单元不受会话
清理影响。

`evolve` 负责巩固采集内容。自动晋升支持多个目标：

- 稳定个人事实进入用户记忆。
- 可泛化技能或课程知识进入匹配的专题记忆。
- 项目特定事实留在项目记忆。
- 临时内容留在会话记忆。
- 无法确定目标的内容保留在来源区域。

晋升需要明确且高置信度的陈述，或至少两个独立来源的证据。敏感内容、冲突内容和证据
不足的内容不自动晋升。晋升创建带 `provenance` 的新单元，来源单元保留。晋升操作幂等，
并写入审计记录。

## 9. 管理台治理与审计

管理台通过后台 `inspect`、`update`、`delete` 和 `audit` 操作治理记忆。用户可以：

- 按区域查看记忆单元、版本、生命周期和来源。
- 使用 `supersede` 更新并保留旧版本。
- 使用 `overwrite` 原地更新，并把旧内容写入审计。
- 使用 `forget`、`archive`、`downweight` 或 `purge` 删除记忆。
- 恢复旧内容。恢复操作创建新版本，不重写审计历史。
- 查看审计事件和来源链。
- 以只读方式导出记忆。导出不构成写入路径。

审计事件保存在 `mem2/audit.sqlite3`。服务重启后，审计记录仍可查询。

## 10. 服务管理

```bash
systemctl --user status personal-agent-memory.service
systemctl --user start personal-agent-memory.service
systemctl --user stop personal-agent-memory.service
systemctl --user restart personal-agent-memory.service
journalctl --user -u personal-agent-memory.service --no-pager
```

健康检查：

```bash
export PERSONAL_AGENT_MEMORY_API_KEY="$(
  cat "$HOME/.local/share/personal-agent-memory/api-key"
)"
curl -sS -o /dev/null -w '%{http_code}\n' \
  -H "Authorization: Bearer $PERSONAL_AGENT_MEMORY_API_KEY" \
  http://127.0.0.1:7331/health/live
unset PERSONAL_AGENT_MEMORY_API_KEY
```

服务以当前用户身份运行，并随用户级 systemd 启动。安装器不启用 linger。用户注销后
服务是否继续运行，取决于系统的用户级 systemd 配置。

## 11. 备份与恢复

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

自定义状态目录时，请对 `--state-dir` 指定的完整目录执行相同操作。

配置和密钥可以单独备份：

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

## 12. 故障排查

### 12.1 嵌入服务返回 NotFoundError

原因通常是 `--embedding-url` 缺少 `/v1`，或模型名不可用。

检查：

```bash
journalctl --user -u personal-agent-memory.service -n 100 --no-pager
```

确认嵌入地址为：

```text
https://api.siliconflow.cn/v1
```

确认模型存在于服务商模型列表中：

```bash
curl -sS \
  -H "Authorization: Bearer $(cat "$HOME/.config/personal-agent-memory/embedding.key")" \
  https://api.siliconflow.cn/v1/models
```

### 12.2 密钥文件缺失或为空

如果 `memory_kernel_detail` 包含 `FileNotFoundError` 或 `credential is empty`，检查：

```bash
stat -c '%n %a %s' \
  "$HOME/.config/personal-agent-memory/embedding.key" \
  "$HOME/.config/personal-agent-memory/graph.key"
```

文件权限应为 `600`，文件大小应大于零。

### 12.3 嵌入维度不匹配

如果 `memory_kernel_detail` 包含 `embedding_dimension_mismatch`，确认
`--embedding-dimension` 与模型实际输出维度一致。

`Qwen/Qwen3-Embedding-0.6B` 应使用 `1024`。

### 12.4 管理台返回 401 或 403

检查 API 密钥：

```bash
ls -l "$HOME/.local/share/personal-agent-memory/api-key"
```

重新读取密钥并登录。API 密钥轮换后，旧管理台会话会失效。

### 12.5 项目检索返回 unbound

当前项目根目录没有项目绑定。打开管理台“项目绑定”页面，创建绑定。工作树继承也需要
显式创建继承绑定。

### 12.6 mem2 版本不兼容

如果日志提示 `mem2 state version` 错误，停止服务并恢复正确的备份。系统不会自动迁移
其他版本的状态目录。

### 12.7 安装器健康检查超时

首次启动需要初始化本地索引。先检查：

```bash
systemctl --user status personal-agent-memory.service
journalctl --user -u personal-agent-memory.service -n 100 --no-pager
```

服务已运行后，可以重新执行 `./install.sh` 完成幂等验收。服务未运行时，先修复日志中的
错误。

## 13. 升级、卸载与发行验收

升级时使用新版本的已审查安装器重新执行：

```bash
./install.sh
```

普通卸载：

```bash
personal-agent-memory-uninstall
```

普通卸载保留平台状态、API 密钥、安装元数据和 `mem2/` 记忆数据。彻底删除状态：

```bash
personal-agent-memory-uninstall --purge
```

`--purge` 会删除记忆、图数据和审计记录。执行前必须完成备份。

发行资产、升级细节、卸载所有权检查和发行验收命令见
[Linux 发行说明](release.md)。

## 14. 相关文档

- [README](../../README.md)：产品介绍和能力概览。
- [Linux 发行说明](release.md)：发行、升级、卸载和验收。
- [领域词汇表](../../CONTEXT.md)：稳定术语和领域边界。
- [架构决策记录](../adr/)：持久架构决策及其权衡。
