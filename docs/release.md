# Personal Agent Memory Linux 发行说明

本文说明 Linux 发行版的生命周期。推荐从同一 GitHub Release 下载 `install.sh`、
`uninstall.sh` 和对应源码归档。先审查本地文件，再执行安装。生产安装不要使用
`curl | sh`。

## 产品范围

本发行版只使用 `JiuwenMemory 0.2.0` 记忆内核。系统通过用户、专题、项目和会话记忆
区域工作。项目根目录通过显式项目绑定关联到项目记忆区域。

长期记忆保存在平台状态目录的 `mem2/` 子目录。该目录包含 `VERSION` 文件，内容为
`jiuwen-memory-0.2.0`。记忆单元使用 SQLite KV。稠密向量、BM25 全文索引和图节点边
使用 Milvus Lite。审计事件使用持久化 SQLite。系统不要求 Docker。

本发行版支持：

- 项目会话、专题会话和普通会话的自动采集。
- 多目标晋升、来源引用和幂等审计。
- 同一项目内的默认召回。
- 用户显式发起的跨区域召回。
- 后台 `inspect`、`update`、`delete` 和 `audit` 治理。
- 持久化审计、图扩展和 10,000 token 预算。
- 本机 Codex Hook、MCP 和管理台。

MCP 只公开读取工具：`memory_search` 和 `memory_get`。系统不提供独立审批队列、
审批 REST 接口或审批 MCP 工具。

## 支持矩阵

已测试的发行目标是 Ubuntu 22.04 和 Ubuntu 24.04 x86_64。

其他主流 x86_64 Linux 发行版可以使用本发行版，但需要具备以下条件：

- 用户级 `systemd`。
- Python 3.11-3.13。
- `uv`。
- Node.js。
- Codex CLI。
- Git。Git 用于验证工作树继承绑定。

其他发行版属于能力兼容目标，不属于已测试承诺。

守护进程只监听回环地址，并以当前用户身份通过 `systemd --user` 运行。安装、升级和
卸载不需要 `sudo`，不调用系统包管理器，也不写系统级服务文件。服务在用户登录后启动。
安装器不启用 systemd linger，也不保证用户注销后的可用性。

Alpine、非 systemd 环境、ARM、Windows 和 macOS 不属于本发行版范围。

## 发行资产

每个 Release 必须发布以下匹配资产：

- `install.sh`：无特权安装和原地升级入口。
- `uninstall.sh`：稳定卸载入口。安装器会把它复制到用户可执行目录。
- 同一标签的源码归档。归档包含 `pyproject.toml`、`uv.lock`、插件目录和两个脚本。

执行安装前，请审查下载的脚本。请确认 `pyproject.toml` 中的版本与插件清单一致。
请保留源码归档以便审计。删除源码归档后，已安装的卸载器仍可使用。

`uv.lock` 必须锁定 `JiuwenMemory==0.2.0`、`milvus-lite==3.2.2`、
`pymilvus==3.0.2` 和其他必需依赖。

## 自动安装

在包含已审查发行资产的目录执行：

```bash
chmod +x install.sh uninstall.sh
./install.sh
```

安装器检查 Linux、x86_64、Python、`uv`、Node.js、Codex CLI、Git 和用户级
systemd 总线。

如果 `uv` 缺失，安装器把 `uv` 安装到用户可执行目录。此过程不使用包管理器，也不
提升权限。其他必需依赖缺失时，安装器在部分安装前停止，并输出修复命令。

安装器执行以下操作：

1. 创建用户级服务。
2. 安装发行版 Python 工具和全局 Codex 插件。
3. 启动服务。
4. 等待带认证的健康检查。
5. 校验 `mem2/VERSION` 和 MCP 读取工具。

安装器不会打开浏览器，也不会打印 API 密钥内容。

默认位置如下：

- 平台状态：`~/.local/share/personal-agent-memory`
- 记忆状态：`~/.local/share/personal-agent-memory/mem2`
- 安装元数据：`~/.config/personal-agent-memory/install.json`
- 用户级服务：`~/.config/systemd/user/personal-agent-memory.service`

需要自定义状态目录或端口时，使用显式参数：

```bash
./install.sh \
  --state-dir "$HOME/data/personal-agent-memory" \
  --port 17331
```

这些路径会写入安装元数据。升级和稳定卸载器会复用这些路径。可选的
`--library-root` 只用于保护既有用户内容，不参与 mem2.0 记忆读写。记忆区域和项目
绑定在管理台创建。

## 手动安装

高级用户可以使用 `uv` 手动安装：

```bash
uv tool install .
codex plugin marketplace add "$PWD"
codex plugin add personal-agent-memory@personal-agent-memory
```

手动路径适合开发或手工管理的服务。手动路径不创建发行版管理的用户级服务，也不创建
稳定卸载器。需要完整生命周期时，请使用发行版安装器。

## 服务管理

安装后使用以下命令管理服务：

```bash
systemctl --user status personal-agent-memory.service
systemctl --user start personal-agent-memory.service
systemctl --user stop personal-agent-memory.service
systemctl --user restart personal-agent-memory.service
journalctl --user -u personal-agent-memory.service --no-pager
```

首次启动会在平台状态目录创建 `api-key` 文件，权限为 `0600`。插件默认读取该文件。
人工检查认证状态时，只在当前 shell 导出密钥：

```bash
export PERSONAL_AGENT_MEMORY_API_KEY="$(
  cat "$HOME/.local/share/personal-agent-memory/api-key"
)"
```

打开 `http://127.0.0.1:7331`，输入同一密钥，并确认服务状态为可用。安装器不会自动
打开浏览器。

## 升级和重装

使用新版本的已审查安装器执行原地升级：

```bash
./install.sh
```

安装器先停止用户服务，再替换 Python 工具、插件和服务单元。随后启动服务，并执行
带认证的健康检查。对同一版本重复执行安装器属于幂等修复。

升级保留以下数据：

- API 密钥。
- `mem2/` 中的记忆单元。
- 项目绑定和记忆区域。
- 图数据和审计记录。
- 会话采集状态。
- 模型配置和安装元数据。

升级失败时，安装器保留这些数据位置，并输出服务状态和日志命令。

本发行版不迁移其他版本的状态目录。需要重建时，请先备份，再由用户显式删除状态目录。

## 卸载

安装器把已验证的卸载器复制到用户可执行目录。卸载器不依赖原发行目录：

```bash
personal-agent-memory-uninstall
```

普通卸载停止并移除用户服务、Python 工具、Codex 插件和安装器管理的市场注册。普通
卸载保留平台状态、API 密钥、安装元数据和 `mem2/` 记忆数据。

如需同时删除平台状态和安装元数据：

```bash
personal-agent-memory-uninstall --purge
```

`--purge` 会删除 `mem2/` 中的记忆、图数据和审计记录。执行前必须完成备份。如果
安装器不能证明路径所有权，卸载器会保留该路径并报告原因。

## 发行验证

发行验收的唯一入口是：

```bash
PAM_INSTALL_ACCEPTANCE_DEDICATED_USER=1 \
  ./scripts/verify-linux-installer.sh
```

请使用一次性普通用户运行该命令，并确保该用户具备可用的 systemd 总线。该入口覆盖：

- 默认和自定义状态目录。
- `uv` 缺失时的用户级引导。
- 同版本重装和跨版本升级。
- Codex 市场和插件注册。
- Node.js Hook。
- 带认证的健康检查。
- MCP 读取工具。
- `mem2/VERSION`。
- 普通卸载和 `--purge`。
- 数据保留和故障放行。
- 项目工作树状态不变。

发行门禁必须在 Ubuntu 22.04 和 Ubuntu 24.04 x86_64 上运行。

需要明确验证 `uv` 引导时，请在未把 `uv` 加入 `PATH` 的一次性用户中运行：

```bash
PAM_INSTALL_ACCEPTANCE_DEDICATED_USER=1 \
PAM_INSTALL_ACCEPTANCE_MISSING_UV=1 \
  ./scripts/verify-linux-installer.sh
```

快速环境检查：

```bash
PAM_INSTALL_ACCEPTANCE_DEDICATED_USER=1 \
  ./scripts/verify-linux-installer.sh --check
```

仓库还保留手动 `uv` 路径的包级验收器：

```bash
./scripts/verify-native-install.sh
```

完整测试套件只使用 `JiuwenMemory 0.2.0` 路径。开发检查命令如下：

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

## 模型配置

嵌入和图抽取使用独立的 OpenAI 兼容服务。每个服务配置 URL、模型名和受保护的密钥文件。
超时、并发数和重试次数使用共享参数。嵌入模型和图抽取模型均为可选。

没有模型服务时，BM25 全文召回仍可使用。向量或图通道故障时，检索返回可用结果并报告
降级状态。发行验收只验证不依赖外部模型的 BM25 路径，不要求付费服务；如需向量或图通道，
请另行配置外部服务。

嵌入服务使用以下参数：

```text
--embedding-url
--embedding-model
--embedding-dimension
--embedding-api-key-file
```

图抽取服务使用以下参数：

```text
--graph-url
--graph-model
--graph-api-key-file
```

建议把模型密钥放在状态目录外，并把密钥文件权限设为 `0600`。服务不会把密钥写入记忆、图数据、审计事件或响应内容。

## 备份责任

`mem2/` 是记忆、图数据和审计记录的唯一持久化位置。用户必须备份整个
`<state_dir>/mem2/` 目录，并保留 `VERSION` 文件。不要只复制单个 SQLite 文件。

为获得一致快照，请先停止服务：

```bash
systemctl --user stop personal-agent-memory.service
umask 077
tar -C "$HOME/.local/share" \
  -czf "$HOME/personal-agent-memory-state.tgz" \
  personal-agent-memory
systemctl --user start personal-agent-memory.service
```

使用自定义状态目录时，请备份 `--state-dir` 指定的完整目录。

配置和密钥可以单独备份：

```bash
umask 077
tar -C "$HOME/.config" \
  -czf "$HOME/personal-agent-memory-config.tgz" \
  personal-agent-memory
```

备份可能包含 API 密钥和模型密钥。请限制备份文件权限，并使用加密存储。

恢复前先停止服务。把备份恢复到原状态目录。确认 `mem2/VERSION` 内容为
`jiuwen-memory-0.2.0`。启动服务后，执行健康检查。

## 已知限制

- Linux x86_64 是受支持目标。Windows 和 macOS 不属于发行目标。
- 服务只监听回环地址，并使用一个本地 API 密钥。
- 嵌入和图抽取依赖外部服务。外部服务故障时，部分召回通道会降级。
- 向量和图数据是可重建索引。服务重启后，持久化记忆和审计记录仍然存在。
- 会话记忆默认保留 30 天。保留策略按区域执行。
- 备份、服务监控、升级和密钥导出由用户或操作系统负责。
- 系统不迁移其他版本的状态目录。

## 明确排除项

以下能力不在本发行版范围内：

- Codex Cloud 和远程工作节点。
- 远程同步。
- 多用户或团队协作。
- 局域网或公网服务。
- OAuth 或 OIDC。
- 移动客户端。
- 不允许审计的 Agent 写入。
- `.deb`、RPM、AppImage、Snap 和 Flatpak。
- 系统级服务和自动后台升级。
