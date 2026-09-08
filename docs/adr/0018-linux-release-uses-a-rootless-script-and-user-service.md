# Linux 发布采用无 root 安装脚本与用户级服务

首个 Linux 发行流程面向 Ubuntu 22.04、Ubuntu 24.04 及其他使用 systemd 的主流 x86_64 Linux，提供 `install.sh` 和 `uninstall.sh`，通过 `systemd --user` 在安装后立即启动记忆守护进程并随用户登录自动启动。安装器不调用 `sudo`：它可以自动安装 `uv`，但只检查 Python、Git、Node.js、Codex 等其他依赖并在缺失时给出明确的人工安装指引；现有 `uv tool install` 方式继续作为高级安装入口。首版不承诺 Alpine、无 systemd 环境、RPM 或 AppImage 支持，待脚本安装与升级流程稳定后再提供 `.deb`。

默认平台状态目录为 `~/.local/share/personal-agent-memory`，默认记忆库根目录为 `~/memory-libraries`，两者均允许通过安装参数覆盖。GitHub Release 同时发布源码归档与可下载审阅的 `install.sh`，文档不推荐直接执行 `curl | sh`。升级通过重新运行新版安装器完成：停止用户服务，升级 Python 程序和 Codex 插件，保留数据与配置，再启动服务并执行健康检查。

`uninstall.sh` 默认仅移除程序、Codex 插件和用户级 systemd 服务，保留平台状态与全部记忆库；显式 `--purge` 只额外删除平台状态，永远不自动删除用户的记忆库。安装完成后不自动打开浏览器，而是在终端显示服务状态、Web 地址、API 密钥文件位置以及启动、停止和日志命令。
