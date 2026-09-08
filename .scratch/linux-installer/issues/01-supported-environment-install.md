# 01 — 受支持环境的一键安装

**What to build:** 为依赖已经就绪的受支持 Linux 用户提供一个无需 root 的安装入口。一次执行应安装 Personal Agent Memory 命令和全局 Codex 插件，准备默认平台状态目录与记忆库根目录，创建并启用用户级 systemd 服务，立即启动记忆守护进程，通过认证健康检查后显示可操作的安装摘要。安装后的真实 Codex Hook 必须能够连接本地服务，同时保持现有 fail-open 行为。

**Blocked by:** None — can start immediately

**Status:** done

- [x] 在依赖已就绪的 Ubuntu/systemd x86_64 隔离用户环境中，一次执行可完成安装且返回成功。
- [x] 安装使用默认平台状态目录 `~/.local/share/personal-agent-memory` 和默认记忆库根目录 `~/memory-libraries`，缺失目录会被安全创建。
- [x] Personal Agent Memory 用户级命令和全局 Codex 插件安装成功，不产生重复插件或 marketplace 记录。
- [x] 用户级 systemd 服务被创建、启用并立即启动，且只使用 `127.0.0.1:7331`。
- [x] 安装器等待 API 密钥生成并通过认证健康检查后才报告成功。
- [x] 安装摘要显示服务状态、Web 地址、API 密钥文件位置以及启动、停止、重启、状态和日志命令，但不显示密钥正文，也不打开浏览器。
- [x] 安装后的真实 Codex Hook 可以使用已安装插件连接守护进程，并保持既有的安静 fail-open 输出契约。
- [x] 完整安装流程不调用 `sudo`，不写入系统级服务位置，也不修改项目仓库状态或 Git remote。
