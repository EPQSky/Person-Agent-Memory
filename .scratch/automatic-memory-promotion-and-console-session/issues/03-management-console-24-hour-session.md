# 03 — 管理台 24 小时本地会话

**What to build:** 为本地管理台增加 24 小时固定有效期的服务端会话，使用户首次输入 API key 后可以持续访问管理台，刷新页面或重启记忆服务都不需要重复输入凭据。

**Blocked by:** None — can start immediately

**Status:** ready-for-agent

- [ ] API key 登录成功后签发固定有效期为 24 小时的管理台会话。
- [ ] 会话通过 HttpOnly、同源限制的 Cookie 传递，API key 不写入 `localStorage`、`sessionStorage`、URL、页面源码或普通页面状态。
- [ ] 会话状态保存在用户级平台状态中，记忆服务重启后仍能在有效期内验证会话。
- [ ] 管理台刷新、普通导航和重新加载页面不会要求再次输入 API key。
- [ ] 会话过期后，管理台显示清晰的认证失效状态并要求重新登录。
- [ ] 显式退出会立即使当前会话失效。
- [ ] API key 轮换会使旧 API key 关联的管理台会话失效。
- [ ] Bearer API key 认证继续支持 Codex Hooks、MCP 和非浏览器 REST 客户端。
- [ ] 会话认证、过期、退出、服务重启和 key 轮换有可控时间的后端测试。
- [ ] 浏览器验收确认刷新后仍可访问、页面存储和 URL 不含 API key，并覆盖过期和退出流程。
- [ ] 中文 README 和管理台登录提示说明 24 小时会话行为。
