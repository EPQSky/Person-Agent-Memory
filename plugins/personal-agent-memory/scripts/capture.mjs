import { createHash, createHmac, randomUUID } from "node:crypto";
import {
  mkdir,
  open,
  opendir,
  readFile,
  rename,
  rm,
  rmdir,
  stat,
  unlink,
  writeFile,
} from "node:fs/promises";
import { isAbsolute, join } from "node:path";

const MAX_INPUT_BYTES = 256 * 1024;
const MAX_CONTENT_BYTES = 64 * 1024;
const MAX_SPOOL_BYTES = 4 * 1024 * 1024;
const MAX_SPOOL_FILES = 256;
const SPOOL_TTL_MS = 7 * 24 * 60 * 60 * 1000;
const TEMP_STALE_MS = 30 * 1000;
const REPLAY_BUDGET_MS = 250;
const REPLAY_RECORD_BUDGET = 8;
const HOOK_BUDGET_MS = 1500;
const MAINTENANCE_SCAN_BUDGET = 512;
const MAX_WRITER_SLOTS = 56;
const MAX_STATE_FILES = 256;
const MAX_STATE_BYTES = 1024 * 1024;
const TURN_TTL_MS = 24 * 60 * 60 * 1000;
const IDENTITY_TTL_MS = 60 * 60 * 1000;
const PERSISTENCE_DEADLINE_HEADER = "x-personal-agent-memory-persistence-deadline-ms";

function inspectSensitive(content) {
  const patterns = [
    ["private-key", /-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----/i],
    ["aws-access-key", /\b(?:AKIA|ASIA)[A-Z0-9]{16}\b/],
    ["github-token", /\bgh[pousr]_[A-Za-z0-9]{20,}\b/],
    ["api-token", /\bsk[-\s_]+[A-Za-z0-9_-]{20,}\b/i],
    ["bearer-token", /\bBearer\s+[A-Za-z0-9._~+/=-]{20,}\b/i],
  ];
  const assignedSecretKey = /(?<![a-z0-9_-])["']?((?:[a-z0-9]+[_-])*(?:api[\s_-]*key|access[\s_-]*token|client[\s_-]*secret|secret[\s_-]*access[\s_-]*key|password|passwd|pass|secret|token))\b["']?\s*(?::|=)\s*/gi;
  const compact = content.replace(/[\s\\]+/g, "");
  const placeholder = /\b(?:example|sample|placeholder|redacted|masked|dummy|fake|not-a-secret|\*{4,})\b/i;
  const assignedPlaceholder = /^(?:example|sample|placeholder|redacted|masked|dummy|fake|not-a-secret|\*{4,})$/i;
  const benignAssignedValue = (value) => {
    const candidate = value.trim().replace(/[,.;)]+$/, "");
    const normalized = candidate.replace(/[\s_]+/g, "-");
    const benign = /^(?:\$\{?[A-Z][A-Z0-9_]*\}?|(?:env|environment)(?::|[._-])?[A-Z][A-Z0-9_]*|vault:\/\/[a-z0-9._-]+(?:\/[a-z0-9._-]+)+#[a-z0-9._-]+|(?:generated|managed|configured|injected|provided|resolved|loaded|fetched)(?:[-_](?:at|by|from|during|via)[-_][a-z0-9_-]+)+)$/i;
    return assignedPlaceholder.test(candidate) || benign.test(candidate) || benign.test(normalized);
  };
  const assignedSecretValues = [];
  for (const match of content.matchAll(assignedSecretKey)) {
    const start = match.index + match[0].length;
    const quote = ['"', "'"].includes(content[start]) ? content[start] : undefined;
    const valueStart = quote ? start + 1 : start;
    let end = valueStart;
    if (quote) {
      let escaped = false;
      while (end < content.length) {
        const character = content[end];
        if ((character === "\n" || character === "\r") && !escaped) break;
        if (character === quote && !escaped) break;
        escaped = character === "\\" && !escaped;
        if (character !== "\\") escaped = false;
        end += 1;
      }
    } else {
      const boundaries = [content.indexOf(";", valueStart), content.indexOf("\n", valueStart)]
        .filter((boundary) => boundary >= 0);
      end = boundaries.length ? Math.min(...boundaries) : content.length;
    }
    assignedSecretValues.push([match[1], content.slice(valueStart, end).trim()]);
  }
  const categories = patterns.filter(([, pattern]) => pattern.test(content)).map(([name]) => name);
  if (assignedSecretValues.some(([, value]) => {
    const candidate = value.trim().replace(/[,.;)]+$/, "");
    return candidate.length > 0 && !benignAssignedValue(candidate);
  })) {
    categories.push("assigned-secret");
  }
  for (const [name, pattern] of [
    ["api-token", /\bsk-[A-Za-z0-9_-]{20,}\b/i],
    ["github-token", /\bgh[pousr]_[A-Za-z0-9]{20,}\b/],
    ["aws-access-key", /\b(?:AKIA|ASIA)[A-Z0-9]{16}\b/],
  ]) {
    if (pattern.test(compact) && !categories.includes(name)) categories.push(name);
  }
  if (categories.length) return { disposition: "discarded", categories };
  const uncertain = /\b(?:credential|api[\s_-]*key|access[\s_-]*token|password|passwd|private[\s_-]*key|secret|token)\b(?:\s+\w+){0,3}\s+(?:may|might|could|possibly)\b(?:\s+\w+){0,3}/i.exec(content);
  if (uncertain && !placeholder.test(uncertain[0])) {
    return { disposition: "quarantined", categories: ["sensitive-language"] };
  }
}

async function quarantineSensitive(content, finding, key, deadline) {
  const directories = await spoolDirectories();
  if (!directories) return;
  const fingerprint = `opaque:${randomUUID()}`;
  const record = {
    disposition: finding.disposition,
    categories: finding.categories,
    summary: `Sensitive content withheld; ${Buffer.byteLength(content)} bytes; reference ${fingerprint}.`,
    fingerprint,
    created_at: new Date().toISOString(),
  };
  const identity = createHmac("sha256", key).update(content).digest("hex");
  const path = join(directories.quarantine, `sensitive-${identity}.json`);
  try {
    await writeFile(path, JSON.stringify(record), {
      encoding: "utf8",
      mode: 0o600,
      flag: "wx",
    });
  } catch (error) {
    if (error?.code !== "EEXIST") throw error;
  }
  const maintained = await withSpoolLock(directories, async () => {
    await pruneSpool(directories, path, deadline);
  }, deadline);
  if (!maintained && Date.now() < deadline) await pruneSpool(directories, path, deadline);
}

async function reportSensitiveCapture(content, finding, key, hookDeadline, eventKind = "user") {
  const remaining = hookDeadline - Date.now();
  if (remaining > 0) {
    const withheld = {
      event_id: randomUUID(),
      session_id: "withheld",
      turn_id: randomUUID(),
      event_kind: eventKind,
      content,
      occurred_at: new Date().toISOString(),
      cwd: "/",
    };
    try {
      if (await request("/api/v1/capture/events", withheld, key, remaining) === "accepted") return;
    } catch {}
  }
  await quarantineSensitive(content, finding, key, hookDeadline).catch(() => {});
}

async function readStdin() {
  const chunks = [];
  let size = 0;
  for await (const chunk of process.stdin) {
    size += chunk.length;
    if (size > MAX_INPUT_BYTES) return;
    chunks.push(chunk);
  }
  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    return;
  }
}

async function apiKey() {
  const direct = process.env.PERSONAL_AGENT_MEMORY_API_KEY;
  if (direct?.trim()) return direct.trim();
  const path =
    process.env.PERSONAL_AGENT_MEMORY_API_KEY_FILE ??
    join(process.env.HOME ?? "", ".local", "share", "personal-agent-memory", "api-key");
  if (!isAbsolute(path)) return;
  try {
    const key = (await readFile(path, "utf8")).trim();
    return key || undefined;
  } catch {
    return;
  }
}

function daemonUrl(path) {
  const configured = process.env.PERSONAL_AGENT_MEMORY_URL ?? "http://127.0.0.1:7331";
  const parsed = new URL(configured);
  if (
    parsed.protocol !== "http:" ||
    !["127.0.0.1", "localhost", "::1", "[::1]"].includes(parsed.hostname) ||
    parsed.username ||
    parsed.password ||
    parsed.search ||
    parsed.hash
  ) {
    throw new Error("capture daemon must be loopback HTTP");
  }
  return new URL(path, `${parsed.href.replace(/\/$/, "")}/`);
}

function dataRoot() {
  const configured = process.env.PERSONAL_AGENT_MEMORY_SPOOL_DIR ?? process.env.PLUGIN_DATA;
  if (!configured || !isAbsolute(configured)) return;
  return join(configured, "capture");
}

function digest(...values) {
  const hash = createHash("sha256");
  for (const value of values) hash.update(String(value)).update("\0");
  return hash.digest("hex");
}

function statePath(root, namespace, key) {
  return join(root, namespace, `${key}.json`);
}

async function readState(root, namespace, key, ttl) {
  try {
    const record = JSON.parse(await readFile(statePath(root, namespace, key), "utf8"));
    if (Date.now() - record.updated_at > ttl) return;
    return record;
  } catch {
    return;
  }
}

async function writeState(root, namespace, key, record) {
  const directory = join(root, namespace);
  await mkdir(directory, { recursive: true, mode: 0o700 });
  await writeFile(
    statePath(root, namespace, key),
    JSON.stringify({ ...record, updated_at: Date.now() }),
    { encoding: "utf8", mode: 0o600 },
  );
}

async function activeTurn(event, kind, content) {
  if (typeof event.turn_id === "string" && event.turn_id.trim()) return event.turn_id.slice(0, 200);
  const root = dataRoot();
  if (!root) return randomUUID();
  const activeKey = digest(event.session_id, event.cwd);
  if (kind === "user") {
    const turnId = randomUUID();
    await writeState(root, "turns", activeKey, { turn_id: turnId });
    return turnId;
  }
  const identityKey = digest(event.session_id, event.cwd, content);
  const completed = await readState(root, "identities", identityKey, IDENTITY_TTL_MS);
  if (typeof completed?.turn_id === "string") return completed.turn_id;
  const active = await readState(root, "turns", activeKey, TURN_TTL_MS);
  return typeof active?.turn_id === "string" ? active.turn_id : randomUUID();
}

async function rememberCompletedIdentity(event, content, turnId, eventId) {
  if (typeof event.turn_id === "string" && event.turn_id.trim()) return;
  const root = dataRoot();
  if (!root) return;
  const identityKey = digest(event.session_id, event.cwd, content);
  await writeState(root, "identities", identityKey, { turn_id: turnId, event_id: eventId });
}

async function pruneState(deadline) {
  const root = dataRoot();
  if (!root) return;
  for (const [namespace, ttl] of [["turns", TURN_TTL_MS], ["identities", IDENTITY_TTL_MS]]) {
    const directory = join(root, namespace);
    try {
      const entries = [];
      let scanned = 0;
      for await (const entry of await opendir(directory)) {
        if (scanned >= MAINTENANCE_SCAN_BUDGET || Date.now() >= deadline) break;
        scanned += 1;
        if (!entry.isFile() || !/^[a-f0-9]{64}\.json$/.test(entry.name)) continue;
        const path = join(directory, entry.name);
        try {
          const metadata = await stat(path);
          const record = JSON.parse(await readFile(path, "utf8"));
          if (
            metadata.size > MAX_STATE_BYTES / MAX_STATE_FILES ||
            Date.now() - record.updated_at > ttl
          ) {
            await unlink(path);
          } else {
            entries.push({ path, size: metadata.size, modified: record.updated_at });
          }
        } catch {
          await unlink(path).catch(() => {});
        }
      }
      entries.sort((left, right) => left.modified - right.modified);
      let total = entries.reduce((sum, entry) => sum + entry.size, 0);
      let count = entries.length;
      for (const entry of entries) {
        if (count <= MAX_STATE_FILES && total <= MAX_STATE_BYTES) break;
        await unlink(entry.path).catch(() => {});
        count -= 1;
        total -= entry.size;
      }
    } catch {}
  }
  const writers = join(root, "writers");
  try {
    let scanned = 0;
    for await (const entry of await opendir(writers)) {
      if (scanned >= MAX_WRITER_SLOTS || Date.now() >= deadline) break;
      scanned += 1;
      if (!entry.isDirectory() || !/^[a-f0-9]{2}$/.test(entry.name)) continue;
      const path = join(writers, entry.name);
      try {
        if (Date.now() - (await stat(path)).mtimeMs >= TEMP_STALE_MS) {
          await rm(path, { recursive: true, force: true });
        }
      } catch {}
    }
  } catch {}
}

function requestTimeout() {
  return Math.min(
    2000,
    Math.max(100, Number.parseInt(process.env.PERSONAL_AGENT_MEMORY_TIMEOUT_MS ?? "500", 10) || 500),
  );
}

async function request(path, body, key, timeout = requestTimeout()) {
  const effectiveTimeout = Math.max(1, Math.min(Math.floor(timeout), requestTimeout()));
  const persistenceDeadline = Date.now() + effectiveTimeout;
  const response = await fetch(daemonUrl(path), {
    method: "POST",
    headers: {
      authorization: `Bearer ${key}`,
      "content-type": "application/json",
      [PERSISTENCE_DEADLINE_HEADER]: String(persistenceDeadline),
    },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(effectiveTimeout),
  });
  if (response.ok) return "accepted";
  if ([400, 404, 409, 413, 422].includes(response.status)) return "permanent";
  return "transient";
}

async function spoolDirectories() {
  const root = dataRoot();
  if (!root) return;
  const spool = join(root, "spool");
  const quarantine = join(root, "quarantine");
  await mkdir(spool, { recursive: true, mode: 0o700 });
  await mkdir(quarantine, { recursive: true, mode: 0o700 });
  const writers = join(root, "writers");
  await mkdir(writers, { recursive: true, mode: 0o700 });
  return { spool, quarantine, writers };
}

async function acquireWriterSlot(directories, deadline) {
  const start = Number.parseInt(randomUUID().slice(0, 8), 16) % MAX_WRITER_SLOTS;
  while (Date.now() < deadline) {
    for (let offset = 0; offset < MAX_WRITER_SLOTS; offset += 1) {
      const index = (start + offset) % MAX_WRITER_SLOTS;
      const slot = join(directories.writers, index.toString(16).padStart(2, "0"));
      try {
        await mkdir(slot, { mode: 0o700 });
        return slot;
      } catch (error) {
        if (error?.code !== "EEXIST") continue;
        try {
          if (Date.now() - (await stat(slot)).mtimeMs >= TEMP_STALE_MS) {
            await rm(slot, { recursive: true, force: true });
          }
        } catch {}
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  throw new Error("capture writer capacity exhausted");
}

async function withSpoolLock(directories, operation, deadline = Number.POSITIVE_INFINITY) {
  const lock = join(directories.spool, ".maintenance-lock");
  for (let attempt = 0; attempt < 20 && Date.now() < deadline; attempt += 1) {
    try {
      await mkdir(lock, { mode: 0o700 });
      try {
        await operation();
        return true;
      } finally {
        await rmdir(lock).catch(() => {});
      }
    } catch (error) {
      if (error?.code !== "EEXIST") return false;
      try {
        if (Date.now() - (await stat(lock)).mtimeMs > 5000) await rmdir(lock);
      } catch {}
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
  }
  return false;
}

async function spoolEntries(directories, deadline = Number.POSITIVE_INFINITY) {
  const entries = [];
  let scanned = 0;
  for await (const entry of await opendir(directories.spool)) {
    if (scanned >= MAINTENANCE_SCAN_BUDGET || Date.now() >= deadline) break;
    scanned += 1;
    if (!entry.isFile() || !/^event-[a-f0-9-]+\.json$/.test(entry.name)) continue;
    const path = join(directories.spool, entry.name);
    try {
      const metadata = await stat(path);
      entries.push({ path, name: entry.name, size: metadata.size, modified: metadata.mtimeMs });
    } catch {}
  }
  entries.sort((left, right) => left.modified - right.modified || left.name.localeCompare(right.name));
  return entries;
}

async function staleTemporaryEntries(directories, deadline = Number.POSITIVE_INFINITY) {
  const entries = [];
  const now = Date.now();
  let scanned = 0;
  for await (const entry of await opendir(directories.spool)) {
    if (scanned >= MAINTENANCE_SCAN_BUDGET || Date.now() >= deadline) break;
    scanned += 1;
    if (!entry.isFile() || !/^event-[a-f0-9-]+\.json\.[a-f0-9-]+\.tmp$/.test(entry.name)) {
      continue;
    }
    const path = join(directories.spool, entry.name);
    try {
      const metadata = await stat(path);
      if (now - metadata.mtimeMs >= TEMP_STALE_MS) {
        entries.push({ path, name: entry.name, size: metadata.size, modified: metadata.mtimeMs });
      }
    } catch {}
  }
  return entries;
}

async function pruneSpool(directories, preservedPath, deadline = Number.POSITIVE_INFINITY) {
  const entries = [
    ...(await spoolEntries(directories, deadline)),
    ...(await staleTemporaryEntries(directories, deadline)),
  ];
  entries.sort((left, right) => left.modified - right.modified || left.name.localeCompare(right.name));
  let total = entries.reduce((sum, entry) => sum + entry.size, 0);
  let count = entries.length;
  const now = Date.now();
  for (const entry of entries) {
    if (now - entry.modified <= SPOOL_TTL_MS && count <= MAX_SPOOL_FILES && total <= MAX_SPOOL_BYTES) {
      continue;
    }
    if (entry.path === preservedPath) continue;
    await unlink(entry.path).catch(() => {});
    total -= entry.size;
    count -= 1;
  }
  const quarantined = [];
  let scanned = 0;
  for await (const entry of await opendir(directories.quarantine)) {
    if (scanned >= MAINTENANCE_SCAN_BUDGET || Date.now() >= deadline) break;
    scanned += 1;
    if (!entry.isFile()) continue;
    const path = join(directories.quarantine, entry.name);
    try {
      const metadata = await stat(path);
      quarantined.push({ path, size: metadata.size, modified: metadata.mtimeMs });
    } catch {}
  }
  quarantined.sort((left, right) => left.modified - right.modified);
  let quarantineBytes = quarantined.reduce((sum, entry) => sum + entry.size, 0);
  let quarantineCount = quarantined.length;
  for (const entry of quarantined) {
    if (
      now - entry.modified <= SPOOL_TTL_MS &&
      quarantineCount <= MAX_SPOOL_FILES &&
      quarantineBytes <= MAX_SPOOL_BYTES
    ) continue;
    await unlink(entry.path).catch(() => {});
    quarantineBytes -= entry.size;
    quarantineCount -= 1;
  }
}

async function spoolEvent(event, deadline) {
  const directories = await spoolDirectories();
  if (!directories) return;
  const serialized = JSON.stringify(event);
  if (Buffer.byteLength(serialized) > MAX_CONTENT_BYTES + 8192) return;
  const finalPath = join(directories.spool, `event-${event.event_id}.json`);
  const slot = await acquireWriterSlot(directories, deadline);
  const temporaryPath = join(slot, "record.tmp");
  try {
    const handle = await open(temporaryPath, "wx", 0o600);
    try {
      await handle.writeFile(serialized, "utf8");
      await handle.sync();
    } finally {
      await handle.close();
    }
    await rename(temporaryPath, finalPath);
  } catch (error) {
    await unlink(temporaryPath).catch(() => {});
    throw error;
  } finally {
    await rmdir(slot).catch(() => {});
  }
  const maintained = await withSpoolLock(directories, async () => {
    await pruneSpool(directories, finalPath, deadline);
  }, deadline);
  if (!maintained && Date.now() < deadline) await pruneSpool(directories, finalPath, deadline);
}

function validCapture(event) {
  return (
    event &&
    typeof event === "object" &&
    typeof event.event_id === "string" &&
    typeof event.session_id === "string" &&
    typeof event.turn_id === "string" &&
    ["user", "assistant"].includes(event.event_kind) &&
    typeof event.content === "string" &&
    Buffer.byteLength(event.content) <= MAX_CONTENT_BYTES &&
    typeof event.occurred_at === "string" &&
    typeof event.cwd === "string" &&
    isAbsolute(event.cwd)
  );
}

async function replay(key, hookDeadline) {
  const directories = await spoolDirectories();
  if (!directories) return;
  const deadline = Math.min(hookDeadline, Date.now() + REPLAY_BUDGET_MS);
  await withSpoolLock(directories, async () => {
    try {
      await pruneSpool(directories, undefined, deadline);
      let processed = 0;
      for (const entry of await spoolEntries(directories, deadline)) {
        if (processed >= REPLAY_RECORD_BUDGET || Date.now() >= deadline) return;
        processed += 1;
        let event;
        try {
          if (entry.size > MAX_CONTENT_BYTES + 8192) throw new Error("oversized spool record");
          event = JSON.parse(await readFile(entry.path, "utf8"));
          if (!validCapture(event)) throw new Error("malformed spool record");
        } catch {
          await rename(entry.path, join(directories.quarantine, entry.name)).catch(() => {});
          continue;
        }
        try {
          const remaining = deadline - Date.now();
          if (remaining <= 0) return;
          const result = await request("/api/v1/capture/events", event, key, remaining);
          if (result === "accepted") {
            await unlink(entry.path).catch(() => {});
            continue;
          }
          if (result === "permanent") {
            await rename(entry.path, join(directories.quarantine, entry.name)).catch(() => {});
            continue;
          }
          return;
        } catch {
          return;
        }
      }
    } finally {
      if (Date.now() < deadline) await pruneSpool(directories, undefined, deadline);
    }
  }, deadline);
}

async function main() {
  const hookDeadline = Date.now() + HOOK_BUDGET_MS;
  const event = await readStdin();
  if (!event || typeof event !== "object") return;
  const key = await apiKey();
  if (!key) return;
  const name = event.hook_event_name;
  if (["PreCompact", "SessionEnd"].includes(name)) {
    await replay(key, hookDeadline);
    const sessionId = typeof event.session_id === "string" ? event.session_id.slice(0, 1000) : undefined;
    const maintenanceMetadata = [
      event.session_id,
      event.cwd,
      event.turn_id,
      event.occurred_at,
      event.timestamp,
      event.hook_event_name,
    ].filter((value) => typeof value === "string").join("\n");
    const sensitive = inspectSensitive(maintenanceMetadata);
    if (sensitive) {
      await reportSensitiveCapture(maintenanceMetadata, sensitive, key, hookDeadline);
      await pruneState(hookDeadline);
      process.stdout.write("{}\n");
      return;
    }
    const remaining = hookDeadline - Date.now();
    if (remaining > 0) {
      await request("/api/v1/capture/consolidate", { session_id: sessionId }, key, remaining).catch(
        () => {},
      );
    }
    await pruneState(hookDeadline);
    process.stdout.write("{}\n");
    return;
  }
  const kind = name === "UserPromptSubmit" ? "user" : name === "Stop" ? "assistant" : undefined;
  const content = kind === "user" ? event.prompt : event.last_assistant_message ?? event.assistant_response;
  if (
    !kind ||
    typeof content !== "string" ||
    !content.trim() ||
    Buffer.byteLength(content) > MAX_CONTENT_BYTES ||
    typeof event.session_id !== "string" ||
    !event.session_id.trim() ||
    typeof event.cwd !== "string" ||
    !isAbsolute(event.cwd)
  ) return;
  const eventTime =
    typeof event.occurred_at === "string"
      ? event.occurred_at
      : typeof event.timestamp === "string"
        ? event.timestamp
        : new Date().toISOString();
  const occurredAt = eventTime.slice(0, 100);
  const inspectedInput = [
    event.session_id,
    event.cwd,
    event.turn_id,
    event.occurred_at,
    event.timestamp,
    event.hook_event_name,
    content,
  ].filter((value) => typeof value === "string").join("\n");
  const sensitive = inspectSensitive(inspectedInput);
  if (sensitive) {
    await reportSensitiveCapture(inspectedInput, sensitive, key, hookDeadline, kind);
    await replay(key, hookDeadline);
    await pruneState(hookDeadline);
    if (name === "Stop") process.stdout.write("{}\n");
    return;
  }
  const turnId = await activeTurn(event, kind, content);
  const eventId = digest(event.session_id, name, turnId, content);
  const capture = {
    event_id: eventId,
    session_id: event.session_id.slice(0, 1000),
    turn_id: turnId,
    event_kind: kind,
    content,
    occurred_at: occurredAt,
    cwd: event.cwd.slice(0, 4096),
  };
  if (kind === "assistant") {
    await rememberCompletedIdentity(event, content, turnId, eventId).catch(() => {});
  }
  try {
    const remaining = hookDeadline - Date.now();
    const result =
      remaining > 0
        ? await request("/api/v1/capture/events", capture, key, remaining)
        : "transient";
    if (result !== "accepted") {
      await spoolEvent(capture, hookDeadline);
    }
  } catch {
    await spoolEvent(capture, hookDeadline).catch(() => {});
  }
  await replay(key, hookDeadline);
  await pruneState(hookDeadline);
  if (name === "Stop") process.stdout.write("{}\n");
}

try {
  await main();
} catch {
  // Capture is fail-open. No event or daemon detail is written to stdout/stderr.
}
