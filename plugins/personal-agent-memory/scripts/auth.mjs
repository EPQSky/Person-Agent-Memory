import { constants } from "node:fs";
import { open } from "node:fs/promises";
import { homedir } from "node:os";
import { isAbsolute, join } from "node:path";

const MAX_API_KEY_BYTES = 4096;
const MAX_INSTALL_METADATA_BYTES = 64 * 1024;

async function readBoundedRegularFile(path, maximumBytes) {
  let file;
  try {
    file = await open(path, constants.O_RDONLY | constants.O_NONBLOCK);
    const metadata = await file.stat();
    if (!metadata.isFile() || metadata.size > maximumBytes) return;
    const content = Buffer.alloc(maximumBytes + 1);
    let offset = 0;
    while (offset < content.length) {
      const { bytesRead } = await file.read(content, offset, content.length - offset, offset);
      if (bytesRead === 0) break;
      offset += bytesRead;
    }
    if (offset > maximumBytes) return;
    return content.subarray(0, offset).toString("utf8");
  } catch {
    return;
  } finally {
    await file?.close().catch(() => {});
  }
}

async function installedApiKeyPath() {
  const configHome = process.env.XDG_CONFIG_HOME || join(homedir(), ".config");
  if (!isAbsolute(configHome)) return;
  try {
    const raw = await readBoundedRegularFile(
      join(configHome, "personal-agent-memory", "install.json"),
      MAX_INSTALL_METADATA_BYTES,
    );
    if (raw === undefined) return;
    const metadata = JSON.parse(raw);
    if (
      metadata?.schema_version !== 1 ||
      metadata?.library_root_ownership !== "user-content-never-delete" ||
      typeof metadata?.state_dir !== "string" ||
      !isAbsolute(metadata.state_dir)
    ) {
      return;
    }
    return join(metadata.state_dir, "api-key");
  } catch {
    return;
  }
}

export async function apiKey() {
  const direct = process.env.PERSONAL_AGENT_MEMORY_API_KEY?.trim();
  if (direct) return direct;

  const configuredPath = process.env.PERSONAL_AGENT_MEMORY_API_KEY_FILE;
  const path =
    configuredPath === undefined
      ? ((await installedApiKeyPath()) ??
        join(homedir(), ".local", "share", "personal-agent-memory", "api-key"))
      : configuredPath;
  if (!isAbsolute(path)) return;
  const value = await readBoundedRegularFile(path, MAX_API_KEY_BYTES);
  return value?.trim() || undefined;
}
