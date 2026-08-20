/** Tiny on-disk cache for downloaded tokenizer assets and the model catalog. */
import { mkdirSync, existsSync, readFileSync, writeFileSync, statSync } from "node:fs";
import { dirname } from "node:path";

export const CACHE_DIR = new URL("../.cache/", import.meta.url).pathname;

function ensureDir(path) {
  mkdirSync(dirname(path), { recursive: true });
}

export function cachePath(name) {
  return CACHE_DIR + name;
}

/** Age of a cached file in milliseconds, or Infinity if absent. */
export function cacheAge(name) {
  const p = cachePath(name);
  if (!existsSync(p)) return Infinity;
  return Date.now() - statSync(p).mtimeMs;
}

/**
 * Read `name` from cache, downloading from `url` when missing or stale.
 * @param {string} name cache filename
 * @param {string} url
 * @param {{maxAgeMs?: number, onDownload?: (name: string) => void,
 *          headers?: Record<string, string>}} [opts]
 */
export async function cachedText(name, url, opts = {}) {
  const { maxAgeMs = Infinity, onDownload, headers } = opts;
  const p = cachePath(name);

  // existsSync first: a missing file has age Infinity, which would otherwise
  // satisfy the default Infinity max age and lead to reading a nonexistent path.
  if (existsSync(p) && cacheAge(name) <= maxAgeMs) return readFileSync(p, "utf8");

  onDownload?.(name);
  // Fall back to a stale copy rather than failing outright — both when the
  // server answers with an error status and when fetch itself rejects
  // (offline, DNS failure), which previously killed runs that had a perfectly
  // usable cached copy on disk.
  let res;
  try {
    res = await fetch(url, headers ? { headers } : undefined);
  } catch (err) {
    if (existsSync(p)) return readFileSync(p, "utf8");
    throw new Error(`fetch ${url} failed: ${err.message}`);
  }
  if (!res.ok) {
    if (existsSync(p)) return readFileSync(p, "utf8");
    throw new Error(`fetch ${url} failed: HTTP ${res.status}`);
  }
  const text = await res.text();
  ensureDir(p);
  writeFileSync(p, text);
  return text;
}

export async function cachedJson(name, url, opts = {}) {
  return JSON.parse(await cachedText(name, url, opts));
}
