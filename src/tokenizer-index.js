/**
 * Index of which tokenizer file each Hugging Face repo publishes, and that file's
 * blob id.
 *
 * The blob id is what makes a full-catalogue comparison affordable: many repos
 * ship byte-identical tokenizers (all the Qwen3 variants, all the GLM-4.5
 * variants, and so on), so keying by blob id collapses ~146 repos down to ~63
 * real tokenizers and halves the bytes fetched. It also means the disk cache
 * stores each tokenizer once regardless of how many models reference it.
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { cachePath } from "./cache.js";

const INDEX_FILE = "tokenizer-index.json";

/** Tokenizer files we know how to parse, in order of preference. */
export const KNOWN_FILES = ["tokenizer.json", "tiktoken.model"];

/** Files that indicate a tokenizer we deliberately do not support. */
const UNSUPPORTED_FILES = ["tokenizer.model", "vocab.json"];

const CONCURRENCY = 12;

async function mapLimit(items, limit, fn, onProgress) {
  const out = [];
  for (let i = 0; i < items.length; i += limit) {
    const chunk = items.slice(i, i + limit);
    out.push(...await Promise.all(chunk.map(fn)));
    onProgress?.(Math.min(i + limit, items.length), items.length);
  }
  return out;
}

async function probe(repo, token) {
  const headers = token ? { authorization: `Bearer ${token}` } : {};
  const entry = { repo, file: null, oid: null, size: null, gated: false, reason: null };

  let info;
  try {
    const res = await fetch(`https://huggingface.co/api/models/${repo}`, { headers });
    if (!res.ok) {
      entry.reason = res.status === 401 || res.status === 403
        ? "repo is private or gated (set HF_TOKEN)"
        : `Hugging Face API returned ${res.status}`;
      entry.gated = res.status === 401 || res.status === 403;
      // Server trouble is not a fact about the repo; retry it next run instead
      // of remembering the failure forever.
      if (res.status === 429 || res.status >= 500) entry.transient = true;
      return entry;
    }
    info = await res.json();
  } catch (err) {
    entry.reason = `Hugging Face API unreachable: ${err.message}`;
    entry.transient = true;
    return entry;
  }

  if (info.gated) {
    entry.gated = true;
    // Gated repos still list files, but downloads need an accepted licence.
    if (!token) {
      entry.reason = "repo is gated (set HF_TOKEN and accept the licence)";
      return entry;
    }
  }

  const names = new Set((info.siblings || []).map((s) => s.rfilename));
  entry.file = KNOWN_FILES.find((f) => names.has(f)) ?? null;
  if (!entry.file) {
    const other = UNSUPPORTED_FILES.find((f) => names.has(f));
    entry.reason = other
      ? `only publishes ${other}, which is not a byte-level BPE tokenizer`
      : "publishes no tokenizer file";
    return entry;
  }

  // paths-info gives the blob id without downloading the blob.
  try {
    const res = await fetch(`https://huggingface.co/api/models/${repo}/paths-info/main`, {
      method: "POST",
      headers: { ...headers, "content-type": "application/json" },
      body: JSON.stringify({ paths: [entry.file] }),
    });
    if (res.ok) {
      const json = await res.json();
      const p = Array.isArray(json) ? json[0] : json;
      entry.oid = p?.lfs?.oid || p?.oid || null;
      entry.size = p?.size ?? p?.lfs?.size ?? null;
    }
  } catch {
    // A missing oid only costs us deduplication, so carry on.
  }

  // Without an oid, fall back to a per-repo key so the tokenizer still resolves.
  if (!entry.oid) entry.oid = `repo:${repo}/${entry.file}`;
  return entry;
}

/**
 * Build (or load) the repo -> tokenizer-file index.
 *
 * @param {string[]} repos
 * @param {{refresh?: boolean, token?: string, onProgress?: (done: number, total: number) => void}} [opts]
 * @returns {Promise<Map<string, object>>}
 */
export async function loadTokenizerIndex(repos, opts = {}) {
  const path = cachePath(INDEX_FILE);
  let cached = {};
  if (!opts.refresh && existsSync(path)) {
    try {
      cached = JSON.parse(readFileSync(path, "utf8"));
    } catch {
      cached = {};
    }
  }

  const token = opts.token || process.env.HF_TOKEN;

  // Re-probe entries whose failure was environmental rather than a fact about
  // the repo: transient network/server errors, and gated repos probed without
  // a token when one is available now.
  const stale = (e) => !e || e.transient || (e.gated && !e.file && Boolean(token));
  const missing = repos.filter((r) => stale(cached[r]));
  if (missing.length) {
    const probed = await mapLimit(missing, CONCURRENCY, (r) => probe(r, token), opts.onProgress);
    for (const entry of probed) cached[entry.repo] = entry;
    writeFileSync(path, JSON.stringify(cached, null, 2));
  }

  return new Map(Object.entries(cached));
}

/**
 * Total bytes that would be downloaded for the given tokenizer specs.
 *
 * Deduplicates on the spec key, which already encodes the blob id, so
 * vocabularies shared by several models are counted once.
 */
export function downloadSize(specs) {
  const byBlob = new Map();
  for (const s of specs) {
    const id = s.oid ?? s.key;
    if (id && s.size) byBlob.set(id, s.size);
  }
  return [...byBlob.values()].reduce((n, v) => n + v, 0);
}
