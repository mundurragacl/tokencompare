/**
 * Model catalog backed by OpenRouter's public /models endpoint.
 *
 * OpenRouter does not expose a tokenizer or token-counting endpoint, but its
 * catalog is the most convenient single source for live per-model pricing,
 * context length and the Hugging Face repo behind each open-weight model, and
 * it needs no API key.
 */
import { cachedJson } from "./cache.js";

const CATALOG_URL = "https://openrouter.ai/api/v1/models";
const CATALOG_FILE = "openrouter-models.json";
const DEFAULT_TTL_MS = 24 * 60 * 60 * 1000;

/**
 * @param {{refresh?: boolean, onDownload?: (n: string) => void}} [opts]
 * @returns {Promise<Map<string, object>>} model id -> catalog entry
 */
export async function loadCatalog(opts = {}) {
  const json = await cachedJson(CATALOG_FILE, CATALOG_URL, {
    maxAgeMs: opts.refresh ? -1 : DEFAULT_TTL_MS,
    onDownload: opts.onDownload,
  });
  const byId = new Map();
  for (const m of json.data || []) byId.set(m.id, m);
  return byId;
}

/** Per-token prices as numbers, or null when the field is absent. */
export function pricing(entry) {
  const num = (v) => (v === undefined || v === null || v === "" ? null : Number(v));
  const p = entry?.pricing || {};
  return {
    inputPerToken: num(p.prompt),
    outputPerToken: num(p.completion),
    cacheReadPerToken: num(p.input_cache_read),
  };
}
