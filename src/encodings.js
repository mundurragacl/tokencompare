/**
 * Map OpenAI model ids to tiktoken encodings.
 *
 * Both tables mirror tiktoken 0.14.0's `model.py` verbatim — MODEL_TO_ENCODING
 * for exact names, MODEL_PREFIX_TO_ENCODING for documented prefixes — minus the
 * `ft:` fine-tune forms and the pre-2023 completion/search/edit families, which
 * OpenRouter has never listed. Keeping the tables verbatim is the point: an id
 * that matches neither has NO published encoding, and the caller reports it as
 * inferred rather than exact. Do not "helpfully" broaden a prefix here.
 */

/** @type {Map<string, string>} exact model name -> encoding */
const MODEL_TO_ENCODING = new Map([
  ["o1", "o200k_base"],
  ["o3", "o200k_base"],
  ["o4-mini", "o200k_base"],
  ["gpt-5", "o200k_base"],
  ["gpt-4.1", "o200k_base"],
  ["gpt-4o", "o200k_base"],
  ["gpt-4", "cl100k_base"],
  ["gpt-3.5-turbo", "cl100k_base"],
  ["gpt-3.5", "cl100k_base"],
  ["gpt-35-turbo", "cl100k_base"],
  ["davinci-002", "cl100k_base"],
  ["babbage-002", "cl100k_base"],
  ["text-embedding-ada-002", "cl100k_base"],
  ["text-embedding-3-small", "cl100k_base"],
  ["text-embedding-3-large", "cl100k_base"],
]);

/**
 * @type {Array<[string, string]>} documented prefix -> encoding, in tiktoken's
 * own order. Note `gpt-5` is genuinely bare in tiktoken (it covers gpt-5.x),
 * while the o-series and 4.x prefixes are dashed.
 */
const MODEL_PREFIX_TO_ENCODING = [
  ["o1-", "o200k_base"],
  ["o3-", "o200k_base"],
  ["o4-mini-", "o200k_base"],
  ["gpt-5", "o200k_base"],
  ["gpt-4.5-", "o200k_base"],
  ["gpt-4.1-", "o200k_base"],
  ["chatgpt-4o-", "o200k_base"],
  ["gpt-4o-", "o200k_base"],
  ["gpt-4-", "cl100k_base"],
  ["gpt-3.5-turbo-", "cl100k_base"],
  ["gpt-35-turbo-", "cl100k_base"],
  ["gpt-oss-", "o200k_harmony"],
];

/** Encoding assumed for current-generation models whose id matches no table. */
export const DEFAULT_OPENAI_ENCODING = "o200k_base";

/**
 * @param {string} modelId an OpenRouter id such as "openai/gpt-4o-mini"
 * @returns {{encoding: string, matched: boolean}} `matched` is true only when
 *   tiktoken documents the mapping; false means the encoding is an assumption
 *   and the caller must report the count as inferred, not exact.
 */
export function encodingForModel(modelId) {
  const bare = modelId.includes("/") ? modelId.slice(modelId.indexOf("/") + 1) : modelId;
  const name = bare.replace(/:.*$/, ""); // drop OpenRouter suffixes like ":free"

  if (MODEL_TO_ENCODING.has(name)) {
    return { encoding: MODEL_TO_ENCODING.get(name), matched: true };
  }
  for (const [prefix, encoding] of MODEL_PREFIX_TO_ENCODING) {
    if (name.startsWith(prefix)) return { encoding, matched: true };
  }
  return { encoding: DEFAULT_OPENAI_ENCODING, matched: false };
}

/** Encodings that gpt-tokenizer ships and we can therefore load. */
export const SUPPORTED_ENCODINGS = new Set([
  "o200k_base",
  "o200k_harmony",
  "cl100k_base",
  "p50k_base",
  "p50k_edit",
  "r50k_base",
]);
