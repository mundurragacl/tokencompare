/**
 * Token counting backends.
 *
 * Counts are computed once per *tokenizer*, not per model. Many models share one
 * (all six GPT-5.6 variants, every Qwen3 checkpoint, each Claude generation), and
 * the registry keys tokenizers by content hash so that sharing is exact rather
 * than assumed.
 */
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { cachePath, cachedText } from "./cache.js";
import { bedrockApiKey, countClaudeExact, resolveCredentials } from "./bedrock.js";
import { tokenizerFromJson } from "./hf.js";
import { kimiTokenizer } from "./kimi.js";

/**
 * Calibration used only when no counting API is reachable for Claude.
 *
 * Anthropic publishes no tokenizer for Claude 3 and later. These factors scale an
 * o200k_base count and were derived by measuring Bedrock CountTokens against
 * o200k_base on ~10KB of mixed prose, code and CJK:
 *
 *   legacy (<= 4.6): 2188 / 1978 = 1.106
 *   next   (>= 4.7): 3261 / 2188 = 1.490 on top of legacy
 *
 * One calibration text cannot represent every workload, so these stay estimates.
 */
export const CLAUDE_ESTIMATE = { legacyFactor: 1.106, nextMultiplier: 1.490 };

const ANTHROPIC_VERSION = "2023-06-01";
const CLAUDE_PROBE_MODEL = { legacy: "claude-opus-4-6", next: "claude-opus-5" };

/**
 * tiktoken encodings, wrapped in "ordinary" semantics.
 *
 * By default both gpt-tokenizer and Python tiktoken throw when the input happens
 * to contain a special-token literal such as `<|endoftext|>`, which would crash
 * this tool on perfectly ordinary input (any document discussing tokenizers).
 * Passing an empty `disallowedSpecial` set treats those literals as plain text,
 * which is exactly what tiktoken's own `encode_ordinary` does, and is the right
 * semantic for counting user-supplied text.
 */
const encodingCache = new Map();
const NO_DISALLOWED = new Set();

async function getEncoding(name) {
  if (!encodingCache.has(name)) {
    const m = await import(`gpt-tokenizer/encoding/${name}`);
    encodingCache.set(name, (text) => m.encode(text, { disallowedSpecial: NO_DISALLOWED }));
  }
  return encodingCache.get(name);
}

/** Cache filename for a tokenizer blob, keyed by content hash so it is stored once. */
function blobCacheName(spec) {
  const safe = String(spec.key).replace(/[^A-Za-z0-9._-]/g, "_");
  return `tok/${safe}.${spec.file}`;
}

/** First line of any thrown value, Error or not. */
const firstLine = (err) => String(err?.message ?? err).split("\n")[0];

/**
 * Verify tokenizer bytes against the blob id the registry keyed this spec by.
 *
 * The index records the oid once and is cached indefinitely, so a repo that
 * later republishes its tokenizer would otherwise desync silently: models
 * would keep collapsing onto the old blob id while actually counting with the
 * new bytes. Hashing every load turns that into a loud error instead.
 *
 * LFS oids are a plain sha256 of the content; non-LFS oids are git blob ids,
 * sha1 over "blob <len>\0" + content. Per-repo fallback keys carry no hash and
 * are skipped.
 */
function verifyBlob(spec, text) {
  const oid = String(spec.key).replace(/^hf:/, "");
  if (!/^([0-9a-f]{40}|[0-9a-f]{64})$/.test(oid)) return;
  const bytes = Buffer.from(text, "utf8");
  const got = oid.length === 64
    ? createHash("sha256").update(bytes).digest("hex")
    : createHash("sha1").update(`blob ${bytes.length}\0`).update(bytes).digest("hex");
  if (got !== oid) {
    throw new Error(
      `${spec.repo}/${spec.file} hashes to ${got.slice(0, 12)}..., expected ${oid.slice(0, 12)}... ` +
      "— the repo has likely republished its tokenizer; run with --refresh-index",
    );
  }
}

const tokenizerCache = new Map();

async function loadHfTokenizer(spec, onDownload) {
  if (tokenizerCache.has(spec.key)) return tokenizerCache.get(spec.key);

  const url = `https://huggingface.co/${spec.repo}/resolve/main/${spec.file}`;
  const headers = process.env.HF_TOKEN ? { authorization: `Bearer ${process.env.HF_TOKEN}` } : undefined;
  const text = await cachedText(blobCacheName(spec), url, {
    onDownload: onDownload && (() => onDownload(spec)),
    headers,
  });
  verifyBlob(spec, text);

  let tok;
  if (spec.file === "tiktoken.model") {
    // The tiktoken.model format carries ranks only; the pre-tokenizer pattern
    // lives in the vendor's Python code. The only pattern this repo knows is
    // Moonshot's, so any other vendor shipping the format must fail loudly
    // rather than be counted with Kimi's pattern and labelled exact.
    if (!spec.repo.startsWith("moonshotai/")) {
      throw new Error(
        `${spec.repo} ships tiktoken.model, but only Moonshot's pre-tokenizer pattern is known here; ` +
        "counting it with Kimi's pattern could be silently wrong",
      );
    }
    tok = kimiTokenizer(text);
  } else {
    tok = tokenizerFromJson(JSON.parse(text));
  }

  tokenizerCache.set(spec.key, tok);
  return tok;
}

/** True when the tokenizer blob is already on disk. */
export function isCached(spec) {
  if (spec.kind !== "hf") return true;
  return existsSync(cachePath(blobCacheName(spec)));
}

async function anthropicCountTokens(text, model, apiKey) {
  const res = await fetch("https://api.anthropic.com/v1/messages/count_tokens", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": ANTHROPIC_VERSION,
    },
    body: JSON.stringify({ model, messages: [{ role: "user", content: text }] }),
  });
  if (!res.ok) {
    throw new Error(`count_tokens HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
  }
  const json = await res.json();
  if (typeof json.input_tokens !== "number") throw new Error("no input_tokens in response");
  return json.input_tokens;
}

async function estimateClaude(text, spec) {
  const encode = await getEncoding("o200k_base");
  const base = encode(text).length;
  const factor = spec.generation === "next"
    ? CLAUDE_ESTIMATE.legacyFactor * CLAUDE_ESTIMATE.nextMultiplier
    : CLAUDE_ESTIMATE.legacyFactor;
  return {
    tokens: Math.round(base * factor),
    exact: false,
    source: "estimate",
    method: `estimate: o200k_base x ${factor.toFixed(3)}`,
  };
}

async function countClaude(text, spec, ctx) {
  const { apiKey, bedrock, awsCredentials, awsError, bedrockKey } = ctx;
  const failures = [];

  if (bedrock?.enabled && spec.bedrock) {
    // An API key needs no local AWS config; IAM covers the runtime endpoint via
    // the AWS CLI even when credential export did not yield a usable object.
    const canCall = bedrockKey || awsCredentials || spec.bedrock.endpoint === "runtime";
    if (canCall) {
      try {
        const res = await countClaudeExact(
          text,
          { ...spec.bedrock, region: bedrock.region || spec.bedrock.region },
          { profile: bedrock.profile, credentials: awsCredentials, apiKey: bedrockKey },
        );
        return {
          tokens: res.tokens, exact: true, source: "bedrock",
          endpoint: spec.bedrock.endpoint, overhead: res.overhead,
          auth: res.auth, method: res.method,
        };
      } catch (err) {
        failures.push(`Bedrock: ${firstLine(err)}`);
      }
    } else if (awsError) {
      failures.push(`AWS credentials: ${awsError}`);
    }
  }

  if (apiKey) {
    try {
      const tokens = await anthropicCountTokens(text, CLAUDE_PROBE_MODEL[spec.generation], apiKey);
      return { tokens, exact: true, source: "anthropic-api", method: "Anthropic count_tokens" };
    } catch (err) {
      failures.push(`Anthropic API: ${firstLine(err)}`);
    }
  }

  const est = await estimateClaude(text, spec);
  return failures.length ? { ...est, error: `estimated instead — ${failures.join("; ")}` } : est;
}

/**
 * Count `text` with every tokenizer in `tokenizers`.
 *
 * @param {string} text
 * @param {Map<string, object>} tokenizers from buildRegistry
 * @param {{apiKey?: string, bedrock?: object, concurrency?: number,
 *          onDownload?: Function, onCounted?: Function, skipUncached?: boolean}} [opts]
 * @returns {Promise<Map<string, object>>}
 */
export async function countTokenizers(text, tokenizers, opts = {}) {
  const { apiKey, bedrock, concurrency = 4, onDownload, onCounted, skipUncached } = opts;
  const out = new Map();
  const specs = [...tokenizers.values()];

  // Resolve credentials once, and only if Claude is in scope. A Bedrock API key
  // wins because it needs no local AWS configuration at all.
  const bedrockKey = bedrockApiKey();
  let awsCredentials = null;
  let awsError = null;
  if (bedrock?.enabled && !bedrockKey && specs.some((s) => s.kind === "claude")) {
    try {
      awsCredentials = await resolveCredentials(bedrock.profile);
    } catch (err) {
      awsError = firstLine(err);
    }
  }

  const runOne = async (spec) => {
    try {
      if (spec.kind === "tiktoken") {
        const encode = await getEncoding(spec.encoding);
        return {
          tokens: encode(text).length, exact: true, source: "tiktoken",
          method: `tiktoken ${spec.encoding}`,
        };
      }
      if (spec.kind === "hf") {
        if (skipUncached && !isCached(spec)) {
          return { tokens: null, exact: false, method: "hf", error: "not cached (run with --fetch)" };
        }
        const tok = await loadHfTokenizer(spec, onDownload);
        return {
          tokens: tok.count(text), exact: true, source: "bpe",
          method: `local BPE (${spec.repo}/${spec.file})`,
        };
      }
      if (spec.kind === "claude") {
        return await countClaude(text, spec, {
          apiKey, bedrock, awsCredentials, awsError, bedrockKey,
        });
      }
      return { tokens: null, exact: false, method: spec.kind, error: "unsupported tokenizer kind" };
    } catch (err) {
      return { tokens: null, exact: false, method: spec.kind, error: firstLine(err) };
    }
  };

  // Claude counting is network-bound while local BPE is CPU-bound, so the two
  // run as separate pipelines in parallel: a slow API round-trip no longer
  // holds up local counting that happens to sit behind it in the queue.
  let done = 0;
  const finish = (spec, result) => {
    out.set(spec.key, result);
    onCounted?.(++done, specs.length);
  };
  const runAll = async (list) => {
    for (let i = 0; i < list.length; i += concurrency) {
      const chunk = list.slice(i, i + concurrency);
      const results = await Promise.all(chunk.map(runOne));
      chunk.forEach((spec, j) => finish(spec, results[j]));
    }
  };

  await Promise.all([
    runAll(specs.filter((s) => s.kind === "claude")),
    runAll(specs.filter((s) => s.kind !== "claude")),
  ]);

  return out;
}
