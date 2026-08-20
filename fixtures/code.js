/**
 * Loader for Hugging Face `tokenizer.json` files (GLM-5.2, DeepSeek-V4-*).
 *
 * Only the byte-level BPE shape is supported, which is what every open model in
 * the comparison set uses. Anything else throws loudly rather than silently
 * producing a wrong count.
 */
import { BpeModel, Tokenizer, splitIsolated } from "./bpe.js";

/**
 * Translate a Rust `regex`/`fancy-regex` pattern into JavaScript syntax.
 *
 * The only construct these tokenizers use that JS lacks is the inline
 * case-insensitive group `(?i:...)`, which we expand by hand into explicit
 * character alternatives.
 */
export function translateRegex(src) {
  let out = "";
  let i = 0;
  while (i < src.length) {
    if (src.startsWith("(?i:", i)) {
      const end = findGroupEnd(src, i);
      out += "(?:" + expandCaseInsensitive(src.slice(i + 4, end)) + ")";
      i = end + 1;
    } else {
      // Copy escapes verbatim so we never re-interpret an escaped char.
      if (src[i] === "\\" && i + 1 < src.length) {
        out += src.slice(i, i + 2);
        i += 2;
      } else {
        out += src[i];
        i++;
      }
    }
  }
  return out;
}

/** Index of the `)` closing the group whose `(` is at `open`. */
function findGroupEnd(src, open) {
  let depth = 0;
  for (let i = open; i < src.length; i++) {
    if (src[i] === "\\") { i++; continue; }
    if (src[i] === "(") depth++;
    else if (src[i] === ")") {
      depth--;
      if (depth === 0) return i;
    }
  }
  throw new Error("unbalanced group in regex: " + src);
}

/** Replace each ASCII letter with a two-case character class. */
function expandCaseInsensitive(body) {
  let out = "";
  for (let i = 0; i < body.length; i++) {
    const ch = body[i];
    if (ch === "\\") { out += body.slice(i, i + 2); i++; continue; }
    if (/[a-zA-Z]/.test(ch)) {
      out += "[" + ch.toLowerCase() + ch.toUpperCase() + "]";
    } else {
      out += ch;
    }
  }
  return out;
}

/** Build the pre-tokenizer function described by a tokenizer.json node. */
function buildPreTokenizer(node) {
  const stages = [];

  const visit = (n) => {
    if (!n) return;
    switch (n.type) {
      case "Sequence":
        for (const child of n.pretokenizers || []) visit(child);
        break;
      case "Split": {
        const pattern = n.pattern?.Regex;
        if (pattern === undefined) {
          throw new Error("unsupported Split pattern: " + JSON.stringify(n.pattern));
        }
        if (n.behavior !== "Isolated") {
          throw new Error("unsupported Split behavior: " + n.behavior);
        }
        if (n.invert) throw new Error("inverted Split is not supported");
        const re = new RegExp(translateRegex(pattern), "gu");
        stages.push((pieces) => pieces.flatMap((p) => splitIsolated(p, re)));
        break;
      }
      case "ByteLevel":
        // Byte-level conversion happens in Tokenizer.count. The only field that
        // changes token counts here is add_prefix_space.
        if (n.add_prefix_space) {
          stages.push((pieces) => pieces.map((p, idx) => (idx === 0 ? " " + p : p)));
        }
        break;
      default:
        throw new Error("unsupported pre_tokenizer type: " + n.type);
    }
  };

  visit(node);

  return (text) => {
    let pieces = [text];
    for (const stage of stages) pieces = stage(pieces);
    return pieces;
  };
}

function assertNoOpNormalizer(normalizer) {
  if (!normalizer) return;
  if (normalizer.type === "Sequence" && (normalizer.normalizers || []).length === 0) return;
  throw new Error("unsupported normalizer: " + normalizer.type);
}

/**
 * Build a Tokenizer from parsed tokenizer.json content.
 * @param {object} json
 */
export function tokenizerFromJson(json) {
  if (json.model?.type !== "BPE") {
    throw new Error("unsupported model type: " + json.model?.type);
  }
  const m = json.model;
  if (m.continuing_subword_prefix) throw new Error("continuing_subword_prefix unsupported");
  if (m.end_of_word_suffix) throw new Error("end_of_word_suffix unsupported");
  if (m.byte_fallback) throw new Error("byte_fallback unsupported");
  assertNoOpNormalizer(json.normalizer);

  const vocab = new Map(Object.entries(m.vocab));

  // merges come either as ["a", "b"] pairs (newer files) or "a b" strings.
  const merges = new Map();
  m.merges.forEach((entry, rank) => {
    let a, b;
    if (Array.isArray(entry)) {
      [a, b] = entry;
    } else {
      const sp = entry.indexOf(" ");
      a = entry.slice(0, sp);
      b = entry.slice(sp + 1);
    }
    merges.set(a + "\u0000" + b, rank);
  });

  const model = new BpeModel(vocab, merges, { ignoreMerges: m.ignore_merges });
  const specials = (json.added_tokens || []).map((t) => t.content);
  return new Tokenizer(buildPreTokenizer(json.pre_tokenizer), model, specials);
}

/**
 * Minimal AWS SigV4 request signing.
 *
 * Needed because the `bedrock-mantle` endpoint that serves token counts for
 * Claude 4.7+ is not exposed by any AWS SDK method, so the request has to be
 * signed by hand. Only static request paths are supported, which avoids the
 * canonical-URI encoding pitfalls that arise when a path segment contains
 * characters like `:`.
 */
import { createHash, createHmac } from "node:crypto";

const ALGORITHM = "AWS4-HMAC-SHA256";

const sha256Hex = (data) => createHash("sha256").update(data).digest("hex");
const hmac = (key, data) => createHmac("sha256", key).update(data).digest();

/**
 * @param {object} params
 * @param {string} params.method
 * @param {string} params.url        full URL; its path must need no escaping
 * @param {string} params.body
 * @param {string} params.service    SigV4 service name, e.g. "bedrock-mantle"
 * @param {string} params.region
 * @param {{accessKeyId: string, secretAccessKey: string, sessionToken?: string}} params.credentials
 * @param {Record<string, string>} [params.headers] additional headers to sign
 * @returns {Record<string, string>} headers including Authorization
 */
export function signRequest({ method, url, body, service, region, credentials, headers = {} }) {
  const parsed = new URL(url);
  if (parsed.pathname !== encodeURI(parsed.pathname)) {
    throw new Error("signRequest only supports paths that require no escaping");
  }

  const now = new Date();
  const amzDate = now.toISOString().replace(/[:-]|\.\d{3}/g, ""); // YYYYMMDDTHHMMSSZ
  const dateStamp = amzDate.slice(0, 8);

  const allHeaders = {
    ...headers,
    host: parsed.host,
    "x-amz-date": amzDate,
  };
  if (credentials.sessionToken) allHeaders["x-amz-security-token"] = credentials.sessionToken;

  // Canonical headers: lowercase names, trimmed values, sorted by name.
  const normalised = Object.entries(allHeaders)
    .map(([k, v]) => [k.toLowerCase(), String(v).trim().replace(/\s+/g, " ")])
    .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));

  const canonicalHeaders = normalised.map(([k, v]) => `${k}:${v}\n`).join("");
  const signedHeaders = normalised.map(([k]) => k).join(";");

  const canonicalQuery = [...parsed.searchParams.entries()]
    .map(([k, v]) => [encodeURIComponent(k), encodeURIComponent(v)])
    .sort((a, b) => (a[0] < b[0] ? -1 : 1))
    .map(([k, v]) => `${k}=${v}`)
    .join("&");

  const canonicalRequest = [
    method,
    parsed.pathname,
    canonicalQuery,
    canonicalHeaders,
    signedHeaders,
    sha256Hex(body),
  ].join("\n");

  const scope = `${dateStamp}/${region}/${service}/aws4_request`;
  const stringToSign = [ALGORITHM, amzDate, scope, sha256Hex(canonicalRequest)].join("\n");

  const signingKey = ["aws4_request"].reduce(
    (key, part) => hmac(key, part),
    [region, service].reduce(
      (key, part) => hmac(key, part),
      hmac("AWS4" + credentials.secretAccessKey, dateStamp),
    ),
  );
  const signature = createHmac("sha256", signingKey).update(stringToSign).digest("hex");

  return {
    ...allHeaders,
    Authorization:
      `${ALGORITHM} Credential=${credentials.accessKeyId}/${scope}, ` +
      `SignedHeaders=${signedHeaders}, Signature=${signature}`,
 