/**
 * Loader for Hugging Face `tokenizer.json` files (GLM-5.2, DeepSeek-V4-*).
 *
 * Only the byte-level BPE shape is supported, which is what every open model in
 * the comparison set uses. Anything else throws loudly rather than silently
 * producing a wrong count.
 */
import { assertByteLevelComplete, BpeModel, Tokenizer, splitIsolated } from "./bpe.js";

/**
 * The GPT-2 pre-tokenizer pattern that HF's ByteLevel stage applies when its
 * `use_regex` field is true (which is the default when the field is absent).
 * Verbatim from the tokenizers crate; the contractions are deliberately
 * case-sensitive.
 */
const GPT2_PRETOKENIZE = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

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
      case "ByteLevel": {
        // Byte-level conversion itself happens in Tokenizer.count. Two fields
        // change token counts here, and both follow HF semantics:
        //   add_prefix_space  prepend " " to each piece not already starting
        //                     with one (HF applies this per split, not once)
        //   use_regex         apply the built-in GPT-2 pattern; it defaults to
        //                     TRUE when absent, and ignoring it would leave the
        //                     whole text as one piece and silently change counts
        const addPrefix = Boolean(n.add_prefix_space);
        const useRegex = n.use_regex !== false;
        if (addPrefix || useRegex) {
          stages.push((pieces) => {
            if (addPrefix) pieces = pieces.map((p) => (p.startsWith(" ") ? p : " " + p));
            if (useRegex) pieces = pieces.flatMap((p) => splitIsolated(p, GPT2_PRETOKENIZE));
            return pieces;
          });
        }
        break;
      }
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
  assertByteLevelComplete(vocab, "tokenizer.json vocab");

  // merges come either as ["a", "b"] pairs (newer files) or "a b" strings.
  const merges = new Map();
  m.merges.forEach((entry, rank) => {
    let a, b;
    if (Array.isArray(entry)) {
      [a, b] = entry;
    } else {
      const sp = entry.indexOf(" ");
      if (sp === -1) {
        throw new Error(`malformed merge entry at rank ${rank}: ${JSON.stringify(entry)}`);
      }
      a = entry.slice(0, sp);
      b = entry.slice(sp + 1);
    }
    merges.set(a + "\u0000" + b, rank);
  });

  const model = new BpeModel(vocab, merges, { ignoreMerges: m.ignore_merges });

  // Added tokens are matched as literals and counted as one token each. HF
  // additionally honours lstrip/rstrip/single_word when matching; none of the
  // supported vocabularies set them, and imitating them wrongly would change
  // counts silently, so their presence fails loudly instead.
  const added = json.added_tokens || [];
  for (const t of added) {
    if (t.lstrip || t.rstrip || t.single_word) {
      throw new Error(
        `added token ${JSON.stringify(t.content)} uses lstrip/rstrip/single_word, which is unsupported`,
      );
    }
  }
  const specials = added.map((t) => t.content);
  return new Tokenizer(buildPreTokenizer(json.pre_tokenizer), model, specials);
}
