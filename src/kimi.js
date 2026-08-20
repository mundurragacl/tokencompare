/**
 * Loader for Moonshot's Kimi tokenizers, which ship a `tiktoken.model` file
 * (base64 token + rank per line) instead of a Hugging Face `tokenizer.json`.
 */
import { assertByteLevelComplete, BYTE_TO_CHAR, TiktokenModel, Tokenizer, splitIsolated } from "./bpe.js";
import { translateRegex } from "./hf.js";

/**
 * Pattern from moonshotai/Kimi-K3 `tokenization_kimi.py`.
 *
 * Written for the Rust regex crate, so it uses two constructs JS spells
 * differently: character-class intersection (`A&&[^B]`) and bare script names
 * (`\p{Han}`). Both are handled by {@link translateKimiRegex}.
 */
export const KIMI_PATTERN = [
  String.raw`[\p{Han}]+`,
  String.raw`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?`,
  String.raw`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?`,
  String.raw`\p{N}{1,3}`,
  String.raw` ?[^\s\p{L}\p{N}]+[\r\n]*`,
  String.raw`\s*[\r\n]+`,
  String.raw`\s+(?!\S)`,
  String.raw`\s+`,
].join("|");

/**
 * Convert the Rust-flavoured Kimi pattern into a JS `v`-mode regex source.
 *
 * - `[X&&[^Y]]` (intersect with complement) becomes `[[X]--[Y]]`, the v-mode
 *   set-difference form.
 * - `\p{Han}` becomes `\p{Script=Han}`, since JS requires scripts to be named
 *   explicitly rather than as a bare property.
 */
export function translateKimiRegex(src) {
  let out = src.replace(/\[([^\[\]]*)&&\[\^([^\[\]]*)\]\]/g, "[[$1]--[$2]]");
  out = out.replace(/\\p\{Han\}/g, "\\p{Script=Han}");
  return translateRegex(out);
}

/**
 * Parse a `tiktoken.model` file into byte-level ranks.
 * @param {string} content
 * @returns {Map<string, number>}
 */
export function parseTiktokenModel(content) {
  const ranks = new Map();
  for (const line of content.split("\n")) {
    if (!line.trim()) continue;
    const sp = line.indexOf(" ");
    const rank = sp === -1 ? NaN : Number(line.slice(sp + 1));
    if (!Number.isInteger(rank)) {
      throw new Error(`malformed tiktoken.model line: ${JSON.stringify(line.slice(0, 40))}`);
    }
    const bytes = Buffer.from(line.slice(0, sp), "base64");
    let key = "";
    for (const b of bytes) key += BYTE_TO_CHAR[b];
    ranks.set(key, rank);
  }
  return ranks;
}

/**
 * Build a Kimi tokenizer from `tiktoken.model` file content.
 *
 * No special tokens are registered on purpose: Moonshot's specials live in
 * Python code rather than in this file, and raw text is counted with plain
 * `encode` semantics (special literals as ordinary text) — the same choice
 * scripts/reference.py makes with `special_tokens={}`, so verification covers
 * this behaviour rather than papering over a mismatch.
 *
 * @param {string} content
 */
export function kimiTokenizer(content) {
  const ranks = parseTiktokenModel(content);
  assertByteLevelComplete(ranks, "tiktoken.model ranks");
  const re = new RegExp(translateKimiRegex(KIMI_PATTERN), "gv");
  const preTokenize = (text) => splitIsolated(text, re);
  return new Tokenizer(preTokenize, new TiktokenModel(ranks));
}
