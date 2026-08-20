/**
 * Byte-level BPE core, shared by the Hugging Face `tokenizer.json` backend and
 * the Kimi `tiktoken.model` backend.
 *
 * Both formats are byte-level BPE: text is split by a pre-tokenizer regex, each
 * piece is reinterpreted as raw UTF-8 bytes, and merges are applied greedily by
 * rank. The only real difference is how the vocabulary is serialised, so the
 * merge machinery lives here and the loaders live in ./hf.js and ./kimi.js.
 */

/**
 * GPT-2's reversible byte <-> unicode mapping. Bytes that are not printable
 * ASCII get shifted into a private range so that a byte sequence can be carried
 * around as a normal JS string (which is how HF stores vocab keys).
 */
function buildByteToUnicode() {
  const bs = [];
  for (let i = 0x21; i <= 0x7e; i++) bs.push(i); // !..~
  for (let i = 0xa1; i <= 0xac; i++) bs.push(i);
  for (let i = 0xae; i <= 0xff; i++) bs.push(i);

  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }

  const byteToChar = new Array(256);
  for (let i = 0; i < bs.length; i++) byteToChar[bs[i]] = String.fromCodePoint(cs[i]);
  return byteToChar;
}

export const BYTE_TO_CHAR = buildByteToUnicode();

export const CHAR_TO_BYTE = (() => {
  const m = new Map();
  for (let b = 0; b < 256; b++) m.set(BYTE_TO_CHAR[b], b);
  return m;
})();

const utf8 = new TextEncoder();

/** Encode a raw string into its byte-level representation. */
export function toByteLevel(text) {
  const bytes = utf8.encode(text);
  let out = "";
  for (let i = 0; i < bytes.length; i++) out += BYTE_TO_CHAR[bytes[i]];
  return out;
}

/** Split a byte-level string into its constituent single-byte characters. */
function splitBytes(byteLevel) {
  // Byte-level chars are all in the BMP, so Array.from is safe and avoids
  // splitting surrogate pairs (there are none by construction).
  return Array.from(byteLevel);
}

/**
 * A byte-level BPE model.
 *
 * @param {Map<string, number>} vocab  byte-level token string -> id
 * @param {Map<string, number>} merges "a\u0000b" -> rank (lower merges first)
 * @param {{ignoreMerges?: boolean}} [opts] when ignoreMerges is set, a piece
 *   that is already a vocab entry is emitted as-is without running merges
 *   (matches HF's `ignore_merges`).
 */
export class BpeModel {
  constructor(vocab, merges, opts = {}) {
    this.vocab = vocab;
    this.merges = merges;
    this.ignoreMerges = Boolean(opts.ignoreMerges);
    this._cache = new Map();
  }

  /**
   * Number of tokens a single pre-token piece becomes.
   * @param {string} byteLevel piece already in byte-level form
   */
  countPiece(byteLevel) {
    if (byteLevel.length === 0) return 0;

    const cached = this._cache.get(byteLevel);
    if (cached !== undefined) return cached;

    let count;
    if (this.ignoreMerges && this.vocab.has(byteLevel)) {
      count = 1;
    } else {
      count = this._mergeCount(byteLevel);
    }

    // Unbounded growth is a real risk on large inputs; cap the cache.
    if (this._cache.size < 500_000) this._cache.set(byteLevel, count);
    return count;
  }

  _mergeCount(byteLevel) {
    const parts = splitBytes(byteLevel);
    if (parts.length === 1) return 1;

    // Greedy lowest-rank merge, repeated until no adjacent pair is mergeable.
    for (;;) {
      let bestRank = Infinity;
      let bestIdx = -1;
      for (let i = 0; i < parts.length - 1; i++) {
        const rank = this.merges.get(parts[i] + "\u0000" + parts[i + 1]);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIdx = i;
        }
      }
      if (bestIdx === -1) break;
      parts.splice(bestIdx, 2, parts[bestIdx] + parts[bestIdx + 1]);
      if (parts.length === 1) break;
    }

    return parts.length;
  }
}

/**
 * A tokenizer: a pre-tokenizer that yields raw text pieces, plus a BPE model.
 *
 * Special/added tokens are matched before pre-tokenization so that literals like
 * `<|endoftext|>` count as one token rather than being split.
 */
export class Tokenizer {
  /**
   * @param {(text: string) => string[]} preTokenize
   * @param {BpeModel} model
   * @param {string[]} [specialTokens]
   */
  constructor(preTokenize, model, specialTokens = []) {
    this.preTokenize = preTokenize;
    this.model = model;
    this.specialPattern = specialTokens.length
      ? new RegExp(specialTokens.map(escapeRegExp).sort((a, b) => b.length - a.length).join("|"), "g")
      : null;
  }

  /** Total token count for `text`, excluding any chat-template scaffolding. */
  count(text) {
    let total = 0;
    for (const segment of this._splitOnSpecials(text)) {
      if (segment.special) {
        total += 1;
        continue;
      }
      for (const piece of this.preTokenize(segment.text)) {
        total += this.model.countPiece(toByteLevel(piece));
      }
    }
    return total;
  }

  _splitOnSpecials(text) {
    if (!this.specialPattern) return [{ text, special: false }];
    const out = [];
    let last = 0;
    this.specialPattern.lastIndex = 0;
    for (let m; (m = this.specialPattern.exec(text)) !== null; ) {
      if (m.index > last) out.push({ text: text.slice(last, m.index), special: false });
      out.push({ text: m[0], special: true });
      last = m.index + m[0].length;
      if (m[0].length === 0) this.specialPattern.lastIndex++; // guard
    }
    if (last < text.length) out.push({ text: text.slice(last), special: false });
    return out;
  }
}

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Assert that a byte-level vocabulary contains all 256 single-byte tokens.
 *
 * The merge loops count any unmergeable leftover part as one token, which is
 * only correct when every single byte is itself a vocabulary entry. That holds
 * for genuine byte-level BPE vocabularies; anything else must fail loudly here
 * rather than miscount quietly.
 *
 * @param {Map<string, number>} vocab byte-level token string -> id or rank
 * @param {string} label for the error message
 */
export function assertByteLevelComplete(vocab, label) {
  for (let b = 0; b < 256; b++) {
    if (!vocab.has(BYTE_TO_CHAR[b])) {
      throw new Error(
        `${label} lacks a token for byte 0x${b.toString(16).padStart(2, "0")}; ` +
        "not a complete byte-level BPE vocabulary",
      );
    }
  }
}

/**
 * Apply a `Split(behavior="Isolated")` pre-tokenizer stage: every regex match
 * becomes its own piece and the text between matches is kept as pieces too.
 */
export function splitIsolated(text, regex) {
  if (text.length === 0) return [];
  const out = [];
  let last = 0;
  regex.lastIndex = 0;
  for (let m; (m = regex.exec(text)) !== null; ) {
    if (m.index > last) out.push(text.slice(last, m.index));
    if (m[0].length > 0) out.push(m[0]);
    last = m.index + m[0].length;
    if (m[0].length === 0) regex.lastIndex++; // zero-width match guard
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

/**
 * A tiktoken-style BPE model.
 *
 * Differs from {@link BpeModel} in how merge priority is decided: there is no
 * merges list, so the pair to merge is the adjacent pair whose *concatenation*
 * has the lowest rank in the vocabulary. Kimi's `tiktoken.model` uses this form.
 *
 * @param {Map<string, number>} ranks byte-level token string -> rank
 */
export class TiktokenModel {
  constructor(ranks) {
    this.ranks = ranks;
    this._cache = new Map();
  }

  countPiece(byteLevel) {
    if (byteLevel.length === 0) return 0;

    const cached = this._cache.get(byteLevel);
    if (cached !== undefined) return cached;

    let count;
    if (this.ranks.has(byteLevel)) {
      count = 1; // whole piece is a single token
    } else {
      const parts = Array.from(byteLevel);
      for (;;) {
        let bestRank = Infinity;
        let bestIdx = -1;
        for (let i = 0; i < parts.length - 1; i++) {
          const rank = this.ranks.get(parts[i] + parts[i + 1]);
          if (rank !== undefined && rank < bestRank) {
            bestRank = rank;
            bestIdx = i;
          }
        }
        if (bestIdx === -1) break;
        parts.splice(bestIdx, 2, parts[bestIdx] + parts[bestIdx + 1]);
        if (parts.length === 1) break;
      }
      count = parts.length;
    }

    if (this._cache.size < 500_000) this._cache.set(byteLevel, count);
    return count;
  }
}
