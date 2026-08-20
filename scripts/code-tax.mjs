/**
 * Is code taxed more heavily than prose?
 *
 * Counts two fixtures of identical character length across every selected model
 * and reports the difference. Because byte-level BPE operates on UTF-8 bytes
 * rather than characters, and prose typically carries more multi-byte
 * punctuation, density is reported per 1,000 characters and per 1,000 bytes.
 *
 * Usage:
 *   node scripts/code-tax.mjs [--group open] [--no-bedrock] [--json]
 *   node scripts/code-tax.mjs --a fixtures/prose.txt --b fixtures/code.js
 */
import { readFileSync } from "node:fs";
import { countTokenizers, isCached } from "../src/backends.js";
import { loadCatalog } from "../src/catalog.js";
import { selectModels } from "../src/models.js";
import { buildRegistry } from "../src/registry.js";

const args = process.argv.slice(2);
const argOf = (name, fallback) => {
  const i = args.indexOf(name);
  return i === -1 ? fallback : args[i + 1];
};
const has = (name) => args.includes(name);

const ROOT = new URL("..", import.meta.url).pathname;
const aPath = argOf("--a", ROOT + "fixtures/prose.txt");
const bPath = argOf("--b", ROOT + "fixtures/code.js");
const aLabel = argOf("--a-label", "PROSE");
const bLabel = argOf("--b-label", "CODE");

const a = readFileSync(aPath, "utf8");
const b = readFileSync(bPath, "utf8");

const groups = [];
for (let i = 0; i < args.length; i++) if (args[i] === "--group") groups.push(args[i + 1]);

const catalog = await loadCatalog();
const { models, tokenizers } = await buildRegistry(catalog);
const selected = selectModels(models, { groups });

const needed = new Map();
for (const m of selected) {
  if (m.tokenizerKey && tokenizers.has(m.tokenizerKey)) needed.set(m.tokenizerKey, tokenizers.get(m.tokenizerKey));
}

const bedrock = {
  enabled: !has("--no-bedrock"),
  profile: process.env.AWS_PROFILE || "default",
  region: "us-east-1",
};
const opts = {
  apiKey: process.env.ANTHROPIC_API_KEY,
  bedrock,
  skipUncached: !has("--fetch"),
  concurrency: 4,
};

const uncached = [...needed.values()].filter((s) => s.kind === "hf" && !isCached(s));
if (uncached.length) console.error(`note: ${uncached.length} tokenizers not cached; pass --fetch to download`);

const [countsA, countsB] = await Promise.all([
  countTokenizers(a, needed, opts),
  countTokenizers(b, needed, opts),
]);

const bytesA = Buffer.byteLength(a, "utf8");
const bytesB = Buffer.byteLength(b, "utf8");

// One row per distinct tokenizer: models sharing a tokenizer give the same answer.
const seen = new Set();
const rows = [];
const skipped = [];
for (const m of selected) {
  const spec = needed.get(m.tokenizerKey);
  if (!spec) {
    skipped.push({ model: m.name, reason: m.reason || "no tokenizer resolved" });
    continue;
  }
  const ca = countsA.get(m.tokenizerKey);
  const cb = countsB.get(m.tokenizerKey);
  if (ca?.tokens == null || cb?.tokens == null) {
    skipped.push({
      model: m.name,
      tokenizer: spec.label,
      reason: ca?.error || cb?.error || "count unavailable",
    });
    continue;
  }
  // One row per tokenizer. Claude models are probed individually and API-derived
  // counts can differ by +/-1 within a generation, because merges may span the
  // boundary between the request envelope and the text; that is measurement
  // noise, not a different tokenizer, so it must not split the row.
  const dedupe = spec.label;
  if (seen.has(dedupe)) continue;
  seen.add(dedupe);
  rows.push({
    tokenizer: spec.label,
    example: m.name,
    aTokens: ca.tokens,
    bTokens: cb.tokens,
    aPerKChar: (ca.tokens / a.length) * 1000,
    bPerKChar: (cb.tokens / b.length) * 1000,
    aPerKByte: (ca.tokens / bytesA) * 1000,
    bPerKByte: (cb.tokens / bytesB) * 1000,
    exact: ca.exact && cb.exact,
  });
}

rows.sort((x, y) => (y.bTokens / y.aTokens) - (x.bTokens / x.aTokens));

if (has("--json")) {
  console.log(JSON.stringify({
    inputs: {
      a: { path: aPath, label: aLabel, chars: a.length, bytes: bytesA },
      b: { path: bPath, label: bLabel, chars: b.length, bytes: bytesB },
    },
    equalChars: a.length === b.length,
    rows,
    skipped,
  }, null, 2));
  process.exit(0);
}

const pad = (s, n, right = false) => (right ? String(s).padStart(n) : String(s).padEnd(n));
const pct = (v) => (v >= 0 ? "+" : "") + v.toFixed(1) + "%";

console.log();
console.log(`${aLabel}: ${a.length.toLocaleString()} chars, ${bytesA.toLocaleString()} bytes  (${aPath.split("/").pop()})`);
console.log(`${bLabel}: ${b.length.toLocaleString()} chars, ${bytesB.toLocaleString()} bytes  (${bPath.split("/").pop()})`);
console.log(a.length === b.length
  ? `Both inputs are exactly ${a.length.toLocaleString()} characters, so token differences are content, not length.`
  : `WARNING: inputs differ in length; compare the per-1k columns, not the raw totals.`);
console.log();

const w = { tok: Math.max(9, ...rows.map((r) => r.tokenizer.length)) };
console.log(
  pad("TOKENIZER", w.tok) + "  " + pad(aLabel, 8, true) + "  " + pad(bLabel, 8, true) +
  "  " + pad("DIFF", 8, true) + "  " + pad(`${aLabel}/1kc`, 9, true) + "  " + pad(`${bLabel}/1kc`, 9, true) +
  "  " + pad(`${aLabel}/1kB`, 9, true) + "  " + pad(`${bLabel}/1kB`, 9, true),
);
console.log("-".repeat(w.tok + 2 + 8 * 3 + 6 + 9 * 4 + 8));

for (const r of rows) {
  const diff = ((r.bTokens / r.aTokens) - 1) * 100;
  console.log(
    pad(r.tokenizer + (r.exact ? "" : " ~"), w.tok) + "  " +
    pad(r.aTokens.toLocaleString(), 8, true) + "  " +
    pad(r.bTokens.toLocaleString(), 8, true) + "  " +
    pad(pct(diff), 8, true) + "  " +
    pad(r.aPerKChar.toFixed(1), 9, true) + "  " +
    pad(r.bPerKChar.toFixed(1), 9, true) + "  " +
    pad(r.aPerKByte.toFixed(1), 9, true) + "  " +
    pad(r.bPerKByte.toFixed(1), 9, true),
  );
}

console.log();
const diffs = rows.map((r) => ((r.bTokens / r.aTokens) - 1) * 100);
if (diffs.length) {
  // Median, not mean, leads: rows are per tokenizer rather than per vendor
  // (Claude alone contributes two generations), so a mean double-weights
  // whoever ships the most tokenizers. Both are printed.
  const sorted = [...diffs].sort((x, y) => x - y);
  const mid = sorted.length / 2;
  const median = sorted.length % 2 ? sorted[Math.floor(mid)] : (sorted[mid - 1] + sorted[mid]) / 2;
  const mean = diffs.reduce((n, v) => n + v, 0) / diffs.length;
  console.log(`Across ${rows.length} distinct tokenizers, ${bLabel.toLowerCase()} uses a median ${pct(median)} tokens`);
  console.log(`relative to ${aLabel.toLowerCase()} for the same character count (range ${pct(sorted[0])} to ${pct(sorted[sorted.length - 1])}).`);
  console.log(`Unweighted mean: ${pct(mean)} — rows are per tokenizer, not per vendor; Claude's two`);
  console.log(`generations contribute two rows, so the mean leans toward vendors with more tokenizers.`);
  console.log();
  console.log(`DIFF is ${bLabel.toLowerCase()} vs ${aLabel.toLowerCase()} on identical character counts.`);
  console.log(`/1kc = tokens per 1,000 characters. /1kB = tokens per 1,000 UTF-8 bytes.`);
  console.log(`~ marks a count that is an estimate rather than exact.`);
  console.log(`Claude counts come from an API and carry +/-1 token of envelope noise.`);
}

if (skipped.length) {
  console.log();
  console.log("Not counted:");
  for (const s of skipped) console.log(`  ${s.model}: ${s.reason}`);
}
