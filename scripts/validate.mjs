/**
 * Diff the JavaScript tokenizers against the reference counts in
 * test/reference.json, which are produced by the vendors' own libraries
 * (see scripts/reference.py). Any mismatch is a bug here.
 *
 * Vocabularies are downloaded from the vendors' Hugging Face repos on first run
 * and cached, so this works from a fresh clone with no prior setup.
 */
import { existsSync, readFileSync } from "node:fs";
import { cachedText } from "../src/cache.js";
import { tokenizerFromJson } from "../src/hf.js";
import { kimiTokenizer } from "../src/kimi.js";

const ROOT = new URL("..", import.meta.url).pathname;
const corpus = JSON.parse(readFileSync(ROOT + "test/corpus.json", "utf8"));
const reference = JSON.parse(readFileSync(ROOT + "test/reference.json", "utf8"));

// Verify the published fixtures too, so the numbers quoted in the README are
// covered rather than only the short synthetic cases.
for (const [name, path] of [["fixture:prose", "fixtures/prose.txt"], ["fixture:code", "fixtures/code.js"]]) {
  if (existsSync(ROOT + path)) corpus.push({ name, text: readFileSync(ROOT + path, "utf8") });
}

/** Fetch a vocabulary file from the vendor's repo, caching it on disk. */
async function vocab(repo, file) {
  const headers = process.env.HF_TOKEN ? { authorization: `Bearer ${process.env.HF_TOKEN}` } : undefined;
  return cachedText(
    repo.replace("/", "__") + "." + file,
    `https://huggingface.co/${repo}/resolve/main/${file}`,
    { headers, onDownload: () => process.stdout.write(`  downloading ${repo}/${file}\n`) },
  );
}

const targets = [
  { repo: "zai-org/GLM-5.2", build: async (r) => tokenizerFromJson(JSON.parse(await vocab(r, "tokenizer.json"))) },
  { repo: "deepseek-ai/DeepSeek-V4-Pro", build: async (r) => tokenizerFromJson(JSON.parse(await vocab(r, "tokenizer.json"))) },
  { repo: "moonshotai/Kimi-K3", build: async (r) => kimiTokenizer(await vocab(r, "tiktoken.model")) },
  { repo: "moonshotai/Kimi-K2.6", build: async (r) => kimiTokenizer(await vocab(r, "tiktoken.model")) },
];

// gpt-tokenizer should match tiktoken's o200k_base exactly. An empty
// disallowedSpecial set selects the same semantics as tiktoken's
// encode_ordinary, so special-token literals in the input count as plain text
// instead of raising.
const { encode: encodeO200k } = await import("gpt-tokenizer/encoding/o200k_base");
const noDisallowed = new Set();
targets.push({
  repo: "o200k_base",
  build: async () => ({ count: (t) => encodeO200k(t, { disallowedSpecial: noDisallowed }).length }),
});

let failures = 0;
let checks = 0;

for (const { repo, build } of targets) {
  const expected = reference[repo];
  if (!expected) {
    // A missing target is a failure, not a skip: skipping would let
    // "N/N checks passed" quietly cover fewer tokenizers than advertised.
    console.log(`\nFAIL ${repo}: no reference counts (run \`pnpm run verify\` to regenerate test/reference.json)`);
    checks += corpus.length;
    failures += corpus.length;
    continue;
  }
  console.log(`\n${repo}`);
  const t0 = Date.now();
  const tok = await build(repo);
  const loadMs = Date.now() - t0;

  let bad = 0;
  let matched = 0;
  for (const c of corpus) {
    const want = expected[c.name];
    checks++;
    if (want === undefined) {
      // The reference predates this corpus case. Failing keeps a new case from
      // silently shrinking coverage until someone regenerates the reference.
      bad++;
      failures++;
      console.log(`  MISSING  ${c.name.padEnd(22)} no reference count; run \`pnpm run verify\``);
      continue;
    }
    const got = tok.count(c.text);
    if (got !== want) {
      bad++;
      failures++;
      console.log(`  MISMATCH ${c.name.padEnd(22)} got=${got} want=${want}`);
    } else {
      matched++;
    }
  }
  console.log(`  ${bad === 0 ? `all ${matched} cases match` : `${bad} of ${corpus.length} cases failed`} (load ${loadMs}ms)`);
}

console.log(`\n${checks - failures}/${checks} checks passed`);
if (failures) console.log("\nA mismatch means the JavaScript tokenizer disagrees with the vendor's library.");
process.exitCode = failures === 0 ? 0 : 1;
