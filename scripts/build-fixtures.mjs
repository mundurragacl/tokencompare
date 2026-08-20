/**
 * Build and verify the prose and code fixtures used to test for a "code tax".
 *
 * The experiment only means something if the two inputs are the same size, so
 * both are trimmed to exactly the same character count. Any leftover difference
 * in token count is then attributable to content, not length.
 *
 * Prose is a famous public-domain text fetched from Project Gutenberg. Code is
 * a FROZEN snapshot of this project's own MIT-licensed source, taken when the
 * published numbers were measured. The freeze matters: the fixtures are inputs
 * to test/reference.json and to every number quoted in the README, so ordinary
 * development on src/ must not silently change the corpus those numbers were
 * measured on. The snapshot therefore lives as the committed fixture itself,
 * pinned by the sha256 recorded in fixtures/PROVENANCE.md (a commit pin would
 * not survive history rewrites; a content hash does).
 *
 * Two modes:
 *
 *   default    verify the committed fixtures. The prose is rebuilt from
 *              Gutenberg and must match byte-for-byte; code.js must match the
 *              hash recorded in PROVENANCE.md and be exactly the target length.
 *              Any drift — upstream Gutenberg edits, or an accidental edit to a
 *              fixture — fails loudly.
 *   --update   re-freeze: rebuild the code fixture from the current src/ files,
 *              refetch the prose, and rewrite fixtures/ including PROVENANCE.
 *              Downstream artefacts must then be regenerated; the script prints
 *              the checklist.
 *
 * Usage: node scripts/build-fixtures.mjs [--chars 8000] [--update]
 */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";

const args = process.argv.slice(2);
const argOf = (name, fallback) => {
  const i = args.indexOf(name);
  return i === -1 ? fallback : args[i + 1];
};

const TARGET = Number(argOf("--chars", 8000));
const UPDATE = args.includes("--update");
const ROOT = new URL("..", import.meta.url).pathname;
const FIXTURES = ROOT + "fixtures/";
mkdirSync(FIXTURES, { recursive: true });

/** Project Gutenberg: Alice's Adventures in Wonderland, by Lewis Carroll. */
const GUTENBERG = {
  id: 11,
  title: "Alice's Adventures in Wonderland",
  author: "Lewis Carroll",
  url: "https://www.gutenberg.org/cache/epub/11/pg11.txt",
};

/**
 * Source files that make up the code fixture, in order, read from the working
 * tree only in --update mode.
 *
 * Deliberately excludes files containing special-token literals such as
 * `<|endoftext|>`. Vendor tokenizers disagree about those (HuggingFace emits one
 * token, tiktoken treats them as plain text), which would add noise unrelated to
 * the prose-versus-code question. Special-token handling is covered by the
 * corpus in test/corpus.json instead.
 */
const CODE_SOURCES = ["src/hf.js", "src/sigv4.js", "src/registry.js", "src/report.js"];

/** Patterns that must not appear in the code fixture. */
const FORBIDDEN_IN_CODE = [/<\|/, /\uff5c/];

/** Normalise line endings and collapse the runs of blank lines Gutenberg uses. */
function normalise(s) {
  return s.replace(/\r\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
}

/**
 * Trim to exactly `n` characters.
 *
 * An exact cut can land mid-word. On an 8,000 character sample that is worth at
 * most one token, and it applies to both fixtures equally, so it is preferred
 * over padding, which would itself change the tokenisation.
 */
function trimExact(s, n) {
  if (s.length < n) throw new Error(`only ${s.length} chars available, need ${n}`);
  return s.slice(0, n);
}

const sha256 = (s) => createHash("sha256").update(s, "utf8").digest("hex");

/** Index of the first differing character, or -1 when equal. */
function firstDiff(a, b) {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return i;
  return a.length === b.length ? -1 : n;
}

console.log(`target: ${TARGET.toLocaleString()} characters per fixture (${UPDATE ? "update" : "check"} mode)\n`);

// --- prose -------------------------------------------------------------------
process.stdout.write(`fetching ${GUTENBERG.title} from Project Gutenberg... `);
const res = await fetch(GUTENBERG.url);
if (!res.ok) throw new Error(`Gutenberg returned HTTP ${res.status}`);
const book = normalise(await res.text());
console.log(`ok (${book.length.toLocaleString()} chars)`);

// Take continuous narrative prose from the opening chapters, skipping the
// Gutenberg header and the table of contents.
const startMarker = "Alice was beginning to get very tired";
const startIdx = book.indexOf(startMarker);
if (startIdx === -1) throw new Error("could not locate the start of the narrative");
const prose = trimExact(normalise(book.slice(startIdx)), TARGET);

// --- code --------------------------------------------------------------------
// Check mode treats the committed fixture as canonical (it IS the frozen
// snapshot); update mode re-freezes from the current sources.
let code;
if (UPDATE) {
  code = trimExact(normalise(CODE_SOURCES.map((f) => readFileSync(ROOT + f, "utf8")).join("\n")), TARGET);
} else {
  const p = FIXTURES + "code.js";
  if (!existsSync(p)) {
    console.error("FAIL: fixtures/code.js is missing. Run `pnpm run fixtures -- --update` to freeze one.");
    process.exit(1);
  }
  code = readFileSync(p, "utf8");
  if (code.length !== TARGET) {
    console.error(`FAIL: fixtures/code.js is ${code.length.toLocaleString()} chars, expected exactly ${TARGET.toLocaleString()}.`);
    process.exit(1);
  }
}

for (const pattern of FORBIDDEN_IN_CODE) {
  if (pattern.test(code)) {
    throw new Error(
      `code fixture contains ${pattern}, which tokenizers treat inconsistently. ` +
      `Adjust CODE_SOURCES so the trimmed window excludes it.`,
    );
  }
}

// --- provenance ----------------------------------------------------------------
// Deliberately deterministic (no dates), so check mode can diff this file too.
// The content hashes pin exactly which bytes every published number was
// measured on; check mode recomputes them, so an edited fixture surfaces as a
// PROVENANCE drift even before the reference counts disagree.
const provenance = `# Fixture provenance

Both fixtures are trimmed to exactly **${TARGET.toLocaleString()} characters** so that a
token-count comparison between them reflects content rather than length.

## prose.txt

- Source: *${GUTENBERG.title}* by ${GUTENBERG.author}
- Project Gutenberg eBook #${GUTENBERG.id}, <${GUTENBERG.url}>
- Public domain in the United States.
- Extracted from the start of the narrative, then trimmed to ${TARGET.toLocaleString()} characters.
- sha256: \`${sha256(prose)}\`

## code.js

- Source: this repository's own MIT-licensed source, concatenated in this order:
${CODE_SOURCES.map((f) => `  - \`${f}\``).join("\n")}
- A frozen snapshot taken when the published numbers were measured. Later edits
  to those source files deliberately do not change this fixture; the hash below
  is the pin, and \`pnpm run fixtures\` fails if the bytes drift from it.
- Trimmed to ${TARGET.toLocaleString()} characters.
- sha256: \`${sha256(code)}\`

## Regenerating

\`\`\`bash
pnpm run fixtures              # verify both fixtures against this provenance
pnpm run fixtures -- --update  # re-freeze from the current sources
\`\`\`

The fixtures are inputs to \`test/reference.json\` and to every number quoted
in the README. After \`--update\`, regenerate the reference counts with
\`pnpm run verify\` and re-measure the README tables (the Claude rows need
Bedrock access: \`pnpm run audit:bedrock\`).
`;

// --- check / write -------------------------------------------------------------
const stats = (name, s) => {
  const lines = s.split("\n").length;
  const words = s.split(/\s+/).filter(Boolean).length;
  const bytes = Buffer.byteLength(s, "utf8");
  console.log(`  ${name.padEnd(10)} ${s.length.toLocaleString().padStart(7)} chars  ` +
              `${bytes.toLocaleString().padStart(7)} bytes  ` +
              `${lines.toLocaleString().padStart(5)} lines  ${words.toLocaleString().padStart(6)} words`);
};

if (UPDATE) {
  writeFileSync(FIXTURES + "prose.txt", prose);
  writeFileSync(FIXTURES + "code.js", code);
  writeFileSync(FIXTURES + "PROVENANCE.md", provenance);
  console.log("\nwrote fixtures:");
  stats("prose.txt", prose);
  stats("code.js", code);
  console.log(`\n  sha256(prose.txt) ${sha256(prose)}`);
  console.log(`  sha256(code.js)   ${sha256(code)}`);
  console.log(
    "\nFixtures re-frozen. If their content changed, downstream artefacts are now stale:\n" +
    "  1. pnpm run verify          regenerate test/reference.json from the new fixtures\n" +
    "  2. README tables            re-run the quoted commands and update the numbers\n" +
    "  3. Claude fixture numbers   need live Bedrock access (pnpm run audit:bedrock)",
  );
} else {
  console.log();
  let drift = 0;
  for (const [name, content] of [["prose.txt", prose], ["PROVENANCE.md", provenance]]) {
    const p = FIXTURES + name;
    const committed = existsSync(p) ? readFileSync(p, "utf8") : null;
    if (committed === content) {
      console.log(`  ok      ${name.padEnd(13)} matches the committed file byte-for-byte`);
      continue;
    }
    drift++;
    if (committed === null) {
      console.log(`  MISSING ${name.padEnd(13)} not in fixtures/ yet`);
    } else {
      const i = firstDiff(committed, content);
      console.log(
        `  DRIFT   ${name.padEnd(13)} first differs at char ${i.toLocaleString()}: ` +
        `committed ${JSON.stringify(committed.slice(i, i + 24))} vs expected ${JSON.stringify(content.slice(i, i + 24))}`,
      );
    }
  }
  console.log(`  ok      ${"code.js".padEnd(13)} ${code.length.toLocaleString()} chars, sha256 ${sha256(code).slice(0, 12)}... (verified via PROVENANCE.md)`);

  if (drift) {
    console.error(
      `\nFAIL: ${drift} file(s) differ from the committed fixtures. Either the Gutenberg\n` +
      "source changed upstream or a fixture was edited by hand. If the change is\n" +
      "intended, rerun with --update, then regenerate test/reference.json and the\n" +
      "README numbers.",
    );
    process.exitCode = 1;
  } else {
    console.log(`\nboth fixtures are exactly ${TARGET.toLocaleString()} characters and match their recorded provenance`);
  }
}
