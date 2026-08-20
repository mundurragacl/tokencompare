/**
 * Audit Claude token counting through Bedrock.
 *
 * Default mode: src/registry.js groups Claude models into two tokenizer
 * generations and probes one representative per group. This script checks that
 * assumption by counting the same text on every model, so a future model that
 * quietly changes tokenizer shows up as a mismatch instead of being silently
 * mislabelled.
 *
 * --sweep mode: substantiates the README claim that Bedrock CountTokens is
 * Anthropic-only. Lists every foundation model the account can see via
 * `aws bedrock list-foundation-models` and probes CountTokens on each
 * non-Anthropic id, expecting a rejection. Exits non-zero if any of them
 * starts answering, because that would obsolete the claim.
 *
 * Usage:
 *   node scripts/bedrock-audit.mjs [--profile NAME] [--region NAME] [--file PATH]
 *   node scripts/bedrock-audit.mjs --sweep [--profile NAME] [--region NAME]
 */
import { execFile } from "node:child_process";
import { readFileSync } from "node:fs";
import { promisify } from "node:util";
import { countClaudeExact, resolveCredentials } from "../src/bedrock.js";

const execFileAsync = promisify(execFile);

const args = process.argv.slice(2);
const arg = (name, fallback) => {
  const i = args.indexOf(name);
  return i === -1 ? fallback : args[i + 1];
};

const profile = arg("--profile", process.env.AWS_PROFILE || "default");
const region = arg("--region", process.env.AWS_REGION || "us-east-1");
const file = arg("--file", new URL("../fixtures/prose.txt", import.meta.url).pathname);
const text = readFileSync(file, "utf8");

/** Every Claude model in the comparison set, with its expected generation. */
const TARGETS = [
  { label: "Opus 4.5", generation: "legacy", endpoint: "runtime", modelId: "anthropic.claude-opus-4-5-20251101-v1:0" },
  { label: "Opus 4.6", generation: "legacy", endpoint: "runtime", modelId: "anthropic.claude-opus-4-6-v1" },
  { label: "Sonnet 4.5", generation: "legacy", endpoint: "runtime", modelId: "anthropic.claude-sonnet-4-5-20250929-v1:0" },
  { label: "Sonnet 4.6", generation: "legacy", endpoint: "runtime", modelId: "anthropic.claude-sonnet-4-6" },
  { label: "Haiku 4.5", generation: "legacy", endpoint: "runtime", modelId: "anthropic.claude-haiku-4-5-20251001-v1:0" },
  { label: "Opus 4.7", generation: "next", endpoint: "mantle", modelId: "anthropic.claude-opus-4-7" },
  { label: "Opus 4.8", generation: "next", endpoint: "mantle", modelId: "anthropic.claude-opus-4-8" },
  { label: "Opus 5", generation: "next", endpoint: "mantle", modelId: "anthropic.claude-opus-5" },
  { label: "Sonnet 5", generation: "next", endpoint: "mantle", modelId: "anthropic.claude-sonnet-5" },
];

// A Bedrock API key needs no local AWS config; fall back to IAM if absent.
const apiKey = process.env.AWS_BEARER_TOKEN_BEDROCK || null;
const credentials = apiKey ? null : await resolveCredentials(profile);

console.log(`auth=${apiKey ? "api-key" : `iam (profile ${profile})`} region=${region}`);

if (args.includes("--sweep")) {
  console.log("sweep: probing CountTokens on every non-Anthropic foundation model\n");

  const listArgs = ["bedrock", "list-foundation-models", "--region", region, "--output", "json"];
  if (!apiKey && profile) listArgs.push("--profile", profile);
  const { stdout } = await execFileAsync("aws", listArgs, {
    maxBuffer: 1 << 26,
    env: { ...process.env, AWS_PAGER: "" },
  });
  const all = JSON.parse(stdout).modelSummaries ?? [];
  const targets = all.filter((m) => m.providerName !== "Anthropic" && !m.modelId.startsWith("anthropic."));
  console.log(`${all.length} model ids listed; ${all.length - targets.length} Anthropic (skipped); probing ${targets.length}\n`);

  const rejected = [];
  const supported = [];
  const inconclusive = [];
  for (const m of targets) {
    try {
      await countClaudeExact("a", { endpoint: "runtime", modelId: m.modelId, region }, { profile, credentials, apiKey });
      supported.push(m.modelId);
      console.log(`  SUPPORTED    ${m.modelId}`);
    } catch (err) {
      const msg = String(err?.message ?? err).split("\n")[0];
      if (/doesn'?t support (counting tokens|token counting)/i.test(msg)) {
        rejected.push(m.modelId);
        console.log(`  rejected     ${m.modelId}`);
      } else {
        // Access denied, throttling, wrong endpoint shape: says nothing about
        // whether the model supports counting, so it neither confirms nor
        // falsifies the claim.
        inconclusive.push(m.modelId);
        console.log(`  inconclusive ${m.modelId}: ${msg.slice(0, 80)}`);
      }
    }
  }

  console.log(
    `\n${targets.length} non-Anthropic model ids: ${rejected.length} reject CountTokens, ` +
    `${supported.length} support it, ${inconclusive.length} inconclusive`,
  );
  if (supported.length) {
    console.log("\nCountTokens is no longer Anthropic-only; update the README Limitations section.");
  }
  process.exit(supported.length ? 1 : 0);
}

console.log(`file=${file.split("/").pop()} (${text.length.toLocaleString()} chars)\n`);
console.log("MODEL        GEN     ENDPOINT  OVERHEAD   RAW TOKENS");
console.log("-".repeat(56));

const byGeneration = new Map();
const failures = [];

for (const t of TARGETS) {
  try {
    const res = await countClaudeExact(text, { ...t, region }, { profile, credentials, apiKey });
    console.log(
      `${t.label.padEnd(12)} ${t.generation.padEnd(7)} ${t.endpoint.padEnd(9)} ` +
      `${String(res.overhead).padStart(8)} ${res.tokens.toLocaleString().padStart(12)}`,
    );
    if (!byGeneration.has(t.generation)) byGeneration.set(t.generation, new Map());
    byGeneration.get(t.generation).set(t.label, res.tokens);
  } catch (err) {
    console.log(`${t.label.padEnd(12)} ${t.generation.padEnd(7)} ${t.endpoint.padEnd(9)}    FAILED  ${err.message.slice(0, 60)}`);
    failures.push(t.label);
  }
}

/**
 * Envelope subtraction can be off by one, because BPE merges may span the
 * boundary between the request envelope and the text. A spread of one token is
 * therefore expected noise; anything wider means the models genuinely disagree
 * and the generation grouping in src/registry.js is wrong.
 */
const ENVELOPE_NOISE = 1;

console.log("\nConsistency within each generation");
let mismatched = 0;
for (const [generation, counts] of byGeneration) {
  const values = [...counts.values()];
  const spread = Math.max(...values) - Math.min(...values);
  const label = generation.padEnd(7);

  if (spread === 0) {
    console.log(`  ${label} all ${counts.size} models agree at ${values[0].toLocaleString()} tokens`);
  } else if (spread <= ENVELOPE_NOISE) {
    console.log(`  ${label} all ${counts.size} models agree within ${spread} token ` +
                `(${Math.min(...values).toLocaleString()}-${Math.max(...values).toLocaleString()}), ` +
                `which is expected envelope noise`);
  } else {
    mismatched++;
    console.log(`  ${label} MISMATCH, spread of ${spread} tokens: ${JSON.stringify(Object.fromEntries(counts))}`);
  }
}

const legacy = [...(byGeneration.get("legacy")?.values() ?? [])][0];
const next = [...(byGeneration.get("next")?.values() ?? [])][0];
if (legacy && next) {
  console.log(`\n  >=4.7 uses ${((next / legacy - 1) * 100).toFixed(1)}% more tokens than <=4.6 on this text`);
}

if (failures.length) console.log(`\n  could not measure: ${failures.join(", ")}`);
process.exitCode = mismatched === 0 ? 0 : 1;
