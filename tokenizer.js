#!/usr/bin/env node
/**
 * Compare how the same text tokenizes, and what it costs, across AI models.
 *
 * Token counts come from the real tokenizer wherever one is published:
 *   - OpenAI            tiktoken encodings (o200k_base, cl100k_base, harmony)
 *   - Claude            Amazon Bedrock CountTokens, or Anthropic count_tokens
 *   - open weights      the model's own tokenizer.json / tiktoken.model
 *
 * Models with no published tokenizer fall back to a same-family stand-in, or are
 * reported as uncountable. The report always says which happened.
 *
 * Pricing and context windows come from OpenRouter's public catalogue.
 */
import { readFileSync } from "node:fs";
import { countTokenizers, isCached } from "./src/backends.js";
import { loadCatalog, pricing } from "./src/catalog.js";
import { GROUPS, selectModels } from "./src/models.js";
import { buildRegistry } from "./src/registry.js";
import {
  renderCoverage, renderGaps, renderModelTable, renderTokenizerTable, setColor,
} from "./src/report.js";
import { downloadSize } from "./src/tokenizer-index.js";

const USAGE = `
Compare token counts and cost across AI models and providers.

Usage
  tokenizer "text to measure"
  tokenizer --file <path>
  cat file.md | tokenizer

Selection
  --all                Every model in the OpenRouter catalogue
  --group <name>       ${Object.keys(GROUPS).join(", ")} (repeatable)
  --provider <name>    Limit to a provider slug, e.g. openai (repeatable)
  --model <substr>     Limit to models matching a substring (repeatable)
  --only <tier>        Keep only exact | family | none resolutions (repeatable)

Input
  --file <path>        Read input from a file

Output
  --by-model           One row per model (default when few models are selected)
  --by-tokenizer       One row per distinct tokenizer (default with --all)
  --baseline <substr>  Model to compare against (default: the GPT-5.6 tokenizer)
  --output <n>         Price n output tokens (default: same count as input)
  --limit <n>          Truncate the per-model table
  --json               Emit JSON
  --no-color           Disable colour

Fetching
  --fetch              Download tokenizers that are not cached yet
  --refresh            Refresh the OpenRouter catalogue
  --refresh-index      Re-probe Hugging Face for tokenizer file locations
  HF_TOKEN             Needed for gated repos (Llama, Gemma, Cohere)

Claude counts
  Claude has no public tokenizer, so counts come from an API. Bedrock is used by
  default via your AWS CLI credentials.
  --aws-profile <name> AWS profile (default: $AWS_PROFILE or "default")
  --aws-region <name>  Region for Bedrock calls (default: us-east-1)
  --no-bedrock         Skip Bedrock
  Fallback order: Bedrock -> ANTHROPIC_API_KEY -> calibrated estimate.
`.trim();

/** A bad invocation: report the message and the usage hint, not a stack trace. */
class UsageError extends Error {}

function parseArgs(argv) {
  const o = {
    groups: [], ids: [], providers: [], resolutions: [], text: null, file: null,
    all: false, json: false, refresh: false, refreshIndex: false, fetch: false,
    baseline: null, output: null, limit: null, view: null,
    bedrock: {
      enabled: true,
      profile: process.env.AWS_PROFILE || "default",
      // Not inheriting AWS_REGION: counts are region independent, but the
      // Claude 4.7+ models only resolve on bedrock-mantle in some Regions, so
      // following the caller's inference Region would silently lose exactness.
      region: "us-east-1",
    },
  };
  // Every value-taking flag validates its value up front. Silently coercing a
  // missing or malformed value used to turn `--limit x` into a no-op and
  // `--file` into readFileSync(undefined).
  const need = (name, v) => {
    if (v === undefined || v.startsWith("--")) throw new UsageError(`${name} requires a value`);
    return v;
  };
  const num = (name, v) => {
    const n = Number(need(name, v));
    if (!Number.isFinite(n) || n < 0) throw new UsageError(`${name} requires a non-negative number, got ${JSON.stringify(v)}`);
    return n;
  };

  const rest = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--help" || a === "-h") return { help: true };
    else if (a === "--file") o.file = need(a, argv[++i]);
    else if (a === "--group") o.groups.push(need(a, argv[++i]));
    else if (a === "--provider") o.providers.push(need(a, argv[++i]));
    else if (a === "--model") o.ids.push(need(a, argv[++i]));
    else if (a === "--only") {
      const v = need(a, argv[++i]);
      if (!["exact", "family", "none"].includes(v)) {
        throw new UsageError(`--only expects exact, family or none, got ${JSON.stringify(v)}`);
      }
      o.resolutions.push(v);
    }
    else if (a === "--baseline") o.baseline = need(a, argv[++i]);
    else if (a === "--output") o.output = num(a, argv[++i]);
    else if (a === "--limit") o.limit = num(a, argv[++i]);
    else if (a === "--all") o.all = true;
    else if (a === "--by-model") o.view = "model";
    else if (a === "--by-tokenizer") o.view = "tokenizer";
    else if (a === "--json") o.json = true;
    else if (a === "--fetch") o.fetch = true;
    else if (a === "--refresh") o.refresh = true;
    else if (a === "--refresh-index") o.refreshIndex = true;
    else if (a === "--no-color") setColor(false);
    else if (a === "--no-bedrock") o.bedrock.enabled = false;
    else if (a === "--aws-profile") o.bedrock.profile = need(a, argv[++i]);
    else if (a === "--aws-region") o.bedrock.region = need(a, argv[++i]);
    else if (a.startsWith("--")) throw new UsageError(`unknown flag ${a}`);
    else rest.push(a);
  }
  if (rest.length) o.text = rest.join(" ");
  return o;
}

function readStdin() {
  return new Promise((resolve) => {
    if (process.stdin.isTTY) return resolve("");
    let data = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (chunk) => (data += chunk));
    process.stdin.on("end", () => resolve(data));
  });
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.help) return void console.log(USAGE);

  let text = opts.text;
  if (opts.file) text = readFileSync(opts.file, "utf8");
  if (!text) text = await readStdin();
  if (!text) {
    console.error(USAGE);
    process.exitCode = 1;
    return;
  }

  const log = (msg) => { if (!opts.json) process.stderr.write(msg + "\n"); };

  const catalog = await loadCatalog({ refresh: opts.refresh, onDownload: () => log("fetching model catalogue...") });
  const { models, tokenizers } = await buildRegistry(catalog, {
    refreshIndex: opts.refreshIndex,
    onProgress: (done, total) => {
      if (!opts.json) process.stderr.write(`\rindexing tokenizers ${done}/${total}`);
      if (done === total && !opts.json) process.stderr.write("\n");
    },
  });

  const selected = selectModels(models, opts);
  if (!selected.length) {
    console.error("No models matched. Groups: " + Object.keys(GROUPS).join(", "));
    process.exitCode = 1;
    return;
  }

  // Only the tokenizers the selection actually needs.
  const needed = new Map();
  for (const m of selected) {
    if (m.tokenizerKey && tokenizers.has(m.tokenizerKey)) {
      needed.set(m.tokenizerKey, tokenizers.get(m.tokenizerKey));
    }
  }

  const uncached = [...needed.values()].filter((s) => s.kind === "hf" && !isCached(s));
  if (uncached.length && !opts.fetch) {
    const mb = (downloadSize(uncached) / 1e6).toFixed(0);
    log(`${uncached.length} tokenizers are not cached (~${mb} MB). Counting without them; pass --fetch to download.`);
  } else if (uncached.length) {
    const mb = (downloadSize(uncached) / 1e6).toFixed(0);
    log(`downloading ${uncached.length} tokenizers (~${mb} MB)...`);
  }

  const counts = await countTokenizers(text, needed, {
    apiKey: process.env.ANTHROPIC_API_KEY,
    bedrock: opts.bedrock,
    skipUncached: !opts.fetch,
    concurrency: 4,
    onCounted: (n, total) => {
      if (!opts.json && total > 8) process.stderr.write(`\rcounting ${n}/${total} tokenizers`);
      if (n === total && !opts.json && total > 8) process.stderr.write("\n");
    },
  });

  const outputOverride = Number.isFinite(opts.output) ? opts.output : null;

  const rows = selected.map((m) => {
    const count = m.tokenizerKey ? counts.get(m.tokenizerKey) : null;
    const spec = m.tokenizerKey ? needed.get(m.tokenizerKey) : null;
    const price = pricing({ pricing: m.pricing });
    const tokens = count?.tokens ?? null;
    const outTokens = outputOverride ?? tokens;
    return {
      ...m,
      tokens,
      exact: count?.exact ?? false,
      method: count?.method ?? null,
      source: count?.source ?? null,
      endpoint: count?.endpoint ?? spec?.endpoint ?? null,
      overhead: count?.overhead ?? null,
      bedrockModelId: spec?.bedrockModelId ?? null,
      error: count?.error ?? null,
      tokenizerLabel: spec?.label ?? null,
      ctxPct: m.contextLength && tokens != null ? (tokens / m.contextLength) * 100 : null,
      inputCost: price.inputPerToken != null && tokens != null ? tokens * price.inputPerToken : null,
      outputCost: price.outputPerToken != null && outTokens != null ? outTokens * price.outputPerToken : null,
      inputPerMTok: price.inputPerToken != null ? price.inputPerToken * 1e6 : null,
      outputPerMTok: price.outputPerToken != null ? price.outputPerToken * 1e6 : null,
    };
  });

  let baseline = null;
  if (opts.baseline) baseline = rows.find((r) => r.id.includes(opts.baseline) && r.tokens != null);
  baseline ??= rows.find((r) => r.tokenizerLabel?.startsWith("o200k_base") && r.tokens != null)
           ?? rows.find((r) => r.tokens != null);

  if (opts.json) {
    console.log(JSON.stringify({
      input: { chars: text.length, bytes: Buffer.byteLength(text, "utf8"), lines: text.split("\n").length },
      baseline: baseline ? { id: baseline.id, tokens: baseline.tokens } : null,
      tokenizers: [...needed.entries()].map(([key, spec]) => ({ key, ...spec, ...counts.get(key) })),
      models: rows,
    }, null, 2));
    return;
  }

  const view = opts.view ?? (rows.length > 40 ? "tokenizer" : "model");

  console.log();
  console.log(
    `Input: ${text.length.toLocaleString()} chars, ` +
    `${Buffer.byteLength(text, "utf8").toLocaleString()} bytes, ` +
    `${text.split("\n").length.toLocaleString()} lines`,
  );
  console.log();

  if (view === "tokenizer") {
    console.log(`Distinct tokenizers across ${rows.length} models`);
    console.log(renderTokenizerTable(rows, { baselineTokens: baseline?.tokens ?? null }));
  } else {
    const shown = opts.limit ? rows.slice(0, opts.limit) : rows;
    console.log(renderModelTable(shown, { baselineTokens: baseline?.tokens ?? null }));
    if (shown.length < rows.length) {
      console.log(`  ... ${rows.length - shown.length} more (raise --limit or use --by-tokenizer)`);
    }
  }
  console.log();

  if (baseline) {
    console.log(`  Baseline: ${baseline.name} (${baseline.tokens?.toLocaleString()} tokens)`);
    console.log(outputOverride != null
      ? `  IN COST prices the input; OUT COST prices ${outputOverride.toLocaleString()} output tokens.`
      : "  IN COST prices the input; OUT COST prices the same count as output.");
    console.log();
  }

  console.log("Coverage");
  console.log(renderCoverage(rows));
  console.log();

  const gaps = renderGaps(rows);
  if (gaps) {
    console.log("Why some models could not be counted");
    console.log(gaps);
    console.log();
  }
}

main().catch((err) => {
  console.error(err instanceof UsageError ? `error: ${err.message} (see --help)` : err);
  process.exitCode = 1;
});
