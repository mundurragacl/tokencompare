/**
 * Resolve every model in the OpenRouter catalogue to a tokenizer.
 *
 * Resolution is tiered, and the tier is reported rather than hidden:
 *
 *   exact   the model's own tokenizer, or a published encoding for it
 *   family  no tokenizer of its own, so a sibling in the same tokenizer family
 *           (as declared by OpenRouter) stands in. Inferred, not verified.
 *   none    no tokenizer is obtainable; pricing is still shown, counts are not
 *
 * Models are keyed to tokenizers by blob id where possible, so models that
 * provably share a tokenizer collapse to a single entry and a single download.
 */
import { encodingForModel, SUPPORTED_ENCODINGS } from "./encodings.js";
import { loadTokenizerIndex } from "./tokenizer-index.js";

/**
 * Claude tokenizer generations, split at 4.7 where Anthropic changed tokenizer.
 *
 * Both lists are explicit and bounded (the trailing `($|[:-])` stops
 * `opus-5` from claiming a future `opus-5.1`). An id matching neither list is
 * a Claude model whose generation has not been audited; it is reported as
 * uncountable rather than silently binned into a generation and labelled
 * exact. `pnpm run audit:bedrock` is the tool for extending the lists.
 */
const CLAUDE_NEXT = /claude-(opus-(4\.7|4\.8|5)|sonnet-5|fable-5|haiku-5|mythos)($|[:-])/;
const CLAUDE_LEGACY = /claude-(instant|[123])([.-]|$)|claude-(opus|sonnet|haiku)-4(\.[0-6])?($|[:-])/;

/** Bedrock probes per Claude generation, verified to agree across the generation. */
const CLAUDE_BEDROCK = {
  legacy: { endpoint: "runtime", modelId: "anthropic.claude-opus-4-6-v1", region: "us-east-1" },
  next: { endpoint: "mantle", modelId: "anthropic.claude-opus-5", region: "us-east-1" },
};

/**
 * Bedrock ids for individual Claude models, so each one can be measured on its
 * own endpoint instead of inheriting a generation-level probe.
 *
 * Models at 4.7 and later are cross-Region-inference only, which is why they sit
 * on bedrock-mantle: CountTokens on bedrock-runtime rejects them outright. Every
 * entry here has been confirmed to answer.
 */
const CLAUDE_MODEL_TARGETS = {
  "anthropic/claude-opus-4.5": { endpoint: "runtime", modelId: "anthropic.claude-opus-4-5-20251101-v1:0" },
  "anthropic/claude-opus-4.6": { endpoint: "runtime", modelId: "anthropic.claude-opus-4-6-v1" },
  "anthropic/claude-opus-4.7": { endpoint: "mantle", modelId: "anthropic.claude-opus-4-7" },
  "anthropic/claude-opus-4.8": { endpoint: "mantle", modelId: "anthropic.claude-opus-4-8" },
  "anthropic/claude-opus-5": { endpoint: "mantle", modelId: "anthropic.claude-opus-5" },
  "anthropic/claude-sonnet-4.5": { endpoint: "runtime", modelId: "anthropic.claude-sonnet-4-5-20250929-v1:0" },
  "anthropic/claude-sonnet-4.6": { endpoint: "runtime", modelId: "anthropic.claude-sonnet-4-6" },
  "anthropic/claude-sonnet-5": { endpoint: "mantle", modelId: "anthropic.claude-sonnet-5" },
  "anthropic/claude-haiku-4.5": { endpoint: "runtime", modelId: "anthropic.claude-haiku-4-5-20251001-v1:0" },
};

/** Bedrock target for a Claude model id, stripping OpenRouter suffixes. */
export function claudeTarget(id) {
  const base = id.replace(/(-fast)?(:.*)?$/, "");
  return CLAUDE_MODEL_TARGETS[base] || CLAUDE_MODEL_TARGETS[id] || null;
}

/** Families where no member publishes a tokenizer and no counting API exists. */
const NO_TOKENIZER_REASON = {
  Grok: "xAI publishes no tokenizer and offers no token-counting API",
  Nova: "Amazon publishes no Nova tokenizer; Bedrock CountTokens is Anthropic-only",
};

function isRealModel(m, { includeBatch }) {
  if (m.id.startsWith("~")) return false;                       // router alias
  if (m.architecture?.tokenizer === "Router") return false;      // router alias
  if (!includeBatch && m.id.endsWith(":batch")) return false;    // price-only variant
  return true;
}

/** "next", "legacy", or null when the id matches no audited generation. */
function claudeGeneration(id) {
  if (CLAUDE_NEXT.test(id)) return "next";
  if (CLAUDE_LEGACY.test(id)) return "legacy";
  return null;
}

/**
 * @param {Map<string, object>} catalog
 * @param {{includeBatch?: boolean, refreshIndex?: boolean, onProgress?: Function}} [opts]
 */
export async function buildRegistry(catalog, opts = {}) {
  const models = [...catalog.values()].filter((m) => isRealModel(m, { includeBatch: Boolean(opts.includeBatch) }));

  // Index every repo referenced by the catalogue in one pass.
  const repos = [...new Set(models.map((m) => m.hugging_face_id).filter(Boolean))];
  const index = await loadTokenizerIndex(repos, {
    refresh: opts.refreshIndex,
    onProgress: opts.onProgress,
  });

  /** family -> the repo entry we will use as a stand-in for repo-less members */
  const familyProxy = new Map();
  for (const m of models) {
    const family = m.architecture?.tokenizer;
    if (!family || !m.hugging_face_id) continue;
    const entry = index.get(m.hugging_face_id);
    if (!entry?.file) continue;
    // First usable repo in catalogue order. That order is not contractual, so
    // the chosen stand-in is named in the row's reason and every count derived
    // from it is reported as "family", never "exact".
    if (!familyProxy.has(family)) familyProxy.set(family, { repo: m.hugging_face_id, entry });
  }

  const tokenizers = new Map();
  const resolved = [];

  const addTokenizer = (key, spec) => {
    if (!tokenizers.has(key)) tokenizers.set(key, { key, ...spec });
    return key;
  };

  for (const m of models) {
    const family = m.architecture?.tokenizer || "(unknown)";
    const provider = m.id.split("/")[0];
    let tokenizerKey = null;
    let resolution = "none";
    let reason = null;

    if (family === "Claude") {
      const generation = claudeGeneration(m.id);
      if (!generation) {
        // Refusing to guess beats silently binning a future model into the
        // wrong generation and calling the count exact.
        reason = "Claude id outside the audited generations (<= 4.6 and >= 4.7); " +
          "extend the lists in src/registry.js and re-run pnpm run audit:bedrock";
      } else {
        const label = generation === "next" ? "Claude (>= 4.7)" : "Claude (<= 4.6)";
        const own = claudeTarget(m.id);

        // Measure each known model on its own endpoint. Models with no Bedrock
        // id of their own still resolve via the generation-level probe, but that
        // count is a sibling's standing in — an inference — so it is reported as
        // "family", never "exact".
        tokenizerKey = own
          ? addTokenizer(`claude:model:${own.modelId}`, {
              kind: "claude",
              generation,
              bedrock: { ...own, region: "us-east-1" },
              label,
              endpoint: own.endpoint,
              bedrockModelId: own.modelId,
            })
          : addTokenizer(`claude:${generation}`, {
              kind: "claude",
              generation,
              bedrock: CLAUDE_BEDROCK[generation],
              label,
              endpoint: CLAUDE_BEDROCK[generation].endpoint,
              bedrockModelId: CLAUDE_BEDROCK[generation].modelId,
            });
        resolution = own ? "exact" : "family";
        if (!own) reason = `no per-model Bedrock id; the ${CLAUDE_BEDROCK[generation].modelId} generation probe stands in`;
      }
    } else if (m.hugging_face_id && index.get(m.hugging_face_id)?.file) {
      const entry = index.get(m.hugging_face_id);
      tokenizerKey = addTokenizer(`hf:${entry.oid}`, {
        kind: "hf",
        repo: entry.repo,
        file: entry.file,
        size: entry.size,
        label: entry.repo.split("/").pop(),
      });
      resolution = "exact";
    } else if (family === "GPT") {
      const { encoding, matched } = encodingForModel(m.id);
      if (SUPPORTED_ENCODINGS.has(encoding)) {
        tokenizerKey = addTokenizer(`tiktoken:${encoding}`, {
          kind: "tiktoken",
          encoding,
          label: `${encoding} (tiktoken)`,
        });
        resolution = matched ? "exact" : "family";
        if (!matched) reason = `id matches no documented tiktoken prefix; assuming ${encoding}`;
      } else {
        reason = `encoding ${encoding} is not available locally`;
      }
    } else if (familyProxy.has(family)) {
      const { repo, entry } = familyProxy.get(family);
      tokenizerKey = addTokenizer(`hf:${entry.oid}`, {
        kind: "hf",
        repo: entry.repo,
        file: entry.file,
        size: entry.size,
        label: entry.repo.split("/").pop(),
      });
      resolution = "family";
      reason = `no tokenizer published; using ${repo} as the ${family}-family stand-in`;
    } else {
      reason = NO_TOKENIZER_REASON[family]
        || (m.hugging_face_id
              ? index.get(m.hugging_face_id)?.reason || "no usable tokenizer file"
              : `no model in the ${family} family publishes a tokenizer`);
    }

    resolved.push({
      id: m.id,
      name: m.name || m.id,
      provider,
      family,
      contextLength: m.context_length ?? null,
      pricing: m.pricing || {},
      huggingFaceId: m.hugging_face_id || null,
      tokenizerKey,
      resolution,
      reason,
    });
  }

  return { models: resolved, tokenizers };
}

