/**
 * Curated selections over the full catalogue.
 *
 * Every model now comes from the live OpenRouter catalogue via src/registry.js;
 * this file only decides which slice to show by default. `--all` bypasses it.
 */

/** Default comparison set: the frontier models across the three big providers. */
export const CURATED = [
  // Anthropic
  "anthropic/claude-opus-4.5",
  "anthropic/claude-opus-4.6",
  "anthropic/claude-opus-4.7",
  "anthropic/claude-opus-4.8",
  "anthropic/claude-opus-5",
  "anthropic/claude-sonnet-4.5",
  "anthropic/claude-sonnet-4.6",
  "anthropic/claude-sonnet-5",
  "anthropic/claude-haiku-4.5",
  // OpenAI
  "openai/gpt-5.6-sol",
  "openai/gpt-5.6-sol-pro",
  "openai/gpt-5.6-terra",
  "openai/gpt-5.6-terra-pro",
  "openai/gpt-5.6-luna",
  "openai/gpt-5.6-luna-pro",
  // Open weights
  "moonshotai/kimi-k3",
  "moonshotai/kimi-k2.6",
  "z-ai/glm-5.2",
  "deepseek/deepseek-v4-pro",
];

/** Groups usable with --group. */
export const GROUPS = {
  claude: (m) => m.family === "Claude",
  openai: (m) => m.provider === "openai",
  open: (m) => Boolean(m.huggingFaceId),
  google: (m) => m.provider === "google",
  qwen: (m) => m.provider === "qwen" || m.family?.startsWith("Qwen"),
  meta: (m) => m.provider === "meta-llama" || m.family?.startsWith("Llama"),
  mistral: (m) => m.provider === "mistralai",
  deepseek: (m) => m.provider === "deepseek",
  grok: (m) => m.provider === "x-ai",
  amazon: (m) => m.provider === "amazon",
};

/**
 * Filter registry models.
 * @param {Array<object>} models
 * @param {{all?: boolean, groups?: string[], ids?: string[], providers?: string[],
 *          resolutions?: string[]}} opts
 */
export function selectModels(models, opts = {}) {
  let out = models;

  if (!opts.all && !opts.groups?.length && !opts.ids?.length && !opts.providers?.length) {
    const wanted = new Set(CURATED);
    // Preserve the curated order rather than catalogue order.
    const order = new Map(CURATED.map((id, i) => [id, i]));
    out = out
      .filter((m) => wanted.has(m.id))
      .sort((a, b) => order.get(a.id) - order.get(b.id));
  } else {
    if (opts.groups?.length) {
      const preds = opts.groups.map((g) => GROUPS[g]).filter(Boolean);
      if (preds.length) out = out.filter((m) => preds.some((p) => p(m)));
    }
    if (opts.providers?.length) {
      out = out.filter((m) => opts.providers.includes(m.provider));
    }
    if (opts.ids?.length) {
      out = out.filter((m) => opts.ids.some((want) => m.id.includes(want)));
    }
  }

  // Applied to every path, including the curated default, so
  // `tokenizer "x" --only exact` filters instead of being silently ignored.
  if (opts.resolutions?.length) {
    out = out.filter((m) => opts.resolutions.includes(m.resolution));
  }
  return out;
}
