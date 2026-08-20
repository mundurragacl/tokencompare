/** Table and summary rendering for the comparison report. */

const DIM = "\x1b[2m";
const BOLD = "\x1b[1m";
const RESET = "\x1b[0m";

let useColor = process.stdout.isTTY && !process.env.NO_COLOR;
export function setColor(on) { useColor = on; }
const c = (code, s) => (useColor ? code + s + RESET : s);

const visibleLength = (s) => String(s).replace(/\x1b\[[0-9;]*m/g, "").length;

export function formatMoney(dollars) {
  if (dollars === null || dollars === undefined) return "-";
  if (dollars === 0) return "$0";
  if (dollars < 0.000001) return "<$0.000001";
  if (dollars < 0.01) return "$" + dollars.toFixed(6);
  if (dollars < 1) return "$" + dollars.toFixed(4);
  return "$" + dollars.toFixed(2);
}

function pad(s, width, align = "left") {
  const fill = Math.max(0, width - visibleLength(s));
  return align === "right" ? " ".repeat(fill) + s : String(s) + " ".repeat(fill);
}

/** Marker showing how a count was obtained. */
export function resolutionMark(row) {
  if (row.tokens == null) return "x";      // no tokenizer available
  if (row.resolution === "family") return "*"; // family stand-in
  if (row.exact === false) return "~";     // estimated
  return " ";
}

/**
 * Render a table from column definitions.
 * @param {Array<object>} rows
 * @param {Array<{key: string, title: string, align?: string}>} cols
 */
export function renderTable(rows, cols) {
  const widths = {};
  for (const col of cols) {
    widths[col.key] = Math.max(
      col.title.length,
      ...rows.map((r) => visibleLength(r[col.key] ?? "")),
    );
  }

  const lines = [];
  lines.push(c(BOLD, cols.map((col) => pad(col.title, widths[col.key], col.align)).join("  ")));
  lines.push(c(DIM, "─".repeat(cols.reduce((n, col) => n + widths[col.key] + 2, -2))));
  for (const row of rows) {
    lines.push(cols.map((col) => pad(row[col.key] ?? "", widths[col.key], col.align)).join("  "));
  }
  return lines.join("\n");
}

/**
 * Short description of where a count came from, so the table is auditable
 * without cross-referencing a legend.
 */
function sourceLabel(row) {
  if (row.tokens == null) return "unavailable";
  if (row.source === "bedrock") {
    const ovh = typeof row.overhead === "number" ? ` ovh ${row.overhead}` : "";
    return `bedrock-${row.endpoint}${ovh}`;
  }
  if (row.source === "anthropic-api") return "anthropic api";
  if (row.source === "estimate") return "estimate";
  if (row.source === "tiktoken") return "tiktoken (local)";
  if (row.source === "bpe") return "own vocab (local)";
  return row.method ?? "-";
}

/** Per-model detail table. */
export function renderModelTable(rows, { baselineTokens, showSource = true }) {
  const cols = [
    { key: "model", title: "MODEL" },
    { key: "tokens", title: "TOKENS", align: "right" },
    { key: "vs", title: "VS BASE", align: "right" },
    { key: "tokenizer", title: "TOKENIZER" },
    ...(showSource ? [{ key: "source", title: "COUNTED VIA" }] : []),
    { key: "inputCost", title: "IN COST", align: "right" },
    { key: "outputCost", title: "OUT COST", align: "right" },
    { key: "ctx", title: "CTX USED", align: "right" },
  ];

  const cells = rows.map((r) => {
    const vs = baselineTokens && r.tokens != null
      ? (r.tokens === baselineTokens
          ? "—"
          : (r.tokens > baselineTokens ? "+" : "") +
            (((r.tokens - baselineTokens) / baselineTokens) * 100).toFixed(1) + "%")
      : "-";
    return {
      model: r.name + resolutionMark(r),
      tokens: r.tokens == null ? "n/a" : r.tokens.toLocaleString(),
      vs,
      inputCost: formatMoney(r.inputCost),
      outputCost: formatMoney(r.outputCost),
      ctx: r.ctxPct == null ? "-" : r.ctxPct < 0.1 ? "<0.1%" : r.ctxPct.toFixed(1) + "%",
      tokenizer: r.tokenizerLabel ?? "-",
      source: sourceLabel(r),
    };
  });

  return renderTable(cells, cols);
}

/**
 * One row per distinct tokenizer, which is the real comparison: models sharing a
 * tokenizer always produce the same count.
 */
export function renderTokenizerTable(rows, { baselineTokens }) {
  // Group by tokenizer identity. For local tokenizers that identity is the
  // spec key (blob id or encoding name), so two distinct vocabularies that
  // merely share a repo basename can never merge. Claude rows group by
  // generation label instead: each model is probed on its own endpoint and
  // counts within a generation may differ by the documented +/-1 envelope
  // noise, which is measurement noise, not a different tokenizer.
  const byTokenizer = new Map();
  for (const r of rows) {
    if (!r.tokenizerKey) continue;
    const key = r.tokenizerKey.startsWith("claude:")
      ? `claude|${r.tokenizerLabel}`
      : r.tokenizerKey;
    if (!byTokenizer.has(key)) {
      byTokenizer.set(key, {
        label: r.tokenizerLabel,
        tokens: r.tokens,
        exact: r.exact,
        resolution: r.resolution,
        error: r.error,
        models: [],
      });
    }
    byTokenizer.get(key).models.push(r);
  }

  const entries = [...byTokenizer.values()].sort((a, b) => {
    if (a.tokens == null && b.tokens == null) return 0;
    if (a.tokens == null) return 1;
    if (b.tokens == null) return -1;
    return a.tokens - b.tokens;
  });

  const cells = entries.map((e) => {
    const vs = baselineTokens && e.tokens != null
      ? (e.tokens === baselineTokens
          ? "—"
          : (e.tokens > baselineTokens ? "+" : "") +
            (((e.tokens - baselineTokens) / baselineTokens) * 100).toFixed(1) + "%")
      : "-";
    // Cheapest and priciest model on this tokenizer, to show the cost spread.
    const priced = e.models.filter((m) => m.inputCost != null).sort((a, b) => a.inputCost - b.inputCost);
    return {
      tokenizer: e.label + (e.tokens == null ? " x" : e.exact === false ? " ~" : ""),
      tokens: e.tokens == null ? "n/a" : e.tokens.toLocaleString(),
      vs,
      models: String(e.models.length),
      cheapest: priced.length ? formatMoney(priced[0].inputCost) : "-",
      priciest: priced.length ? formatMoney(priced[priced.length - 1].inputCost) : "-",
      example: e.models[0].name.slice(0, 34),
    };
  });

  return renderTable(cells, [
    { key: "tokenizer", title: "TOKENIZER" },
    { key: "tokens", title: "TOKENS", align: "right" },
    { key: "vs", title: "VS BASE", align: "right" },
    { key: "models", title: "MODELS", align: "right" },
    { key: "cheapest", title: "CHEAPEST IN", align: "right" },
    { key: "priciest", title: "PRICIEST IN", align: "right" },
    { key: "example", title: "E.G." },
  ]);
}

/** Coverage and legend footer. */
export function renderCoverage(rows) {
  const total = rows.length;
  const counted = rows.filter((r) => r.tokens != null);
  const exact = counted.filter((r) => r.exact !== false && r.resolution === "exact");
  const family = counted.filter((r) => r.resolution === "family");
  const estimated = counted.filter((r) => r.exact === false);
  const missing = rows.filter((r) => r.tokens == null);

  const pct = (n) => ((n / total) * 100).toFixed(0) + "%";
  const lines = [];
  lines.push(`  ${String(exact.length).padStart(4)} exact       ${pct(exact.length).padStart(4)}  the model's own tokenizer or published encoding`);
  if (family.length) lines.push(`  ${String(family.length).padStart(4)} family  *   ${pct(family.length).padStart(4)}  a sibling model's tokenizer or probe stands in (inferred)`);
  if (estimated.length) lines.push(`  ${String(estimated.length).padStart(4)} estimate ~  ${pct(estimated.length).padStart(4)}  scaled from o200k_base`);
  if (missing.length) lines.push(`  ${String(missing.length).padStart(4)} none    x   ${pct(missing.length).padStart(4)}  no tokenizer obtainable`);
  lines.push(`  ${String(total).padStart(4)} models total`);
  return lines.join("\n");
}

/** Group the reasons why models could not be counted. */
export function renderGaps(rows) {
  const byReason = new Map();
  for (const r of rows) {
    if (r.tokens != null) continue;
    const reason = r.error || r.reason || "unknown";
    if (!byReason.has(reason)) byReason.set(reason, []);
    byReason.get(reason).push(r);
  }
  if (!byReason.size) return "";

  const lines = [];
  for (const [reason, list] of [...byReason.entries()].sort((a, b) => b[1].length - a[1].length)) {
    const providers = [...new Set(list.map((m) => m.provider))];
    lines.push(`  ${String(list.length).padStart(3)} models  ${reason}`);
    lines.push(c(DIM, `             ${providers.slice(0, 5).join(", ")}${providers.length > 5 ? " ..." : ""}`));
  }
  return lines.join("\n");
}
