# tokencompare

Compare how the same text tokenizes, and what it costs, across AI models and providers — using **each vendor's own tokenizer**, not an approximation.

```
MODEL                            TOKENS  VS BASE  TOKENIZER              COUNTED VIA               IN COST   OUT COST  CTX USED
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Anthropic: Claude Opus 4.5        2,077    +4.0%  Claude (<= 4.6)        bedrock-runtime ovh 7     $0.0104    $0.0519      1.0%
Anthropic: Claude Opus 4.6        2,077    +4.0%  Claude (<= 4.6)        bedrock-runtime ovh 24    $0.0104    $0.0519      0.2%
Anthropic: Claude Opus 4.7        2,754   +37.9%  Claude (>= 4.7)        bedrock-mantle ovh 11     $0.0138    $0.0689      0.3%
Anthropic: Claude Opus 4.8        2,754   +37.9%  Claude (>= 4.7)        bedrock-mantle ovh 6      $0.0138    $0.0689      0.3%
Claude Opus 5                     2,753   +37.9%  Claude (>= 4.7)        bedrock-mantle ovh 6      $0.0138    $0.0688      0.3%
Anthropic: Claude Sonnet 4.5      2,077    +4.0%  Claude (<= 4.6)        bedrock-runtime ovh 23  $0.006231    $0.0312      0.2%
Anthropic: Claude Sonnet 4.6      2,077    +4.0%  Claude (<= 4.6)        bedrock-runtime ovh 24  $0.006231    $0.0312      0.2%
Anthropic: Claude Sonnet 5        2,754   +37.9%  Claude (>= 4.7)        bedrock-mantle ovh 6    $0.005508    $0.0275      0.3%
Anthropic: Claude Haiku 4.5       2,077    +4.0%  Claude (<= 4.6)        bedrock-runtime ovh 23  $0.002077    $0.0104      1.0%
OpenAI: GPT-5.6 Sol               1,997        —  o200k_base (tiktoken)  tiktoken (local)        $0.004993    $0.0300      0.2%
OpenAI: GPT-5.6 Terra             1,997        —  o200k_base (tiktoken)  tiktoken (local)        $0.003994    $0.0240      0.2%
OpenAI: GPT-5.6 Luna              1,997        —  o200k_base (tiktoken)  tiktoken (local)        $0.000399  $0.002396      0.2%
MoonshotAI: Kimi K3               2,012    +0.8%  Kimi-K3                own vocab (local)       $0.006036    $0.0302      0.2%
MoonshotAI: Kimi K2.6             2,012    +0.8%  Kimi-K3                own vocab (local)       $0.001911  $0.008048      0.8%
Z.ai: GLM 5.2                     2,003    +0.3%  GLM-5.2                own vocab (local)       $0.001935  $0.006081      0.2%
DeepSeek: DeepSeek V4 Pro         2,024    +1.4%  DeepSeek-V4-Pro        own vocab (local)       $0.003238  $0.006477      0.2%

Coverage
    19 exact       100%  the model's own tokenizer or published encoding
```

## Why you can trust the numbers

Most token calculators reimplement or approximate a vendor's tokenizer. This one doesn't:

| Provider | Source of truth | Exact? |
| --- | --- | --- |
| Claude (Opus/Sonnet/Haiku 4.5 → 5) | Amazon Bedrock `CountTokens` — Anthropic's own service | yes |
| OpenAI GPT-5.x, GPT-4.x, o-series | OpenAI's published `o200k_base` / `cl100k_base` encodings | yes |
| GLM, DeepSeek, Qwen, Llama, Mistral… | the vocabulary file from the vendor's own Hugging Face repo | yes |
| Kimi K3 / K2.6 | Moonshot's own `tiktoken.model` vocabulary | yes |
| Grok, Amazon Nova, Gemini | no tokenizer published anywhere | **no — reported as `n/a`** |

No vocabulary in this repo is hand-written or scraped. Open-weight vocabularies are downloaded from the vendor's Hugging Face repo at run time and cached.

The BPE merge loop is our own JavaScript, so it comes with a proof obligation. `pnpm run verify` recounts everything with the **vendors' own reference libraries** — HuggingFace `tokenizers` and OpenAI's Python `tiktoken` — and fails if a single count differs:

```
$ pnpm run verify
1/2  generating counts with the official reference libraries
2/2  diffing the JavaScript implementation against them
zai-org/GLM-5.2              all 20 cases match
deepseek-ai/DeepSeek-V4-Pro  all 20 cases match
moonshotai/Kimi-K3           all 20 cases match
moonshotai/Kimi-K2.6         all 20 cases match
o200k_base                   all 20 cases match
100/100 checks passed
The JavaScript tokenizers match the vendors' own libraries exactly.
```

The 20 cases cover English, Chinese, Japanese, Arabic, Cyrillic, emoji, source code, JSON, whitespace runs, special-token literals, and both 8,000-character fixtures the tables above are built from. (Kimi K3 and K2.6 currently ship byte-identical vocabulary files, so those two targets exercise one vocabulary — keeping both catches the day they diverge. The o200k_base target verifies the bundled `gpt-tokenizer` package rather than this repo's BPE loop, which the other four targets cover.) So "your package is wrong" is a claim you can settle yourself in one command.

## Two findings worth knowing

**1. Claude changed tokenizer at 4.7, and it is expensive.** Anthropic's docs mention roughly 30% more tokens. Measured through Bedrock on 8,000 characters of prose, it is **+32.6%** (2,077 → 2,754), and on English prose versus GPT-5.6 it is **+37.9%**. There are exactly two Claude tokenizer generations, verified by measuring all nine models individually:

```
$ pnpm run audit:bedrock
auth=iam (profile default) region=us-east-1
file=prose.txt (8,000 chars)

MODEL        GEN     ENDPOINT  OVERHEAD   RAW TOKENS
--------------------------------------------------------
Opus 4.5     legacy  runtime          7        2,077
Opus 4.6     legacy  runtime         24        2,077
Sonnet 4.5   legacy  runtime         23        2,077
Sonnet 4.6   legacy  runtime         24        2,077
Haiku 4.5    legacy  runtime         23        2,077
Opus 4.7     next    mantle          11        2,754
Opus 4.8     next    mantle           6        2,754
Opus 5       next    mantle           6        2,753
Sonnet 5     next    mantle           6        2,754

Consistency within each generation
  legacy  all 5 models agree at 2,077 tokens
  next    all 4 models agree within 1 token (2,753-2,754), which is expected envelope noise

  >=4.7 uses 32.6% more tokens than <=4.6 on this text
```

Note the `OVERHEAD` column: the request envelope varies from 6 to 24 tokens depending on model and endpoint. Subtracting it is what makes these counts comparable with the local tokenizers.

**2. Code costs more than prose, and how much depends on the vendor.** See below.

## Does code pay a token tax?

Yes — and the size of the tax varies about fourfold between vendors. Both fixtures are trimmed to **exactly 8,000 characters**, so the difference is content, not length:

```
$ pnpm run codetax

PROSE: 8,000 chars, 8,158 bytes  (prose.txt)
CODE:  8,000 chars, 8,000 bytes  (code.js)
Both inputs are exactly 8,000 characters, so token differences are content, not length.

TOKENIZER                 PROSE      CODE      DIFF  PROSE/1kc   CODE/1kc  PROSE/1kB   CODE/1kB
-------------------------------------------------------------------------------------------------
Claude (<= 4.6)           2,077     2,688    +29.4%      259.6      336.0      254.6      336.0
Claude (>= 4.7)           2,754     3,458    +25.6%      344.3      432.3      337.6      432.3
DeepSeek-V4-Pro           2,024     2,255    +11.4%      253.0      281.9      248.1      281.9
OpenAI: o200k_base        1,997     2,160     +8.2%      249.6      270.0      244.8      270.0
Kimi-K3                   2,012     2,158     +7.3%      251.5      269.8      246.6      269.8
GLM-5.2                   2,003     2,139     +6.8%      250.4      267.4      245.5      267.4

Across 6 distinct tokenizers, code uses a median +9.8% tokens
relative to prose for the same character count (range +6.8% to +29.4%).
Unweighted mean: +14.8% — rows are per tokenizer, not per vendor; Claude's two
generations contribute two rows, so the mean leans toward vendors with more tokenizers.
```

Reading it:

- Every tokenizer charges more for code. Indentation, punctuation and `camelCase` fragment into more tokens than ordinary words.
- **Claude's tax is the outlier**: +29.4% versus GLM-5.2's +6.8%.
- In absolute terms the spread is wider still. On identical code, Claude ≥4.7 spends **432 tokens per 1,000 characters** against GLM-5.2's **267** — 62% more tokens for the same file, before any price difference.
- On prose the six tokenizers sit within ~1% of each other (except Claude). Tokenizer choice barely matters for prose; for code it matters a lot.

Characters and bytes are both reported because byte-level BPE operates on UTF-8 bytes. The prose fixture is 8,158 bytes despite being 8,000 characters, because Project Gutenberg uses curly quotes and em dashes. The `/1kB` columns remove that effect; the conclusion is unchanged.

### Fixtures

| File | Content | Licence |
| --- | --- | --- |
| `fixtures/prose.txt` | *Alice's Adventures in Wonderland*, Lewis Carroll — [Project Gutenberg #11](https://www.gutenberg.org/cache/epub/11/pg11.txt) | public domain |
| `fixtures/code.js` | this repository's own source | MIT |

`pnpm run fixtures` verifies both: the prose is rebuilt from Gutenberg and must match the committed file byte-for-byte, and `code.js` — a frozen snapshot of this repo's own source, taken when the numbers were measured — must match the sha256 pinned in `fixtures/PROVENANCE.md`. Any drift fails, so the corpus behind the numbers above cannot change silently; ordinary development on `src/` deliberately does not touch it. `pnpm run fixtures -- --update` re-freezes the snapshot from the current sources and prints which downstream artefacts (reference counts, README tables, Claude measurements) must be regenerated with it.

Compare any two inputs of your own:

```bash
node scripts/code-tax.mjs --a my-essay.md --b my-service.py --a-label ESSAY --b-label PYTHON
```

## Conclusion: a token is not a unit

Everything above in one table — the same 8,000 characters, counted by each
vendor's own tokenizer, compared against the o200k_base baseline:

| Tokenizer | Prose | Prose vs base | Code | Code vs base | Code vs prose |
| --- | ---: | ---: | ---: | ---: | ---: |
| Claude (>= 4.7) | 2,754 | +37.9% | 3,458 | +60.1% | +25.6% |
| Claude (<= 4.6) | 2,077 | +4.0% | 2,688 | +24.4% | +29.4% |
| DeepSeek-V4-Pro | 2,024 | +1.4% | 2,255 | +4.4% | +11.4% |
| o200k_base (base) | 1,997 | — | 2,160 | — | +8.2% |
| Kimi-K3 | 2,012 | +0.8% | 2,158 | -0.1% | +7.3% |
| GLM-5.2 | 2,003 | +0.3% | 2,139 | -1.0% | +6.8% |

*(Derived from the fixture counts above: open tokenizers from `test/reference.json`, Claude measured through Bedrock.)*

Three things this table settles:

- **The exchange rate between vendors' tokens is not a constant.** On prose,
  five of the six tokenizers agree within ~4%; on code the agreement collapses
  to a spread from -1.0% to +60.1%. The same file is 2,139 tokens or 3,458
  tokens — 62% apart — depending on who counts it.
- **There is no stable conversion factor.** Claude <= 4.6 costs +4.0% versus the
  baseline on prose but +24.4% on code; GLM-5.2 flips sign entirely. Any
  "model X uses n% more tokens" claim is a property of one input, not of the
  models.
- **Price per token therefore cannot rank models.** Opus 4.6 and Opus 5 both
  list at ~$5.01 per million input tokens — an identical sticker price — yet
  the same prose costs 33% more on Opus 5, because its tokenizer spends 33%
  more tokens on it. Sonnet 5's sticker is 33% below Sonnet 4.6's ($2 vs $3
  per MTok); the real saving on this text is 12%.

A token is a vendor-private unit whose size depends on the tokenizer and on the
content. $/MTok is a price quoted in that private currency, at an exchange rate
that floats with your workload — comparing models by it is comparing prices in
different currencies without converting. The comparison that does hold is
**cost per task**: the same input, each model's own tokenizer, its own price,
in dollars. That is the number the `IN COST` column computes, and the reason
this tool insists on exact vendor tokenizers rather than approximations.

## Install

Requires **Node 20+**. Python 3 is optional and only needed for `pnpm run verify`.

```bash
git clone https://github.com/<you>/tokencompare.git
cd tokencompare
pnpm install          # or npm install
node tokenizer.js "hello world"
```

Nothing else is required for OpenAI and open-weight models. Their tokenizers download on first use and cache in `.cache/`.

## Setting up Claude counts

Claude has no public tokenizer, so exact counts come from Amazon Bedrock. Pick **either** option.

### Option A — Bedrock API key (simplest)

Create a Bedrock API key in the AWS console under **Amazon Bedrock → API keys**, then:

```bash
export AWS_BEARER_TOKEN_BEDROCK="your-bedrock-api-key"
node tokenizer.js --file fixtures/prose.txt
```

No AWS CLI, no profile, no IAM policy authoring. This is the recommended path.

### Option B — IAM (profiles, SSO, instance roles)

Uses your existing AWS CLI configuration, so it also covers SSO, assumed roles and EC2/Lambda instance roles.

```bash
aws configure           # or aws sso login
node tokenizer.js --aws-profile myprofile --file fixtures/prose.txt
```

The IAM identity needs:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["bedrock:CountTokens", "bedrock-mantle:CountTokens"],
      "Resource": "*"
    }
  ]
}
```

`bedrock:CountTokens` covers Claude 4.6 and earlier. `bedrock-mantle:CountTokens` covers 4.7 and later. Scope `Resource` to specific model ARNs if you prefer; token counting itself is free of charge.

### If you skip both

Claude rows fall back to a calibrated estimate, clearly marked `~`, and the tool tells you why. Everything else stays exact. You can also force this with `--no-bedrock`.

## Usage

```bash
node tokenizer.js "some text"                 # inline
node tokenizer.js --file README.md            # a file
cat file.md | node tokenizer.js               # stdin

node tokenizer.js --group claude --file x.md  # one provider
node tokenizer.js --model opus-5 --model luna # specific models
node tokenizer.js --all --by-tokenizer        # the whole catalogue
node tokenizer.js --json --file x.md          # machine readable
node tokenizer.js --output 4000 --file x.md   # price 4,000 output tokens
```

Useful flags:

| Flag | Effect |
| --- | --- |
| `--all` | every model in the OpenRouter catalogue (~335) |
| `--by-tokenizer` | one row per distinct tokenizer instead of per model |
| `--only exact` | hide models whose counts are inferred or unavailable |
| `--baseline <substr>` | choose what `VS BASE` compares against |
| `--fetch` | download tokenizers that are not cached yet |
| `--no-bedrock` | skip Bedrock and estimate Claude instead |
| `--refresh` | refresh the cached pricing catalogue |

`HF_TOKEN` is needed for gated Hugging Face repos (Llama, Gemma, Cohere).

## How it works

1. **Catalogue.** Model list, live pricing and context windows come from OpenRouter's public `/api/v1/models` endpoint. No key required. OpenRouter has no token-counting endpoint, so it is used only for metadata.
2. **Tokenizer resolution.** Each model resolves to a tokenizer: a published tiktoken encoding, a Bedrock probe for Claude, or a vocabulary file from the vendor's Hugging Face repo.
3. **Deduplication by content hash.** Tokenizers are keyed by their Hugging Face blob id, fetched via the `paths-info` API without downloading the blob. Many repos ship byte-identical vocabularies (every Qwen3 checkpoint, every GLM-4.5 variant), which collapses ~146 repos to ~63 real tokenizers and halves the bytes fetched. Each vocabulary is stored once no matter how many models use it, and every downloaded blob is re-hashed against the recorded id, so a repo that republishes its tokenizer fails loudly instead of silently desyncing the dedup.
4. **Counting.** Byte-level BPE runs locally for open models; Claude is measured through Bedrock. Counts are computed once per *tokenizer*, not per model.
5. **Envelope correction.** Bedrock returns the token count of a whole request, including a fixed per-model envelope that ranges from 6 to 24 tokens. It is measured with a single-token probe and subtracted, leaving raw-text counts comparable with every other row.

## Limitations

These are real and worth stating plainly.

- **Bedrock `CountTokens` is Anthropic-only.** Non-Anthropic models served on Bedrock — DeepSeek, Qwen, Mistral, Llama, Nova, Cohere, Titan — all reject it with `The provided model doesn't support counting tokens`. Reproduce that yourself: `pnpm run audit:bedrock -- --sweep` probes CountTokens on every model id `list-foundation-models` returns (45 ids at the time of writing) and fails if any non-Anthropic model starts answering. Counting everything through one AWS API is not possible.
- **Claude counts carry ±1 token of noise.** BPE merges can span the boundary between the request envelope and the text, so envelope subtraction is occasionally off by one. Models in the same generation may report e.g. 2,753 versus 2,754.
- **Grok, Amazon Nova and Gemini cannot be counted exactly.** No tokenizer is published and no counting API is offered. They are reported as `n/a` rather than guessed at.
- **`--all` uses family stand-ins.** A model with no published tokenizer borrows a sibling's from the same OpenRouter tokenizer family. Marked `*` and inferred, not verified. The same honesty applies to Claude: an id measured through a generation probe rather than its own Bedrock id is marked `*`, and an id outside the two audited generations is reported `n/a` rather than guessed at. The curated default set is 100% exact.
- **Kimi's pre-tokenizer pattern is transcribed, not imported.** Moonshot publishes it as Python source (`tokenization_kimi.py`), so the pattern string is transcribed into both the JS tokenizer and the Python verifier. The two translations are independent, which catches conversion bugs, but a transcription error common to both would pass verification.
- **Vendors disagree about special-token literals.** Given input containing `<|endoftext|>`, HuggingFace emits one token while tiktoken treats it as plain text. Each backend follows its own vendor's library; the code fixture deliberately excludes such literals so the prose-versus-code result is not confounded.
- **Chat scaffolding is excluded.** These are raw text counts. Real requests add system prompts, tool definitions and role markers.
- **Pricing changes.** Prices come from a catalogue cached for 24 hours; `--refresh` updates it.

## Commands

| Command | Purpose |
| --- | --- |
| `pnpm run compare` | the comparison table |
| `pnpm test` | diff the JS tokenizers against cached reference counts |
| `pnpm run verify:setup` | create the Python reference environment |
| `pnpm run verify` | regenerate reference counts with the official libraries and diff |
| `pnpm run codetax` | prose versus code at identical character counts |
| `pnpm run fixtures` | verify the fixtures against their recorded provenance (append `-- --update` to re-freeze) |
| `pnpm run audit:bedrock` | measure every Claude model individually |

## Licence

MIT. Bundled prose fixture is public domain; see `fixtures/PROVENANCE.md`.
