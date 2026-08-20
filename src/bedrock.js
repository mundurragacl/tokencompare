/**
 * Exact Claude token counts via Amazon Bedrock.
 *
 * Two endpoints are involved, because which one serves a model depends on how
 * that model is offered:
 *
 *   bedrock-runtime  CountTokens works for Claude 4.6 and earlier.
 *   bedrock-mantle   Claude 4.7 and later are cross-Region-inference only, so
 *                    CountTokens on bedrock-runtime rejects them outright. AWS
 *                    documents Anthropic's count_tokens API at
 *                    /anthropic/v1/messages/count_tokens on this endpoint.
 *
 * Two authentication modes are supported:
 *
 *   API key  A Bedrock API key in AWS_BEARER_TOKEN_BEDROCK. Simplest to set up
 *            and needs no local AWS config, so it is tried first.
 *   IAM      SigV4 signing with credentials from the AWS CLI, which also covers
 *            instance roles, SSO and assumed roles.
 *
 * Both endpoints report the token count of a whole request, which includes a
 * fixed per-model envelope. That envelope differs between models and endpoints,
 * so it is measured with a single-token probe and subtracted, leaving raw text
 * counts that are comparable with every other provider in the report.
 */
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { signRequest } from "./sigv4.js";

const execFileAsync = promisify(execFile);

/** Probe whose Claude token count is 1, used to measure envelope overhead. */
const PROBE_TEXT = "a";
const PROBE_TOKENS = 1;

const ANTHROPIC_VERSION = "2023-06-01";

/** Read a Bedrock API key from the environment, if the user configured one. */
export function bedrockApiKey() {
  return process.env.AWS_BEARER_TOKEN_BEDROCK || null;
}

/**
 * Describe how Bedrock will authenticate, for reporting.
 * @returns {"api-key"|"iam"|"none"}
 */
export function authMode({ apiKey, credentials } = {}) {
  if (apiKey) return "api-key";
  if (credentials) return "iam";
  return "none";
}

/** Resolve IAM credentials for a profile through the AWS CLI. */
export async function resolveCredentials(profile) {
  const args = ["configure", "export-credentials", "--format", "process"];
  if (profile) args.push("--profile", profile);
  const { stdout } = await execFileAsync("aws", args, { maxBuffer: 1 << 20 });
  const json = JSON.parse(stdout);
  return {
    accessKeyId: json.AccessKeyId,
    secretAccessKey: json.SecretAccessKey,
    sessionToken: json.SessionToken,
  };
}

/** Extract a useful message from either error envelope Bedrock may return. */
async function describeError(res) {
  const raw = await res.text();
  try {
    const j = JSON.parse(raw);
    // bedrock-mantle uses Anthropic's shape; bedrock-runtime uses AWS's.
    if (j.error?.message) return `${j.error.type || "error"}: ${j.error.message}`;
    if (j.message) return j.message;
  } catch {}
  return raw.slice(0, 300);
}

/**
 * CountTokens on bedrock-runtime.
 *
 * With an API key this is a plain authenticated POST. With IAM we delegate to the
 * AWS CLI rather than signing here, because the model id sits in the request path
 * and contains characters (`:`) whose canonical-URI encoding is easy to get wrong.
 */
async function countViaRuntime(text, { modelId, region, profile, apiKey }) {
  const payload = { converse: { messages: [{ role: "user", content: [{ text }] }] } };

  if (apiKey) {
    const url = `https://bedrock-runtime.${region}.amazonaws.com/model/${encodeURIComponent(modelId)}/count-tokens`;
    const res = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${apiKey}` },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`bedrock-runtime HTTP ${res.status}: ${await describeError(res)}`);
    const json = await res.json();
    if (typeof json.inputTokens !== "number") throw new Error("no inputTokens in response");
    return json.inputTokens;
  }

  const args = [
    "bedrock-runtime", "count-tokens",
    "--model-id", modelId,
    "--input", JSON.stringify(payload),
    "--region", region,
    "--output", "json",
    "--cli-connect-timeout", "10",
    "--cli-read-timeout", "120",
  ];
  if (profile) args.push("--profile", profile);

  const { stdout } = await execFileAsync("aws", args, {
    maxBuffer: 1 << 24,
    env: { ...process.env, AWS_PAGER: "" },
  });
  const tokens = JSON.parse(stdout).inputTokens;
  if (typeof tokens !== "number") throw new Error("count-tokens returned no inputTokens");
  return tokens;
}

/** Anthropic count_tokens on bedrock-mantle, via API key or SigV4. */
async function countViaMantle(text, { modelId, region, credentials, apiKey }) {
  const url = `https://bedrock-mantle.${region}.api.aws/anthropic/v1/messages/count_tokens`;
  const body = JSON.stringify({ model: modelId, messages: [{ role: "user", content: text }] });
  const base = { "content-type": "application/json", "anthropic-version": ANTHROPIC_VERSION };

  let headers;
  if (apiKey) {
    headers = { ...base, "x-api-key": apiKey };
  } else if (credentials) {
    headers = signRequest({
      method: "POST", url, body, service: "bedrock-mantle", region, credentials, headers: base,
    });
  } else {
    throw new Error("bedrock-mantle needs either AWS_BEARER_TOKEN_BEDROCK or IAM credentials");
  }

  const res = await fetch(url, { method: "POST", headers, body });
  if (!res.ok) throw new Error(`bedrock-mantle HTTP ${res.status}: ${await describeError(res)}`);
  const json = await res.json();
  if (typeof json.input_tokens !== "number") throw new Error("no input_tokens in response");
  return json.input_tokens;
}

/**
 * Exact raw-text token count for one Claude model.
 *
 * @param {string} text
 * @param {{modelId: string, endpoint: "runtime"|"mantle", region: string}} target
 * @param {{profile?: string, credentials?: object, apiKey?: string}} ctx
 * @returns {Promise<{tokens: number, overhead: number, auth: string, method: string}>}
 */
export async function countClaudeExact(text, target, ctx = {}) {
  const { endpoint, modelId, region } = target;
  const apiKey = ctx.apiKey ?? bedrockApiKey();

  const call = endpoint === "mantle"
    ? (t) => countViaMantle(t, { modelId, region, credentials: ctx.credentials, apiKey })
    : (t) => countViaRuntime(t, { modelId, region, profile: ctx.profile, apiKey });

  // Two calls: the text itself, plus a probe to remove the request envelope.
  const [full, probe] = await Promise.all([call(text), call(PROBE_TEXT)]);
  const overhead = probe - PROBE_TOKENS;
  const auth = apiKey ? "api-key" : "iam";

  return {
    tokens: full - overhead,
    overhead,
    auth,
    method: `Bedrock ${endpoint} CountTokens (${modelId}, ${region}, ${auth})`,
  };
}
