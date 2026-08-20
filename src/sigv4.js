/**
 * Minimal AWS SigV4 request signing.
 *
 * Needed because the `bedrock-mantle` endpoint that serves token counts for
 * Claude 4.7+ is not exposed by any AWS SDK method, so the request has to be
 * signed by hand. Only static request paths are supported, which avoids the
 * canonical-URI encoding pitfalls that arise when a path segment contains
 * characters like `:`.
 */
import { createHash, createHmac } from "node:crypto";

const ALGORITHM = "AWS4-HMAC-SHA256";

const sha256Hex = (data) => createHash("sha256").update(data).digest("hex");
const hmac = (key, data) => createHmac("sha256", key).update(data).digest();

/**
 * @param {object} params
 * @param {string} params.method
 * @param {string} params.url        full URL; its path must need no escaping
 * @param {string} params.body
 * @param {string} params.service    SigV4 service name, e.g. "bedrock-mantle"
 * @param {string} params.region
 * @param {{accessKeyId: string, secretAccessKey: string, sessionToken?: string}} params.credentials
 * @param {Record<string, string>} [params.headers] additional headers to sign
 * @returns {Record<string, string>} headers including Authorization
 */
export function signRequest({ method, url, body, service, region, credentials, headers = {} }) {
  const parsed = new URL(url);
  if (parsed.pathname !== encodeURI(parsed.pathname)) {
    throw new Error("signRequest only supports paths that require no escaping");
  }

  const now = new Date();
  const amzDate = now.toISOString().replace(/[:-]|\.\d{3}/g, ""); // YYYYMMDDTHHMMSSZ
  const dateStamp = amzDate.slice(0, 8);

  const allHeaders = {
    ...headers,
    host: parsed.host,
    "x-amz-date": amzDate,
  };
  if (credentials.sessionToken) allHeaders["x-amz-security-token"] = credentials.sessionToken;

  // Canonical headers: lowercase names, trimmed values, sorted by name.
  const normalised = Object.entries(allHeaders)
    .map(([k, v]) => [k.toLowerCase(), String(v).trim().replace(/\s+/g, " ")])
    .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));

  const canonicalHeaders = normalised.map(([k, v]) => `${k}:${v}\n`).join("");
  const signedHeaders = normalised.map(([k]) => k).join(";");

  const canonicalQuery = [...parsed.searchParams.entries()]
    .map(([k, v]) => [encodeURIComponent(k), encodeURIComponent(v)])
    .sort((a, b) => (a[0] < b[0] ? -1 : 1))
    .map(([k, v]) => `${k}=${v}`)
    .join("&");

  const canonicalRequest = [
    method,
    parsed.pathname,
    canonicalQuery,
    canonicalHeaders,
    signedHeaders,
    sha256Hex(body),
  ].join("\n");

  const scope = `${dateStamp}/${region}/${service}/aws4_request`;
  const stringToSign = [ALGORITHM, amzDate, scope, sha256Hex(canonicalRequest)].join("\n");

  const signingKey = ["aws4_request"].reduce(
    (key, part) => hmac(key, part),
    [region, service].reduce(
      (key, part) => hmac(key, part),
      hmac("AWS4" + credentials.secretAccessKey, dateStamp),
    ),
  );
  const signature = createHmac("sha256", signingKey).update(stringToSign).digest("hex");

  return {
    ...allHeaders,
    Authorization:
      `${ALGORITHM} Credential=${credentials.accessKeyId}/${scope}, ` +
      `SignedHeaders=${signedHeaders}, Signature=${signature}`,
  };
}
