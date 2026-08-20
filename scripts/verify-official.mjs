/**
 * Prove the fast JavaScript tokenizers agree with the vendors' own libraries.
 *
 * This is the answer to "how do I know your package counts correctly?". It
 * recounts every corpus case and both fixtures with the official reference
 * implementations and diffs them against the JavaScript path:
 *
 *   HuggingFace `tokenizers`  for GLM, DeepSeek and any tokenizer.json model
 *   OpenAI `tiktoken`         for the o200k_base encoding and Kimi's tiktoken.model
 *
 * Any disagreement is a bug here and fails the run.
 *
 * Requires the Python reference environment: `pnpm run verify:setup`.
 */
import { execFile } from "node:child_process";
import { existsSync } from "node:fs";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const ROOT = new URL("..", import.meta.url).pathname;
const VENV = ROOT + ".venv/bin/python";

if (!existsSync(VENV)) {
  console.error("Python reference environment missing. Run:\n\n  pnpm run verify:setup\n");
  process.exitCode = 1;
} else {
  console.log("1/2  generating counts with the official reference libraries");
  const ref = await execFileAsync(VENV, [ROOT + "scripts/reference.py"], { maxBuffer: 1 << 26 });
  process.stdout.write(ref.stdout.split("\n").filter((l) => l.includes("total=") || l.includes("wrote")).join("\n") + "\n");

  console.log("\n2/2  diffing the JavaScript implementation against them");
  try {
    const out = await execFileAsync(process.execPath, [
      "--max-old-space-size=4096", ROOT + "scripts/validate.mjs",
    ], { maxBuffer: 1 << 26 });
    process.stdout.write(out.stdout);
    console.log("\nThe JavaScript tokenizers match the vendors' own libraries exactly.");
  } catch (err) {
    process.stdout.write(err.stdout || "");
    process.stderr.write(err.stderr || "");
    console.error("\nMISMATCH: the JavaScript implementation disagrees with the reference.");
    process.exitCode = 1;
  }
}
