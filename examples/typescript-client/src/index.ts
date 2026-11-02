// Run from the directory that holds jambu.yaml:
//   npm --prefix examples/typescript-client install
//   node --experimental-strip-types examples/typescript-client/src/index.ts "Say hi"

import OpenAI from "openai";
import { Gpu, type GpuConnection } from "./gpu.ts";

const gpu = new Gpu({ cwd: process.env.JAMBU_PROJECT_DIR ?? process.cwd() });

function client(conn: GpuConnection): OpenAI {
  // A short timeout: a stopped machine should fail fast and be restarted.
  return new OpenAI({ baseURL: conn.baseURL, apiKey: conn.apiKey, maxRetries: 0, timeout: 120_000 });
}

const prompt = process.argv[2] ?? "Say hello in one short sentence.";
console.error("> making sure the GPU is up (first call can take minutes)...");
const started = Date.now();
const conn = await gpu.connection();
console.error(`+ ${conn.model} at ${conn.baseURL} after ${Math.round((Date.now() - started) / 1000)}s`);

const reply = await gpu.call((c) =>
  client(c).chat.completions.create({
    model: c.model,
    messages: [{ role: "user", content: prompt }],
    max_tokens: 64,
  }),
);
console.log(reply.choices[0]?.message?.content);

// Nothing to clean up: the watchdog on the machine stops it once it has been
// idle for lifecycle.idle_timeout. Call stopGpu() to stop it sooner.
