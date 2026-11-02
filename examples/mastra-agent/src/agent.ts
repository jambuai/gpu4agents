// Points a normal Mastra Agent at the instance `gpu setup` just
// provisioned. gpu4agents implements no agent runtime of its own - `run`
// exports JAMBU_GPU_ENDPOINT / JAMBU_MODEL_ID / OPENAI_API_KEY, and Mastra's
// custom-URL model config (@mastra/core's model router) is all the glue
// needed to reach vLLM's OpenAI-compatible server from there.

import { Agent } from "@mastra/core/agent";
import { weatherTool } from "./tools/weather-tool.ts";

const endpoint = process.env.JAMBU_GPU_ENDPOINT;
const modelId = process.env.JAMBU_MODEL_ID;

if (!endpoint || !modelId) {
  throw new Error(
    "JAMBU_GPU_ENDPOINT / JAMBU_MODEL_ID are not set. Run this through " +
      "`gpu run` (from a project with `gpu setup` already done) " +
      "so it can inject them - see ../../README.md#agentic-flows-nothing-to-build.",
  );
}

export const weatherAgent = new Agent({
  id: "weather-agent",
  name: "Weather Agent",
  instructions:
    "You are a helpful weather assistant. Always call weatherTool to get " +
    "current data before answering a weather question.",
  model: {
    // Mastra's model router only splits on the FIRST "/" - "gpu4agents/" here
    // is just a routing label, so a HuggingFace id with its own slash
    // (e.g. "Qwen/Qwen2.5-7B-Instruct") still reaches vLLM intact as
    // modelId.
    id: `gpu4agents/${modelId}`,
    url: `${endpoint}/v1`,
    apiKey: process.env.OPENAI_API_KEY ?? "jambu-local",
    // "chat" (the default) hits vLLM's /v1/chat/completions and works on
    // every vLLM version. Switch to "responses" to use the OpenAI Responses
    // API instead - vLLM serves that too, including MCP tool routing when
    // config.yml sets `runtime.tool_server` (docs/CONFIG.md).
    api: "chat",
  },
  tools: { weatherTool },
});
