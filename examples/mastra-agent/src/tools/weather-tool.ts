// Straight from Mastra's own docs on defining tools:
// https://github.com/mastra-ai/mastra/blob/main/docs/src/content/en/docs/agents/tools.mdx
// Nothing gpu4agents-specific here - this is a normal Mastra tool.

import { createTool } from "@mastra/core/tools";
import { z } from "zod";

export const weatherTool = createTool({
  id: "weather-tool",
  description: "Fetches the current weather for a city.",
  inputSchema: z.object({
    city: z.string().describe("City name, e.g. 'Lisbon'"),
  }),
  outputSchema: z.object({
    city: z.string(),
    temperatureCelsius: z.number(),
    conditions: z.string(),
  }),
  execute: async ({ city }, { abortSignal }) => {
    const response = await fetch(`https://wttr.in/${encodeURIComponent(city)}?format=j1`, {
      signal: abortSignal,
    });
    if (!response.ok) {
      throw new Error(`weather lookup failed for ${city}: HTTP ${response.status}`);
    }
    const data = (await response.json()) as {
      current_condition: [{ temp_C: string; weatherDesc: [{ value: string }] }];
    };
    const current = data.current_condition[0];
    return {
      city,
      temperatureCelsius: Number(current.temp_C),
      conditions: current.weatherDesc[0].value,
    };
  },
});
