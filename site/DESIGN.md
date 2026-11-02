# jambu-gpu: site and VSL design

Brand: Jambu.ai (the leaf-and-circuit mark). The product page and the video share these values.

## Palette

| Token | Hex | Use |
| --- | --- | --- |
| ground | `#0A0F0B` | page and video background (green-tinted black) |
| ground-2 | `#111912` | panels, cards, terminal |
| line | `#22302A` | rules, borders |
| ink | `#ECF2E8` | primary text |
| muted | `#9CAA9D` | secondary text, labels |
| leaf | `#5CBF3A` | brand accent: running, ready, "it turns off" |
| lime | `#B4E06C` | second leaf tone, highlights |
| ember | `#FF7A4D` | the problem only: billing, idle, denied |

One accent (leaf). Ember appears only where money is being lost.

## Type

- Display and body: **Bricolage Grotesque** (300 / 800 contrast).
- Data, code, labels, YAML: **JetBrains Mono** (400 / 700).

## Rules

- Dark canvas everywhere, texture from radial glows, hairline grids and oversized ghost type.
- Code and config shown are real: `gpu_setup`, `gpu_offers`, `lifecycle.idle_timeout: 15m`, `budget.max_hourly_cost_usd`.
- No gradient text, no purple, no stock "AI" imagery.
