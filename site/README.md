# GPU for Agents: site and VSL

- `web/`: the static site (Cloudflare Pages project `gpu4agents`, served at gpu4agents.jambu.ai). No build step.
  `web/docs/index.html` is the documentation at `/docs/`: architecture, process and data flow, the watchdog, and the contracts, with inline SVG diagrams and two charts built from a real session's event log.
- `video/`: the 116-second VSL as a HyperFrames composition (`video/index.html`).
- `DESIGN.md`: palette and type shared by both.

## Video

```bash
cd video
npm run check                                   # lint, layout, contrast
npx hyperframes render --quality high --output renders/gpu4agents-vsl.mp4
```

The story is in `video/narration/lines.json`: one entry per spoken line, with the scene it belongs to,
the caption (`text`) and what the voice says (`say`, with ElevenLabs v3 audio tags such as `[whispers]`
or `[excited]` for emotion). The voice is ElevenLabs "George" (`JBFqnCBsd6RMkjVDRZzb`), model `eleven_v3`:

```bash
set -a; . ../.env; set +a                        # ELEVENLABS_API_KEY
python video/narration/eleven.py JBFqnCBsd6RMkjVDRZzb <workdir>/vo < video/narration/lines.json
python video/narration/eleven.py JBFqnCBsd6RMkjVDRZzb <workdir>/vo d2 < video/narration/lines.json   # redo one line (delete its mp3 first)
cd video/narration && python build.py <workdir>/vo ../assets   # narration.wav, timing.json, vsl-audio.wav (needs numpy, scipy, soundfile)
```

`build.py` lays the lines on a 120 BPM grid and writes the music to follow the story: a soft intro (what the
agent can't do alone), a heartbeat under the manual way, the full beat once the doors open on GPU for Agents,
silence when the GPU stops.
After a line changes, paste `assets/timing.json` into `index.html` between `/*TIMING*/` and `/*END*/`.
The fox, the waveform, the transcript, the setup log and the event log on screen are real outputs from the
0.9.0 validation run.

## Site

```bash
cp video/renders/gpu4agents-vsl.mp4 web/media/   # after a render (web-optimized, see below)
npx wrangler pages deploy web --project-name gpu4agents --branch main   # run from site/: functions/ serves byte ranges for the video
```
