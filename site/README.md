# GPU for Agents: site and VSL

- `web/`: the static site (Cloudflare Pages project `gpu4agents`, served at gpu4agents.jambu.ai). No build step.
- `video/`: the 64-second VSL as a HyperFrames composition (`video/index.html`).
- `DESIGN.md`: palette and type shared by both.

## Video

```bash
cd video
npm run check                                   # lint, layout, contrast
npx hyperframes render --quality high --output renders/gpu4agents-vsl.mp4
```

Narration is Kokoro TTS (`am_michael`), one clip per line so scenes and captions sync to it:

```bash
# venv with kokoro-onnx, soundfile, numpy, phonemizer-fork; system espeak-ng
python narration/synth.py am_michael 1.0 <out>/vo < narration/lines.json > <out>/durs.json
python narration/build.py <workdir> assets      # -> assets/narration.wav, assets/timing.json
python narration/music.py assets                # pad, ticks, whooshes, ducking -> assets/vsl-audio.wav
```

`build.py` reads `<workdir>/tts/lines.json`, `durs.json` and `vo/*.wav`. When a line changes, update the
caption times in `index.html` (`CAPS`) from `assets/timing.json`.

## Site

```bash
cp video/renders/gpu4agents-vsl.mp4 web/media/   # after a render (web-optimized, see below)
npx wrangler pages deploy web --project-name gpu4agents --branch main   # run from site/: functions/ serves byte ranges for the video
```
