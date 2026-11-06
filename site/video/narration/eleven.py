"""ElevenLabs v3 narration, one clip per line, so scenes and captions sync to it.

usage: eleven.py VOICE_ID OUT_DIR [ids...] < lines.json   (ELEVENLABS_API_KEY in the environment)
Each line's "say" (text with v3 audio tags such as [whispers]) is spoken; "text" is the caption.
Writes OUT_DIR/<id>.mp3 and prints {id: seconds}. Lines already rendered are skipped.
"""
import json, os, subprocess, sys, urllib.request

voice, out = sys.argv[1], sys.argv[2]
only = set(sys.argv[3:])
os.makedirs(out, exist_ok=True)
key = os.environ["ELEVENLABS_API_KEY"]
durs = {}
for line in json.load(sys.stdin):
    if only and line["id"] not in only:
        continue
    path = os.path.join(out, line["id"] + ".mp3")
    if not os.path.exists(path):
        body = {"text": line.get("say") or line["text"], "model_id": "eleven_v3",
                "voice_settings": {"stability": line.get("stability", 0.5)}}
        req = urllib.request.Request(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=mp3_44100_128",
            data=json.dumps(body).encode(), headers={"xi-api-key": key, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
            f.write(r.read())
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                           capture_output=True, text=True)
    durs[line["id"]] = round(float(probe.stdout), 3)
print(json.dumps(durs))
