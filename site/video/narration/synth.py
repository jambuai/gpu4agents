import sys, json, os
import espeakng_loader, kokoro_onnx, soundfile as sf
from kokoro_onnx.config import EspeakConfig
C = os.path.expanduser("~/.cache/hyperframes/tts")
m = kokoro_onnx.Kokoro(f"{C}/models/kokoro-v1.0.onnx", f"{C}/voices/voices-v1.0.bin",
    espeak_config=EspeakConfig(lib_path="/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1", data_path="/usr/lib/x86_64-linux-gnu/espeak-ng-data"))
# args: voice speed outdir  ; stdin: json list of {id,text}
voice, speed, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
os.makedirs(out, exist_ok=True)
res = []
for item in json.load(sys.stdin):
    s, r = m.create(item["text"], voice=voice, speed=speed, lang="en-us")
    p = os.path.join(out, item["id"] + ".wav"); sf.write(p, s, r)
    res.append({"id": item["id"], "dur": round(len(s)/r, 3)})
print(json.dumps(res))
