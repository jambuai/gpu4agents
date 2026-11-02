"""Rebuild narration keeping scenes 1-5 at their old times (the animation is keyed to them)
and laying out scene 6 fresh. Usage: build_pinned.py <workdir> <assets> <old timing.json>"""
import json, sys, numpy as np, soundfile as sf
SP, OUT, OLD = sys.argv[1:4]
sr = 24000
lines = {l["id"]: l["text"] for l in json.load(open(f"{SP}/tts/lines.json"))}
old = json.load(open(OLD))
def trim(x):
    idx = np.where(np.abs(x) > 0.01)[0]
    return x[max(0, idx[0]-240): idx[-1]+2400] if len(idx) else x
clips = {k: trim(sf.read(f"{SP}/tts/vo/{k}.wav")[0]) for k in lines}
plan = []
for sc in old["scenes"][:5]:
    caps = []
    for l in sc["lines"]:
        d = len(clips[l["id"]]) / sr
        caps.append({"id": l["id"], "text": lines[l["id"]], "start": l["start"], "end": round(l["start"] + d, 3)})
    plan.append({"scene": sc["scene"], "start": sc["start"], "end": sc["end"], "lines": caps})
s6 = old["scenes"][5]; cur = s6["start"] + 0.6; caps = []
for k in ["s6a", "s6b", "s6c", "s6d"]:
    d = len(clips[k]) / sr
    caps.append({"id": k, "text": lines[k], "start": round(cur, 3), "end": round(cur + d, 3)})
    cur += d + (0.55 if k in ("s6a", "s6b") else 0.35)
end = round(cur + 2.4, 3)
plan.append({"scene": 6, "start": s6["start"], "end": end, "lines": caps})
for a, b in zip([l for s in plan for l in s["lines"]][:-1], [l for s in plan for l in s["lines"]][1:]):
    assert a["end"] <= b["start"], (a, b)
total = end
buf = np.zeros(int(total * sr) + sr)
for s in plan:
    for l in s["lines"]:
        x = clips[l["id"]]; i = int(l["start"] * sr); buf[i:i+len(x)] += x
sf.write(f"{OUT}/narration.wav", buf[:int(total*sr)], sr)
json.dump({"total": total, "scenes": plan}, open(f"{OUT}/timing.json", "w"), indent=1)
print(total)
for s in plan[3:]:
    for l in s["lines"]: print(l["start"], l["end"], l["text"])
