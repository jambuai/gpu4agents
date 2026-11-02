import json, sys, numpy as np, soundfile as sf
SP = sys.argv[1]; OUT = sys.argv[2]
lines = {l["id"]: l["text"] for l in json.load(open(f"{SP}/tts/lines.json"))}
durs = {d["id"]: d["dur"] for d in json.load(open(f"{SP}/tts/durs.json"))}
scenes = [["s1a","s1b","s1c"],["s2a","s2b","s2c"],["s3a","s3b"],["s4a","s4b","s4c","s4d"],["s5a"],["s6a","s6b","s6c"]]
LEAD, GAP, TAIL = 0.6, 0.3, 0.9      # voice starts LEAD into scene; GAP between lines; TAIL after last line
extra_tail = {2: 1.6, 5: 2.6}         # let the "billing" beat and the CTA breathe
t = 0.0; out = []; plan = []
sr = 24000; audio = []
def trim(x):
    a = np.abs(x); idx = np.where(a > 0.01)[0]
    return x[max(0, idx[0]-240): idx[-1]+2400] if len(idx) else x
clips = {}
for k in durs:
    x, r = sf.read(f"{SP}/tts/vo/{k}.wav"); assert r == sr; clips[k] = trim(x)
for i, sc in enumerate(scenes):
    start = t; cur = start + LEAD; caps = []
    for k in sc:
        d = len(clips[k]) / sr
        caps.append({"id": k, "text": lines[k], "start": round(cur, 3), "end": round(cur + d, 3)})
        out.append((cur, clips[k])); cur += d + GAP
    end = cur - GAP + TAIL + extra_tail.get(i, 0)
    plan.append({"scene": i + 1, "start": round(start, 3), "end": round(end, 3), "lines": caps})
    t = end
total = round(t, 2)
buf = np.zeros(int(total * sr) + sr)
for s, x in out:
    i = int(s * sr); buf[i:i+len(x)] += x
sf.write(f"{OUT}/narration.wav", buf[:int(total*sr)], sr)
json.dump({"total": total, "scenes": plan}, open(f"{OUT}/timing.json", "w"), indent=1)
print(total); [print(p["scene"], p["start"], p["end"]) for p in plan]
