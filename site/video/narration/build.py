"""Lay the narration on a 120 BPM bar grid, then write a beat that follows the story.

usage: build.py <voice dir> <out dir>
Reads lines.json (id, scene, text) and <voice dir>/<id>.mp3 or .wav (eleven.py writes them).
Writes narration.wav, timing.json (scenes, lines, the GPU stop moment) and vsl-audio.wav.

Story and sound: 1 is 2028 (pad and a soft arp), 2-3 are the problem in 2026 (a heartbeat kick,
no arp), 4 is the turn (riser, then the full beat), 6 goes quiet when the GPU stops, 7 resolves.
"""
import json, os, subprocess, sys
import numpy as np, soundfile as sf
from scipy.signal import butter, lfilter, resample_poly

VO, OUT = sys.argv[1], sys.argv[2]
SR = 44100
BPM = 120.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
RAW = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lines.json")))
LINES = {l["id"]: l["text"] for l in RAW}
SCENES = [[l["id"] for l in RAW if l["scene"] == k] for k in sorted({l["scene"] for l in RAW})]
LEAD, GAP = BEAT, 0.32
# silence after the last line of each scene, before the bar rounds it up
TAIL = {0: 0.9, 1: 0.5, 2: 0.2, 3: 0.4, 4: 0.5, 5: 1.0, 6: 3.4}
PAIN, TURN, STOP, CTA_SCENE = (1, 2), 3, 5, 6   # 0-based scene indexes
STOP_LINE = "f1"                                 # "...the GPU stops."


def trim(x):
    idx = np.where(np.abs(x) > 0.01)[0]
    return x[max(0, idx[0] - 240): idx[-1] + 1200] if len(idx) else x


def load(k):
    wav = f"{VO}/{k}.wav"
    if not os.path.exists(wav):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", f"{VO}/{k}.mp3", "-ac", "1", "-ar", str(SR), wav], check=True)
    x, r = sf.read(wav)
    x = x.mean(axis=1) if x.ndim > 1 else x
    return resample_poly(trim(x), SR, r) if r != SR else trim(x)


clips = {k: load(k) for k in LINES}

t, plan, placed = 0.0, [], []
for i, ids in enumerate(SCENES):
    start = t
    cur = start + LEAD
    caps = []
    for k in ids:
        d = len(clips[k]) / SR
        caps.append({"id": k, "text": LINES[k], "start": round(cur, 3), "end": round(cur + d, 3)})
        placed.append((cur, clips[k]))
        cur += d + GAP
    end = np.ceil((cur - GAP + TAIL[i] - start) / BAR) * BAR + start
    plan.append({"scene": i + 1, "start": round(start, 3), "end": round(end, 3), "lines": caps})
    t = end
total = t
# the GPU stops on the beat nearest the end of "...the GPU stops."
sl = next(l for l in plan[STOP]["lines"] if l["id"] == STOP_LINE)
stop_at = plan[STOP]["start"] + np.round((sl["end"] - 0.25 - plan[STOP]["start"]) / BEAT) * BEAT

n = int(total * SR) + SR
vo = np.zeros(n)
for s, x in placed:
    i = int(s * SR)
    vo[i:i + len(x)] += x
vo = vo[: int(total * SR)]
n = len(vo)
sf.write(f"{OUT}/narration.wav", vo, SR)
json.dump({"total": round(total, 3), "bpm": BPM, "stop_at": round(float(stop_at), 3), "scenes": plan},
          open(f"{OUT}/timing.json", "w"), indent=1)

# ---------------------------------------------------------------- music
rng = np.random.default_rng(11)
tt = np.arange(n) / SR


def lp(x, fc, order=2):
    b, a = butter(order, fc / (SR / 2), "low")
    return lfilter(b, a, x)


def hp(x, fc, order=2):
    b, a = butter(order, fc / (SR / 2), "high")
    return lfilter(b, a, x)


def hz(m):
    return 440 * 2 ** ((m - 69) / 12)


def add(buf, at, x, gain=1.0):
    i = int(at * SR)
    if i >= len(buf) or i < 0:
        return
    j = min(len(buf), i + len(x))
    buf[i:j] += x[: j - i] * gain


kick_s = np.arange(int(0.42 * SR)) / SR
kick = np.sin(2 * np.pi * (48 * kick_s + (140 - 48) * 0.035 * (1 - np.exp(-kick_s / 0.035)))) * np.exp(-kick_s / 0.16)
kick += 0.4 * np.exp(-kick_s / 0.004) * rng.standard_normal(len(kick_s))  # click
clap_s = np.arange(int(0.25 * SR)) / SR
clap = hp(lp(rng.standard_normal(len(clap_s)), 3800), 900) * (np.exp(-clap_s / 0.07) + 0.6 * np.exp(-np.maximum(clap_s - 0.012, 0) / 0.02) * (clap_s > 0.012))
hat_s = np.arange(int(0.07 * SR)) / SR
hat = hp(rng.standard_normal(len(hat_s)), 7000) * np.exp(-hat_s / 0.018)
ohat_s = np.arange(int(0.25 * SR)) / SR
ohat = hp(rng.standard_normal(len(ohat_s)), 6500) * np.exp(-ohat_s / 0.09)

# i - VI - III - VII in A minor, one chord per bar; the CTA resolves to C major
PROG = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]
CTA = [[48, 52, 55], [53, 57, 60], [55, 59, 62], [48, 52, 55]]

drums, bass, arp, pad, fx = (np.zeros(n) for _ in range(5))
s1, s2, s6 = plan[0]["start"], plan[TURN]["start"], plan[CTA_SCENE]["start"]
pain = (plan[PAIN[0]]["start"], plan[PAIN[-1]]["end"])
bars = int(np.ceil(total / BAR))
for b in range(bars):
    t0 = b * BAR
    in_cta = t0 >= s6 - 1e-6
    chord = (CTA if in_cta else PROG)[(b - int(round(s6 / BAR)) if in_cta else b) % 4]
    stopped = stop_at - 1e-6 <= t0 < stop_at + BAR - 1e-6
    tail_bars = t0 >= total - 2 * BAR - 1e-6
    full = (t0 >= s2 - 1e-6) and not stopped and not tail_bars
    in_pain = pain[0] - 1e-6 <= t0 < pain[1] - 1e-6
    if in_pain:  # heartbeat: two kicks at the top of the bar, nothing else
        add(drums, t0, kick, 0.7)
        add(drums, t0 + 0.32, kick, 0.45)
    for q in range(4):
        tb = t0 + q * BEAT
        if full:
            add(drums, tb, kick, 0.95)
            if q in (1, 3):
                add(drums, tb, clap, 0.42)
        for e in range(2):
            th = tb + e * BEAT / 2
            if full and not stopped:
                add(drums, th, ohat if e == 1 and full else hat, 0.20 if e else 0.12)
        # bass: root on the off-beats, octave pop on beat 4
        if (full or in_cta) and not stopped and not tail_bars:
            for e in range(2):
                note = chord[0] - 24 + (12 if (q == 3 and e == 1) else 0)
                ln = int(BEAT / 2 * 0.9 * SR)
                st = np.arange(ln) / SR
                saw = sum(np.sin(2 * np.pi * hz(note) * k * st) / k for k in range(1, 7))
                add(bass, tb + e * BEAT / 2, saw * np.exp(-st / 0.18) * (0.55 if e else 0.8))
    # arp: 16ths through the chord, plucked
    if not stopped and not in_pain and t0 >= s1 + BAR - 1e-6 and not tail_bars:
        seq = [chord[0] + 12, chord[1] + 12, chord[2] + 12, chord[1] + 24]
        for s16 in range(16):
            note = seq[s16 % 4]
            ln = int(0.2 * SR)
            st = np.arange(ln) / SR
            pl = (np.sin(2 * np.pi * hz(note) * st) + 0.3 * np.sin(4 * np.pi * hz(note) * st)) * np.exp(-st / 0.07)
            soft = 0.5 if t0 < pain[0] else 1.0   # 2028 intro: a quieter arp
            add(arp, t0 + s16 * BEAT / 4, pl, (0.22 if s16 % 4 == 0 else 0.14) * soft)
    # pad, every bar
    st = np.arange(int(BAR * SR)) / SR
    env = np.minimum(1, st / 0.3) * np.minimum(1, (BAR - st) / 0.3)
    p = sum(np.sin(2 * np.pi * hz(m + det) * st + rng.uniform(0, 6.28)) for m in chord for det in (-0.08, 0, 0.07))
    add(pad, t0, p * env, 0.18 if stopped else (0.5 if in_pain else 0.35))

bass = lp(bass, 700)
arp = lp(arp, 4200)
pad = lp(pad, 1400)
# risers into scene 2 and into the CTA, an impact on each scene change, a "power down" at the stop
for at in (s2, s6):
    ln = int(BAR * SR)
    st = np.arange(ln) / SR
    noise = rng.standard_normal(ln)
    rise = np.zeros(ln)
    for k in range(8):  # stepped band to fake a filter sweep
        seg = slice(k * ln // 8, (k + 1) * ln // 8)
        rise[seg] = hp(lp(noise, 600 + 1200 * k), 300)[seg]
    add(fx, at - BAR, rise * (st / BAR) ** 2, 0.5)
for sc in plan[1:]:
    st = np.arange(int(0.6 * SR)) / SR
    boom = np.sin(2 * np.pi * 55 * st) * np.exp(-st / 0.25)
    add(fx, sc["start"], boom, 0.55)
st = np.arange(int(1.2 * SR)) / SR
f = 220 * np.exp(-st * 2.4)
add(fx, stop_at, np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-st / 0.5), 0.35)

# sidechain pump from the kick, and duck everything under the voice
pump = np.ones(n)
for b in range(bars):
    for q in range(4):
        tb = b * BAR + q * BEAT
        i = int(tb * SR)
        ln = int(BEAT * SR)
        st = np.arange(ln) / SR
        seg = 1 - 0.45 * np.exp(-st / 0.09)
        j = min(n, i + ln)
        if i < n:
            pump[i:j] = np.minimum(pump[i:j], seg[: j - i])
voice_env = lp(np.abs(vo), 8, order=1)
duck = 1 - 0.5 * np.clip(voice_env / 0.04, 0, 1)

music = drums * 0.9 + bass * 0.55 + (arp * 0.8 + pad * 0.16) * pump + fx
music /= np.max(np.abs(music)) + 1e-9
mix = vo / (np.max(np.abs(vo)) + 1e-9) * 0.9 + music * 0.30 * duck
fade = np.clip((total - tt) / 1.6, 0, 1)
mix *= fade
mix /= max(1.0, np.max(np.abs(mix)) / 0.95)
sf.write(f"{OUT}/vsl-audio.wav", mix, SR)
print(json.dumps({"total": round(total, 2), "stop_at": float(stop_at),
                  "scenes": [(p["scene"], p["start"], p["end"]) for p in plan]}))
