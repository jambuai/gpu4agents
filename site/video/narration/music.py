import json, numpy as np, soundfile as sf, sys
OUT = sys.argv[1]; sr = 24000
T = json.load(open(f"{OUT}/timing.json")); total = T["total"]; n = int(total*sr)
t = np.arange(n)/sr
rng = np.random.default_rng(7)
def note(f): return 440*2**((f-69)/12)
# chord per scene (midi notes)
chords = [[45,57,60,64,71],[41,57,60,65,69],[38,50,53,57],[48,55,60,64,67],[43,55,59,62,67],[48,55,60,64,67,72]]
pad = np.zeros(n)
for sc, ch in zip(T["scenes"], chords):
    a, b = sc["start"]-0.8, sc["end"]+0.8
    m = (t>=max(a,0))&(t<b); tt = t[m]
    env = np.clip((tt-a)/1.2,0,1)*np.clip((b-tt)/1.2,0,1)
    s = np.zeros(len(tt))
    for f in ch:
        for det in (-0.12, 0.0, 0.11):
            s += np.sin(2*np.pi*note(f+det)*tt + rng.uniform(0,6.28))*(0.6 if f<50 else 0.35)
    s *= (0.85+0.15*np.sin(2*np.pi*0.18*tt))
    pad[m] += s*env
# soften: one-pole lowpass
def lp(x, fc):
    a = np.exp(-2*np.pi*fc/sr); y = np.empty_like(x); acc = 0.0
    for i in range(len(x)): acc = (1-a)*x[i] + a*acc; y[i] = acc
    return y
pad = lp(pad, 900)
pad /= np.max(np.abs(pad))+1e-9
# clock ticks during the billing beat (scene 3)
s3 = T["scenes"][2]; tick = np.zeros(n)
k = s3["start"]+0.5
while k < s3["end"]-0.2:
    i = int(k*sr); L = int(0.03*sr)
    tick[i:i+L] += np.sin(2*np.pi*2200*np.arange(L)/sr)*np.exp(-np.arange(L)/(0.004*sr))
    k += 0.5
# whooshes at scene boundaries
wh = np.zeros(n)
for sc in T["scenes"][1:]:
    c = sc["start"]; a = int((c-0.5)*sr); L = int(0.8*sr)
    if a < 0: continue
    noise = rng.standard_normal(L); env = np.sin(np.linspace(0,np.pi,L))**2
    wh[a:a+L] += lp(noise, 1400)*env
wh /= np.max(np.abs(wh))+1e-9
vo, _ = sf.read(f"{OUT}/narration.wav")
vo = vo[:n] if len(vo)>=n else np.pad(vo,(0,n-len(vo)))
# duck music under voice
act = lp(np.abs(vo), 6); duck = 1 - 0.55*np.clip(act/0.05,0,1)
mix = vo/ (np.max(np.abs(vo))+1e-9)*0.85 + pad*0.16*duck + tick*0.10 + wh*0.10
fade = np.clip((total - t)/1.5, 0, 1); mix *= fade
mix /= max(1.0, np.max(np.abs(mix))/0.95)
sf.write(f"{OUT}/vsl-audio.wav", mix, sr)
print("ok", total)
