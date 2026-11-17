"""Random search over pipeline parameters on the train split. Appends JSON lines to a log."""
import sys, json, time; sys.path.insert(0, '..')
import numpy as np, evalkit as ek
from nr.pipeline import enhance
log = sys.argv[1]; n = int(sys.argv[2]); seed = int(sys.argv[3])
rng = np.random.default_rng(seed)
space = lambda: dict(
    a=float(rng.uniform(0.96, 0.995)), gmin_db=float(rng.uniform(-40, -22)), xi_min_db=float(rng.uniform(-35, -15)),
    rho=float(rng.uniform(0.2, 0.8)), tsnr_mix=float(rng.uniform(0.2, 1.0)), sfrac=float(rng.uniform(0.05, 0.2)),
    spp_alpha=float(rng.uniform(0.7, 0.95)), xi_h1=float(rng.uniform(10, 20)), nscale=float(rng.uniform(0.8, 1.3)),
    floor_cap_db=float(rng.uniform(-50, -40)), hop=int(rng.choice([128, 256])), n_fft=int(rng.choice([768, 1024, 1280])),
    floor=True, nmode="geo")
for k in range(n):
    cfg = space(); t = time.time()
    r = ek.evaluate(enhance, 'train', **cfg)
    rec = dict(cfg=cfg, pesq=float(np.nanmean(r['pesq'])), snr=float(np.mean(r['snr'])), sec=time.time() - t)
    with open(log, 'a') as f: f.write(json.dumps(rec) + '\n')
    print(k, rec['pesq'], flush=True)
