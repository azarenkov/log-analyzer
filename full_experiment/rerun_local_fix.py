"""Rerun ONLY the neural local configurations whose source is BGL or Thunderbird (affected by the missing checkpoint restore
in the original loop). Trains the source model exactly as run_experiment.py does (same seeds, data, epochs) and evaluates it
on the source test partition BEFORE any adaptation. Writes out_fix/results_local_fix.csv.
Usage: python3 rerun_local_fix.py <cache_dir> <data_dir> <out_fix_dir>"""
import sys, time, os, numpy as np, torch, pandas as pd
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3], "0"]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_experiment.py")).read().split("# ------------------------------------------------------------------ main loop")[0]
exec(src)
rows = []
for seed in [0, 1, 2, 3, 4]:
    for model in ["bilstm", "transformer"]:
        for s in ["BGL", "Thunderbird"]:
            torch.manual_seed(seed); np.random.seed(seed); t0 = time.time()
            vocab = Vocab(s); net = NEURAL[model](len(vocab.i) + 2)
            fit(net, vocab, s, D[s]["tr"], (s, D[s]["va"]), EPOCHS, seed); ts = time.time() - t0
            te = D[s]["te"]; m = metrics(D[s]["y"][te], predict(net, vocab, s, te))
            rows.append(dict(model=model, setting="local", source=s, target=s, seed=seed, train_s=ts, n_adapt="", adapt_anom="", **m))
            p(f"[local-fix] {model:<11} {s:<11} seed={seed} F1={m['F1']:.3f} FPR={m['FPR']:.3f} train_s={ts:.1f}")
            pd.DataFrame(rows).to_csv(f"{OUT}/results_local_fix.csv", index=False)
p("done")
