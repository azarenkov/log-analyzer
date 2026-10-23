"""Figure: F1 per direction for each model (local/zero-shot vs 5% adaptation). Usage: python3 fig.py results_summary.csv out.png"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
r = pd.read_csv(sys.argv[1])
dirs = [("HDFS", "HDFS"), ("BGL", "BGL"), ("HDFS", "BGL"), ("BGL", "HDFS")]
names = {"logreg": "Logistic regression", "bilstm": "Attention Bi-LSTM", "transformer": "Transformer"}
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
for ax, m in zip(axes, names):
    x = np.arange(len(dirs))
    for off, sets, col, lab in [(-0.18, ["local", "zero-shot"], "#4C78A8", "local / zero-shot"), (0.18, ["adapt5"], "#F58518", "5% adaptation")]:
        vals, errs = [], []
        for s, t in dirs:
            g = r[(r.model == m) & (r.source == s) & (r.target == t) & r.setting.isin(sets)]
            vals.append(g.F1_mean.iloc[0] if len(g) else 0); errs.append(g.F1_sd.iloc[0] if len(g) and pd.notna(g.F1_sd.iloc[0]) else 0)
        ax.bar(x + off, vals, 0.36, yerr=errs, capsize=2, color=col, label=lab)
    ax.set_xticks(x, ["HDFS\nlocal", "BGL\nlocal", "HDFS→\nBGL", "BGL→\nHDFS"], fontsize=8)
    ax.set_title(names[m], fontsize=9); ax.set_ylim(0, 1); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("F1 on target test"); axes[0].legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=200)
