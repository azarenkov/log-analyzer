"""Step 3: aggregate results over seeds, test H1/H2, draw figures.
H1: zero-shot F1 (source->target) < local F1 of the target           (one-sided Wilcoxon signed-rank, alternative="less")
H2: adapt5 F1 (source->target)   > zero-shot F1 (source->target)       (alternative="greater")
Paired units: 3 models x 6 directed pairs = 18 seed-averaged F1 values. Bootstrap 95% CI of the mean paired difference.
Usage: python3 analyze_results.py <out_dir>
"""
import sys, numpy as np, pandas as pd, matplotlib
from scipy.stats import wilcoxon
matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = sys.argv[1]
r = pd.read_csv(f"{OUT}/results_runs.csv")
MODELS = {"logreg": "Logistic regression", "bilstm": "Attention Bi-LSTM", "transformer": "Transformer"}
DS = ["HDFS", "BGL", "Thunderbird"]
lines = []
def say(s=""): print(s); lines.append(s)

seeds = sorted(r.seed.unique()); say(f"runs={len(r)} seeds={seeds}")
agg = (r.groupby(["model", "setting", "source", "target"])
         .agg(runs=("F1", "size"), F1_mean=("F1", "mean"), F1_sd=("F1", "std"), P_mean=("P", "mean"), R_mean=("R", "mean"),
              FPR_mean=("FPR", "mean"), FPR_sd=("FPR", "std"), train_s=("train_s", "mean")).reset_index())
agg.to_csv(f"{OUT}/results_summary.csv", index=False)

# ---- paired comparisons
def m(model, setting, s, t):
    return agg[(agg.model == model) & (agg.setting == setting) & (agg.source == s) & (agg.target == t)].F1_mean.iloc[0]
pairs = []
for model in MODELS:
    for s in DS:
        for t in DS:
            if s == t: continue
            pairs.append(dict(model=model, source=s, target=t, local_target=m(model, "local", t, t),
                              zero_shot=m(model, "zero-shot", s, t), adapt5=m(model, "adapt5", s, t)))
pairs = pd.DataFrame(pairs)
pairs["d_H1"] = pairs.zero_shot - pairs.local_target
pairs["d_H2"] = pairs.adapt5 - pairs.zero_shot
pairs.to_csv(f"{OUT}/paired_differences.csv", index=False)

rng = np.random.default_rng(0)
def boot_ci(d, n=10000):
    b = rng.choice(d, (n, len(d)), replace=True).mean(1); return np.percentile(b, [2.5, 97.5])
def rank_biserial(d):
    d = d[d != 0]; rk = pd.Series(np.abs(d)).rank().values
    return (rk[d > 0].sum() - rk[d < 0].sum()) / rk.sum()
tests = []
for h, col, alt in [("H1", "d_H1", "less"), ("H2", "d_H2", "greater")]:
    d = pairs[col].values
    w = wilcoxon(d, alternative=alt)
    lo, hi = boot_ci(d)
    row = dict(hypothesis=h, comparison="zero-shot - local" if h == "H1" else "adapt5 - zero-shot", n_pairs=len(d),
               mean_diff=d.mean(), median_diff=np.median(d), ci95_low=lo, ci95_high=hi, W=w.statistic, p_value=w.pvalue,
               rank_biserial=rank_biserial(d), pairs_in_predicted_direction=int((d < 0).sum() if h == "H1" else (d > 0).sum()),
               decision="reject H0 (supported)" if w.pvalue < 0.05 else "fail to reject H0")
    tests.append(row)
    say(f"{h}: mean diff={d.mean():+.3f} (95% CI {lo:+.3f}..{hi:+.3f}), median={np.median(d):+.3f}, W={w.statistic:.1f}, "
        f"p={w.pvalue:.2e}, r_rb={row['rank_biserial']:+.2f}, {row['pairs_in_predicted_direction']}/18 in predicted direction -> {row['decision']}")
    for model in MODELS:  # per-model sensitivity (n=6, descriptive)
        dm = pairs[pairs.model == model][col].values
        pm = wilcoxon(dm, alternative=alt).pvalue
        say(f"   {model:<11} mean diff={dm.mean():+.3f}, {int((dm<0).sum() if h=='H1' else (dm>0).sum())}/6 in predicted direction, exact p={pm:.3f}")
pd.DataFrame(tests).to_csv(f"{OUT}/hypothesis_tests.csv", index=False)

# ---- model ranks within each setting (rank 1 = best F1), averaged over targets/directions
rk = []
for setting in ["local", "zero-shot", "adapt5"]:
    sub = agg[agg.setting == setting].copy()
    sub["rank"] = sub.groupby(["source", "target"]).F1_mean.rank(ascending=False)
    for model in MODELS:
        x = sub[sub.model == model]
        rk.append(dict(setting=setting, model=model, mean_rank=x["rank"].mean(), mean_F1=x.F1_mean.mean(),
                       mean_FPR=x.FPR_mean.mean(), mean_train_s=x.train_s.mean()))
rk = pd.DataFrame(rk); rk.to_csv(f"{OUT}/model_ranks.csv", index=False)
say(); say(rk.round(3).to_string(index=False))

# ---- degenerate predictions (all sessions flagged anomalous / none flagged)
deg = r.assign(all_pos=(r.TN == 0) & (r.FN == 0), all_neg=(r.TP == 0) & (r.FP == 0)).groupby("setting")[["all_pos", "all_neg"]].sum()
say(); say("degenerate runs (all positive / all negative):"); say(deg.to_string())
open(f"{OUT}/analysis_summary.txt", "w").write("\n".join(lines))

# ---- Figure 1: F1 per direction and setting, one panel per model
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
dirs = [(s, t) for s in DS for t in DS if s != t]
lab = {"HDFS": "HDFS", "BGL": "BGL", "Thunderbird": "TB"}
for ax, model in zip(axes, MODELS):
    x = np.arange(len(dirs)); w = 0.27
    for k, (setting, col, name) in enumerate([("local", "#4C78A8", "Local (target)"), ("zero-shot", "#E45756", "Zero-shot"), ("adapt5", "#F58518", "5% adaptation")]):
        vals, errs = [], []
        for s, t in dirs:
            a = agg[(agg.model == model) & (agg.setting == setting) & (agg.source == (t if setting == "local" else s)) & (agg.target == t)].iloc[0]
            vals.append(a.F1_mean); errs.append(0 if np.isnan(a.F1_sd) else a.F1_sd)
        ax.bar(x + (k - 1) * w, vals, w, yerr=errs, capsize=1.5, color=col, label=name, error_kw=dict(lw=0.7))
    ax.set_xticks(x, [f"{lab[s]}→\n{lab[t]}" for s, t in dirs], fontsize=8)
    ax.set_title(MODELS[model], fontsize=10); ax.set_ylim(0, 1.05); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("F1 on target test partition")
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=3, fontsize=9, frameon=False)
plt.tight_layout(rect=(0, 0, 1, 0.92)); plt.savefig(f"{OUT}/fig1_f1_by_direction.png", dpi=220); plt.close()

# ---- Figure 2: paired differences for H1 and H2
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
mk = {"logreg": "o", "bilstm": "s", "transformer": "^"}
for ax, (h, col, title) in zip(axes, [("H1", "d_H1", "H1: zero-shot − local F1"), ("H2", "d_H2", "H2: 5% adaptation − zero-shot F1")]):
    for i, model in enumerate(MODELS):
        d = pairs[pairs.model == model][col].values
        ax.scatter(np.full(len(d), i) + np.linspace(-0.15, 0.15, len(d)), d, marker=mk[model], s=28, color="#4C78A8")
    ax.axhline(0, color="grey", lw=0.8); ax.set_xticks(range(3), [MODELS[k] for k in MODELS], fontsize=8)
    t = [x for x in tests if x["hypothesis"] == h][0]
    ax.set_title(f"{title}\nmean {t['mean_diff']:+.2f}, Wilcoxon p = {t['p_value']:.1e}", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Paired F1 difference (seed mean)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_paired_differences.png", dpi=220); plt.close()

# ---- Figure 3: false-positive rate by setting
fig, ax = plt.subplots(figsize=(6, 3.2))
for k, setting in enumerate(["local", "zero-shot", "adapt5"]):
    v = r[r.setting == setting].groupby(["model", "source", "target"]).FPR.mean().values
    ax.boxplot(v, positions=[k], widths=0.5, showfliers=True)
ax.set_xticks(range(3), ["Local", "Zero-shot", "5% adaptation"]); ax.set_ylabel("False-positive rate"); ax.set_ylim(-0.02, 1.02)
ax.spines[["top", "right"]].set_visible(False); plt.tight_layout(); plt.savefig(f"{OUT}/fig3_fpr_by_setting.png", dpi=220); plt.close()
print("figures written")
