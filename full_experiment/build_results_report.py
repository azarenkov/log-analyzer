"""Step 4: build the results report (.docx) from the CSV outputs. Usage: python3 build_results_report.py <out_dir> <report.docx>"""
import sys, pandas as pd, numpy as np
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT, DOCX = sys.argv[1], sys.argv[2]
ds = pd.read_csv(f"{OUT}/dataset_summary.csv"); agg = pd.read_csv(f"{OUT}/results_summary.csv")
tests = pd.read_csv(f"{OUT}/hypothesis_tests.csv").set_index("hypothesis"); ranks = pd.read_csv(f"{OUT}/model_ranks.csv")
pairs = pd.read_csv(f"{OUT}/paired_differences.csv"); runs = pd.read_csv(f"{OUT}/results_runs.csv")
NS = runs.seed.nunique()
MN = {"logreg": "LogReg", "bilstm": "Bi-LSTM", "transformer": "Transformer"}
SN = {"local": "Local", "zero-shot": "Zero-shot", "adapt5": "5% adapt."}
DSL = ["HDFS", "BGL", "Thunderbird"]

d = Document(); st = d.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(12)
for s in d.sections: s.left_margin = Cm(2.5); s.right_margin = Cm(2); s.top_margin = s.bottom_margin = Cm(2)
def P(t, b=False, i=False):
    p = d.add_paragraph(); r = p.add_run(t); r.bold = b; r.italic = i; p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(6); return p
def H(t, l=1): d.add_heading(t, l)
def B(t): d.add_paragraph(t, style="List Bullet").alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
def T(rows, hdr, fs=8.5, cap=None):
    if cap: P(cap, b=True)
    t = d.add_table(rows=1, cols=len(hdr)); t.style = "Table Grid"
    for k, h in enumerate(hdr):
        c = t.rows[0].cells[k]; c.text = ""; r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(fs)
    for row in rows:
        cs = t.add_row().cells
        for k, v in enumerate(row):
            cs[k].text = ""; r = cs[k].paragraphs[0].add_run(str(v)); r.font.size = Pt(fs)
    d.add_paragraph()
def F(path, cap, w=16):
    d.add_picture(path, width=Cm(w)); P(cap, i=True)
g = lambda m, s, a, b: agg[(agg.model == m) & (agg.setting == s) & (agg.source == a) & (agg.target == b)].iloc[0]
f2 = lambda x: f"{x:.3f}"
pm = lambda r: f"{r.F1_mean:.3f} ± {0 if np.isnan(r.F1_sd) else r.F1_sd:.3f}"
h1, h2 = tests.loc["H1"], tests.loc["H2"]

tp = d.add_paragraph(); tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = tp.add_run("Cross-System Generalization of Log Anomaly Detectors on Loghub Benchmarks"); r.bold = True; r.font.size = Pt(15)
tp = d.add_paragraph(); tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
tp.add_run("Full experiment: results and hypothesis tests (completion of the Assignment 3 pilot)\nRauan Kuntuganov, Alexey Azarenkov · SE-2401")

H("1 What was completed")
P(f"The Assignment 3 pilot ran on small Loghub samples, so H1 and H2 could not be tested. This report completes the experiment on the full labelled HDFS, BGL and Thunderbird datasets with all three model representatives, all three settings, all six directed source→target pairs and {NS} random seeds ({len(runs)} runs in total), and tests both hypotheses with the planned one-sided Wilcoxon signed-rank test.")
H("2 Data and deviations from the pilot protocol")
P("Source of the full data. The original Loghub archives are hosted on Zenodo. We used the full pre-processed versions published with the FSE 2024 benchmark study by Landauer, Skopik and Wurzenberger (A critical review of common log data sets used for evaluation of sequence-based anomaly detection techniques, Proc. ACM Softw. Eng., vol. 1, FSE, 2024, doi: 10.1145/3660768; repository ait-aecid/anomaly-detection-log-datasets, commit 5296693). These files contain every labelled session of HDFS (Loghub), BGL (Loghub) and Thunderbird (CFDR) as sequences of event-type identifiers, together with the template files that map identifiers to template text. The event totals match the Loghub figures exactly (Table 1), which confirms that the complete datasets were used.")
T([[r.dataset, f"{r.sessions:,}", f"{r.anomalous:,} ({r.anomaly_pct:.1f}%)", r.event_types, r.template_words,
    f"{r.train:,} ({r.train_anom:,})", f"{r.val:,} ({r.val_anom:,})", f"{r.test:,} ({r.test_anom:,})"] for r in ds.itertuples()],
  ["Dataset", "Sessions", "Anomalous", "Event types", "Template words", "Train (anom.)", "Validation (anom.)", "Test (anom.)"],
  cap="Table 1. Full datasets after preprocessing (dataset_summary.csv). Total events: HDFS 11,175,629; BGL 4,747,963; Thunderbird 211,212,192.")
P("Three changes to the Assignment 3 protocol were necessary and are reported openly:")
B("Unit of analysis. In the published files BGL and Thunderbird are grouped by node identifier and labelled per node (a node session is anomalous if any of its lines is anomalous), as in Landauer et al. Fixed 20-line windows therefore could not be formed; the unit is the session for all datasets (HDFS block, BGL/Thunderbird node).")
B("Split. The files carry no timestamps, so the planned chronological 70/10/20 split was replaced by a stratified random 70/10/20 split at session level with a fixed seed (42), identical for all models, settings and seeds.")
B("Size caps. To keep CPU training time feasible, the training partition was capped at 20,000 sessions and the validation partition at 5,000 (stratified samples); the test partition is used in full (115,013 HDFS sessions). The 5% adaptation budget is 5% of the (capped) target training partition: 1,000 sessions for HDFS and BGL, 205 for Thunderbird.")
P("Neural models read, per session, the first 64 distinct event types in order of first occurrence; each event is represented by the words of its template (same as in the pilot). Logistic regression uses TF-IDF of template words weighted by event counts over the whole session. One Thunderbird event identifier (6425) has no template line and contributes no words.")

H("3 Results")
rows = []
for m in MN:
    for s in DSL:
        for t in DSL:
            for setting in (["local"] if s == t else ["zero-shot", "adapt5"]):
                a = g(m, setting, s, t)
                rows.append([MN[m], SN[setting], s if s == t else f"{s}→{t}", pm(a), f2(a.P_mean), f2(a.R_mean), f2(a.FPR_mean), f"{a.train_s:.1f}"])
T(rows, ["Model", "Setting", "Direction", "F1 (mean ± SD)", "Precision", "Recall", "FPR", "Train s"], fs=7.5,
  cap=f"Table 2. Results on the target test partition, mean over {NS} seeds (results_summary.csv). For zero-shot runs the training time is that of the source model.")
F(f"{OUT}/fig1_f1_by_direction.png", f"Figure 1. F1 on the target test partition for every directed pair. Blue: model trained on the target itself (local reference); red: zero-shot transfer from the source; orange: source model adapted with 5% of the target training data. Error bars: SD over {NS} seeds.")
loc = agg[agg.setting == "local"]
P(f"Within-system training. All three models detect anomalies almost perfectly inside each system: local F1 ranges from {loc.F1_mean.min():.3f} to {loc.F1_mean.max():.3f}. This agrees with the very high scores in the literature and with Landauer et al.'s observation that these benchmarks are comparatively easy when train and test come from the same system.")
zs = agg[agg.setting == "zero-shot"]; ad = agg[agg.setting == "adapt5"]
P(f"Zero-shot transfer. Applying a source model to another system without any target data reduces F1 sharply: the mean zero-shot F1 over all models and directions is {zs.F1_mean.mean():.3f}, compared with {loc.F1_mean.mean():.3f} for local training. Many zero-shot runs degenerate: the model flags every target session as anomalous (FPR = 1) or none of them (F1 = 0). The high zero-shot F1 values that appear for some directions into Thunderbird must be read together with FPR: 74% of Thunderbird sessions are anomalous, so a model that labels everything as anomalous already reaches F1 ≈ 0.85 while being useless (Figure 3).")
P(f"5% adaptation. Fine-tuning or refitting with 5% of the target training data recovers most of the loss: mean F1 rises to {ad.F1_mean.mean():.3f} and mean FPR falls from {zs.FPR_mean.mean():.3f} (zero-shot) to {ad.FPR_mean.mean():.3f}. Recovery is weakest when HDFS is the target ({ad[ad.target=='HDFS'].F1_mean.mean():.3f} on average), where the 1,000-session budget contains only about 30 anomalies (2.9%).")
F(f"{OUT}/fig3_fpr_by_setting.png", "Figure 3. False-positive rate per model–direction combination (seed mean) in each setting. Zero-shot transfer produces many runs with FPR close to 1, i.e., alarms on almost all normal sessions.", w=11)

H("4 Hypothesis tests")
T([[h, r.comparison, int(r.n_pairs), f"{r.mean_diff:+.3f}", f"{r.median_diff:+.3f}", f"[{r.ci95_low:+.3f}, {r.ci95_high:+.3f}]", f"{r.W:.0f}",
    f"{r.p_value:.1e}", f"{r.rank_biserial:+.2f}", f"{int(r.pairs_in_predicted_direction)}/18", r.decision] for h, r in tests.iterrows()],
  ["Hyp.", "Paired difference", "n", "Mean", "Median", "Bootstrap 95% CI", "W", "p (one-sided)", "r (rank-biserial)", "In predicted direction", "Decision"],
  fs=7.5, cap="Table 3. One-sided Wilcoxon signed-rank tests on 18 seed-averaged model–direction pairs, α = 0.05 (hypothesis_tests.csv).")
P(f"H1 is supported. Zero-shot F1 was lower than the local F1 of the target in {int(h1.pairs_in_predicted_direction)} of 18 model–direction pairs; the mean paired difference is {h1.mean_diff:+.3f} (95% CI {h1.ci95_low:+.3f} to {h1.ci95_high:+.3f}), W = {h1.W:.0f}, one-sided p = {h1.p_value:.1e}. The effect size (matched-pairs rank-biserial correlation {h1.rank_biserial:+.2f}) is the largest possible: transfer without target data never matched local training.")
P(f"H2 is supported. F1 after 5% adaptation was higher than zero-shot F1 in {int(h2.pairs_in_predicted_direction)} of 18 pairs; mean paired difference {h2.mean_diff:+.3f} (95% CI {h2.ci95_low:+.3f} to {h2.ci95_high:+.3f}), W = {h2.W:.0f}, one-sided p = {h2.p_value:.1e}, rank-biserial r = {h2.rank_biserial:+.2f}.")
F(f"{OUT}/fig2_paired_differences.png", "Figure 2. Paired F1 differences used in the tests; each point is one directed dataset pair for one model (seed mean). Points below zero in the left panel and above zero in the right panel agree with H1 and H2.", w=14)
P("Per-model checks (n = 6 directions each, exact Wilcoxon test, smallest attainable one-sided p = 0.016) point in the same direction for every model; they are descriptive because six pairs give little power.")

H("5 Comparison of model representatives")
T([[SN[r.setting], MN[r.model], f"{r.mean_rank:.2f}", f2(r.mean_F1), f2(r.mean_FPR), f"{r.mean_train_s:.1f}"] for r in ranks.itertuples()],
  ["Setting", "Model", "Mean rank (1 = best)", "Mean F1", "Mean FPR", "Mean train s"], cap="Table 4. Model ranks within each setting (model_ranks.csv).")
P("Differences between the model representatives are much smaller than differences between settings, and no representative is best in all settings. Logistic regression trains in about a second, compared with tens of seconds for the Bi-LSTM and minutes for the Transformer, while reaching comparable local and adapted F1. In line with the analysis plan, these are descriptive rankings of one implementation per family, not conclusions about classical, recurrent or Transformer models in general.")

H("6 Threats to validity")
B("Random instead of chronological split: sessions from the same period can appear in training and test, which may inflate local scores; this affects all settings equally because the test partitions are identical.")
B("Label granularity: HDFS is labelled per block, BGL and Thunderbird per node; a whole Thunderbird node is anomalous if one line is. Part of the transfer loss reflects different anomaly definitions and very different anomaly ratios (2.9%, 45.3%, 73.6%), not only model weakness.")
B("Training caps (20,000 sessions) and the 64-event truncation for neural models may reduce neural performance on long node sessions.")
B("Adaptation differs by model type (refit for logistic regression, fine-tuning for neural models), and the adaptation sample is random, not stratified; sample anomaly counts are logged in results_runs.csv.")
B("One representative per model family, one budget (5%) and CPU-sized models limit generalization of the findings.")

H("7 Reproducibility")
P("All steps are scripted: prepare_data.py (session cache), run_experiment.py (all runs; resumable), analyze_results.py (aggregation, Wilcoxon tests, bootstrap CIs, figures) and build_results_report.py (this report). Raw per-run results with confusion counts (results_runs.csv), aggregated results, paired differences, test results and the run log are saved with the code. Seeds: split 42; model and adaptation-sample seeds 0–4.")
d.save(DOCX); print("saved", DOCX)
