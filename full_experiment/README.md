# Full experiment (completion of the Assignment 3 pilot)

Full labelled HDFS (575,061 sessions), BGL (69,252 node sessions) and Thunderbird (5,855 node sessions, 211M events),
3 models x {local, zero-shot, 5% adaptation} x 6 directed pairs x 5 seeds = 225 runs; H1/H2 tested with one-sided Wilcoxon.

Data: pre-processed full datasets from Landauer et al., FSE 2024 (doi:10.1145/3660768),
https://github.com/ait-aecid/anomaly-detection-log-datasets (commit 5296693, GPL-3.0) -
files hdfs_loghub/*, bgl_loghub/*, thunderbird_cfdr/* (unpack the two .tar.gz) and templates/{HDFS,BGL,Thunderbird}_templates.csv into data/.

```bash
pip install -r requirements.txt
python prepare_data.py data cache            # ~1 min
python run_experiment.py cache data out      # ~60 min on 2 CPU cores, resumable
python analyze_results.py out                # tests + figures
python build_results_report.py out report.docx
```
Outputs in out/: results_runs.csv (every run, confusion counts), results_summary.csv, paired_differences.csv,
hypothesis_tests.csv, model_ranks.csv, analysis_summary.txt, run_log.txt, fig1-3 PNGs.

## Correction (checkpoint restore) and rerun

A code audit found that, for neural models with BGL or Thunderbird as source, the local evaluation ran after the
model had been fine-tuned on another target (the loop did not restore the source checkpoint). `run_experiment.py`
now restores the checkpoint before local prediction. `rerun_local_fix.py` refits exactly the four affected
configurations (Bi-LSTM / Transformer x BGL / Thunderbird, seeds 0-4, same data, split and hyper-parameters).

- `out/` - archived original run (kept unchanged)
- `out_fix/results_local_fix.csv` - the 20 corrected local records
- `out_corrected/` - merged results (archived records with the 20 replaced) and the final analysis:
  H1 18/18 pairs, mean -0.615, exact one-sided Wilcoxon p = 3.8e-6 (Holm 7.6e-6);
  H2 17/18 pairs, mean +0.312, p = 1.1e-5.
