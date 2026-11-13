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
