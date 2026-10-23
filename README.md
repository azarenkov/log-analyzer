# Cross-System Generalization of Log Anomaly Detectors — Assignment 3 pilot

Rauan Kuntuganov, Alexey Azarenkov · SE-2401

Pilot for comparing a classical (logistic regression), recurrent (attention Bi-LSTM) and
Transformer detector under within-system training, zero-shot transfer and 5% target adaptation
on Loghub HDFS, BGL and Thunderbird samples.

## Reproduce

```bash
pip install -r requirements.txt
mkdir -p data out
curl -L -o data/BGL_2k.log_structured.csv https://raw.githubusercontent.com/logpai/loghub/master/BGL/BGL_2k.log_structured.csv
curl -L -o data/Thunderbird_2k.log_structured.csv https://raw.githubusercontent.com/logpai/loghub/master/Thunderbird/Thunderbird_2k.log_structured.csv
curl -L -o data/HDFS_100k.log_structured.csv https://raw.githubusercontent.com/logpai/loglizer/master/data/HDFS/HDFS_100k.log_structured.csv
curl -L -o data/anomaly_label.csv https://raw.githubusercontent.com/logpai/loglizer/master/data/HDFS/anomaly_label.csv
python pilot_pipeline.py data out      # ~10 min on 2 CPU cores
python fig.py out/results_summary.csv out/f1_pilot.png
```

## Files
- `pilot_pipeline.py` — preprocessing, chronological 70/10/20 split, three models, three settings, metrics
- `fig.py` — Figure 1 of the report
- `build_report.py` — regenerates the .docx report from the output files
- `out/` — outputs of the reported run: `dataset_summary.csv`, `results_runs.csv` (every run),
  `results_summary.csv` (mean ± SD), `run_log.txt`, `console.txt`, `f1_pilot.png`

Notes: F1/precision/recall are N/A (empty) when the target test partition has no anomalies
(Thunderbird_2k). Neural results can differ slightly across hardware/PyTorch builds.
