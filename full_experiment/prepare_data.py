"""Step 1: read the full pre-processed Loghub sequences (Landauer et al., 2024; github.com/ait-aecid/anomaly-detection-log-datasets)
and cache, per session: label, total length, event-type counts and the order of first occurrence of event types.

Input files per dataset: <ds>_train, <ds>_test_normal (label 0) and <ds>_test_abnormal (label 1); each line "session_id,e1 e2 e3 ...".
The repository's own train/test division is NOT used: all sessions are pooled and re-split in run_experiment.py.
Usage: python3 prepare_data.py <data_dir> <cache_dir>
"""
import sys, pickle, time
from collections import Counter

DATA, CACHE = sys.argv[1], sys.argv[2]
MAX_ORDER = 64  # first 64 distinct event types in order of first occurrence

for ds, prefix in [("HDFS", "hdfs"), ("BGL", "bgl"), ("Thunderbird", "thunderbird")]:
    t0 = time.time(); sessions = []
    for part, label in [("train", 0), ("test_normal", 0), ("test_abnormal", 1)]:
        with open(f"{DATA}/{prefix}_{part}") as f:
            for line in f:
                sid, _, seq = line.rstrip("\n").partition(",")
                ev = seq.split()
                cnt = Counter(ev)
                order = list(dict.fromkeys(ev))[:MAX_ORDER]
                sessions.append({"id": sid, "y": label, "n_events": len(ev), "counts": dict(cnt), "order": order})
    pickle.dump(sessions, open(f"{CACHE}/{ds}.pkl", "wb"))
    n_anom = sum(s["y"] for s in sessions)
    print(f"{ds}: sessions={len(sessions)} anomalous={n_anom} ({100*n_anom/len(sessions):.1f}%) "
          f"events={sum(s['n_events'] for s in sessions)} event_types={len({e for s in sessions for e in s['counts']})} "
          f"time={time.time()-t0:.0f}s", flush=True)
