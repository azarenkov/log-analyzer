"""Step 2: full cross-system experiment (3 models x 3 settings x all source->target pairs x 5 seeds).

Unit            : session (HDFS block; BGL / Thunderbird node), labelled anomalous if any of its lines is anomalous.
Representation  : words of the event templates (letters only, lower case, len>1) - shared across systems.
Split           : stratified random 70/10/20 per dataset (fixed, random_state=42); the pre-processed data carry no timestamps.
                  Training partition capped at 20,000 sessions and validation at 5,000 (stratified) to bound CPU time;
                  the test partition is used in full.
Settings        : local     - train on target train, early stopping on target validation, test on target test
                  zero-shot - the source model (trained on source train / source validation) applied to the target test
                  adapt5    - source model + random 5% of the TARGET training partition (seed-dependent sample):
                              logistic regression refitted on source+sample, neural models fine-tuned on the sample only
Models          : logreg (TF-IDF of template words, class_weight=balanced), bilstm (attention Bi-LSTM), transformer
Usage           : python3 run_experiment.py <cache_dir> <data_dir> <out_dir> [seeds, default 0,1,2,3,4]
"""
import sys, re, time, pickle, platform, os
import numpy as np, pandas as pd, scipy.sparse as sp, sklearn, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

CACHE, DATA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SEEDS = [int(s) for s in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0, 1, 2, 3, 4]
SPLIT_SEED, TRAIN_CAP, VAL_CAP, ADAPT_FRAC = 42, 20000, 5000, 0.05
MAX_EVENTS, MAX_WORDS, EMB, HID, EPOCHS, ADAPT_EPOCHS, BATCH = 64, 12, 32, 32, 8, 5, 128
DATASETS = ["HDFS", "BGL", "Thunderbird"]
torch.set_num_threads(os.cpu_count())
LOG = open(f"{OUT}/run_log.txt", "a")
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

def words(t):
    return [w.lower() for w in re.findall(r"[a-zA-Z]+", str(t).replace("<*>", " ")) if len(w) > 1]

# ------------------------------------------------------------------ load
def load_templates(ds):
    # event id = 1-based line number in the template file (as in the source repository's parsers)
    with open(f"{DATA}/{ds}_templates.csv", encoding="utf-8", errors="replace") as f:
        return {str(i + 1): words(line.rstrip("\n")) for i, line in enumerate(f)}

D = {}
for ds in DATASETS:
    S = pickle.load(open(f"{CACHE}/{ds}.pkl", "rb"))
    tmpl = load_templates(ds)
    ev_keys = sorted({e for s in S for e in s["counts"]}, key=lambda x: (len(x), x))
    ev_idx = {e: i for i, e in enumerate(ev_keys)}
    missing = [e for e in ev_keys if e not in tmpl]
    y = np.array([s["y"] for s in S], dtype=np.int64)
    rows, cols, vals = [], [], []
    for r, s in enumerate(S):
        for e, c in s["counts"].items():
            rows.append(r); cols.append(ev_idx[e]); vals.append(c)
    C = sp.csr_matrix((vals, (rows, cols)), shape=(len(S), len(ev_keys)), dtype=np.float64)
    order = np.full((len(S), MAX_EVENTS), -1, dtype=np.int32)
    for r, s in enumerate(S):
        o = [ev_idx[e] for e in s["order"]]; order[r, :len(o)] = o
    ev_words = [tmpl.get(e, []) for e in ev_keys]
    idx = np.arange(len(S))
    tr, rest = train_test_split(idx, test_size=0.3, stratify=y, random_state=SPLIT_SEED)
    va, te = train_test_split(rest, test_size=2 / 3, stratify=y[rest], random_state=SPLIT_SEED)
    if len(tr) > TRAIN_CAP: tr = train_test_split(tr, train_size=TRAIN_CAP, stratify=y[tr], random_state=SPLIT_SEED)[0]
    if len(va) > VAL_CAP: va = train_test_split(va, train_size=VAL_CAP, stratify=y[va], random_state=SPLIT_SEED)[0]
    D[ds] = dict(y=y, C=C, order=order, ev_words=ev_words, tr=np.sort(tr), va=np.sort(va), te=np.sort(te))
    p(f"{ds}: sessions={len(S)} anom={y.sum()} event_types={len(ev_keys)} unmapped_event_ids={missing[:5]}{'...' if len(missing)>5 else ''} "
      f"train={len(tr)}({y[tr].sum()}) val={len(va)}({y[va].sum()}) test={len(te)}({y[te].sum()})")
    del S

# global word index for the classical model: session x word counts = (session x event counts) @ (event x word)
ALLW = sorted({w for ds in DATASETS for ws in D[ds]["ev_words"] for w in ws}); WIDX = {w: i for i, w in enumerate(ALLW)}
for ds in DATASETS:
    r, c = [], []
    for i, ws in enumerate(D[ds]["ev_words"]):
        for w in ws: r.append(i); c.append(WIDX[w])
    W = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(D[ds]["ev_words"]), len(ALLW)))
    D[ds]["Xw"] = (D[ds]["C"] @ W).tocsr()
    D[ds]["words"] = {w for ws in D[ds]["ev_words"] for w in ws}
summ = []
for a in DATASETS:
    for b in DATASETS:
        if a < b:
            sh = D[a]["words"] & D[b]["words"]
            p(f"template-word overlap {a}-{b}: {len(sh)} shared of {len(D[a]['words'] | D[b]['words'])}")
for ds in DATASETS:
    y = D[ds]["y"]
    summ.append(dict(dataset=ds, sessions=len(y), anomalous=int(y.sum()), anomaly_pct=round(100 * y.mean(), 2),
                     event_types=len(D[ds]["ev_words"]), template_words=len(D[ds]["words"]),
                     train=len(D[ds]["tr"]), train_anom=int(y[D[ds]["tr"]].sum()), val=len(D[ds]["va"]), val_anom=int(y[D[ds]["va"]].sum()),
                     test=len(D[ds]["te"]), test_anom=int(y[D[ds]["te"]].sum())))
pd.DataFrame(summ).to_csv(f"{OUT}/dataset_summary.csv", index=False)

# ------------------------------------------------------------------ metrics
def metrics(y, yp):
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else float("nan")
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    return dict(P=P, R=R, F1=F1, FPR=fp / (fp + tn) if fp + tn else float("nan"), TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn))

# ------------------------------------------------------------------ classical
def lr_fit_predict(train_parts, test):
    X = sp.vstack([D[ds]["Xw"][ix] for ds, ix in train_parts]); y = np.concatenate([D[ds]["y"][ix] for ds, ix in train_parts])
    tf = TfidfTransformer().fit(X)
    m = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0).fit(tf.transform(X), y)
    ds, ix = test
    return m.predict(tf.transform(D[ds]["Xw"][ix]))

# ------------------------------------------------------------------ neural
class Vocab:
    def __init__(self, ds):
        self.i = {w: k + 2 for k, w in enumerate(sorted(D[ds]["words"]))}  # 0 pad, 1 unknown
    def event_table(self, ds):
        T = np.zeros((len(D[ds]["ev_words"]) + 1, MAX_WORDS), dtype=np.int64)  # last row = padding event
        for e, ws in enumerate(D[ds]["ev_words"]):
            ids = [self.i.get(w, 1) for w in ws[:MAX_WORDS]] or [1]
            T[e, :len(ids)] = ids
        return torch.tensor(T)
    def batches(self, ds, ix, bs):
        T = self.event_table(ds); pad = T.shape[0] - 1
        for b in range(0, len(ix), bs):
            o = torch.tensor(D[ds]["order"][ix[b:b + bs]].astype(np.int64)); o[o < 0] = pad
            yield T[o]

class EventEncoder(nn.Module):
    def __init__(self, nv):
        super().__init__(); self.emb = nn.Embedding(nv, EMB, padding_idx=0)
    def forward(self, x):
        m = (x > 0).float().unsqueeze(-1)
        return (self.emb(x) * m).sum(2) / m.sum(2).clamp(min=1), (x > 0).any(-1)

class BiLSTMAtt(nn.Module):
    def __init__(self, nv):
        super().__init__(); self.enc = EventEncoder(nv)
        self.lstm = nn.LSTM(EMB, HID, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * HID, 1); self.out = nn.Linear(2 * HID, 1)
    def forward(self, x):
        v, mask = self.enc(x); h, _ = self.lstm(v)
        a = self.att(h).squeeze(-1).masked_fill(~mask, -1e9).softmax(-1)
        return self.out((a.unsqueeze(-1) * h).sum(1)).squeeze(-1)

class TransformerClf(nn.Module):
    def __init__(self, nv):
        super().__init__(); self.enc = EventEncoder(nv)
        self.pos = nn.Parameter(torch.zeros(1, MAX_EVENTS, EMB))
        layer = nn.TransformerEncoderLayer(EMB, nhead=4, dim_feedforward=64, dropout=0.1, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False); self.out = nn.Linear(EMB, 1)
    def forward(self, x):
        v, mask = self.enc(x); h = self.tr(v + self.pos, src_key_padding_mask=~mask)
        m = mask.float().unsqueeze(-1)
        return self.out((h * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)

NEURAL = {"bilstm": BiLSTMAtt, "transformer": TransformerClf}

def predict(model, vocab, ds, ix):
    model.eval()
    with torch.no_grad():
        return np.concatenate([(model(xb) > 0).int().numpy() for xb in vocab.batches(ds, ix, 1024)])

def fit(model, vocab, ds, ix, val, epochs, seed):
    g = torch.Generator().manual_seed(seed)
    y_all = torch.tensor(D[ds]["y"][ix], dtype=torch.float32)
    npos = float(y_all.sum()); nneg = len(y_all) - npos
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(nneg / npos if npos > 0 else 1.0))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3); best, best_state = -1.0, None
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(ix), generator=g).numpy()
        for b in range(0, len(ix), BATCH):
            sel = perm[b:b + BATCH]
            xb = next(vocab.batches(ds, ix[sel], len(sel)))
            opt.zero_grad(); lossf(model(xb), y_all[sel]).backward(); opt.step()
        if val is not None:  # early stopping on validation F1 of data the setting may use
            f = metrics(D[val[0]]["y"][val[1]], predict(model, vocab, *val))["F1"]
            if f > best: best, best_state = f, {k: v.clone() for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    return model

# ------------------------------------------------------------------ main loop
p("python", platform.python_version(), "| torch", torch.__version__, "| sklearn", sklearn.__version__, "| numpy", np.__version__,
  "| threads", torch.get_num_threads(), "| seeds", SEEDS)
res_path = f"{OUT}/results_runs.csv"
done = set()
if os.path.exists(res_path):
    old = pd.read_csv(res_path); done = set(zip(old.model, old.setting, old.source, old.target, old.seed))
COLS = ["model", "setting", "source", "target", "seed", "train_s", "n_adapt", "adapt_anom", "P", "R", "F1", "FPR", "TP", "FP", "FN", "TN"]
def record(row, y, yp):
    r = {"n_adapt": "", "adapt_anom": "", **row, **metrics(y, yp)}
    pd.DataFrame([r])[COLS].to_csv(res_path, mode="a", header=not os.path.exists(res_path), index=False)
    p(f"[{r['setting']:<9}] {r['model']:<11} {r['source']}->{r['target']:<11} seed={r['seed']} F1={r['F1']:.3f} P={r['P']:.3f} "
      f"R={r['R']:.3f} FPR={r['FPR']:.3f} train_s={r['train_s']:.1f}" + (f" adapt={r['n_adapt']}({r['adapt_anom']})" if r['n_adapt'] != "" else ""))

for seed in SEEDS:
    for model in ["logreg", "bilstm", "transformer"]:
        for s in DATASETS:
            todo = [(st, s, t) for t in DATASETS for st in (["local"] if s == t else ["zero-shot", "adapt5"])]
            if all((model, st, a, b, seed) in done for st, a, b in todo): continue
            torch.manual_seed(seed); np.random.seed(seed)
            t0 = time.time()
            if model != "logreg":
                vocab = Vocab(s); net = NEURAL[model](len(vocab.i) + 2)
                fit(net, vocab, s, D[s]["tr"], (s, D[s]["va"]), EPOCHS, seed)
                base_state = {k: v.clone() for k, v in net.state_dict().items()}
            t_base = time.time() - t0
            for t in DATASETS:
                te = D[t]["te"]; yt = D[t]["y"][te]
                if s == t:
                    if model != "logreg": net.load_state_dict(base_state)  # FIX: restore the source checkpoint (earlier adapt5 branches fine-tune net in place)
                    yp = lr_fit_predict([(s, D[s]["tr"])], (t, te)) if model == "logreg" else predict(net, vocab, t, te)
                    record(dict(model=model, setting="local", source=s, target=t, seed=seed, train_s=time.time() - t0 if model == "logreg" else t_base), yt, yp)
                    continue
                if model == "logreg":
                    t1 = time.time(); yp = lr_fit_predict([(s, D[s]["tr"])], (t, te))
                    record(dict(model=model, setting="zero-shot", source=s, target=t, seed=seed, train_s=time.time() - t1), yt, yp)
                else:
                    net.load_state_dict(base_state)
                    record(dict(model=model, setting="zero-shot", source=s, target=t, seed=seed, train_s=t_base), yt, predict(net, vocab, t, te))
                ttr = D[t]["tr"]; n_ad = int(round(ADAPT_FRAC * len(ttr)))
                ad = np.sort(np.random.RandomState(1000 + seed).choice(ttr, n_ad, replace=False))
                t1 = time.time()
                if model == "logreg":
                    yp = lr_fit_predict([(s, D[s]["tr"]), (t, ad)], (t, te))
                else:
                    net.load_state_dict(base_state)
                    fit(net, vocab, t, ad, None, ADAPT_EPOCHS, seed); yp = predict(net, vocab, t, te)
                record(dict(model=model, setting="adapt5", source=s, target=t, seed=seed, train_s=time.time() - t1,
                            n_adapt=n_ad, adapt_anom=int(D[t]["y"][ad].sum())), yt, yp)
p("done")
