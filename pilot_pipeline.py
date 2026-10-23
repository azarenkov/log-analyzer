"""Assignment 3 pilot v2: cross-system log anomaly detection with three model representatives.

Models : classical  = logistic regression on TF-IDF of template words
         recurrent  = LogRobust-style attention Bi-LSTM over semantic event vectors
         transformer= Transformer encoder over the same semantic event vectors
Settings: local (target train), zero-shot (source only), adapt5 (source + 5% of target train)
Split   : chronological 70/10/20 (train/validation/test) per dataset
Data    : Loghub samples - logpai/loghub BGL_2k, Thunderbird_2k; logpai/loglizer HDFS_100k + anomaly_label
Usage   : python3 pilot_pipeline.py <data_dir> <out_dir>
"""
import sys, re, time, json, hashlib, platform
import numpy as np, pandas as pd, sklearn, torch, torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

DATA, OUT = sys.argv[1], sys.argv[2]
WINDOW, SPLIT, ADAPT_FRAC = 20, (0.7, 0.1, 0.2), 0.05
HDFS_MAX = None  # use all block sessions of the 100k-line sample
SEEDS = [0, 1, 2]
MAX_EVENTS, MAX_WORDS, EMB, HID, EPOCHS, ADAPT_EPOCHS = 50, 12, 32, 32, 8, 5
torch.set_num_threads(2)
log = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.append(s)

def words(t):
    return [w.lower() for w in re.findall(r"[a-zA-Z]+", str(t).replace("<*>", " ")) if len(w) > 1]

# ---------------------------------------------------------------- data
def load_hdfs():
    d = pd.read_csv(f"{DATA}/HDFS_100k.log_structured.csv")
    d["blk"] = d.Content.str.extract(r"(blk_-?\d+)")[0]
    d = d.dropna(subset=["blk"])
    lab = pd.read_csv(f"{DATA}/anomaly_label.csv").set_index("BlockId").Label
    g = d.groupby("blk", sort=False)  # order of first appearance = chronological
    u = pd.DataFrame({"events": g.EventId.apply(list), "templates": g.EventTemplate.apply(list)})
    u["y"] = (lab.reindex(u.index) == "Anomaly").astype(int).values
    return u.iloc[:HDFS_MAX].reset_index(drop=True), len(d)

def load_windowed(name):
    d = pd.read_csv(f"{DATA}/{name}_2k.log_structured.csv")
    d["anom"] = (d.Label != "-").astype(int)
    rows = [{"events": list(w.EventId), "templates": list(w.EventTemplate), "y": int(w.anom.max())}
            for w in (d.iloc[i:i + WINDOW] for i in range(0, len(d) - WINDOW + 1, WINDOW))]
    return pd.DataFrame(rows), len(d)

def split(u):
    a = int(len(u) * SPLIT[0]); b = int(len(u) * (SPLIT[0] + SPLIT[1]))
    return [u.iloc[x:y].reset_index(drop=True) for x, y in ((0, a), (a, b), (b, len(u)))]

def text(df):
    return [" ".join(" ".join(words(t)) for t in ts) for ts in df.templates]

# ---------------------------------------------------------------- metrics
def metrics(y, yp):
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    pos = tp + fn
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / pos if pos else float("nan")
    F1 = (2 * P * R / (P + R) if P + R else 0.0) if pos else float("nan")  # undefined without positives -> N/A
    return dict(P=P if pos else float("nan"), R=R, F1=F1, FPR=fp / (fp + tn) if fp + tn else float("nan"),
                TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn))

# ---------------------------------------------------------------- classical
def run_lr(train, val, test):
    """Refit on all allowed data; threshold fixed at 0.5 (val used only to report val-F1)."""
    vec = TfidfVectorizer().fit(text(train))
    m = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0).fit(vec.transform(text(train)), train.y)
    return m.predict(vec.transform(text(test)))

# ---------------------------------------------------------------- neural
class Vocab:
    def __init__(self, dfs):
        ws = sorted({w for df in dfs for ts in df.templates for t in ts for w in words(t)})
        self.i = {w: k + 2 for k, w in enumerate(ws)}  # 0 pad, 1 unknown
    def encode(self, df):
        X = np.zeros((len(df), MAX_EVENTS, MAX_WORDS), dtype=np.int64)
        for n, ts in enumerate(df.templates):
            for e, t in enumerate(ts[:MAX_EVENTS]):
                for k, w in enumerate(words(t)[:MAX_WORDS]):
                    X[n, e, k] = self.i.get(w, 1)
        return torch.tensor(X)

class EventEncoder(nn.Module):  # semantic event vector = mean of word embeddings (LogRobust idea, learned here)
    def __init__(self, nv):
        super().__init__(); self.emb = nn.Embedding(nv, EMB, padding_idx=0)
    def forward(self, x):
        m = (x > 0).float().unsqueeze(-1)
        v = (self.emb(x) * m).sum(2) / m.sum(2).clamp(min=1)
        return v, (x > 0).any(-1)

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
        self.tr = nn.TransformerEncoder(layer, num_layers=2); self.out = nn.Linear(EMB, 1)
    def forward(self, x):
        v, mask = self.enc(x); h = self.tr(v + self.pos, src_key_padding_mask=~mask)
        m = mask.float().unsqueeze(-1)
        return self.out((h * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)

def fit(model, X, y, Xv, yv, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    y = torch.tensor(np.asarray(y), dtype=torch.float32)
    pw = torch.tensor(max(1.0, float((y == 0).sum()) / max(1.0, float(y.sum()))))  # class imbalance weight
    opt = torch.optim.Adam(model.parameters(), lr=2e-3); lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    best, best_state = -1, None
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(X))
        for b in range(0, len(X), 64):
            idx = perm[b:b + 64]; opt.zero_grad()
            lossf(model(X[idx]), y[idx]).backward(); opt.step()
        if Xv is not None and len(np.unique(yv)) > 1:  # early stopping on validation F1 of allowed data
            f = metrics(np.asarray(yv), predict(model, Xv))["F1"]
            if f > best: best, best_state = f, {k: v.clone() for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    return model

def predict(model, X):
    model.eval()
    with torch.no_grad():
        return (torch.cat([model(X[b:b + 256]) for b in range(0, len(X), 256)]) > 0).int().numpy()

NEURAL = {"bilstm": BiLSTMAtt, "transformer": TransformerClf}

# ---------------------------------------------------------------- main
p("python", platform.python_version(), "| torch", torch.__version__, "| sklearn", sklearn.__version__,
  "| pandas", pd.__version__, "| numpy", np.__version__)
for f in ["HDFS_100k.log_structured.csv", "anomaly_label.csv", "BGL_2k.log_structured.csv", "Thunderbird_2k.log_structured.csv"]:
    p("sha256", f, hashlib.sha256(open(f"{DATA}/{f}", "rb").read()).hexdigest()[:16])
ds = {}
ds["HDFS"], n = load_hdfs(); p(f"HDFS: {n} lines -> {len(ds['HDFS'])} block sessions, anomalies={ds['HDFS'].y.sum()}")
for name in ["BGL", "Thunderbird"]:
    ds[name], n = load_windowed(name); p(f"{name}: {n} lines -> {len(ds[name])} windows of {WINDOW}, anomalous={ds[name].y.sum()}")
parts = {k: split(v) for k, v in ds.items()}
summ = []
for k, (tr, va, te) in parts.items():
    summ.append(dict(dataset=k, train=len(tr), train_anom=int(tr.y.sum()), val=len(va), val_anom=int(va.y.sum()),
                     test=len(te), test_anom=int(te.y.sum()), templates=len({e for ev in ds[k].events for e in ev})))
    p(summ[-1])
pd.DataFrame(summ).to_csv(f"{OUT}/dataset_summary.csv", index=False)
for a in ds:
    for b in ds:
        if a < b:
            wa = {w for ts in ds[a].templates for t in ts for w in words(t)}; wb = {w for ts in ds[b].templates for t in ts for w in words(t)}
            p(f"template-word overlap {a}-{b}: {len(wa & wb)} shared of {len(wa | wb)}")

results = []
def record(**kw):
    r = metrics(kw.pop("y"), kw.pop("yp")); row = {**kw, **r}; results.append(row)
    p(f"[{row['setting']}] {row['model']:<11} {row['source']}->{row['target']} seed={row['seed']} "
      f"F1={r['F1']:.3f} P={r['P']:.3f} R={r['R']:.3f} FPR={r['FPR']:.3f} TP={r['TP']} FP={r['FP']} FN={r['FN']} TN={r['TN']} t={row['train_s']:.1f}s")

def single_class(df): return df.y.nunique() < 2

for model in ["logreg", "bilstm", "transformer"]:
    seeds = [0] if model == "logreg" else SEEDS  # logistic regression is deterministic
    for s in ds:
        str_, sva, _ = parts[s]
        if single_class(str_):
            p(f"[skip] {model} source={s}: training partition has one class -> local and transfer from {s} are N/A"); continue
        for t in ds:
            ttr, tva, tte = parts[t]
            for seed in seeds:
                if s == t:   # local: train on target train, early-stop on target val
                    settings = [("local", tr_ := str_, sva)]
                else:
                    n_ad = max(1, int(round(ADAPT_FRAC * len(ttr))))
                    ad = ttr.sample(n=n_ad, random_state=seed)
                    settings = [("zero-shot", str_, sva), ("adapt5", ad, None)]
                base = None
                for setting, train, val in settings:
                    t0 = time.time()
                    if model == "logreg":
                        trn = train if setting != "adapt5" else pd.concat([str_, train])  # refit source + 5% target
                        yp = run_lr(trn, val, tte)
                    else:
                        if setting != "adapt5":
                            vocab = Vocab([str_]); net = NEURAL[model](len(vocab.i) + 2)
                            Xv = vocab.encode(val) if val is not None else None
                            base = fit(net, vocab.encode(train), train.y, Xv, None if val is None else val.y, EPOCHS, seed)
                            yp = predict(base, vocab.encode(tte))
                        else:  # fine-tune the source model on the 5% target sample (source vocabulary, unseen words -> UNK)
                            net = fit(base, vocab.encode(train), train.y, None, None, ADAPT_EPOCHS, seed)
                            yp = predict(net, vocab.encode(tte))
                    extra = {"n_adapt": len(train), "adapt_anom": int(train.y.sum())} if setting == "adapt5" else {}
                    record(model=model, setting=setting, source=s, target=t, seed=seed, train_s=time.time() - t0,
                           y=tte.y.values, yp=yp, **extra)

res = pd.DataFrame(results); res.to_csv(f"{OUT}/results_runs.csv", index=False)
agg = (res.groupby(["model", "setting", "source", "target"], sort=False)
          .agg(runs=("F1", "size"), F1_mean=("F1", "mean"), F1_sd=("F1", "std"), P_mean=("P", "mean"),
               R_mean=("R", "mean"), FPR_mean=("FPR", "mean"), train_s=("train_s", "mean")).reset_index())
agg.to_csv(f"{OUT}/results_summary.csv", index=False)
open(f"{OUT}/run_log.txt", "w").write("\n".join(log))
p("done")
