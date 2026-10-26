"""
Cutines, Adrian Carl B.
2021-07168

CMSC 191 -  Special Topics: Fundamentals of Neural Computing
Title -  A Neural Network Approach to Predicting the Approval or Rejection of a Credit Card Application

test.py - data preparation and scoring for backprop.bas (CMSC 191)

Usage:
  python test.py          builds the file PROBLEM (train + validation + test rows)
  python test.py score    scores predictions.csv written by backprop.exe

Preprocessing: Type_Occupation is dropped; 365243 in Employed_days is recoded to 0;
missing values are filled with the median (numeric) or mode (categorical); binary
attributes are mapped to 0/1; categorical attributes are one-hot encoded; all inputs
are min-max scaled to [0, 1].

Evaluation design:
  1. The data are split 70/15/15 into training, validation and test sets. Rows with the
     same person key (same stable attributes, different Ind_ID) are kept in one split.
  2. Medians, modes and min/max values are computed from the training rows only. The list
     of category values comes from the whole file, so there are always 32 inputs.
  3. Training examples are weighted by class (N / (2 * n_class)) to offset the imbalance.
  4. Scoring reports AUC, accuracy, precision, recall and F1.

Settings are read from network.txt, one value per line:
  hidden nodes, kappa, phi, theta, mu, jump (0/1), max epochs, patience, seed,
  split (0 = repeated applicants kept together, 1 = plain random split),
  restore (optional, default 1: 1 = use best-validation weights, 0 = use final weights)

Each "python test.py score" appends a row to results_log.csv and saves a training curve
(curve_<settings>.png).
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)


def find_csv(keyword):
    hits = sorted(glob.glob(f"*{keyword}*.csv"))
    if not hits:
        sys.exit(f"Cannot find a .csv containing '{keyword}' in {HERE}")
    return hits[0]


def read_settings():
    v = [float(l) for l in open("network.txt").read().split() if l.strip()]
    names = ["hidden", "kappa", "phi", "theta", "mu", "jump", "epochs", "patience", "seed", "split", "restore"]
    cfg = dict(zip(names, v))
    cfg.setdefault("split", 0.0)
    cfg.setdefault("restore", 1.0)
    return cfg


def run_id(cfg):
    rb = "" if int(cfg["restore"]) == 1 else "_rb0"     # rb0 = final weights kept (no best-epoch restore)
    return (f"h{int(cfg['hidden'])}_k{cfg['kappa']:g}_f{cfg['phi']:g}_t{cfg['theta']:g}_m{cfg['mu']:g}"
            f"_j{int(cfg['jump'])}_s{int(cfg['seed'])}_sp{int(cfg['split'])}{rb}")


# ------------------------------------------------------------------ preparation
NUM = ["Annual_income", "Birthday_count", "Employed_days"]
BIN = {"GENDER": "M", "Car_Owner": "Y", "Propert_Owner": "Y"}
CAT = ["Type_Income", "EDUCATION", "Marital_status", "Housing_type"]
PASS = ["CHILDREN", "Mobile_phone", "Work_Phone", "Phone", "EMAIL_ID", "Family_Members"]


def person_key(df):
    cols = ["Car_Owner", "Propert_Owner", "Type_Income", "EDUCATION", "Marital_status",
            "Housing_type", "Employed_days", "Family_Members", "CHILDREN"]
    return df[cols].astype(str).agg("|".join, axis=1)


def grouped_split(keys, y, fracs, rng):
    """Split whole groups (same person) into len(fracs) parts.  Groups are visited in
    RANDOM order; a group goes to the split that is furthest below its target share
    (positives first, then the rest), so every split gets a fair mix of single-row and
    repeated applicants."""
    g = pd.DataFrame({"g": keys.values, "y": y}).groupby("g")["y"].agg(["size", "sum"])
    g = g.iloc[rng.permutation(len(g))]
    tn, tp = np.array(fracs) * len(y), np.array(fracs) * y.sum()
    cn, cp, assign = np.zeros(len(fracs)), np.zeros(len(fracs)), {}
    for key, r in pd.concat([g[g["sum"] > 0], g[g["sum"] == 0]]).iterrows():
        deficit = (tp - cp) / tp + 0.01 * (tn - cn) / tn if r["sum"] > 0 else (tn - cn) / tn
        s = int(np.argmax(deficit))
        assign[key] = s
        cn[s] += r["size"]
        cp[s] += r["sum"]
    return keys.map(assign).to_numpy()


def fit_stats(tr, full):
    # category LISTS come from the whole file (they are just the possible values, so the
    # input count is always the same); every STATISTIC still comes from TRAIN only.
    d = tr.copy()
    d["Employed_days"] = d["Employed_days"].replace(365243, 0)
    st = {"med": {c: float(d[c].median()) for c in NUM},
          "mode": {c: d[c].mode()[0] for c in list(BIN) + CAT},
          "cats": {c: sorted(full[c].dropna().unique()) for c in CAT}}
    X = raw_matrix(tr, st)
    st["min"], st["max"] = X.min(axis=0), X.max(axis=0)
    return st


def raw_matrix(df, st):
    d = df.copy()
    d["Employed_days"] = d["Employed_days"].replace(365243, 0)
    cols = {}
    for c, pos in BIN.items():
        cols[c] = (d[c].fillna(st["mode"][c]) == pos).astype(float)
    for c in PASS:
        cols[c] = d[c].astype(float)
    for c in NUM:
        cols[c] = d[c].fillna(st["med"][c]).astype(float)
    for c in CAT:
        v = d[c].fillna(st["mode"][c])
        for cat in st["cats"][c]:
            cols[f"{c}_{cat}"] = (v == cat).astype(float)
    return pd.DataFrame(cols).to_numpy(dtype=float)


def scale(df, st):
    X = raw_matrix(df, st)
    rng = st["max"] - st["min"]
    rng[rng == 0] = 1.0
    return np.clip((X - st["min"]) / rng, 0.0, 1.0)


def class_weights(t):
    n1, n0 = max(int(t.sum()), 1), max(int((t == 0).sum()), 1)
    return np.where(t == 1, len(t) / (2 * n1), len(t) / (2 * n0))


def prepare():
    cfg = read_settings()
    df = pd.merge(pd.read_csv(find_csv("INPUTS")), pd.read_csv(find_csv("OUTPUT")), on="Ind_ID")
    df = df.drop(columns=["Ind_ID", "Type_Occupation"])
    y = df["label"].to_numpy()
    rng = np.random.default_rng(int(cfg["seed"]))
    if int(cfg["split"]) == 1:        # plain random stratified split (duplicates NOT kept together)
        split = np.zeros(len(y), int)
        for c in (0, 1):
            idx = rng.permutation(np.where(y == c)[0])
            cut = [int(0.70 * len(idx)), int(0.85 * len(idx))]
            split[idx[cut[0]:cut[1]]] = 1
            split[idx[cut[1]:]] = 2
        print("NOTE: random split (split=1) - duplicate applicants can be in train AND test")
    else:
        split = grouped_split(person_key(df), y, [0.70, 0.15, 0.15], rng)
    parts = [df[split == s] for s in range(3)]
    st = fit_stats(parts[0], df)                  # statistics from TRAIN only
    blocks = []
    for i, d in enumerate(parts):
        t = d["label"].to_numpy()
        w = class_weights(t) if i < 2 else np.ones(len(t))
        blocks.append((scale(d, st), t, w))
    ins = blocks[0][0].shape[1]
    with open("PROBLEM", "w") as f:
        f.write(f"{ins}\n1\n{len(blocks[0][0])}\n{len(blocks[1][0])}\n{len(blocks[2][0])}\n")
        for X, t, w in blocks:
            for xi, ti, wi in zip(X, t, w):
                f.write(" ".join(f"{v:.5f}" for v in xi) + "\n")
                f.write(f"{int(ti)}\n{wi:.5f}\n")
    print("Successfully generated 'PROBLEM'")
    print(f"Inputs: {ins} | train {len(blocks[0][0])} | validation {len(blocks[1][0])} | "
          f"test {len(blocks[2][0])} | positives in test: {int(blocks[2][1].sum())}")
    print("Next: run backprop.exe, then  python test.py score")


# ---------------------------------------------------------------------- scoring
def auc(t, s):
    r = pd.Series(np.concatenate([s[t == 1], s[t == 0]])).rank().to_numpy()
    n1, n0 = (t == 1).sum(), (t == 0).sum()
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float("nan")


def metrics(t, s, thr):
    p = (s >= thr).astype(int)
    tp, fp = int(((t == 1) & (p == 1)).sum()), int(((t == 0) & (p == 1)).sum())
    tn, fn = int(((t == 0) & (p == 0)).sum()), int(((t == 1) & (p == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return dict(TP=tp, FP=fp, TN=tn, FN=fn, accuracy=(tp + tn) / len(t), precision=pr, recall=rc,
                f1=2 * pr * rc / (pr + rc) if pr + rc else 0.0, auc=auc(t, s))


def score():
    cfg = read_settings()
    pr = pd.read_csv("predictions.csv")
    va, te = pr[pr.set == "V"], pr[pr.set == "T"]
    tv, sv, tt, st_ = (va.target.to_numpy(), va.score.to_numpy(),
                       te.target.to_numpy(), te.score.to_numpy())
    thr = max(np.unique(np.round(sv, 4)), key=lambda c: metrics(tv, sv, c)["f1"])
    print(f"Run: {run_id(cfg)}")
    print(f"Decision threshold chosen on VALIDATION (max F1): {thr:.4f}\n")
    res = {}
    for name, (t, s) in {"VALIDATION": (tv, sv), "TEST (never seen)": (tt, st_)}.items():
        m = metrics(t, s, thr)
        res[name[:4]] = m
        print(f"{name}:  n={len(t)}")
        print(f"  confusion  TP={m['TP']} FP={m['FP']} TN={m['TN']} FN={m['FN']}")
        print(f"  accuracy {m['accuracy']:.3f}  precision {m['precision']:.3f}  "
              f"recall {m['recall']:.3f}  F1 {m['f1']:.3f}  AUC {m['auc']:.3f}\n")
    print(f"Baseline 'always predict 0' accuracy on test: {np.mean(tt == 0):.3f}  (AUC of chance = 0.500)")

    ep = best = None
    curve = None
    try:
        lg = pd.read_csv("trainlog.csv")
        ep, best = len(lg), int(lg.val_wmse.idxmin()) + 1
        print(f"Epochs trained: {ep}   best epoch (lowest validation error): {best}")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(lg.epoch, lg.train_wmse, label="train")
        plt.plot(lg.epoch, lg.val_wmse, label="validation")
        plt.axvline(best, ls="--", c="gray", label="best epoch")
        plt.xlabel("epoch"); plt.ylabel("weighted MSE"); plt.legend(); plt.title(run_id(cfg), fontsize=8)
        plt.tight_layout()
        curve = f"curve_{run_id(cfg)}.png"
        plt.savefig(curve, dpi=150)
        print(f"Saved {curve}")
    except Exception as e:
        print("(no curve:", e, ")")

    row = {"run": run_id(cfg), **{k: cfg[k] for k in ("hidden", "kappa", "phi", "theta", "mu", "jump",
           "epochs", "patience", "seed", "split", "restore")}, "epochs_trained": ep, "best_epoch": best,
           "threshold": thr}
    for tag, key in (("val", "VALI"), ("test", "TEST")):
        for k in ("TP", "FP", "TN", "FN", "accuracy", "precision", "recall", "f1", "auc"):
            row[f"{tag}_{k}"] = round(res[key][k], 4) if isinstance(res[key][k], float) else res[key][k]
    log = pd.read_csv("results_log.csv") if os.path.exists("results_log.csv") else pd.DataFrame()
    if len(log) and "run" in log:
        log = log[log["run"] != row["run"]]          # re-running the same settings replaces its row
    pd.concat([log, pd.DataFrame([row])], ignore_index=True).to_csv("results_log.csv", index=False)
    print("Logged to results_log.csv")


if __name__ == "__main__":
    score() if len(sys.argv) > 1 and sys.argv[1] == "score" else prepare()
