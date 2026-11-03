"""
Dataset loaders for the revision experiments.

Every loader returns (X: DataFrame of numeric features, y: Series of string labels) for one
(task, seed). Class balancing is a per-class cap drawn with the run's seed; nothing here looks
at a train/test split, so there is no information flow from test rows into preprocessing.

Raw data lives outside the repo (large files); set REVISION_DATA to override.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("REVISION_DATA", "/mnt/Projects/datasets")
CACHE = os.path.join(DATA_ROOT, "_cache")

PREBUILT_DIR = os.path.join(REPO, "legacy", "Datasets", "IOT Top 20 features datasets")

# Device order used in the manuscript (Table 1).
NBAIOT_DEVICES = {
    "1": "Danmini_Doorbell",
    "2": "Ecobee_Thermostat",
    "3": "Ennio_Doorbell",
    "4": "Philips_B120N10_Baby_Monitor",
    "5": "Provision_PT_737E_Security_Camera",
    "6": "Provision_PT_838_Security_Camera",
    "7": "Samsung_SNH_1011_N_Webcam",
    "8": "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "9": "SimpleHome_XCS7_1003_WHT_Security_Camera",
}
NBAIOT_CAP = 20000          # per class, as in the manuscript
POOL_CAP = 60000            # per class kept in the on-disk cache (drawn once, seed 12345)
EDGE_CAP = 10000
CIC_CAP = 10000


def _cap(X, y, cap, seed):
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        keep.append(rng.choice(idx, size=min(cap, len(idx)), replace=False))
    keep = np.sort(np.concatenate(keep))
    return X.iloc[keep].reset_index(drop=True), pd.Series(np.asarray(y)[keep], name="label")


# ---------------------------------------------------------------------------- N-BaIoT
def nbaiot_prebuilt(device: str):
    df = pd.read_csv(os.path.join(PREBUILT_DIR, f"device{device}_top_20_features.csv"))
    return df.drop(columns=["label"]), df["label"].astype(str)


def _nbaiot_raw_dir(device):
    base = os.path.join(DATA_ROOT, "nbaiot", "extracted")
    cands = glob.glob(os.path.join(base, "**", NBAIOT_DEVICES[device]), recursive=True)
    if not cands:
        raise FileNotFoundError(f"raw N-BaIoT folder for device {device} not under {base}")
    return cands[0]


def nbaiot_raw_pool(device: str):
    """All 115 features. Returns a per-class pool (<= POOL_CAP rows per class), cached on disk."""
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, f"nbaiot_dev{device}_pool.pkl")
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    d = _nbaiot_raw_dir(device)
    frames = []
    for f in sorted(glob.glob(os.path.join(d, "**", "*.csv"), recursive=True)):
        rel = os.path.relpath(f, d).replace(os.sep, "/").lower()
        stem = os.path.splitext(os.path.basename(f))[0].lower()
        if stem == "benign_traffic":
            lab = "benign"
        elif "gafgyt" in rel:
            lab = f"gafgyt_{stem}"
        elif "mirai" in rel:
            lab = f"mirai_{stem}"
        else:
            continue
        df = pd.read_csv(f, dtype=np.float32)
        if len(df) > POOL_CAP:
            df = df.sample(POOL_CAP, random_state=12345)
        df["label"] = lab
        frames.append(df)
    pool = pd.concat(frames, ignore_index=True)
    pool.to_pickle(cache)
    return pool


def nbaiot_raw(device: str, seed: int):
    pool = nbaiot_raw_pool(device)
    return _cap(pool.drop(columns=["label"]), pool["label"].to_numpy(), NBAIOT_CAP, seed)


# ---------------------------------------------------------------------------- Edge-IIoTset
EDGE_DROP = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
             "http.file_data", "http.request.full_uri", "icmp.transmit_timestamp",
             "http.request.uri.query", "tcp.options", "tcp.payload", "tcp.srcport", "tcp.dstport",
             "udp.port", "mqtt.msg", "Attack_label"]
EDGE_ONEHOT = ["http.request.method", "http.referer", "http.request.version", "dns.qry.name.len",
               "mqtt.conack.flags", "mqtt.protoname", "mqtt.topic"]


def _safe_names(cols):
    import re
    out, seen = [], set()
    for c in cols:
        n = re.sub(r"[^0-9A-Za-z]+", "_", c).strip("_")[:40] or "f"
        base, k = n, 1
        while n in seen:
            k += 1
            n = f"{base}_{k}"
        seen.add(n)
        out.append(n)
    return out


def edgeiiot_full():
    """Preprocessing follows the dataset authors' published ML notebook: drop identifiers and
    free-text payload columns, drop duplicates / missing rows, one-hot the categorical columns."""
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, "edgeiiot_clean.pkl")
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    df = pd.read_csv(os.path.join(DATA_ROOT, "edgeiiot", "ML-EdgeIIoT-dataset.csv"), engine="pyarrow")
    df = df.drop(columns=[c for c in EDGE_DROP if c in df.columns])
    df = df.dropna().drop_duplicates()
    y = df.pop("Attack_type").astype(str)
    cats = [c for c in EDGE_ONEHOT if c in df.columns]
    df = pd.get_dummies(df, columns=cats, dtype=np.float32)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.loc[:, df.notna().all()]
    df = df.loc[:, df.nunique() > 1].astype(np.float32)
    df.columns = _safe_names(df.columns)
    df["label"] = y.values
    df.reset_index(drop=True).to_pickle(cache)
    return df


def edgeiiot(seed):
    df = edgeiiot_full()
    return _cap(df.drop(columns=["label"]), df["label"].to_numpy(), EDGE_CAP, seed)


# ---------------------------------------------------------------------------- CIC-IoT-2023
CIC_GROUPS = {
    "Benign": ["Benign_Final"],
    "DDoS": ["DDoS-ACK_Fragmentation", "DDoS-HTTP_Flood", "DDoS-ICMP_Flood", "DDoS-ICMP_Fragmentation",
             "DDoS-PSHACK_FLOOD", "DDoS-RSTFINFLOOD", "DDoS-SYN_Flood", "DDoS-SlowLoris",
             "DDoS-SynonymousIP_Flood", "DDoS-TCP_Flood", "DDoS-UDP_Flood", "DDoS-UDP_Fragmentation"],
    "DoS": ["DoS-HTTP_Flood", "DoS-SYN_Flood", "DoS-TCP_Flood", "DoS-UDP_Flood"],
    "Mirai": ["Mirai-greeth_flood", "Mirai-greip_flood", "Mirai-udpplain"],
    "Recon": ["Recon-HostDiscovery", "Recon-OSScan", "Recon-PingSweep", "Recon-PortScan",
              "VulnerabilityScan"],
    "Spoofing": ["DNS_Spoofing", "MITM-ArpSpoofing"],
    "Web": ["Backdoor_Malware", "BrowserHijacking", "CommandInjection", "SqlInjection",
            "Uploading_Attack", "XSS"],
    "BruteForce": ["DictionaryBruteForce"],
}


def ciciot_full():
    """8-category grouping of the 34 CIC-IoT-2023 classes (as in the dataset paper).
    Within a category, sub-attacks are sampled evenly so no single sub-attack dominates."""
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, "ciciot2023_pool.pkl")
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    base = os.path.join(DATA_ROOT, "ciciot2023", "CSV")
    rng = np.random.default_rng(12345)
    frames = []
    for group, subs in CIC_GROUPS.items():
        per_sub = int(np.ceil(3 * CIC_CAP / len(subs)))       # pool 3x the cap per category
        for s in subs:
            files = sorted(glob.glob(os.path.join(base, s, "*.csv")))
            if not files:
                raise FileNotFoundError(f"CIC-IoT-2023 folder missing: {s}")
            df = pd.concat([pd.read_csv(f, dtype=np.float32) for f in files], ignore_index=True)
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            if len(df) > per_sub:
                df = df.iloc[rng.choice(len(df), per_sub, replace=False)]
            df["label"] = group
            df["sub_label"] = s
            frames.append(df)
    pool = pd.concat(frames, ignore_index=True)
    pool.columns = [c.strip().replace(" ", "_") for c in pool.columns]
    pool.to_pickle(cache)
    return pool


def ciciot(seed):
    df = ciciot_full()
    return _cap(df.drop(columns=["label", "sub_label"]), df["label"].to_numpy(), CIC_CAP, seed)


# ---------------------------------------------------------------------------- round 3
# Round-3 datasets are deduplicated during data preparation: rows identical in every feature
# *and* the label are dropped, so copies of a test row cannot sit in the training split.
# Identical feature vectors with different labels are kept; they are part of the data.
CIC34_CAP = 5000            # smallest class (Uploading_Attack) has about 1,250 rows
UNSW_CAP = 10000


def ciciot34_full():
    """CIC-IoT-2023 with all 34 sub-attack labels, deduplicated."""
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, "ciciot2023_34_dedup_pool.pkl")
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    base = os.path.join(DATA_ROOT, "ciciot2023", "CSV")
    rng = np.random.default_rng(12345)
    frames = []
    for s in sorted(sub for subs in CIC_GROUPS.values() for sub in subs):
        files = sorted(glob.glob(os.path.join(base, s, "*.csv")))
        if not files:
            raise FileNotFoundError(f"CIC-IoT-2023 folder missing: {s}")
        df = pd.concat([pd.read_csv(f, dtype=np.float32) for f in files], ignore_index=True)
        df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
        if len(df) > 3 * CIC34_CAP:                            # pool 3x the cap per class
            df = df.iloc[rng.choice(len(df), 3 * CIC34_CAP, replace=False)]
        df["label"] = s
        frames.append(df)
    pool = pd.concat(frames, ignore_index=True)
    pool.columns = [c.strip().replace(" ", "_") for c in pool.columns]
    pool.to_pickle(cache)
    return pool


def ciciot34(seed):
    df = ciciot34_full()
    return _cap(df.drop(columns=["label"]), df["label"].to_numpy(), CIC34_CAP, seed)


def unswnb15_full():
    """UNSW-NB15, official training + testing files pooled, 10 attack categories, deduplicated.
    proto / service / state are ordinal-coded from their sorted distinct values (label-free)."""
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, "unswnb15_dedup.pkl")
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    base = os.path.join(DATA_ROOT, "unswnb15")
    df = pd.concat([pd.read_csv(os.path.join(base, f), encoding="utf-8-sig") for f in ("train.csv", "test.csv")],
                   ignore_index=True)
    label = df["attack_cat"].astype(str).str.strip()
    df = df.drop(columns=["id", "label", "attack_cat"])
    for c in df.select_dtypes("object").columns:
        v = df[c].astype(str)
        df[c] = pd.Categorical(v, categories=sorted(v.unique())).codes
    df = df.astype(np.float32)
    df["label"] = label.to_numpy()
    df = df.drop_duplicates().reset_index(drop=True)
    df.to_pickle(cache)
    return df


def unswnb15(seed):
    df = unswnb15_full()
    return _cap(df.drop(columns=["label"]), df["label"].to_numpy(), UNSW_CAP, seed)


# ---------------------------------------------------------------------------- dispatcher
def load(task, seed):
    if task.dataset == "nbaiot":
        if task.feature_mode == "prebuilt20":
            return nbaiot_prebuilt(task.unit)     # already balanced, 20k/class, fixed rows
        return nbaiot_raw(task.unit, seed)
    if task.dataset == "edgeiiot":
        return edgeiiot(seed)
    if task.dataset == "ciciot2023":
        return ciciot(seed)
    if task.dataset == "ciciot34":
        return ciciot34(seed)
    if task.dataset == "unswnb15":
        return unswnb15(seed)
    raise ValueError(task.dataset)
