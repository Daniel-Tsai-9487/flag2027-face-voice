"""FLAG 2027 資料讀取：官方預先擷取特徵（VGGFace 4096 維、ECAPA 192 維）、
dev／eval 配對清單、依說話者切出的本地驗證集。"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = DATA / "cache"

FACE_DIM, VOICE_DIM = 4096, 192
TRACKS = ("no_gender", "gender")
HEARD, UNHEARD = "English", "Bangla"


def _find(name):
    """在 data/ 底下找唯一一個檔名相符的檔案，避免寫死解壓後的資料夾層級。"""
    hits = [p for p in DATA.rglob(name) if "downloads" not in p.parts and "cache" not in p.parts]
    if len(hits) != 1:
        raise FileNotFoundError(f"{name}: 找到 {len(hits)} 個，預期 1 個 {hits}")
    return hits[0]


# 測試配對清單所在的資料夾。eval 是 Evaluation 階段（2026-11-11 釋出）的輸入，
# 假設解壓後與 dev_set 結構相同；實際資料夾名稱不同時，用 split_dir 參數指定。
SPLIT_DIRS = {"dev": DATA / "dev_set", "eval": DATA / "eval_set"}


def split_root(split, split_dir=None):
    root = Path(split_dir) if split_dir else SPLIT_DIRS[split]
    if not root.is_dir():
        raise SystemExit(f"找不到 {split} 資料夾 {root}；請解壓到這裡，或用 --split_dir 指定")
    return root


def _find_in(root, track, name):
    """在 root 底下找 track 資料夾（gender 或 no_gender）中唯一一個名為 name 的檔案。"""
    hits = [p for p in root.rglob(name) if track in p.relative_to(root).parts]
    if len(hits) != 1:
        raise FileNotFoundError(f"{root}/{track}/{name}: 找到 {len(hits)} 個，預期 1 個 {hits}")
    return hits[0]


def _load_csv(path, cache_name):
    """讀 CSV 並快取成 .npy。快取檔名含來源路徑、大小與修改時間的雜湊，
    換了來源檔（例如 eval 資料解壓到別處、或重新下載）就不會誤用舊快取。"""
    import hashlib

    CACHE.mkdir(parents=True, exist_ok=True)
    st = path.stat()
    tag = hashlib.md5(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:8]
    npy = CACHE / f"{cache_name}_{tag}.npy"
    if npy.exists():
        return np.load(npy)
    arr = pd.read_csv(path, header=None).to_numpy(dtype=np.float32)
    np.save(npy, arr)
    return arr


V2 = DATA / "features_v2"
# 特徵組以「+」串接多個來源：official（官方 VGGFace／ECAPA）、v2（ArcFace／WavLM 說話者向量）、
# clip（CLIP 影像嵌入，只有臉部）、wavlmh（WavLM 隱藏層平均，只有語音）。concat 是 official+v2 的舊名。
# clipl、wavlml 是同樣做法的大模型版（CLIP ViT-L/14、WavLM Large）。
FEATURE_PARTS = ("official", "v2", "clip", "wavlmh", "clipl", "wavlml")
FACE_ONLY, VOICE_ONLY = {"clip", "clipl"}, {"wavlmh", "wavlml"}
_SUFFIX = {"clip": "_clip", "wavlmh": "_wavlmh", "clipl": "_clipl", "wavlml": "_wavlml"}


def check_feature_set(s):
    parts = ("official+v2" if s == "concat" else s).split("+")
    if not parts or any(p not in FEATURE_PARTS for p in parts):
        raise SystemExit(f"不認得的特徵組 {s}；可用 {FEATURE_PARTS} 以 + 串接，或 concat")
    return parts


def _combine(official, v2_name, feature_set):
    """依 feature_set 逐列串接各來源。v2、clip 的列順序與官方清單相同（由 extract_features.py 保證）。"""
    out = []
    for part in check_feature_set(feature_set):
        if part == "official":
            arr = official
        else:
            name = v2_name if part == "v2" else v2_name.replace(".npy", f"{_SUFFIX[part]}.npy")
            if not (V2 / name).exists():
                split = name.split("_")[0]
                only = ("faces" if "_faces" in name else "voices") if part == "v2" else part
                raise SystemExit(f"缺少 {V2 / name}；請先執行 extract_features.py --only {only} --splits {split}")
            arr = np.load(V2 / name).astype(np.float32)
            if len(arr) != len(official):
                raise SystemExit(f"{name} 有 {len(arr)} 列，官方特徵有 {len(official)} 列")
        out.append(arr)
    return out[0] if len(out) == 1 else np.hstack(out)


def load_train(feature_set="official", voice_set=None):
    """回傳 face、voice、說話者 id 字串 (N,)、性別對照 dict。
    feature_set 決定臉部特徵；voice_set 未指定時與臉部相同。"""
    voice_set = voice_set or feature_set
    face = _load_csv(_find("train_English_faces.csv"), "train_faces")
    voice = _load_csv(_find("train_English_voices.csv"), "train_voices")
    if not np.array_equal(face[:, -1], voice[:, -1]):
        raise SystemExit("臉部與語音 CSV 的標籤欄不一致，兩檔不是逐列對齊")
    labels = face[:, -1].astype(np.int64)
    face = _combine(face[:, :FACE_DIM], "train_faces.npy", feature_set)
    voice = _combine(voice[:, :VOICE_DIM], "train_voices.npy", voice_set)

    # train_English.txt 每列：key、1、語音路徑、臉部路徑、說話者 id、類別編號
    ids = []
    with open(_find("train_English.txt"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.append(line.split()[4])
    ids = np.asarray(ids)
    if len(ids) != len(labels):
        raise SystemExit(f"train_English.txt 有 {len(ids)} 列，特徵有 {len(labels)} 列")
    # 同一個數字標籤必須只對應一個說話者 id
    for lab in np.unique(labels):
        if len(set(ids[labels == lab])) != 1:
            raise SystemExit(f"標籤 {lab} 對應到多個說話者 id，列順序與清單不一致")

    meta = pd.read_csv(DATA / "downloads" / "meta_file_train_set.csv", header=None, names=["id", "gender"])
    gender = dict(zip(meta["id"].str.strip(), meta["gender"].str.strip()))
    return face, voice, ids, gender


def split_speakers(ids, gender, n_val=10, seed=0):
    """依性別分層抽 n_val 位說話者作本地驗證，其餘訓練。"""
    rng = np.random.default_rng(seed)
    speakers = sorted(set(ids))
    val = []
    for g in ("f", "m"):
        pool = [s for s in speakers if gender.get(s) == g]
        val += list(rng.choice(pool, n_val // 2, replace=False))
    return sorted(set(speakers) - set(val)), sorted(val)


def make_val_pairs(face, voice, ids, gender, seed=0):
    """每個驗證樣本的語音配一張同人（不同樣本）的臉作正例、一張他人的臉作負例。
    no_gender 的負例可為任何人；gender 的負例限同性別，對應官方兩個賽道。
    回傳 {track: (face_rows, voice_rows, label)}，label 1 表示同一人。"""
    rng = np.random.default_rng(seed)
    # 排序後再建表：set 的走訪順序依程序的雜湊種子而變，不排序就無法重現配對
    by_spk = {s: np.flatnonzero(ids == s) for s in sorted(set(ids))}
    out = {}
    for track in TRACKS:
        f_idx, v_idx, lab = [], [], []
        for i in range(len(ids)):
            s = ids[i]
            same = by_spk[s][by_spk[s] != i]
            others = [t for t in by_spk if t != s and (track == "no_gender" or gender[t] == gender[s])]
            f_idx += [rng.choice(same), rng.choice(by_spk[rng.choice(others)])]
            v_idx += [i, i]
            lab += [1, 0]
        out[track] = (face[f_idx], voice[v_idx], np.asarray(lab))
    return out


def load_dev(track, lang, feature_set="official", voice_set=None, split="dev", split_dir=None):
    """回傳測試配對（dev 或 eval）的 keys、face、voice。
    快取與 features_v2 的檔名以 split 開頭（dev_*、eval_*），與 extract_features.py 一致。"""
    voice_set = voice_set or feature_set
    root = split_root(split, split_dir)
    keys = []
    with open(_find_in(root, track, f"{lang}_test.txt"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                keys.append(line.split()[0])
    pre = f"{split}_{track}_{lang}"
    face = _load_csv(_find_in(root, track, f"{lang}_test_faces.csv"), f"{pre}_faces")[:, :FACE_DIM]
    voice = _load_csv(_find_in(root, track, f"{lang}_test_voices.csv"), f"{pre}_voices")[:, :VOICE_DIM]
    face = _combine(face, f"{pre}_faces.npy", feature_set)
    voice = _combine(voice, f"{pre}_voices.npy", voice_set)
    if not (len(keys) == len(face) == len(voice)):
        raise SystemExit(f"{track}/{lang}: keys {len(keys)}、face {len(face)}、voice {len(voice)} 列數不一致")
    return keys, face, voice


def eer(label, dist):
    """官方 eval_submission.py 的算法：分數為距離，越小越像同一人。"""
    from scipy.interpolate import interp1d
    from scipy.optimize import brentq
    from sklearn import metrics

    fpr, tpr, _ = metrics.roc_curve(label, -np.asarray(dist))
    return brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
