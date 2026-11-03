"""固定的性別一致項（Rules & FAQ 第 4、5 條）：只加在 no_gender 配對的距離上。

  d' = d + λ · m,  m = pf·(1 − pv) + (1 − pf)·pv
  pf：臉部男性機率，來自公開、凍結的 insightface genderage 模型（extract_attributes.py --only face）。
  pv：語音男性機率，來自只用官方訓練資料訓練的邏輯迴歸（WavLM Base+ SV 說話者向量，訓練集統計量標準化）。

λ 與分類器都只用訓練資料決定：
  fit   ：在 10 位保留驗證說話者上（分類器只用其餘 60 位訓練），用 20 個模型第一階段 checkpoint 的距離挑 λ；
          再用全部 70 位重訓分類器，連同 λ 存成 runs/gender_term.json。
  submit：讀取固定的 json，對 no_gender 兩格加上一致項，gender 兩格不變（負例同性別，此項只會引入誤判）。
不從 dev／eval 資料推出任何參數或統計量；每個配對獨立評分。

需要的特徵：v2 語音（extract_features.py --only voices）與臉部屬性（extract_attributes.py --only face），
train 給 fit 用，dev／eval 給 submit 用。Evaluation 當天由 eval_day.py 自動補齊並呼叫 submit。
  .venv/Scripts/python -I -X utf8 gender_term.py fit
  .venv/Scripts/python -I -X utf8 gender_term.py submit --split eval --out final
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "baseline"))
sys.path.insert(0, str(ROOT))
import flag_data  # noqa: E402
from make_submission import distances, load_model  # noqa: E402

CFG = ROOT / "runs" / "gender_term.json"
V2 = flag_data.V2
C_REG = 0.1


def fit_voice_gender(voice, is_male):
    from sklearn.linear_model import LogisticRegression

    mu, sd = voice.mean(0), voice.std(0) + 1e-6
    m = LogisticRegression(C=C_REG, max_iter=5000).fit((voice - mu) / sd, is_male)
    return {"mean": mu.tolist(), "std": sd.tolist(), "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0])}


def predict_voice_gender(clf, voice):
    z = (voice - np.asarray(clf["mean"])) / np.asarray(clf["std"])
    return 1.0 / (1.0 + np.exp(-(z @ np.asarray(clf["coef"]) + clf["intercept"])))


def mismatch(pf, pv):
    return pf * (1 - pv) + (1 - pf) * pv


def read_models(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            tag, w = (line.split() + ["1"])[:2]
            out.append((tag, float(w)))
    return out


def ensemble_dist(models, face_by_set, voice_by_set, rows_f, rows_v, weights):
    ds = []
    for model, ck in models:
        cfg = ck["cfg"]
        fs, vs = cfg.get("features", "official"), cfg.get("voice_features") or cfg.get("features", "official")
        ds.append(distances(model, ck, face_by_set[fs][rows_f], voice_by_set[vs][rows_v]))
    return np.average(ds, axis=0, weights=weights)


def fit(args):
    specs = read_models(args.models)
    models = [load_model(ROOT / "runs" / t / "stage1_best.pt") for t, _ in specs]
    weights = [w for _, w in specs]
    sets = {(m[1]["cfg"].get("features", "official"), m[1]["cfg"].get("voice_features") or m[1]["cfg"].get("features", "official")) for m in models}
    face_by, voice_by = {}, {}
    for fs, vs in sets:
        f, v, ids, gender = flag_data.load_train(fs, vs)
        face_by[fs], voice_by[vs] = f, v
    _, _, ids, gender = flag_data.load_train("official", "v2")
    v2_voice = flag_data.load_train("official", "v2")[1]
    face_attr = np.load(V2 / "train_face_attr.npy")
    is_male = np.array([gender[s] == "m" for s in ids]).astype(int)

    tr_spk, va_spk = flag_data.split_speakers(ids, gender, 10, seed=0)
    tr = np.isin(ids, tr_spk)
    va_rows = np.flatnonzero(~tr)
    clf60 = fit_voice_gender(v2_voice[tr], is_male[tr])

    # 驗證配對：與 train_fop 相同的切分與配對（no_gender 賽道），取列索引
    idx = np.arange(len(ids))
    pairs = flag_data.make_val_pairs(idx[va_rows][:, None].astype(np.float32), idx[va_rows][:, None].astype(np.float32),
                                     ids[va_rows], gender, seed=0)
    f_rows = pairs["no_gender"][0][:, 0].astype(int)
    v_rows = pairs["no_gender"][1][:, 0].astype(int)
    label = pairs["no_gender"][2]
    d = ensemble_dist(models, face_by, voice_by, f_rows, v_rows, weights)
    m = mismatch(face_attr[f_rows, 0], predict_voice_gender(clf60, v2_voice[v_rows]))
    print(f"驗證配對 {len(label)} 組；臉部性別準確率 {np.mean((face_attr[f_rows,0]>0.5)==is_male[f_rows]):.3f}；"
          f"語音性別準確率（分類器沒看過這些人）{np.mean((predict_voice_gender(clf60, v2_voice[v_rows])>0.5)==is_male[v_rows]):.3f}")
    grid = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    res = {lam: flag_data.eer(label, d + lam * m) for lam in grid}
    for lam, e in res.items():
        print(f"  λ={lam:<5} no_gender 驗證 EER {100*e:.2f}")
    lam = min(grid, key=lambda x: (res[x], x))
    clf70 = fit_voice_gender(v2_voice, is_male)
    cfg_path = Path(args.cfg)
    cfg_path.write_text(json.dumps({"lambda": lam, "val_eer": {str(k): v for k, v in res.items()},
                               "voice_gender_clf": clf70, "models": specs}, ensure_ascii=False), encoding="utf-8")
    print(f"選定 λ={lam}，寫出 {cfg_path}")


def submit(args):
    cfg = json.loads(Path(args.cfg).read_text(encoding="utf-8"))
    lam, clf = cfg["lambda"], cfg["voice_gender_clf"]
    specs = read_models(args.models)
    if [list(x) for x in specs] != [list(x) for x in cfg["models"]]:
        raise SystemExit("模型清單或權重與 fit 時不同；λ 是針對 fit 時的組合決定的，請重新 fit")
    models = [load_model(ROOT / "runs" / t / args.ckpt) for t, _ in specs]
    weights = np.asarray([w for _, w in specs])
    name = args.out
    if args.split == "eval" and not name.endswith("_eval"):  # 與 make_submission.py 相同的命名
        name += "_eval"
    out = ROOT / "submissions" / f"{name}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for track in flag_data.TRACKS:
            for lang, cond in ((flag_data.HEARD, "heard"), (flag_data.UNHEARD, "unheard")):
                ds, keys = [], None
                for model, ck in models:
                    c = ck["cfg"]
                    k, face, voice = flag_data.load_dev(track, lang, c.get("features", "official"), c.get("voice_features"),
                                                        args.split, args.split_dir)
                    keys = keys or k
                    ds.append(distances(model, ck, face, voice))
                d = np.average(ds, axis=0, weights=weights)
                if track == "no_gender":
                    pre = f"{args.split}_{track}_{lang}"
                    attr = V2 / f"{pre}_face_attr.npy"
                    if not attr.exists():
                        raise SystemExit(f"缺少 {attr}；請先執行 extract_attributes.py --only face --splits {args.split}")
                    pf = np.load(attr)[:, 0]
                    _, _, v2 = flag_data.load_dev(track, lang, "official", "v2", args.split, args.split_dir)
                    if not len(pf) == len(v2) == len(keys):
                        raise SystemExit(f"{pre}: 臉部屬性 {len(pf)}、v2 語音 {len(v2)}、配對 {len(keys)} 列數不一致")
                    d = d + lam * mismatch(pf, predict_voice_gender(clf, v2))
                zf.writestr(f"{track}/sub_score_v4_{lang}_{cond}.txt", "".join(f"{k} {x:f}\n" for k, x in zip(keys, d)))
                print(f"{track}/{lang}: {len(keys)} 列" + (f"（加性別項 λ={lam}）" if track == "no_gender" else ""))
    print(f"寫出 {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["fit", "submit"])
    p.add_argument("--models", default=str(ROOT / "final_models.txt"))
    p.add_argument("--split", choices=["dev", "eval"], default="dev")
    p.add_argument("--split_dir", default=None)
    p.add_argument("--out", default="ens20_gender")
    p.add_argument("--cfg", default=str(CFG), help="λ 與分類器的存放檔；試其他模型組合時另指定，以免覆蓋最終系統的設定")
    p.add_argument("--ckpt", default="final.pt", help="submit 用的 runs/<tag>/ 檔名（fit 固定用 stage1_best.pt）")
    args = p.parse_args()
    fit(args) if args.cmd == "fit" else submit(args)


if __name__ == "__main__":
    main()
