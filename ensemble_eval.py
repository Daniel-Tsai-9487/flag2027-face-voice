"""在本地驗證集上評估多個第一階段模型的集成（平均距離）。

用法：ensemble_eval.py tagA,tagB,tagC
各模型用自己的 stage1_best.pt、特徵組與標準化統計量；驗證配對與 train_fop.py 相同（說話者切分 seed 0、配對 seed 0）。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "baseline"))
sys.path.insert(0, str(ROOT))
import flag_data  # noqa: E402
from make_submission import distances, load_model  # noqa: E402

_cache = {}


def val_pairs_raw(features, voice_features, n_val):
    """未標準化的驗證配對；make_val_pairs 只依說話者與 seed 取列，與特徵值無關，所以能先配對再標準化。"""
    key = (features, voice_features, n_val)
    if key not in _cache:
        face, voice, ids, gender = flag_data.load_train(features, voice_features)
        _, va_spk = flag_data.split_speakers(ids, gender, n_val, seed=0)
        va = np.isin(ids, va_spk)
        _cache[key] = flag_data.make_val_pairs(face[va], voice[va], ids[va], gender, seed=0)
    return _cache[key]


def main():
    tags = [t.strip() for t in sys.argv[1].split(",") if t.strip()]
    per_model, label = [], {}
    for t in tags:
        model, ck = load_model(ROOT / "runs" / t / "stage1_best.pt")
        cfg = ck["cfg"]
        pairs = val_pairs_raw(cfg.get("features", "official"), cfg.get("voice_features"), cfg.get("n_val", 10))
        d = {}
        for track, (face, voice, lab) in pairs.items():
            d[track] = distances(model, ck, face, voice)
            if track in label and not np.array_equal(label[track], lab):
                raise SystemExit("各模型的驗證配對不一致")
            label[track] = lab
        per_model.append(d)
        print(f"{t:28s} " + "  ".join(f"{tr} {100 * flag_data.eer(label[tr], d[tr]):.1f}" for tr in flag_data.TRACKS)
              + f"  平均 {100 * np.mean([flag_data.eer(label[tr], d[tr]) for tr in flag_data.TRACKS]):.1f}")
    ens = {tr: np.mean([d[tr] for d in per_model], axis=0) for tr in flag_data.TRACKS}
    e = {tr: flag_data.eer(label[tr], ens[tr]) for tr in flag_data.TRACKS}
    print(f"{'集成（' + str(len(tags)) + ' 個）':26s} " + "  ".join(f"{tr} {100 * e[tr]:.1f}" for tr in flag_data.TRACKS)
          + f"  平均 {100 * np.mean(list(e.values())):.1f}")


if __name__ == "__main__":
    main()
