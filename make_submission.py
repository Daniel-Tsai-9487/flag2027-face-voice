"""用訓練好的 FOP 產生 Codabench 提交檔。

輸出 submissions/<名稱>.zip，內含 no_gender/ 與 gender/ 兩個資料夾，各有
sub_score_v4_English_heard.txt 與 sub_score_v4_Bangla_unheard.txt；每列「key 距離」，
距離越小越像同一人（與官方 baseline/computeScore.py、challenge/eval_submission.py 一致）。

--tag 可用逗號列多個 run，提交分數為各模型距離的平均（集成）。
每個 run 可寫成 名稱:權重 做加權平均，例如 a:0.3,b:0.7；沒寫權重的預設為 1。
"""
import argparse
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "baseline"))
sys.path.insert(0, str(ROOT))
import flag_data  # noqa: E402
from retrieval_model import FOP  # noqa: E402
from train_fop import apply_norm, pair_score  # noqa: E402


def load_model(path):
    """回傳 (模型, checkpoint)。輸入維度由權重形狀推得。"""
    ck = torch.load(path, map_location="cuda", weights_only=False)
    cfg, sd = ck["cfg"], ck["state_dict"]
    model = FOP(SimpleNamespace(dim_embed=cfg["dim_embed"], fusion=cfg["fusion"], cuda=True),
                sd["face_branch.fc1.0.weight"].shape[1], sd["voice_branch.fc1.0.weight"].shape[1],
                ck["n_class"]).cuda()
    model.load_state_dict(sd)
    model.eval()
    return model, ck


@torch.no_grad()
def distances(model, ck, face, voice):
    """原始特徵進、距離出：套用該 checkpoint 自己的標準化與分數設定。"""
    face, voice = apply_norm(ck.get("norm"), face, voice)
    _, f, v = model(torch.from_numpy(face).cuda(), torch.from_numpy(voice).cuda())
    return pair_score(f.cpu().numpy(), v.cpu().numpy(), ck["cfg"].get("score", "l2"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="fop_baseline", help="runs/ 底下的名稱，逗號分隔多個即集成")
    p.add_argument("--ckpt", default="final.pt", help="runs/<tag>/ 底下的檔名")
    p.add_argument("--out", default=None, help="輸出 zip 名稱，預設為 tag（多個時以 + 連接）；eval 會自動加 _eval")
    p.add_argument("--split", choices=["dev", "eval"], default="dev",
                   help="dev：Progress 階段；eval：Evaluation 階段（預設讀 data/eval_set）")
    p.add_argument("--split_dir", default=None, help="測試資料實際所在的資料夾，與預設不同時才需要")
    args = p.parse_args()

    specs = [t.strip() for t in args.tag.split(",") if t.strip()]
    if not specs:
        raise SystemExit("--tag 至少要有一個 run 名稱")
    tags = [s.split(":")[0] for s in specs]
    weights = np.asarray([float(s.split(":")[1]) if ":" in s else 1.0 for s in specs])
    if (weights < 0).any() or weights.sum() <= 0:
        raise SystemExit("權重不能為負，且總和要大於 0")
    models = [load_model(ROOT / "runs" / t / args.ckpt) for t in tags]

    name = args.out or "+".join(tags)
    if args.split == "eval" and not name.endswith("_eval"):
        name += "_eval"
    out = ROOT / "submissions" / f"{name}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for track in flag_data.TRACKS:
            for lang, cond in ((flag_data.HEARD, "heard"), (flag_data.UNHEARD, "unheard")):
                dists, keys = [], None
                for model, ck in models:
                    cfg = ck["cfg"]
                    k, face, voice = flag_data.load_dev(track, lang, cfg.get("features", "official"),
                                                        cfg.get("voice_features"), args.split, args.split_dir)
                    if keys is not None and k != keys:
                        raise SystemExit(f"{track}/{lang}: 各模型讀到的 key 順序不一致")
                    keys = k
                    dists.append(distances(model, ck, face, voice))
                dist = np.average(dists, axis=0, weights=weights)
                text = "".join(f"{k} {d:f}\n" for k, d in zip(keys, dist))
                zf.writestr(f"{track}/sub_score_v4_{lang}_{cond}.txt", text)
                print(f"{track}/sub_score_v4_{lang}_{cond}.txt  {len(keys)} 列  "
                      f"{len(models)} 個模型  距離平均 {dist.mean():.4f}")
    print(f"寫出 {out}")


if __name__ == "__main__":
    main()
