"""官方 FOP baseline 訓練（沿用 baseline/retrieval_model.py 的模型與 baseline/main.py 的設定）。

第一階段：保留 10 位說話者作本地驗證，以兩賽道驗證 EER 平均值早停，得到最佳 epoch 數。
第二階段：用全部 70 位說話者重訓同樣 epoch 數，存成 runs/<tag>/final.pt 供產生提交檔。
"""
import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "baseline"))
import flag_data  # noqa: E402
from retrieval_model import FOP  # noqa: E402


class OrthogonalProjectionLoss(nn.Module):
    """與 baseline/main.py 相同。"""

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        labels = labels[:, None]
        mask = torch.eq(labels, labels.t()).bool()
        eye = torch.eye(mask.shape[0], mask.shape[1], dtype=torch.bool, device=mask.device)
        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot = features @ features.t()
        pos = (mask_pos * dot).sum() / (mask_pos.sum() + 1e-6)
        neg = torch.abs(mask_neg * dot).sum() / (mask_neg.sum() + 1e-6)
        return (1.0 - pos) + 0.7 * neg


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def build_model(cfg, n_class, face_dim, voice_dim):
    model = FOP(SimpleNamespace(dim_embed=cfg.dim_embed, fusion=cfg.fusion, cuda=True),
                face_dim, voice_dim, n_class)
    model.apply(init_weights)
    return model.cuda()


def optimizer_for(model, cfg):
    # 與 baseline/main.py 相同的參數群組（gated fusion）
    groups = [{"params": model.face_branch.fc1.parameters()},
              {"params": model.voice_branch.fc1.parameters()},
              {"params": model.logits_layer.parameters()},
              {"params": model.fusion_layer.attention.parameters()}]
    return torch.optim.Adam(groups, lr=cfg.lr, weight_decay=0.01)


class CrossModalTripletLoss(nn.Module):
    """臉與語音嵌入之間的 batch-hard triplet（雙向）。官方 FOP 只訓練融合後的分類與 OPL，
    沒有直接拉近同人的臉與聲音；這一項直接優化測試時用的配對距離。
    neg='same_gender' 時，負例只取與錨點同性別的他人，對應 gender 賽道。"""

    def __init__(self, margin, neg):
        super().__init__()
        self.margin, self.neg = margin, neg

    def _one_side(self, sim, same, neg_mask):
        pos = (sim * same).sum(1) / same.sum(1)
        hard = sim.masked_fill(~neg_mask, float("-inf")).max(1).values
        ok = neg_mask.any(1)
        if not ok.any():  # 整批沒有可用負例時回傳零，避免空張量取平均得到 NaN
            return sim.sum() * 0.0
        return F.relu(self.margin + hard[ok] - pos[ok]).mean()

    def forward(self, face_emb, voice_emb, labels, gender):
        f, v = F.normalize(face_emb, dim=1), F.normalize(voice_emb, dim=1)
        sim = v @ f.t()  # 列：語音錨點；行：臉
        same = (labels[:, None] == labels[None, :]).float()
        neg_mask = labels[:, None] != labels[None, :]
        if self.neg == "same_gender":
            neg_mask &= gender[:, None] == gender[None, :]
        return 0.5 * (self._one_side(sim, same, neg_mask) + self._one_side(sim.t(), same, neg_mask))


def fit_norm(face, voice):
    return {"face_mean": face.mean(0), "face_std": face.std(0) + 1e-6,
            "voice_mean": voice.mean(0), "voice_std": voice.std(0) + 1e-6}


def apply_norm(norm, face, voice):
    if norm is None:
        return face, voice
    return ((face - norm["face_mean"]) / norm["face_std"]).astype(np.float32), \
           ((voice - norm["voice_mean"]) / norm["voice_std"]).astype(np.float32)


def pair_score(f, v, kind):
    """距離型分數，越小越像同一人。l2 與官方相同；cos 為 1 減餘弦相似度。"""
    if kind == "l2":
        return np.linalg.norm(f - v, axis=1)
    f = f / np.linalg.norm(f, axis=1, keepdims=True)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return 1.0 - (f * v).sum(1)


@torch.no_grad()
def embed(model, face, voice):
    model.eval()
    _, f, v = model(torch.from_numpy(face).cuda(), torch.from_numpy(voice).cuda())
    return f.cpu().numpy(), v.cpu().numpy()


def val_eer(model, val_pairs, score):
    """回傳以 score 計的兩賽道 EER 與平均，另附另一種分數的平均供比較。"""
    out = {}
    for track, (face, voice, label) in val_pairs.items():
        f, v = embed(model, face, voice)
        for kind in ("l2", "cos"):
            out[f"{track}_{kind}"] = flag_data.eer(label, pair_score(f, v, kind))
    for kind in ("l2", "cos"):
        out[f"mean_{kind}"] = float(np.mean([out[f"{t}_{kind}"] for t in flag_data.TRACKS]))
    for t in flag_data.TRACKS:
        out[t] = out[f"{t}_{score}"]
    out["mean"] = out[f"mean_{score}"]
    return out


def train(cfg, face, voice, labels, gender, n_epochs, val_pairs=None, log=None):
    """依 baseline/main.py：只在讀檔時洗牌一次、捨棄最後不足一批的資料。"""
    n_class = int(labels.max()) + 1
    model = build_model(cfg, n_class, face.shape[1], voice.shape[1])
    opt = optimizer_for(model, cfg)
    ce, opl = nn.CrossEntropyLoss(), OrthogonalProjectionLoss()
    xm = CrossModalTripletLoss(cfg.xm_margin, cfg.xm_neg)

    order = list(range(len(labels)))
    random.shuffle(order)
    face_t = torch.from_numpy(face[order]).cuda()
    voice_t = torch.from_numpy(voice[order]).cuda()
    lab_t = torch.from_numpy(labels[order]).cuda()
    gen_t = torch.from_numpy(gender[order]).cuda()
    n_batch = len(order) // cfg.batch_size

    best = {"mean": 1.0, "epoch": 0, "state": None}
    since = 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        total = 0.0
        for b in range(n_batch):
            sl = slice(b * cfg.batch_size, (b + 1) * cfg.batch_size)
            (feats, logits), f_emb, v_emb = model.train_forward(face_t[sl], voice_t[sl], lab_t[sl])
            loss = ce(logits, lab_t[sl]) + cfg.alpha * opl(feats, lab_t[sl])
            if cfg.xm_weight > 0:
                loss = loss + cfg.xm_weight * xm(f_emb, v_emb, lab_t[sl], gen_t[sl])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        msg = f"epoch {epoch:3d}  loss {total / n_batch:.4f}"
        if val_pairs is not None:
            r = val_eer(model, val_pairs, cfg.score)
            msg += (f"  val EER({cfg.score}) no_gender {r['no_gender']:.4f}  gender {r['gender']:.4f}"
                    f"  mean {r['mean']:.4f}  [l2 {r['mean_l2']:.4f} cos {r['mean_cos']:.4f}]")
            if r["mean"] < best["mean"] - cfg.min_delta:
                best = {**r, "epoch": epoch, "state": {k: v.clone() for k, v in model.state_dict().items()}}
                since = 0
            else:
                since += 1
        print(msg, flush=True)
        if log:
            log.write(msg + "\n")
            log.flush()
        if val_pairs is not None and since >= cfg.patience:
            print(f"early stop：{cfg.patience} 個 epoch 無改善，最佳 epoch {best['epoch']}", flush=True)
            break
    return model, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="fop_baseline")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_epoch", type=int, default=200)
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--dim_embed", type=int, default=128)
    p.add_argument("--fusion", default="gated")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--min_delta", type=float, default=0.0005)
    p.add_argument("--n_val", type=int, default=10)
    p.add_argument("--standardize", action="store_true", help="臉與語音特徵各自以訓練資料做 z-score")
    p.add_argument("--score", choices=["l2", "cos"], default="l2", help="驗證早停與提交用的分數")
    p.add_argument("--xm_weight", type=float, default=0.0, help="跨模態 triplet 損失權重，0 為關閉")
    p.add_argument("--xm_neg", choices=["any", "same_gender"], default="same_gender")
    p.add_argument("--xm_margin", type=float, default=0.2)
    p.add_argument("--no_stage2", action="store_true", help="只跑第一階段，比較設定時用")
    p.add_argument("--features", default="official",
                   help="臉部特徵組：official、v2（ArcFace）、clip，或以 + 串接，例如 official+clip；concat 即 official+v2")
    p.add_argument("--voice_features", default=None,
                   help="語音特徵組：official、v2（WavLM）或 official+v2；未指定時與 --features 相同")
    cfg = p.parse_args()
    face_parts = set(flag_data.check_feature_set(cfg.features))
    voice_parts = set(flag_data.check_feature_set(cfg.voice_features or cfg.features))
    if face_parts & flag_data.VOICE_ONLY:
        raise SystemExit(f"{face_parts & flag_data.VOICE_ONLY} 只有語音特徵，不能放在 --features（臉部）")
    if voice_parts & flag_data.FACE_ONLY:
        raise SystemExit(f"{voice_parts & flag_data.FACE_ONLY} 只有臉部特徵，--voice_features 不能含它；"
                         "臉部用到它時請另外指定 --voice_features")

    def seed_all():
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)

    run = ROOT / "runs" / cfg.tag
    run.mkdir(parents=True, exist_ok=True)
    face, voice, ids, gender = flag_data.load_train(cfg.features, cfg.voice_features)
    print(f"特徵 face {cfg.features}（{face.shape[1]} 維）、voice {cfg.voice_features or cfg.features}"
          f"（{voice.shape[1]} 維）", flush=True)
    tr_spk, va_spk = flag_data.split_speakers(ids, gender, cfg.n_val, seed=0)
    print(f"train {len(face)} 筆、{len(set(ids))} 位說話者；驗證說話者 {[str(s) for s in va_spk]}", flush=True)

    tr = np.isin(ids, tr_spk)
    va = ~tr
    gen01 = np.asarray([1 if gender[s] == "m" else 0 for s in ids], dtype=np.int64)
    norm1 = fit_norm(face[tr], voice[tr]) if cfg.standardize else None
    face1, voice1 = apply_norm(norm1, face, voice)
    val_pairs = flag_data.make_val_pairs(face1[va], voice1[va], ids[va], gender, seed=0)
    tr_labels = np.searchsorted(np.asarray(tr_spk), ids[tr])

    with open(run / "train_log.txt", "w", encoding="utf-8") as log:
        log.write(f"[stage1] {vars(cfg)}\n")
        seed_all()
        _, best = train(cfg, face1[tr], voice1[tr], tr_labels, gen01[tr], cfg.max_epoch, val_pairs, log)
        torch.save({"state_dict": best["state"], "n_class": len(tr_spk), "cfg": vars(cfg), "norm": norm1},
                   run / "stage1_best.pt")

        if not cfg.no_stage2:
            all_spk = np.asarray(sorted(set(ids)))
            all_labels = np.searchsorted(all_spk, ids)
            norm2 = fit_norm(face, voice) if cfg.standardize else None
            face2, voice2 = apply_norm(norm2, face, voice)
            log.write(f"[stage2] 全部 {len(all_spk)} 位說話者重訓 {best['epoch']} 個 epoch\n")
            print(f"[stage2] 全部 {len(all_spk)} 位說話者重訓 {best['epoch']} 個 epoch", flush=True)
            seed_all()
            model, _ = train(cfg, face2, voice2, all_labels, gen01, best["epoch"], None, log)
            torch.save({"state_dict": model.state_dict(), "n_class": len(all_spk), "cfg": vars(cfg),
                        "norm": norm2}, run / "final.pt")

    summary = {"best_epoch": best["epoch"], "val_eer_no_gender": best["no_gender"],
               "val_eer_gender": best["gender"], "val_eer_mean": best["mean"],
               "val_eer_mean_l2": best["mean_l2"], "val_eer_mean_cos": best["mean_cos"],
               "val_speakers": [str(s) for s in va_spk], "cfg": vars(cfg)}
    (run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
