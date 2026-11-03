"""從原始 jpg、wav 重新擷取臉部與語音特徵。

臉部：insightface buffalo_l（SCRFD 偵測 + 5 點對齊 + ArcFace R50，512 維）。偵測不到臉時，
     把整張 224×224 縮到 112×112 直接送辨識模型，並記錄在 face_fallback。
語音：microsoft/wavlm-base-plus-sv（WavLM + x-vector 頭，512 維，VoxCeleb 微調的說話者驗證模型）。

輸出 data/features_v2/<split>_{faces,voices}.npy，列順序與官方清單相同：
  train                    ← train_set/train_set/train_English.txt
  dev_{track}_{lang}       ← dev_set/{track}/{lang}_test.txt
  eval_{track}_{lang}      ← eval_set/{track}/{lang}_test.txt（Evaluation 階段；--splits eval）
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = DATA / "features_v2"
# --only 名稱 → 預訓練模型；輸出檔名後綴與名稱相同（faces_clip、voices_wavlmh ...）
CLIP_MODELS = {"clip": "openai/clip-vit-base-patch16", "clipl": "openai/clip-vit-large-patch14"}
WAVLM_MODELS = {"wavlmh": "microsoft/wavlm-base-plus", "wavlml": "microsoft/wavlm-large"}


def lists(splits=("train", "dev"), eval_dir=None):
    """回傳 [(輸出名稱, 圖檔路徑清單, 音檔路徑清單)]。splits 可含 train、dev、eval。"""
    out = []
    if "train" in splits:
        base = DATA / "train_set" / "train_set"
        rows = [l.split() for l in open(base / "train_English.txt", encoding="utf-8") if l.strip()]
        out.append(("train", [base / r[3] for r in rows], [base / r[2] for r in rows]))
    for split in ("dev", "eval"):
        if split not in splits:
            continue
        root = Path(eval_dir) if (split == "eval" and eval_dir) else DATA / f"{split}_set"
        for track in ("no_gender", "gender"):
            for lang in ("English", "Bangla"):
                base = root / track
                rows = [l.split() for l in open(base / f"{lang}_test.txt", encoding="utf-8") if l.strip()]
                out.append((f"{split}_{track}_{lang}", [base / r[2] for r in rows], [base / r[1] for r in rows]))
    return out


def extract_faces(paths, app, rec):
    import cv2
    from insightface.utils import face_align

    feats, fallback = np.zeros((len(paths), 512), np.float32), np.zeros(len(paths), bool)
    for i, p in enumerate(paths):
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        faces = app.get(img)
        if faces:
            # 多張臉時取面積最大的一張
            f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            crop = face_align.norm_crop(img, landmark=f.kps, image_size=112)
        else:
            crop = cv2.resize(img, (112, 112))
            fallback[i] = True
        feats[i] = rec.get_feat(crop).ravel()
        if (i + 1) % 1000 == 0:
            print(f"  faces {i + 1}/{len(paths)}  fallback {fallback[: i + 1].sum()}", flush=True)
    return feats, fallback


def extract_voices(paths, fe, model, device, max_sec):
    import soundfile as sf
    import torch

    feats = np.zeros((len(paths), 512), np.float32)
    for i, p in enumerate(paths):
        wav, sr = sf.read(str(p), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        assert sr == 16000, f"{p}: {sr} Hz"
        wav = wav[: int(max_sec * sr)]
        inp = fe(wav, sampling_rate=sr, return_tensors="pt", padding=False)
        with torch.no_grad():
            emb = model(**{k: v.to(device) for k, v in inp.items()}).embeddings
        feats[i] = emb[0].float().cpu().numpy()
        if (i + 1) % 1000 == 0:
            print(f"  voices {i + 1}/{len(paths)}", flush=True)
    return feats


def extract_wavlm_hidden(paths, fe, model, device, max_sec):
    """WavLM（未做說話者驗證微調）各層 Transformer 輸出先對時間取平均，再對層取平均（Base+ 12 層 768 維、Large 24 層 1024 維）。
    說話者驗證向量會刻意壓掉年齡、口音等與身分無關的資訊；這裡保留這些屬性，對應臉部的 CLIP。"""
    import soundfile as sf
    import torch

    feats = np.zeros((len(paths), model.config.hidden_size), np.float32)
    for i, p in enumerate(paths):
        wav, sr = sf.read(str(p), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        assert sr == 16000, f"{p}: {sr} Hz"
        inp = fe(wav[: int(max_sec * sr)], sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            hs = model(**{k: v.to(device) for k, v in inp.items()}, output_hidden_states=True).hidden_states
        feats[i] = torch.stack([h[0].mean(0) for h in hs[1:]]).mean(0).float().cpu().numpy()
        if (i + 1) % 1000 == 0:
            print(f"  wavlm-hidden {i + 1}/{len(paths)}", flush=True)
    return feats


def extract_clip(paths, proc, model, device, batch=64):
    """整張 224×224 頭肩裁切送 CLIP 影像編碼器，保留髮型、膚色、年齡等外觀屬性（不做人臉對齊）。"""
    import torch
    from PIL import Image

    feats = []
    for i in range(0, len(paths), batch):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
        inp = proc(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.get_image_features(**inp)
        # transformers 5 起回傳 BaseModelOutputWithPooling，投影後的影像嵌入在 pooler_output
        emb = out if torch.is_tensor(out) else out.pooler_output
        feats.append(emb.float().cpu().numpy())
    return np.concatenate(feats).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["faces", "voices", "both", "clip", "wavlmh", "clipl", "wavlml"],
                   default="both")
    p.add_argument("--max_sec", type=float, default=20.0, help="語音最長取幾秒")
    p.add_argument("--splits", default="train,dev", help="逗號分隔：train、dev、eval")
    p.add_argument("--eval_dir", default=None, help="eval 資料夾不是 data/eval_set 時指定")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = lists(tuple(s.strip() for s in args.splits.split(",")), args.eval_dir)
    stats = {}

    if args.only in ("faces", "both"):
        import onnxruntime as ort

        # onnxruntime-gpu 1.31 需要 CUDA 13 的 DLL（由 onnxruntime-gpu[cuda,cudnn] 裝進 site-packages），
        # 與 PyTorch cu128 附的 CUDA 12 分開；先載入才能用 CUDAExecutionProvider
        ort.preload_dlls()
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                           providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(224, 224))
        rec = app.models["recognition"]
        print("face providers:", rec.session.get_providers(), flush=True)
        for name, imgs, _ in jobs:
            print(f"[faces] {name} {len(imgs)}", flush=True)
            feats, fb = extract_faces(imgs, app, rec)
            np.save(OUT / f"{name}_faces.npy", feats)
            stats[f"{name}_face_fallback"] = int(fb.sum())

    if args.only in ("voices", "both"):
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector

        device = "cuda"
        fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device).eval()
        for name, _, wavs in jobs:
            print(f"[voices] {name} {len(wavs)}", flush=True)
            np.save(OUT / f"{name}_voices.npy", extract_voices(wavs, fe, model, device, args.max_sec))

    if args.only in CLIP_MODELS:
        from transformers import CLIPModel, CLIPProcessor

        name = CLIP_MODELS[args.only]
        proc = CLIPProcessor.from_pretrained(name)
        model = CLIPModel.from_pretrained(name).to("cuda").eval()
        for split, imgs, _ in jobs:
            print(f"[{args.only}] {split} {len(imgs)}", flush=True)
            np.save(OUT / f"{split}_faces_{args.only}.npy", extract_clip(imgs, proc, model, "cuda"))

    if args.only in WAVLM_MODELS:
        from transformers import AutoFeatureExtractor, WavLMModel

        name = WAVLM_MODELS[args.only]
        fe = AutoFeatureExtractor.from_pretrained(name)
        model = WavLMModel.from_pretrained(name).to("cuda").eval()
        for split, _, wavs in jobs:
            print(f"[{args.only}] {split} {len(wavs)}", flush=True)
            np.save(OUT / f"{split}_voices_{args.only}.npy",
                    extract_wavlm_hidden(wavs, fe, model, "cuda", args.max_sec))

    old = OUT / "stats.json"
    merged = json.loads(old.read_text(encoding="utf-8")) if old.exists() else {}
    merged.update(stats)
    old.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print("done", merged, flush=True)


if __name__ == "__main__":
    sys.exit(main())
