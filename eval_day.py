"""Evaluation 階段當天的一鍵流程。

1. 檢查 eval 資料夾結構（四個配對清單與官方特徵 CSV）。
2. 從 final_models.txt 列出的模型設定，判斷需要哪些額外特徵（clip、wavlmh、v2），缺的才擷取。
   最終系統含性別一致項（runs/gender_term.json，由 gender_term.py fit 產生），另需 v2 語音特徵與臉部屬性。
3. 用 gender_term.py submit 產生提交檔（加 --no_gender_term 時改用 make_submission.py）。
4. 檢查提交檔：四個分數檔都在、列數與 key 和配對清單一致、分數沒有 NaN。

用法：
  .venv/Scripts/python -I -X utf8 eval_day.py                        # 讀 data/eval_set，寫 submissions/final_eval.zip
  .venv/Scripts/python -I -X utf8 eval_day.py --eval_dir D:/x/eval    # 資料夾在別處
  .venv/Scripts/python -I -X utf8 eval_day.py --split dev --out _rehearsal   # 用 dev 預演整個流程
"""
import argparse
import math
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import flag_data  # noqa: E402

PY = sys.executable
GENDER_CFG = ROOT / "runs" / "gender_term.json"
CELLS = [(t, lang, cond) for t in flag_data.TRACKS
         for lang, cond in ((flag_data.HEARD, "heard"), (flag_data.UNHEARD, "unheard"))]
# 特徵來源 → extract_features.py 的 --only 參數與 features_v2 檔名
EXTRA = {("face", "v2"): ("faces", "faces"), ("voice", "v2"): ("voices", "voices"),
         ("face", "clip"): ("clip", "faces_clip"), ("voice", "wavlmh"): ("wavlmh", "voices_wavlmh")}


def read_models(path):
    specs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            tag, w = (line.split() + ["1"])[:2]
            specs.append((tag, float(w)))
    if not specs:
        raise SystemExit(f"{path} 沒有列出任何模型")
    return specs


def needed_extras(specs, ckpt):
    import torch

    need = set()
    for tag, _ in specs:
        p = ROOT / "runs" / tag / ckpt
        if not p.exists():
            raise SystemExit(f"找不到模型 {p}")
        cfg = torch.load(p, map_location="cpu", weights_only=False)["cfg"]
        face_set = cfg.get("features", "official")
        for part in flag_data.check_feature_set(face_set):
            if part != "official":
                need.add(("face", part))
        for part in flag_data.check_feature_set(cfg.get("voice_features") or face_set):
            if part != "official":
                need.add(("voice", part))
    return need


def check_structure(root):
    missing = []
    for track in flag_data.TRACKS:
        for lang in (flag_data.HEARD, flag_data.UNHEARD):
            for name in (f"{lang}_test.txt", f"{lang}_test_faces.csv", f"{lang}_test_voices.csv"):
                try:
                    flag_data._find_in(root, track, name)
                except FileNotFoundError as e:
                    missing.append(str(e))
    if missing:
        raise SystemExit("eval 資料夾結構不符：\n  " + "\n  ".join(missing))


def run(cmd):
    print("$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode:
        raise SystemExit(f"指令失敗（exit {r.returncode}）")


def verify_zip(zpath, root):
    with zipfile.ZipFile(zpath) as zf:
        names = set(zf.namelist())
        for track, lang, cond in CELLS:
            n = f"{track}/sub_score_v4_{lang}_{cond}.txt"
            if n not in names:
                raise SystemExit(f"提交檔缺少 {n}")
            lines = [l.split() for l in zf.read(n).decode().splitlines() if l.strip()]
            keys = [l.split()[0] for l in open(flag_data._find_in(root, track, f"{lang}_test.txt"),
                                                   encoding="utf-8") if l.strip()]
            if [l[0] for l in lines] != keys:
                raise SystemExit(f"{n} 的 key 與配對清單不一致")
            if any(len(l) != 2 or not math.isfinite(float(l[1])) for l in lines):
                raise SystemExit(f"{n} 有格式錯誤或非有限的分數")
            print(f"  OK {n}: {len(lines)} 列")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=str(ROOT / "final_models.txt"))
    p.add_argument("--split", choices=["dev", "eval"], default="eval")
    p.add_argument("--eval_dir", default=None, help="eval 資料不在 data/eval_set 時指定")
    p.add_argument("--out", default="final")
    p.add_argument("--ckpt", default="final.pt")
    p.add_argument("--no_gender_term", action="store_true", help="不加性別一致項，只用 make_submission.py")
    args = p.parse_args()
    use_gt = not args.no_gender_term
    if use_gt and not GENDER_CFG.exists():
        raise SystemExit(f"找不到 {GENDER_CFG}；請先執行 gender_term.py fit，或加 --no_gender_term")

    root = flag_data.split_root(args.split, args.eval_dir)
    print(f"[1/4] 檢查 {root}", flush=True)
    check_structure(root)

    specs = read_models(args.models)
    print(f"[2/4] {len(specs)} 個模型{'＋性別一致項' if use_gt else ''}；檢查額外特徵", flush=True)
    need = needed_extras(specs, args.ckpt)
    if use_gt:
        need.add(("voice", "v2"))  # 語音性別分類器的輸入（WavLM Base+ SV 說話者向量）
    for modality, part in sorted(need):
        only, suffix = EXTRA[(modality, part)]
        files = [flag_data.V2 / f"{args.split}_{t}_{lang}_{suffix}.npy" for t, lang, _ in CELLS]
        if all(f.exists() for f in files):
            print(f"  {only}：已存在，略過", flush=True)
            continue
        cmd = [PY, "-I", "-X", "utf8", "-W", "ignore", "extract_features.py", "--only", only, "--splits", args.split]
        if args.eval_dir:
            cmd += ["--eval_dir", args.eval_dir]
        run(cmd)
    if use_gt:
        files = [flag_data.V2 / f"{args.split}_{t}_{lang}_face_attr.npy" for t, lang, _ in CELLS if t == "no_gender"]
        if all(f.exists() for f in files):
            print("  face_attr：已存在，略過", flush=True)
        else:
            # 臉部屬性用 ONNX Runtime，與 PyTorch 的 cuDNN 衝突，必須是獨立程序
            cmd = [PY, "-I", "-X", "utf8", "-W", "ignore", "extract_attributes.py", "--only", "face", "--splits", args.split]
            if args.eval_dir:
                cmd += ["--eval_dir", args.eval_dir]
            run(cmd)

    print("[3/4] 產生提交檔", flush=True)
    if use_gt:
        cmd = [PY, "-I", "-X", "utf8", "gender_term.py", "submit", "--models", args.models, "--split", args.split,
               "--out", args.out, "--ckpt", args.ckpt]
    else:
        spec = ",".join(f"{t}:{w:g}" for t, w in specs)
        cmd = [PY, "-I", "-X", "utf8", "make_submission.py", "--tag", spec, "--split", args.split,
               "--out", args.out, "--ckpt", args.ckpt]
    if args.eval_dir:
        cmd += ["--split_dir", args.eval_dir]
    run(cmd)

    name = args.out + ("_eval" if args.split == "eval" and not args.out.endswith("_eval") else "")
    zpath = ROOT / "submissions" / f"{name}.zip"
    print(f"[4/4] 檢查 {zpath}", flush=True)
    verify_zip(zpath, root)
    print(f"完成：{zpath}" + ("（上傳到 Codabench 的 Evaluation FLAG 階段）" if args.split == "eval" else ""))


if __name__ == "__main__":
    main()
