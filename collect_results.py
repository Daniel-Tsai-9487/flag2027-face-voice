"""彙整 runs/*/summary.json 成表格（EER %），方便貼進 README。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
prefix = sys.argv[1] if len(sys.argv) > 1 else ""

rows = []
for p in sorted((ROOT / "runs").glob(f"{prefix}*/summary.json")):
    s = json.loads(p.read_text(encoding="utf-8"))
    c = s["cfg"]
    rows.append((s["val_eer_mean"], p.parent.name, c, s))

print("| tag | 臉部特徵 | 語音特徵 | lr | 標準化 | 分數 | 跨模態損失 | no_gender | gender | 平均 | 平均(l2) | 平均(cos) | 最佳 epoch |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for _, tag, c, s in sorted(rows, key=lambda r: r[0]):
    xm = "無" if not c.get("xm_weight") else f"{c['xm_weight']}×{c['xm_neg']}"
    pct = lambda k: f"{100 * s[k]:.1f}" if k in s else "-"
    face_set = c.get("features", "official")
    voice_set = c.get("voice_features") or face_set
    print(f"| {tag} | {face_set} | {voice_set} | {c['lr']:g} | {'是' if c.get('standardize') else '否'} "
          f"| {c.get('score', 'l2')} | {xm} "
          f"| {pct('val_eer_no_gender')} | {pct('val_eer_gender')} | {pct('val_eer_mean')} "
          f"| {pct('val_eer_mean_l2')} | {pct('val_eer_mean_cos')} | {s['best_epoch']} |")
