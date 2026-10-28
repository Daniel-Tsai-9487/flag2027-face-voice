#!/usr/bin/env bash
# Reproduce the final development-set submission (dev average EER 28.43%):
#   10 seeds with the official embeddings (VGGFace faces, ECAPA-TDNN voices) and
#   10 seeds with CLIP face embeddings + WavLM Base+ hidden-state voice embeddings, averaged with equal weights.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}

$PY -I -X utf8 extract_features.py --only clip
$PY -I -X utf8 extract_features.py --only wavlmh

for s in 1 2 3 4 5 6 7 8 9 10; do
  $PY -I -X utf8 train_fop.py --tag off_s$s   --seed $s --standardize --lr 1e-4 --features official
  $PY -I -X utf8 train_fop.py --tag clipw_s$s --seed $s --standardize --lr 1e-4 --features clip --voice_features wavlmh
done

$PY -I -X utf8 eval_day.py --split dev --out final_dev      # uses final_models.txt; writes and checks submissions/final_dev.zip
