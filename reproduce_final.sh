#!/usr/bin/env bash
# Reproduce the final development-set submission (dev average EER 28.24%):
#   10 seeds with the official embeddings (VGGFace faces, ECAPA-TDNN voices) and
#   10 seeds with CLIP face embeddings + WavLM Base+ hidden-state voice embeddings, averaged with equal weights,
#   plus a fixed gender-consistency term on the no_gender tracks (lambda and classifier fixed on training data).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}

$PY -I -X utf8 extract_features.py --only clip
$PY -I -X utf8 extract_features.py --only wavlmh
$PY -I -X utf8 extract_features.py --only voices        # WavLM Base+ SV embeddings for the voice-gender classifier
$PY -I -X utf8 extract_attributes.py --only face        # InsightFace genderage; runs in its own process (ONNX Runtime)

for s in 1 2 3 4 5 6 7 8 9 10; do
  $PY -I -X utf8 train_fop.py --tag off_s$s   --seed $s --standardize --lr 1e-4 --features official
  $PY -I -X utf8 train_fop.py --tag clipw_s$s --seed $s --standardize --lr 1e-4 --features clip --voice_features wavlmh
done

$PY -I -X utf8 gender_term.py fit                        # chooses lambda on held-out training speakers; writes runs/gender_term.json
$PY -I -X utf8 eval_day.py --split dev --out final_dev   # uses final_models.txt; writes and checks submissions/final_dev.zip
