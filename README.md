# FLAG 2027 – Team geniusturtle6174

Code for our submission to the [FLAG 2027 Grand Challenge](https://mavceleb.github.io/dataset/competition.html)
(face–voice association across languages and gender, ICASSP 2027 GC-5).

The system keeps the FOP baseline model and changes the inputs and the training recipe:

1. Two groups of models with different inputs:
   - **Official embeddings**: VGGFace 4096-d faces and ECAPA-TDNN 192-d voices, as released by the organizers.
   - **Appearance and voice-attribute embeddings**: CLIP ViT-B/16 image embeddings of the full head-and-shoulder crop
     (512-d) for faces, and the time- and layer-averaged hidden states of WavLM Base+ (768-d) for voices.
2. Embeddings are standardized with training-set statistics; Adam learning rate 1e-4 instead of 1e-5.
3. Training length is chosen on 10 held-out training speakers, then the model is retrained on all 70.
4. Ten seeds per group; the final score is the mean distance of the 20 models.

Every face–voice pair is scored independently. No encoder is fine-tuned and no external training data is used.

## Development-set results (Codabench progress phase, EER %)

| System | NG-En | NG-Bn | G-En | G-Bn | Avg. |
|---|---|---|---|---|---|
| FOP baseline (official) | 32.54 | 38.12 | 32.99 | 44.01 | 36.92 |
| Official embeddings, 5 seeds | 26.39 | 30.44 | 30.75 | 40.74 | 32.08 |
| CLIP face + ECAPA voice, 5 seeds | 26.59 | 25.04 | 41.14 | 40.05 | 33.20 |
| VGGFace face + WavLM hidden voice, 5 seeds | 27.78 | 33.57 | 37.07 | 44.41 | 35.71 |
| CLIP face + WavLM hidden voice, 10 seeds | 22.22 | 24.47 | 33.40 | 39.10 | 29.80 |
| **Final: last row + official embeddings, 10 seeds each** | 22.02 | 24.75 | 30.55 | 36.38 | **28.43** |

NG: no_gender, G: gender, En: English (heard), Bn: Bangla (unheard).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
git clone https://github.com/SwapnilKhandoker101/FLAG_2027 baseline   # provides retrieval_model.py (FOP)
```

We used the official repository at commit `4871141c`.

## Data

Download the training and development sets from the official
[download folder](https://drive.google.com/drive/folders/1YFVLHIWu0yBQYOIgjTvfK7d_M_fsfzn8)
and unzip them into `data/`, and put `meta_file_train_set.csv` (training-set gender annotations) in `data/downloads/`:

```
data/
├── downloads/meta_file_train_set.csv
├── train_set/{features, train_set}/...
└── dev_set/{gender, no_gender}/...
```

The data is for research within the challenge only and is not redistributed here.

## Reproduce

```bash
bash reproduce_final.sh   # extracts CLIP and WavLM features, trains 20 models, writes submissions/final_dev.zip
```

Each model trains in about two minutes on an 8 GB laptop GPU (RTX 5060); feature extraction takes about 40 minutes.

## Evaluation phase

Unzip the evaluation inputs into `data/eval_set/` (same layout as `dev_set/`; otherwise pass `--eval_dir`), then:

```bash
python -I -X utf8 eval_day.py      # checks the folder, extracts missing features, writes and verifies submissions/final_eval.zip
```

`eval_day.py` reads the model list from `final_models.txt` (the 20 models trained by `reproduce_final.sh`).

## Files

| File | Purpose |
|---|---|
| `src/flag_data.py` | Loads features and pair lists, builds the speaker-disjoint validation pairs, computes EER |
| `extract_features.py` | Extracts CLIP face (`--only clip`) and WavLM hidden-state voice (`--only wavlmh`) embeddings, plus the ArcFace / WavLM speaker-verification features we also tried |
| `train_fop.py` | Trains FOP (`--features`, `--voice_features`, `--standardize`, `--lr`, `--seed`) |
| `make_submission.py` | Writes the Codabench zip; `--tag a:0.7,b:0.3` averages several models with weights |
| `eval_day.py` | One-command submission from the models in `final_models.txt` (`--split dev` or `eval`) |
| `ensemble_eval.py` | Evaluates an ensemble on the local validation pairs |
| `collect_results.py` | Tabulates `runs/*/summary.json` |

## Pretrained models

- VGGFace and ECAPA-TDNN (official challenge embeddings)
- CLIP ViT-B/16 (`openai/clip-vit-base-patch16`)
- WavLM Base+ (`microsoft/wavlm-base-plus`)

All encoders are used frozen. This repository contains no model weights or challenge data; the pretrained models
are downloaded from their original sources and remain under their own licenses. In particular, the insightface
`buffalo_l` models used by the optional ArcFace experiments (`extract_features.py --only faces`) are released for
non-commercial research only.

## License

Copyright 2026 Min-Yu Tsai and Chung-Che Wang, Department of Electrical Engineering, Yuan Ze University.

The code in this repository is licensed under the [Apache License 2.0](LICENSE).
