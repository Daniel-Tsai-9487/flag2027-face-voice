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
4. Ten seeds per group; the ensemble score is the mean distance of the 20 models.
5. A fixed gender-consistency term on the no_gender tracks only:
   `d' = d + λ·[p_f·(1 − p_v) + (1 − p_f)·p_v]`, where `p_f` is the probability that the face is male (frozen
   InsightFace `buffalo_l` genderage model) and `p_v` the probability that the voice is male (logistic regression on
   WavLM Base+ SV embeddings, trained on the official training set only). λ = 0.1 is chosen on the 10 held-out
   training speakers with the classifier trained on the other 60; the classifier is then retrained on all 70.
   The gender tracks are unchanged, because their negatives share the gender of the positive speaker.

Every face–voice pair is scored independently. No encoder is fine-tuned, no external training data is used, and no
parameter or statistic is estimated on development or evaluation data.

## Development-set results (Codabench progress phase, EER %)

| System | NG-En | NG-Bn | G-En | G-Bn | Avg. |
|---|---|---|---|---|---|
| FOP baseline (official) | 32.54 | 38.12 | 32.99 | 44.01 | 36.92 |
| Official embeddings, 5 seeds | 26.39 | 30.44 | 30.75 | 40.74 | 32.08 |
| CLIP face + ECAPA voice, 5 seeds | 26.59 | 25.04 | 41.14 | 40.05 | 33.20 |
| VGGFace face + WavLM hidden voice, 5 seeds | 27.78 | 33.57 | 37.07 | 44.41 | 35.71 |
| CLIP face + WavLM hidden voice, 10 seeds | 22.22 | 24.47 | 33.40 | 39.10 | 29.80 |
| Last row + official embeddings, 10 seeds each | 22.02 | 24.75 | 30.55 | 36.38 | 28.43 |
| **Final: previous row + gender-consistency term** | 21.43 | 24.61 | 30.55 | 36.38 | **28.24** |

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
bash reproduce_final.sh   # extracts features and face attributes, trains 20 models, fits the gender term, writes submissions/final_dev.zip
```

Each model trains in about two minutes on an 8 GB laptop GPU (RTX 5060); feature extraction takes about an hour.
`gender_term.py fit` uses the stage-1 checkpoints of the models in `final_models.txt` and stores λ, the voice-gender
classifier and the model list in `runs/gender_term.json`; rerun it whenever `final_models.txt` changes.

## Evaluation phase

Unzip the evaluation inputs into `data/eval_set/` (same layout as `dev_set/`; otherwise pass `--eval_dir`), then:

```bash
python -I -X utf8 eval_day.py      # checks the folder, extracts missing features, writes and verifies submissions/final_eval.zip
```

`eval_day.py` reads the model list from `final_models.txt` (the 20 models trained by `reproduce_final.sh`) and adds the
gender-consistency term from `runs/gender_term.json` (`--no_gender_term` gives the plain ensemble).

## Files

| File | Purpose |
|---|---|
| `src/flag_data.py` | Loads features and pair lists, builds the speaker-disjoint validation pairs, computes EER |
| `extract_features.py` | Extracts CLIP face (`--only clip`), WavLM hidden-state voice (`--only wavlmh`) and WavLM Base+ SV voice (`--only voices`) embeddings, plus the ArcFace features we also tried (`--only faces`) |
| `extract_attributes.py` | Face gender and age with the frozen InsightFace genderage model (`--only face`); the voice age/gender model (`--only voice`) was tried but not used |
| `train_fop.py` | Trains FOP (`--features`, `--voice_features`, `--standardize`, `--lr`, `--seed`) |
| `make_submission.py` | Writes the Codabench zip; `--tag a:0.7,b:0.3` averages several models with weights |
| `gender_term.py` | `fit` fixes λ and the voice-gender classifier on training speakers; `submit` writes the zip with the gender term |
| `eval_day.py` | One-command submission from the models in `final_models.txt` plus the gender term (`--split dev` or `eval`) |
| `ensemble_eval.py` | Evaluates an ensemble on the local validation pairs |
| `collect_results.py` | Tabulates `runs/*/summary.json` |

## Pretrained models

- VGGFace and ECAPA-TDNN (official challenge embeddings)
- CLIP ViT-B/16 (`openai/clip-vit-base-patch16`)
- WavLM Base+ (`microsoft/wavlm-base-plus`)
- Gender-consistency term: InsightFace `buffalo_l` (SCRFD detector and genderage model) and WavLM Base+ SV
  (`microsoft/wavlm-base-plus-sv`)

All pretrained models are used frozen. This repository contains no model weights or challenge data; the pretrained
models are downloaded from their original sources and remain under their own licenses. In particular, the InsightFace
`buffalo_l` models (used for face gender and by the ArcFace experiments) are released for non-commercial research
only.

## License

Copyright 2026 Min-Yu Tsai and Chung-Che Wang, Department of Electrical Engineering, Yuan Ze University.

The code in this repository is licensed under the [Apache License 2.0](LICENSE).
