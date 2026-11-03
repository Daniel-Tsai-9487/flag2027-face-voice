"""用公開、凍結的性別／年齡模型，逐樣本預測性別與年齡（Rules & FAQ 第 4 條允許，須在系統說明中寫明）。

臉部：insightface buffalo_l 的 genderage.onnx（SCRFD 偵測後裁切；偵測不到臉時用整張圖）。
語音：audeering/wav2vec2-large-robust-6-ft-age-gender（CC BY-NC-SA 4.0；訓練資料為 aGender、Common Voice、TIMIT、VoxCeleb2）。

輸出 data/features_v2/<split>_{face,voice}_attr.npy，每列 [男性機率, 年齡（歲）]，列順序與官方清單相同。
每個樣本獨立處理，不使用其他測試樣本的任何資訊。

用法（臉部與語音要分兩次執行：ONNX Runtime 用 CUDA 13 的 cuDNN、PyTorch 用 CUDA 12 的，同一程序只能載入一份）：
  .venv/Scripts/python -I -X utf8 extract_attributes.py --only face --splits train,dev
  .venv/Scripts/python -I -X utf8 extract_attributes.py --only voice --splits train,dev
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from extract_features import OUT, lists  # noqa: E402

VOICE_MODEL = "audeering/wav2vec2-large-robust-6-ft-age-gender"


def face_attributes(paths):
    import cv2
    import onnxruntime as ort

    ort.preload_dlls()
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align

    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "genderage"],
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(224, 224))
    ga = app.models["genderage"]
    out = np.zeros((len(paths), 2), np.float32)
    for i, p in enumerate(paths):
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        faces = app.models["detection"].detect(img, max_num=0, metric="default")[0]
        if len(faces):
            b = max(faces, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[:4]
        else:
            b = np.array([0, 0, img.shape[1], img.shape[0]], np.float32)
        # 與 insightface Attribute.get 相同的裁切與前處理，但保留原始輸出以取得機率
        w, h = b[2] - b[0], b[3] - b[1]
        center = ((b[2] + b[0]) / 2, (b[3] + b[1]) / 2)
        aimg, _ = face_align.transform(img, center, ga.input_size[0], ga.input_size[0] / (max(w, h) * 1.5), 0)
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / ga.input_std, tuple(aimg.shape[0:2][::-1]),
                                     (ga.input_mean,) * 3, swapRB=True)
        pred = ga.session.run(ga.output_names, {ga.input_name: blob})[0][0]
        g = pred[:2].astype(np.float64)
        # genderage 的前兩個輸出若已是機率（介於 0 與 1 且總和為 1）就直接用，否則視為 logit 做 softmax
        if not (np.all((g >= 0) & (g <= 1)) and np.isclose(g.sum(), 1.0, atol=1e-3)):
            g = np.exp(g - g.max()) / np.exp(g - g.max()).sum()
        out[i] = (g[1], pred[2] * 100.0)  # insightface：索引 1 為男性
        if (i + 1) % 2000 == 0:
            print(f"  face attr {i + 1}/{len(paths)}", flush=True)
    return out


def voice_attributes(paths, max_sec=20.0):
    import soundfile as sf
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Processor
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class ModelHead(nn.Module):
        def __init__(self, config, num_labels):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, num_labels)

        def forward(self, x):
            return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))

    class AgeGenderModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.age = ModelHead(config, 1)
            self.gender = ModelHead(config, 3)
            self.post_init()

        def forward(self, input_values):
            hidden = self.wav2vec2(input_values)[0].mean(dim=1)
            return self.age(hidden), torch.softmax(self.gender(hidden), dim=1)

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import Wav2Vec2Config

    proc = Wav2Vec2Processor.from_pretrained(VOICE_MODEL)
    # transformers 5 的 from_pretrained 不會把舊式 weight_norm 參數（pos_conv weight_g／weight_v）轉成新的
    # parametrizations 名稱，位置卷積會變成隨機值、輸出近乎亂猜；改用 PyTorch 的 load_state_dict（有轉換掛鉤），
    # 並要求所有參數都對上。
    model = AgeGenderModel(Wav2Vec2Config.from_pretrained(VOICE_MODEL))
    model.load_state_dict(load_file(hf_hub_download(VOICE_MODEL, "model.safetensors")), strict=True)
    model = model.to("cuda").eval()
    out = np.zeros((len(paths), 2), np.float32)
    for i, p in enumerate(paths):
        wav, sr = sf.read(str(p), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        assert sr == 16000, f"{p}: {sr} Hz"
        x = proc(wav[: int(max_sec * sr)], sampling_rate=sr, return_tensors="pt").input_values.to("cuda")
        with torch.no_grad():
            age, gender = model(x)
        child, female, male = gender[0].float().cpu().numpy()
        out[i] = (male / (male + female + 1e-9), float(age[0, 0]) * 100.0)
        if (i + 1) % 2000 == 0:
            print(f"  voice attr {i + 1}/{len(paths)}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", default="train,dev")
    p.add_argument("--eval_dir", default=None)
    p.add_argument("--only", choices=["face", "voice"], required=True)
    args = p.parse_args()
    jobs = lists(tuple(s.strip() for s in args.splits.split(",")), args.eval_dir)
    for name, imgs, wavs in jobs:
        if args.only == "face":
            print(f"[face_attr] {name} {len(imgs)}", flush=True)
            np.save(OUT / f"{name}_face_attr.npy", face_attributes(imgs))
        if args.only == "voice":
            print(f"[voice_attr] {name} {len(wavs)}", flush=True)
            np.save(OUT / f"{name}_voice_attr.npy", voice_attributes(wavs))
    print("done", flush=True)


if __name__ == "__main__":
    main()
