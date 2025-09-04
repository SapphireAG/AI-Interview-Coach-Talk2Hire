import torch, librosa, numpy as np, pathlib
from pathlib import Path
from Prepare_Data import ClassifierCNN
import Prepare_Data
import os 
# ── 1.  Load the trained network ───────────────────────────────────────
LABELS = ["angry", "disgust", "fear",
          "happy", "neutral", "sad",
          "surprise", "pleasant_surprised"]
label2idx = {l: i for i, l in enumerate(LABELS)}
idx2label = {i: l for l, i in label2idx.items()}

device = "cpu"          # or "cuda" if you have a GPU
model  = ClassifierCNN(num_classes=len(LABELS)).to(device)

ckpt_path = "/Users/amangolani/Downloads/checkpoints/cnn_mfcc_best.pth"
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state["model_state"])
model.eval()

# ── 2.  Helper: extract MFCC exactly as during training ───────────────
SR = 16_000
WIN_LEN, HOP_LEN, N_MFCC, FIX_SECONDS = 1024, 512, 40, 2
N_FRAMES = int(np.ceil(FIX_SECONDS * SR / HOP_LEN))

def mfcc_tensor(path):
    wav, _ = librosa.load(path, sr=SR, mono=True)
    target = SR * FIX_SECONDS
    if len(wav) < target:  wav = np.pad(wav, (0, target-len(wav)))
    else:                  wav = wav[:target]

    mfcc = librosa.feature.mfcc(y=wav, sr=SR,
                                n_fft=WIN_LEN, hop_length=HOP_LEN,
                                n_mfcc=N_MFCC)
    if mfcc.shape[1] < N_FRAMES:
        mfcc = np.pad(mfcc, ((0,0),(0,N_FRAMES-mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :N_FRAMES]

    mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / \
           (mfcc.std(axis=1, keepdims=True) + 1e-9)

    x = torch.from_numpy(mfcc.astype(np.float32)).T  # (98,40)
    x = x.unsqueeze(0).unsqueeze(0)                  # (1,1,98,40)
    return x

# ── 3.  Inference on one file ─────────────────────────────────────────
input_path = "/Users/amangolani/Downloads/test_my_audio.m4a"
output_path= "/Users/amangolani/Downloads/testing_mfcc_pipeline"
#os.makedirs(output_path,exist_ok=True)

wav_path=os.path.join(output_path,"test_my_audio.wav")
Prepare_Data.convert_to_wav(input_path,wav_path)
x = mfcc_tensor(wav_path).to(device)

with torch.no_grad():
    logits = model(x)                 # (1, 8)
    probs  = torch.softmax(logits, 1)[0]

pred_idx = int(probs.argmax())
print(f"Prediction: {idx2label[pred_idx]}  "
      f"(p={probs[pred_idx]:.3f})")

# Optional: show all class probabilities
for i, p in enumerate(probs):
    print(f"{idx2label[i]:>20}: {p:.3f}")
