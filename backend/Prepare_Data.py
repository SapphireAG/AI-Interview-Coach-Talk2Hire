#import kagglehub

import pandas as pd
import os
from pydub import AudioSegment

# Download latest version
#path = kagglehub.dataset_download("ejlok1/toronto-emotional-speech-set-tess")

#print("Path to dataset files:", path)



def convert_to_wav(input_path, output_path):
    # Load audio file
    audio = AudioSegment.from_file(input_path)

    # Set parameters: mono, 16kHz sample rate, 16-bit PCM
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_sample_width(2)  # 2 bytes = 16-bit

    # Export as .wav with PCM encoding
    audio.export(output_path, format="wav", codec="pcm_s16le")
    print(f"Converted and saved to: {output_path}")

# Example 
SUPPORTED = (".wav", ".m4a", ".mp3", ".flac", ".ogg")
def convert(root_dir,output_dir ):
                os.makedirs(output_dir,exist_ok=True)

                #root_dir="/Users/amangolani/.cache/kagglehub/datasets/ejlok1/toronto-emotional-speech-set-tess/versions/1/TESS Toronto emotional speech set data"
             # Walk through each subfolder
                for i,(subdir, dirs, files) in enumerate(os.walk(root_dir)):
                    for file in files:
                        if not file.lower().endswith(SUPPORTED):
                            continue
                        input_path = os.path.join(subdir, file)
                        rel_path=os.path.relpath(subdir,root_dir)
                        target_dir=os.path.join(output_dir,rel_path)
                        os.makedirs(target_dir, exist_ok=True)
                            # output_path=os.path.join(subdir,f"{file}{i}.wav")
                        base=pathlib.Path(file).stem+".wav"
                        output_path = os.path.join(target_dir, base)
                        try:
                                # Extract features
                            convert_to_wav(input_path,output_path)

                        except Exception as e:
                                print(f"Error with file {file}: {e}")

#convert(input_file,"/Users/amangolani/Downloads/converted_tess")



import librosa, numpy as np, pandas as pd, os, pathlib, json

#input .wav files and then recieve mfcc axtracted features 
#pass thru the CNN and classify them into the 
#'disgust': 400, 'surprise': 400, 'happy': 400, 'sad': 400, 'neutral': 400, 'fear': 400, 'angry': 400})

SR          = 16_000        # sampling rate
WIN_LEN     = 1_024         # FFT window
HOP_LEN     = 512
N_MFCC      = 40
FIX_SECONDS = 2             # pad / trim to 2 s
N_FRAMES    = int(np.ceil(FIX_SECONDS * SR / HOP_LEN))

def extract_mfcc(filepath: str,
                 sr: int = SR,
                 n_mfcc: int = N_MFCC,
                 max_frames: int = N_FRAMES) -> np.ndarray:
    """Return a (n_mfcc, max_frames) float32 tensor."""
    wav, _ = librosa.load(filepath, sr=sr, mono=True)

    # 1️⃣  Fix clip length (exactly 2 s => 32 000 samples)
    target_len = sr * FIX_SECONDS
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    else:
        wav = wav[:target_len]

    # 2️⃣  MFCC
    mfcc = librosa.feature.mfcc(y=wav, sr=sr,
                                n_mfcc=n_mfcc,
                                n_fft=WIN_LEN,
                                hop_length=HOP_LEN)          # (40, ~98)

    # 3️⃣  Length-lock in the time axis
    if mfcc.shape[1] < max_frames:
        mfcc = np.pad(mfcc,
                      ((0, 0), (0, max_frames - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :max_frames]

    # 4️⃣  Per-coefficient z-score (helps training)
    mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / \
           (mfcc.std(axis=1, keepdims=True) + 1e-9)

    return mfcc.astype(np.float32)          # (40, 98)
def extract_emotion(folder_name: str) -> str:
    """
    Turn things like
      'YAF_angry'                -> angry
      'OAF_neutral'              -> neutral
      'YAF_pleasant_surprised'   -> pleasant_surprised
      
      'YAF_ps'                   -> pleasant_surprised  (alias)
    """
    # 1️⃣  Drop the speaker prefix (everything before the first '_')
    tokens = folder_name.split("_")[1:]          # ['pleasant','surprised']
    emotion_raw = "_".join(tokens).lower()       # 'pleasant_surprised'

    # 2️⃣  Normalise aliases / typos
    alias_map = {
        "ps": "pleasant_surprised",
        "surprised": "surprise",                 # collapse -ed → -e
        "pleasant": "pleasant_surprised",
        "pleasant_surprise":"pleasant_surprised",
    }
    emotion = alias_map.get(emotion_raw, emotion_raw)
    return emotion


def build_mfcc_dataset(root_dir:str,
                       save_dir:str,
                       csv_out:str | None=None,
                       flatten_to_csv:bool=True):
     """
     root_dir : path containing all the *.wav files 
     save_dir: where to dump the individual npy tensors (if flattem_to_csv=False
     csv_out : manifest or feature file to write 
     flatten_to_csv : True -> write a single big csv of shape (N,)
     """

     records=[]
     os.makedirs(save_dir,exist_ok=True)

     for subdir, _, files in os.walk(root_dir):
          for file in files:
                if not file.lower().endswith(".wav"):
                    continue
                fp=os.path.join(subdir,file)
                try:
                     features = extract_mfcc(fp) 
                    #  label=pathlib.Path(subdir).name.split("_")[-1].lower()
                     folder = pathlib.Path(subdir).name
                     label  = extract_emotion(folder)
                     if flatten_to_csv:
                          records.append({
                               **{f"f{i}": v for i,v in enumerate(features.flatten())},
                               "emotion":label,
                               "filename":file
                          })
                     else:
                        npy_name = pathlib.Path(file).with_suffix(".npy").name
                        np.save(os.path.join(save_dir, npy_name), features)
                        records.append({
                             "npy": npy_name,
                             "emotion": label,
                             "filename": file
                         })

                except Exception as e:
                     print(f"⚠️  Skipping {file}: {e}")

                    
     df=pd.DataFrame.from_records(records)
     if csv_out:
          df.to_csv(csv_out,index=False)       
     print(f"DONE. Extracted {len(df)} files")      




import os, pandas as pd, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

# ──────────────────────────────
#  label vocabulary → index
# ──────────────────────────────
LABELS = ["angry", "disgust", "fear",
          "happy", "neutral", "sad",
         "surprise","pleasant_surprised"]
label2idx = {lbl: i for i, lbl in enumerate(LABELS)}

# ──────────────────────────────
#  Dataset
# ──────────────────────────────
class MFCCNPYDataset(Dataset):
    """
    Reads (40, 98) MFCC tensors saved as .npy files
    and returns them in CNN-ready shape (1, 98, 40).
    """
    def __init__(self, manifest_csv: str, npy_root: str, indices=None):
        df = pd.read_csv(manifest_csv)
        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)

        self.paths  = df["npy"].tolist()
        self.labels = df["emotion"].map(label2idx).tolist()
        self.root   = npy_root

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # 1. Load (40, 98) float32
        mfcc = np.load(os.path.join(self.root, self.paths[idx]))    # ndarray
        # 2. Torch tensor & reshape → (1, 98, 40)
        x = torch.from_numpy(mfcc).transpose(0, 1).unsqueeze(0)     # CHW
        # 3. Label → int64 tensor
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y            # ready for nn.CrossEntropyLoss



# manifest = "/Users/amangolani/Downloads/tess_manifest.csv"
# npy_root = "/Users/amangolani/Downloads/tess_mfcc_npy"

# # stratified 80 / 20
# df = pd.read_csv(manifest)
# splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2,
#                                   random_state=42)
# train_idx, val_idx = next(splitter.split(df["npy"], df["emotion"]))

# train_ds = MFCCNPYDataset(manifest, npy_root, train_idx)
# val_ds   = MFCCNPYDataset(manifest, npy_root, val_idx)

# train_loader = DataLoader(train_ds, batch_size=64,
#                           shuffle=True,  num_workers=1, pin_memory=True)
# val_loader   = DataLoader(val_ds,   batch_size=64,
#                           shuffle=False, num_workers=1)


import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassifierCNN(nn.Module):
    """
    Tiny MFCC-based CNN that fits easily in <1 MB int8 TFLite.
    Input  : (B, 1, 98, 40)   # 2-s window, 40-coef MFCC
    Output : logits of size (B, num_classes)
    """

    def __init__(self, num_classes: int=len(LABELS)):
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                      # (49, 20)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                      # (24, 10)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))          # (1, 1)
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),                         # (B, 128)
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)            # logits
        )

    def forward(self, x):
        """
        x: torch.Tensor of shape (B, 1, 98, 40)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x          # raw logits (preferred for nn.CrossEntropyLoss)
"""
So I would take the input audio, get the file in the mono 11khz etc, extract MFCC using librosa, pass through CNN classify the emotion and the export the code as a tflite optimized file for mobile
"""



def train(model, train_loader, val_loader, opt, crit, scheduler, device):
    best_acc = 0.0
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(20):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

        # ── quick validation ─────────────────────────────────────────────
        model.eval()
        correct = tot = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                tot += y.size(0)
        val_acc = correct / tot
        lr_now  = scheduler.get_last_lr()[0]
        print(f"epoch {epoch:02d} | val acc={val_acc:.3f} | lr={lr_now:.6f}")

        # ── save best ────────────────────────────────────────────────────
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = f"{ckpt_dir}/cnn_mfcc_best.pth"
            torch.save({
                "model_state": model.state_dict(),
                "acc": best_acc,
                "epoch": epoch,
                "label2idx": label2idx          # handy for prod
            }, ckpt_path)
            print(f"👍  New best ({best_acc:.3f}) — saved to {ckpt_path}")

        scheduler.step()


# Initialize openSMILE extractor
# def OpenSmileCSV():
#                 smile = opensmile.Smile(
#                     feature_set=opensmile.FeatureSet.eGeMAPSv02,
#                     feature_level=opensmile.FeatureLevel.Functionals,
#                 )

#                 # Root directory of your TESS dataset
#                 root_dir = "/Users/amangolani/Downloads/converted_tess"
#                 results = []

#                 # Walk through each subfolder
#                 for subdir, dirs, files in os.walk(root_dir):
#                     for file in files:
#                         if file.lower().endswith(".wav"):
#                             filepath = os.path.join(subdir, file)
#                             try:
#                                 # Extract features
#                                 features = smile.process_file(filepath)

#                                 # Derive label from folder name
#                                 emotion = os.path.basename(subdir).split("_")[-1].lower()  # e.g., OAF_angry → angry

#                                 features["filename"] = file
#                                 features["emotion"] = emotion
#                                 results.append(features)

#                             except Exception as e:
#                                 print(f"Error with file {file}: {e}")

#                 # Combine all into one DataFrame
#                 all_features = pd.concat(results).reset_index(drop=True)
#                 all_features.drop_duplicates(inplace=True)
#                 # Save to CSV
#                 all_features.to_csv("/Users/amangolani/Downloads/tess_opensmile_features.csv", index=False)

#                 print("✅ Feature extraction and CSV export complete.")


# OpenSmileCSV()

def main():
    
    raw_root = "/Users/amangolani/.cache/kagglehub/datasets/ejlok1/toronto-emotional-speech-set-tess/versions/1/TESS Toronto emotional speech set data"

    root_dir = "/Users/amangolani/Downloads/converted_tess"
    convert(raw_root,root_dir)

    build_mfcc_dataset(root_dir,
                   save_dir="/Users/amangolani/Downloads/tess_mfcc_npy",
                   csv_out="/Users/amangolani/Downloads/tess_manifest.csv",
                   flatten_to_csv=False)



    manifest = "/Users/amangolani/Downloads/tess_manifest.csv"
    df = pd.read_csv(manifest)

    unknown = sorted({lbl for lbl in df["emotion"].unique()
                  if lbl not in label2idx})
    print("🚨  Unknown labels:", unknown)  
    npy_root = "/Users/amangolani/Downloads/tess_mfcc_npy"

    # stratified 80 / 20
    df = pd.read_csv(manifest)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2,
                                    random_state=42)
    train_idx, val_idx = next(splitter.split(df["npy"], df["emotion"]))

    train_ds = MFCCNPYDataset(manifest, npy_root, train_idx)
    val_ds   = MFCCNPYDataset(manifest, npy_root, val_idx)

    train_loader = DataLoader(train_ds, batch_size=64,
                            shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=64,
                            shuffle=False, num_workers=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = ClassifierCNN(num_classes=len(LABELS)).to(device)
    opt    = torch.optim.AdamW(model.parameters(), 1e-3)
    crit   = torch.nn.CrossEntropyLoss()
    scheduler  = torch.optim.lr_scheduler.StepLR(opt, 7, gamma=0.5)
    train(model, train_loader, val_loader, opt, crit, scheduler, device)


if __name__=="__main__":
     main()