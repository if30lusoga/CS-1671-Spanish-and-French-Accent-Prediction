import os
import numpy as np, pandas as pd, librosa, parselmouth
from joblib import Parallel, delayed

def extract_features(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        snd = parselmouth.Sound(y.astype("float64"), sampling_frequency=sr)
        f0 = snd.to_pitch().selected_array["frequency"]
        f0 = f0[f0 > 0]  # 0 = no pitch detected
        pitch = [f0.mean(), f0.std(), f0.max() - f0.min()] if len(f0) else [0, 0, 0]
        return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), pitch])
    except Exception as e:
        print("failed:", path, e)
        return np.full(29, np.nan)

def run(csv, audio_dir, out):
    df = pd.read_csv(csv)
    feats = Parallel(n_jobs=-1, verbose=5)(
        delayed(extract_features)(os.path.join(audio_dir, p)) for p in df.path)
    np.save(out, np.vstack(feats))

if __name__ == "__main__":
    run("data/dev_filtered.csv", "data/dev", "data/dev_features.npy")
    run("data/test_filtered.csv", "data/test", "data/test_features.npy")