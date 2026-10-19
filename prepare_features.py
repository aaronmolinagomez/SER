import argparse
import zipfile
from pathlib import Path
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

ATTRS = ["valence", "arousal", "dominance"]
SR = 16000

def find_labels(data_dir: Path) -> Path:
    candidates = list(data_dir.rglob("labels.csv"))
    if not candidates:
        raise FileNotFoundError(f"No se encontró labels.csv en {data_dir}")
    return candidates[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--cache", type=str, required=True)
    ap.add_argument("--zip", type=str, default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    cache = Path(args.cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    if args.zip and not any(data_dir.rglob("labels.csv")):
        print(f"Extrayendo {args.zip} en {data_dir} ...")
        data_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(Path(args.zip).expanduser()) as z:
            z.extractall(data_dir)

    labels_path = find_labels(data_dir)
    audio_root = labels_path.parent
    df = pd.read_csv(labels_path)

    ATTRS = [c for c in df.columns if c not in ["path", "partition"]]
    print(f"Detectadas columnas de etiquetas: {ATTRS}")
    if df[ATTRS].isna().any().any():
        raise ValueError("Hay etiquetas NaN en labels.csv")

    print(f"labels.csv: {labels_path} | {len(df)} filas | particiones: {df.partition.value_counts().to_dict()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = torchaudio.pipelines.WAVLM_BASE_PLUS
    ssl = bundle.get_model().to(device).eval()

    def logmel(wave):
        with torch.no_grad():
            wave_tensor = wave.unsqueeze(0) if wave.ndim == 1 else wave
            layers, _ = ssl.extract_features(wave_tensor.to(device))
            h = (torch.stack(layers).mean(dim=0)[0].T if __import__("os").environ.get("LAYER","avg")=="avg" else layers[int(__import__("os").environ["LAYER"])][0].T)
        return h.half().cpu()

    for part in ["train", "validation", "test"]:
        sub = df[df.partition == part].reset_index(drop=True)
        if len(sub) == 0:
            continue
        
        features = []
        labels = []
        paths = []

        print(f"Procesando partición {part} ({len(sub)} muestras)...")
        for idx, row in tqdm(sub.iterrows(), total=len(sub)):
            audio_path = audio_root / row["path"]
            if not audio_path.exists():
                raise FileNotFoundError(f"No existe el audio: {audio_path}")
            
            waveform, sr = torchaudio.load(audio_path)
            if sr != SR:
                waveform = torchaudio.functional.resample(waveform, sr, SR)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            else:
                waveform = waveform.squeeze(0)

            feat = logmel(waveform)
            features.append(feat)
            labels.append(torch.tensor(row[ATTRS].values.astype("float32")))
            paths.append(str(row["path"]))

        cache_file = cache / f"{part}.pt"
        torch.save((features, torch.stack(labels), paths), cache_file)
        print(f"Guardado: {cache_file}")

if __name__ == "__main__":
    main()
