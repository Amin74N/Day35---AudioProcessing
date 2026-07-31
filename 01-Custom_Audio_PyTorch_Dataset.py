import os
import torchaudio
from torch.utils.data import Dataset
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
class UrbanSoundDataset(Dataset):
    def __init__(self, annotation_file, audio_dir):
        self.annotations = pd.read_csv(annotation_file)
        self.audio_dir = audio_dir
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        return signal, label
    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index,5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index,0])
        return path
    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]
# ----------------------------------------------------------------------------------------------------------------------
ANNOTATIONS_FILE = "data/UrbanSound8K/UrbanSound8K.csv"
AUDIO_DIR = "data/UrbanSound8K"
usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR)
print(f"There are {len(usd)} samples in the UrbanSound8K dataset")
signal, label = usd[0]
print("Signal Shape:", signal.shape)
print("Label:", label)
