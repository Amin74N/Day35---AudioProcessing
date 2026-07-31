import os
import torchaudio
import torch
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------
class UrbanSoundDataset(Dataset):
    def __init__(self, annotation_file, audio_dir, transformation, target_sample_rate, num_samples, device):
        self.annotations = pd.read_csv(annotation_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label
    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index,5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index,0])
        return path
    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
# ----------------------------------------------------------------------------------------------------------------------
ANNOTATIONS_FILE = "data/UrbanSound8K/UrbanSound8K.csv"
AUDIO_DIR = "data/UrbanSound8K"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
device = "cuda" if torch.cuda.is_available() else "cpu"
mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)
usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)
# ----------------------------------------------------------------------------------------------------------------------
classes = {0: "air_conditioner",
           1: "car_horn",
           2: "children_playing",
           3: "dog_bark",
           4: "drilling",
           5: "engine_idling",
           6: "gun_shot",
           7: "jackhammer",
           8: "siren",
           9: "street_music"}
print("Dataset Size:", len(usd))
signal, label = usd[0]
print("Signal Shape:", signal.shape)
print("Label:", label)
print("Device:", signal.device)
print("Data Type:", signal.dtype)
print("Dimensions:", signal.ndim)
print("Mel Spectrogram:\n", signal)
plt.imshow(signal.squeeze().numpy(), origin="lower", aspect="auto", cmap="magma")
plt.colorbar()
plt.title(f"Label: {classes[label]}")
plt.xlabel("Time (s)")
plt.ylabel("Mel Bands")
plt.show()
