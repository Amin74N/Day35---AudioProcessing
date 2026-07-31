import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
import random
# ----------------------------------------------------------------------------------------------------------------------
# 1)Hyper-Parameters:
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
# ----------------------------------------------------------------------------------------------------------------------
# 2)Dataset:
ANNOTATIONS_FILE = "data/UrbanSound8K/UrbanSound8K.csv"
AUDIO_DIR = "data/UrbanSound8K"
CLASSES = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
           "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]
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
device = "cuda" if torch.cuda.is_available() else "cpu"
mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)
usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)
train_size = int(0.8 * len(usd))
test_size = len(usd) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(usd, [train_size, test_size])
# ----------------------------------------------------------------------------------------------------------------------
# 3)Data-Loader:
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
# ----------------------------------------------------------------------------------------------------------------------
# 4)CNN Model:
class CNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
                                   , nn.ReLU()
                                   , nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
                                   , nn.ReLU()
                                   , nn.MaxPool2d(kernel_size=2))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
                                   , nn.ReLU()
                                   , nn.MaxPool2d(kernel_size=2))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
                                   , nn.ReLU()
                                   , nn.MaxPool2d(kernel_size=2))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=128*4*2, out_features=10)
    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        return logits
model = CNNNetwork().to(device)
# ----------------------------------------------------------------------------------------------------------------------
# 5)Loss & Optimizer:
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# ----------------------------------------------------------------------------------------------------------------------
# 6)Training Loop:
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    train_correct = 0
    train_total = 0
    for spectrogram, labels in train_loader:
        spectrogram = spectrogram.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(spectrogram)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * spectrogram.size(0)
        _, predicted = torch.max(output, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    epoch_loss /= train_total
    train_accuracy = 100 * train_correct / train_total
    print(f'Epoch: {epoch + 1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Accuracy = {train_accuracy:.2f}%')
# ----------------------------------------------------------------------------------------------------------------------
# 7)Prediction:
model.eval()
index = random.randint(0, len(test_dataset)-1)
with torch.no_grad():
    signal, label = test_dataset[index]
    signal = signal.unsqueeze(0).to(device)
    output = model(signal)
    _, prediction = torch.max(output,1)
print(f"Predicted Label : {CLASSES[prediction.item()]}")
print(f"True Label      : {CLASSES[label]}")
