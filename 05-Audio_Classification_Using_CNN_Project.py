from pathlib import Path
import random
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Subset, Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# ----------------------------------------------------------------------------------------------------------------------
# 1)Hyper-Parameters:
DATASET_PATH = Path("data/GoogleSpeechCommands")
SAMPLE_RATE = 16000
NUM_SAMPLES = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------------------------------------------------------------------------------------
# 2)Dataset:
class GoogleSpeechCommandsDataset(Dataset):
    def __init__(self, dataset_path, num_samples, sample_rate, n_fft, hop_length, n_mels):
        self.dataset_path = Path(dataset_path)
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.class_names = sorted(folder.name for folder in self.dataset_path.iterdir() if folder.is_dir())
        self.class_to_index = {class_name: index for index, class_name in enumerate(self.class_names)}
        self.audio_files = []
        for class_name in self.class_names:
            class_folder = self.dataset_path / class_name
            for audio_path in class_folder.glob("*.wav"):
                self.audio_files.append((audio_path, self.class_to_index[class_name]))
    def __len__(self):
        return len(self.audio_files)
    def __getitem__(self, index):
        audio_path, label = self.audio_files[index]
        signal = self._load_audio(audio_path)
        signal = self._fix_length(signal)
        mel = self._create_mel_spectrogram(signal)
        mel = self._normalize(mel)
        mel = self._to_tensor(mel)
        return mel, label
    def _load_audio(self, audio_path):
        signal, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        return signal
    def _fix_length(self, signal):
        signal = librosa.util.fix_length(signal, size=self.num_samples)
        return signal
    def _create_mel_spectrogram(self, signal):
        mel = librosa.feature.melspectrogram(y=signal, sr=self.sample_rate, n_fft=self.n_fft,
                                             hop_length=self.hop_length, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db
    def _normalize(self, mel):
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
        return mel
    def _to_tensor(self, mel):
        mel = torch.tensor(mel, dtype=torch.float32)
        mel = mel.unsqueeze(0)
        return mel
dataset = GoogleSpeechCommandsDataset(DATASET_PATH, NUM_SAMPLES, SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS)
# ----------------------------------------------------------------------------------------------------------------------
# 3)Train/Test Split:
indices = list(range(len(dataset)))
labels = [label for _, label in dataset.audio_files]
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)
# ----------------------------------------------------------------------------------------------------------------------
# 4)Data-Loader:
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
# ----------------------------------------------------------------------------------------------------------------------
# 5)CNN Model:
class AudioClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
                                   , nn.ReLU()
                                   , nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
                                   , nn.ReLU()
                                   , nn.MaxPool2d(kernel_size=2))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
                                   , nn.ReLU()
                                   , nn.MaxPool2d(kernel_size=2))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(64*8*4, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x
model = AudioClassifierCNN(num_classes=len(dataset.class_names)).to(DEVICE)
# ----------------------------------------------------------------------------------------------------------------------
# 6)Loss & Optimizer:
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# ----------------------------------------------------------------------------------------------------------------------
# 7)Training Loop:
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
best_test_accuracy = 0
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for mel, labels in train_loader:
        mel = mel.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(mel)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * mel.size(0)
        _, predictions = torch.max(outputs, dim=1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)
    train_loss /= train_total
    train_accuracy = (train_correct/ train_total) * 100
# ----------------------------------------------------------------------------------------------------------------------
# 8)Evaluation:
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for mel, labels in test_loader:
            mel = mel.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(mel)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * mel.size(0)
            _, predictions = torch.max(outputs, 1)
            test_correct += (predictions == labels).sum().item()
            test_total += labels.size(0)
    test_loss /= test_total
    test_accuracy = 100 * test_correct / test_total
    print(f'epoch: {epoch + 1}/{EPOCHS} | Train Loss = {train_loss:.4f} | Train Accuracy = {train_accuracy:.2f}% | Test Loss = {test_loss:.4f} | Test Accuracy = {test_accuracy:.2f}%')
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        torch.save(model.state_dict(), "best_audio_classifier.pth")
# ----------------------------------------------------------------------------------------------------------------------
# 9)Performance Analysis (Confusion-Matrix, Loss-Curve, and Accuracy-Curve):
model.load_state_dict(torch.load("best_audio_classifier.pth", weights_only=True))
model.eval()
all_labels = []
all_predictions = []
with torch.no_grad():
    for mel, labels in test_loader:
        mel = mel.to(DEVICE)
        outputs = model(mel)
        _, predictions = torch.max(outputs, dim=1)
        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())
cm = confusion_matrix(all_labels, all_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.class_names)
fig, ax = plt.subplots(figsize=(20, 14))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=90)
plt.title("Confusion Matrix")
ax.set_aspect("auto")
plt.tight_layout()
plt.savefig("Confusion_Matrix.png", dpi=300, bbox_inches="tight")
plt.show()
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("Loss_Curve.png")
plt.show()
plt.figure(figsize=(8,5))
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)
plt.savefig("Accuracy_Curve.png")
plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# 10)Prediction:
model.eval()
index = random.randint(0, len(test_dataset) - 1)
with torch.no_grad():
    mel, label = test_dataset[index]
    mel = mel.unsqueeze(0).to(DEVICE)
    output = model(mel)
    probabilities = torch.softmax(output, dim=1)
    confidence, prediction = torch.max(probabilities, dim=1)
print(f"Sample Index    : {index}")
print(f"True Label      : {dataset.class_names[label]}")
print(f"Predicted Label : {dataset.class_names[prediction.item()]}")
print(f"Confidence      : {confidence.item() * 100:.2f}%")
