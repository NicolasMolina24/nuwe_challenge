from torch.functional import _return_output
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
from pathlib import Path
import pandas as pd
import torchvision
import numpy as np
import torchaudio
import librosa
import torch
import os

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Setup device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_STATE = 42
NUM_WORKERS = os.cpu_count()
BATCH_SIZE = 32
EPOCHS = 10

class SoundDataset(Dataset):

    def __init__(self, data, y):
        self.data = data
        self.y = y
        self.SAMPLE_RATE = 22050
        self.SECONDS = 5
        self.NUM_SAMPLES = self.SAMPLE_RATE * self.SECONDS
        self.N_FFT = 1024
        self.HOP_LENGTH = 512
        self.N_MELS = 64
        self.IMG_SIZE = (224, 224)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load audio file
        path = Path(__file__).parent
        path = path.joinpath("..", self.data.iloc[idx]["Path"])
        audio, sr = torchaudio.load(path)
        label = self.y.iloc[idx]

        # Preprocess audio
        audio = self._preproces_audio(audio, sr)
        sr = self.SAMPLE_RATE

        melspect = self._get_melspectrogram(audio)
        contrast = self._get_spectral_contrast(audio)
        chroma = self._get_chroma_stft(audio)
        data = torch.cat((melspect, contrast, chroma)).float()
        return data, label
    
    def get_classes(self):
        return self.y.unique()

    def _preproces_audio(self, audio, sr):
        # Preprocess audio
        audio = self._resample_audio(audio, sr)
        # stereo to mono
        audio = self._stereo_to_mono(audio)
        # equal samples
        audio = self._same_samples_audio(audio)
        return audio

    def _resample_audio(self, audio, sr):
        audio = (
            audio
            if sr == self.SAMPLE_RATE
            else torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.SAMPLE_RATE)(
                audio
            )
        )
        return audio

    def _stereo_to_mono(self, audio):
        audio = audio if audio.shape[0] <= 1 else audio.mean(dim=0, keepdim=True)
        return audio

    def _same_samples_audio(self, audio):
        if audio.shape[1] > self.NUM_SAMPLES:
            audio = audio[:, :self.NUM_SAMPLES]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.NUM_SAMPLES - audio.shape[1]))
        return audio

    def _get_melspectrogram(self, audio):
        # Get mel spectrogram
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.SAMPLE_RATE, n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH, n_mels=self.N_MELS
        )
        melspect_image = mel_spectrogram(audio)
        melspect_image = self._resize_transform_img(melspect_image)
        return melspect_image

    def _get_spectral_contrast(self, audio):
        # Get spectral contrast
        S = np.abs(librosa.stft(audio.numpy()[0]))
        contrast = librosa.feature.spectral_contrast(S=S, sr=self.SAMPLE_RATE)
        # to tensor
        contrast = torch.from_numpy(contrast)
        # add dimension
        contrast = contrast.unsqueeze(0)
        contrast_image = self._resize_transform_img(contrast)
        return contrast_image

    def _get_chroma_stft(self, audio):
        # Get chroma stft
        chroma = librosa.feature.chroma_stft(y=audio.numpy()[0], sr=self.SAMPLE_RATE)
        # to tensor
        chroma = torch.from_numpy(chroma)
        # add dimension
        chroma = chroma.unsqueeze(0)
        chroma_image = self._resize_transform_img(chroma)
        return chroma_image

    def _resize_transform_img(self, tensor):
        # resize image 
        resize_img = torchvision.transforms.Resize(self.IMG_SIZE)
        tensor = resize_img(tensor)
        return tensor


class CNNModel(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # load pretrained model
        self.weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights
        self.model = torchvision.models.efficientnet_b0(weights=self.weights).to(DEVICE)

        # Freeze all base layers
        for param in self.model.features.parameters():
            param.requires_grad = False

        # Get the length of class_names (one output unit for each class)
        self.output_shape = len(num_classes)

        # Recreate the classifier layer and seed it to the target device
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(in_features=1280,
                            out_features=self.output_shape, # same number of output units as our number of classes
                            bias=True)).to(DEVICE)

    def forward(self, x):
        return self.model(x)
    

def create_dataloader(dataset, batch_size, shuffle=True):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_step(model, train_loader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        prediction = model(data)
        loss = loss_fn(prediction, target)
        running_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        
    return running_loss / len(train_loader)

def eval_step(model, val_loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)

            prediction = model(data)
            loss = loss_fn(prediction, target)
            running_loss += loss.item()

            _, predicted = torch.max(prediction, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
    return running_loss / len(val_loader), correct / total, np.array(all_targets), np.array(all_preds)

def predict(model, data_loader, device):
    model.eval()
    all_idx = []
    all_preds = []
    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            data = data.to(device)

            prediction = model(data)
            _, predicted = torch.max(prediction, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_idx.extend(target)

    return {t: p for t,p  in zip(all_idx, all_preds)}


def train_model(model, train_loader, val_loader, loss_fn, optimizer, device, epochs):
    for epoch in range(epochs):
        train_loss = train_step(model, train_loader, loss_fn, optimizer, device)
        val_loss, accuracy, targets, preds = eval_step(model, val_loader, loss_fn, device)
        f1 = f1_score(targets, preds, average='macro')  # Use 'macro' for multiclass, 'binary' for binary classification
        print(f'Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, Accuracy: {accuracy}, F1 Score: {f1}')
    return model



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)