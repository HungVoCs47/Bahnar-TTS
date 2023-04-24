import numpy as np
import torch
from torch import nn
import librosa, librosa.display
import matplotlib.pyplot as plt
from base import BaseModule
import math

time_steps = 216
num_mels = 128

def melspectrogram(wav_file, window_size=2048, num_frames=216):
    signal, sr = librosa.load(wav_file)
    mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr, hop_length=math.ceil(signal.shape[0]/num_frames), n_fft=window_size)
    power_to_db = librosa.power_to_db(mel_signal, ref=np.max)
    return torch.Tensor(np.reshape(power_to_db, (1,1,power_to_db.shape[1],power_to_db.shape[0])))


class ReferenceEncoder(BaseModule):
    def __init__(self, time_steps, num_mels, dP =128, filter_channels=(32,32,64,64,128,128), kernel_size=3, stride=2):
        super(ReferenceEncoder, self).__init__()
        self.time_steps = time_steps
        self.num_mels = num_mels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_layers = len(filter_channels)
        self.dP = dP

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        self.conv_layers.append(nn.Conv2d(1, filter_channels[0], kernel_size, stride))
        self.norm_layers.append(nn.BatchNorm2d(filter_channels[0]))

        for i in range(self.n_layers - 1):
            self.conv_layers.append(nn.Conv2d(filter_channels[i], filter_channels[i+1], kernel_size, stride))
            self.norm_layers.append(nn.BatchNorm2d(filter_channels[i+1]))

        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size=self.num_mels, hidden_size=128)
        self.linear = nn.Linear(128, dP)
        self.tanh = nn.Tanh()

        for i in range(self.n_layers):
            self.time_steps = math.floor((self.time_steps - kernel_size)/stride + 1)
            self.num_mels = math.floor((self.num_mels - kernel_size)/stride + 1)

    def forward(self, X):
        for i in range(self.n_layers):
            X = self.conv_layers[i](X)
            X = self.norm_layers[i](X)
            X = self.relu(X)

        X = torch.reshape(X, (self.time_steps, 1, 128*self.num_mels))
        _, X = self.gru(X)
        X = self.linear(X)
        X = self.tanh(X)
        return X


if __name__ == "__main__":
    X = melspectrogram("/host/ubuntu/data/tungtk2/Speech-Backbones/Grad-TTS/data/internal/internal audio/0001.wav",window_size=2048, num_frames=time_steps)
    print(X.shape)
    ref_encoder = ReferenceEncoder(time_steps,num_mels)
    print(ref_encoder(X))
