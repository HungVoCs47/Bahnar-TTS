import librosa
from librosa.core import load
from librosa.filters import mel as librosa_mel_fn
import numpy as np
import sys

sys.path.append("speaker_encoder/")
from encoder import inference as spk_encoder
from pathlib import Path


mel_basis = librosa_mel_fn(22050, 1024, 80, 0, 8000)


def get_mel(wav_path):
    wav, _ = load(wav_path, sr=22050)
    wav = wav[: (wav.shape[0] // 256) * 256]
    wav = np.pad(wav, 384, mode="reflect")
    stft = librosa.core.stft(
        wav, n_fft=1024, hop_length=256, win_length=1024, window="hann", center=False
    )
    stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
    mel_spectrogram = np.matmul(mel_basis, stftm)
    log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
    return log_mel_spectrogram


def get_embed(wav_path):
    wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
    embed = spk_encoder.embed_utterance(wav_preprocessed)
    return embed


def noise_median_smoothing(x, w=5):
    y = np.copy(x)
    x = np.pad(x, w, "edge")
    for i in range(y.shape[0]):
        med = np.median(x[i : i + 2 * w + 1])
        y[i] = min(x[i + w + 1], med)
    return y


def mel_spectral_subtraction(
    mel_synth, mel_source, spectral_floor=0.02, silence_window=5, smoothing_window=5
):
    mel_len = mel_source.shape[-1]
    energy_min = 100000.0
    i_min = 0
    for i in range(mel_len - silence_window):
        energy_cur = np.sum(np.exp(2.0 * mel_source[:, i : i + silence_window]))
        if energy_cur < energy_min:
            i_min = i
            energy_min = energy_cur
    estimated_noise_energy = np.min(
        np.exp(2.0 * mel_synth[:, i_min : i_min + silence_window]), axis=-1
    )
    if smoothing_window is not None:
        estimated_noise_energy = noise_median_smoothing(
            estimated_noise_energy, smoothing_window
        )
    mel_denoised = np.copy(mel_synth)
    for i in range(mel_len):
        signal_subtract_noise = np.exp(2.0 * mel_synth[:, i]) - estimated_noise_energy
        estimated_signal_energy = np.maximum(
            signal_subtract_noise, spectral_floor * estimated_noise_energy
        )
        mel_denoised[:, i] = np.log(np.sqrt(estimated_signal_energy))
    return mel_denoised
