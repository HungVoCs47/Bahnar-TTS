import argparse
import json
import os
import numpy as np
import IPython.display as ipd
from tqdm import tqdm
from scipy.io.wavfile import write

import torch
use_gpu = torch.cuda.is_available()

import librosa
from librosa.core import load
from librosa.filters import mel as librosa_mel_fn
mel_basis = librosa_mel_fn(22050, 1024, 80, 0, 8000)

import params
from model import DiffVC

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

import sys
sys.path.append('hifi-gan/')
from models import Generator as HiFiGAN

sys.path.append('speaker_encoder/')
from encoder import inference as spk_encoder
from pathlib import Path


from vc_func import get_mel, get_embed, noise_median_smoothing, mel_spectral_subtraction


vc_path = 'checkpts/vc/vc_libritts_wodyn.pt' # path to voice conversion model

generator = DiffVC(params.n_mels, params.channels, params.filters, params.heads, 
                   params.layers, params.kernel, params.dropout, params.window_size, 
                   params.enc_dim, params.spk_dim, params.use_ref_t, params.dec_dim, 
                   params.beta_min, params.beta_max)
if use_gpu:
    generator = generator.cuda()
    generator.load_state_dict(torch.load(vc_path))
else:
    generator.load_state_dict(torch.load(vc_path, map_location='cpu'))
generator.eval()

print(f'Number of parameters: {generator.nparams}')


hfg_path = 'checkpts/vocoder/' # HiFi-GAN path

with open(hfg_path + 'config.json') as f:
    h = AttrDict(json.load(f))

if use_gpu:
    hifigan_universal = HiFiGAN(h).cuda()
    hifigan_universal.load_state_dict(torch.load(hfg_path + 'generator')['generator'])
else:
    hifigan_universal = HiFiGAN(h)
    hifigan_universal.load_state_dict(torch.load(hfg_path + 'generator',  map_location='cpu')['generator'])

_ = hifigan_universal.eval()
hifigan_universal.remove_weight_norm()

enc_model_fpath = Path('checkpts/spk_encoder/pretrained.pt') # speaker encoder path
if use_gpu:
    spk_encoder.load_model(enc_model_fpath, device="cuda")
else:
    spk_encoder.load_model(enc_model_fpath, device="cpu")
    

    
src_path = 'example/KT_XH02_1_chunk1.wav' # path to source utterance
tgt_path = 'example/0004.wav' # path to reference utterance

mel_source = torch.from_numpy(get_mel(src_path)).float().unsqueeze(0)
if use_gpu:
    mel_source = mel_source.cuda()
mel_source_lengths = torch.LongTensor([mel_source.shape[-1]])
if use_gpu:
    mel_source_lengths = mel_source_lengths.cuda()

mel_target = torch.from_numpy(get_mel(tgt_path)).float().unsqueeze(0)
if use_gpu:
    mel_target = mel_target.cuda()
mel_target_lengths = torch.LongTensor([mel_target.shape[-1]])
if use_gpu:
    mel_target_lengths = mel_target_lengths.cuda()

embed_target = torch.from_numpy(get_embed(tgt_path)).float().unsqueeze(0)
if use_gpu:
    embed_target = embed_target.cuda()
    
mel_encoded, mel_ = generator.forward(mel_source, mel_source_lengths, mel_target, mel_target_lengths, embed_target, n_timesteps=10, mode='ml')
mel_synth_np = mel_.cpu().detach().squeeze().numpy()
mel_source_np = mel_.cpu().detach().squeeze().numpy()
mel = torch.from_numpy(mel_spectral_subtraction(mel_synth_np, mel_source_np, smoothing_window=1)).float().unsqueeze(0)
if use_gpu:
    mel = mel.cuda()
    
with torch.no_grad():
    audio = (hifigan_universal.forward(mel).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
    write(f'./out/sample_{1}.wav', 22050, audio)
#ipd.display(ipd.Audio(audio, rate=22050))
