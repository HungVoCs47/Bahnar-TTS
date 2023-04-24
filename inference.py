# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import argparse
import json
import datetime as dt
import numpy as np
from scipy.io.wavfile import write
import torch
from tqdm import tqdm
import params
from model import GradTTS
from text import text_to_sequence, bndict
from text.symbols import symbols
from utils import intersperse
import librosa
from librosa.core import load
from librosa.filters import mel as librosa_mel_fn
import sys

sys.path.append(r"hifi-gan/")
from models import Generator as HiFiGAN

sys.path.append('speaker_encoder/')
from encoder import inference as spk_encoder
from pathlib import Path

import params_vc
from model import DiffVC
from vc_func import get_mel, get_embed, noise_median_smoothing, mel_spectral_subtraction



class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
HIFIGAN_CHECKPT = './checkpts/hifigan.pt'
vc_path = 'checkpts_vc/vc/vc_libritts_wodyn.pt'
hfg_path = 'checkpts_vc/vocoder/'# HiFi-GAN path


    
mel_basis = librosa_mel_fn(22050, 1024, 80, 0, 8000)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True, help='path to a file with texts to synthesize')
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to a checkpoint of Grad-TTS')
    parser.add_argument('-t', '--timesteps', type=int, required=False, default=10, help='number of timesteps of reverse diffusion')
    parser.add_argument('-s', '--speaker_id', type=int, required=False, default=None, help='speaker id for multispeaker model')
    parser.add_argument('-vc', '--voice_conversion', type=str, required=False, help = 'path to the target audio voice')
    args = parser.parse_args()
    
    if not isinstance(args.speaker_id, type(None)):
        assert params.n_spks > 1, "Ensure you set right number of speakers in `params.py`."
        spk = torch.LongTensor([args.speaker_id])
    else:
        spk = None
    
    print('Initializing Grad-TTS...')
    generator = GradTTS(len(symbols)+1, params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)

    generator.load_state_dict(torch.load(args.checkpoint, map_location=lambda loc, storage: loc))
    
    if args.voice_conversion is not None:
        print('Initializing Voice Conversion...')
        generator_vc = DiffVC(params_vc.n_mels, params_vc.channels, params_vc.filters, params_vc.heads, params_vc.layers, params_vc.kernel, params_vc.dropout, params_vc.window_size,params_vc.enc_dim, params_vc.spk_dim, params_vc.use_ref_t, params_vc.dec_dim, params_vc.beta_min, params_vc.beta_max)
        
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            generator_vc = generator_vc.cuda()
            generator_vc.load_state_dict(torch.load(vc_path))
        else:
            generator_vc.load_state_dict(torch.load(vc_path, map_location='cpu'))
    
        with open(hfg_path + 'config.json') as f:
            h_uni = AttrDict(json.load(f))
        if use_gpu:
            hifigan_universal = HiFiGAN(h_uni).cuda()
            hifigan_universal.load_state_dict(torch.load(hfg_path + 'generator')['generator'])
        else:
            hifigan_universal = HiFiGAN(h_uni)
            hifigan_universal.load_state_dict(torch.load(hfg_path + 'generator',  map_location='cpu')['generator'])

    
        hifigan_universal.remove_weight_norm()

        enc_model_fpath = Path('checkpts_vc/spk_encoder/pretrained.pt') # speaker encoder path
        if use_gpu:
            spk_encoder.load_model(enc_model_fpath, device="cuda")
        else:
            spk_encoder.load_model(enc_model_fpath, device="cpu")
        
        ___ = hifigan_universal.eval()    
        __ = generator_vc.eval()
        print(f'Number of Voice Conversion parameters: {generator_vc.nparams}')
    _ = generator.eval()
    print(f'Number of GRAD-TTS parameters: {generator.nparams}')


    
    print('Initializing HiFi-GAN...')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.eval()
    vocoder.remove_weight_norm()
    
    with open(args.file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    cmu = bndict.BNDict('data/bahnar_lexicon.txt')
    
    with torch.no_grad():
        for i, text in enumerate(texts):
            print(f'Synthesizing {i} text...', end=' ')
            x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols)))[None]
            x_lengths = torch.LongTensor([x.shape[-1]])
            
            t = dt.datetime.now()
            y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=args.timesteps, temperature=1.3,
                                                   stoc=False, spk=spk, length_scale=0.91)
            t = (dt.datetime.now() - t).total_seconds()
            print(f'Grad-TTS RTF: {t * 22050 / (y_dec.shape[-1] * 256)}')

            audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
            
            write(f'./out/sample_GradTTS_{i}.wav', 22050, audio)
            
            
            if args.voice_conversion is not None:
                src_path = f'out/sample_GradTTS_{i}.wav' # path to source utterance
                tgt_path = args.voice_conversion
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
                mel_encoded, mel_ = generator_vc.forward(mel_source, mel_source_lengths, mel_target, mel_target_lengths, embed_target, n_timesteps=5, mode='ml')
                mel_synth_np = mel_.cpu().detach().squeeze().numpy()
                mel_source_np = mel_.cpu().detach().squeeze().numpy()
                mel = torch.from_numpy(mel_spectral_subtraction(mel_synth_np, mel_source_np, smoothing_window=1)).float().unsqueeze(0)
                if use_gpu:
                    mel = mel.cuda()
    
                with torch.no_grad():
                    audio = (hifigan_universal.forward(mel).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
                    write(f'./out/sample_VC_{i}.wav', 22050, audio)
            
    

    print('Done. Check out `out` folder for samples.')