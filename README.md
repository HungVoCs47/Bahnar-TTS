<p align="center">
    <img src="resources/reverse-diffusion.gif" alt="drawing" width="500"/>
</p>


# Grad-TTS

This is an End-to-end implementation of the Grad-TTS model and Voice Conversion model based on Diffusion Probabilistic Modelling.


## Abstract

**Demo page** with voiced abstract: [link](https://grad-tts.github.io/).

Recently, denoising diffusion probabilistic models and generative score matching have shown high potential in modelling complex data distributions while stochastic calculus has provided a unified point of view on these techniques allowing for flexible inference schemes. In this paper we introduce Grad-TTS, a novel text-to-speech model with score-based decoder producing mel-spectrograms by gradually transforming noise predicted by encoder and aligned with text input by means of Monotonic Alignment Search. The framework of stochastic differential equations helps us to generalize conventional diffusion probabilistic models to the case of reconstructing data from noise with different parameters and allows to make this reconstruction flexible by explicitly controlling trade-off between sound quality and inference speed. Subjective human evaluation shows that Grad-TTS is competitive with state-of-the-art text-to-speech approaches in terms of Mean Opinion Score.

## Installation

Firstly, install all Python package requirements:

```bash
pip install -r requirements.txt
```

Secondly, build `monotonic_align` code (Cython):

```bash
cd model/monotonic_align; python setup.py build_ext --inplace; cd ../..
```

**Note**: code is tested on Python==3.6.9.


# Voice Conversion

Official implementation of the paper "Diffusion-Based Voice Conversion with Fast Maximum Likelihood Sampling Scheme" (ICLR 2022, Oral). [Link](https://arxiv.org/abs/2109.13821).


## Abstract

**Demo page** with voiced abstract: [link](https://diffvc-fast-ml-solver.github.io/).

Voice conversion is a common speech synthesis task which can be solved in different ways depending on a particular real-world scenario. The most challenging one often referred to as one-shot many-to-many voice conversion consists in copying the target voice from only one reference utterance in the most general case when both source and target speakers do not belong to the training dataset. We present a scalable high-quality solution based on diffusion probabilistic modeling and demonstrate its superior quality compared to state-of-the-art one-shot voice conversion approaches. Moreover, focusing on real-time applications, we investigate general principles which can make diffusion models faster while keeping synthesis quality at a high level. As a result, we develop a novel Stochastic Differential Equations solver suitable for various diffusion model types and generative tasks as shown through empirical studies and justify it by theoretical analysis.



## Inference End-to-End model:
# GradTTS setup:
You should create `Bahnar-TTS\logs\bahnar_exp ` directory and `Bahnar-TTS\checkpts\ ` directory:
1) You can download Grad-TTS trained on Bahnar datasets (22kHz) from [here](https://drive.google.com/drive/u/1/folders/1OMXQ9_t0Vnw7oxdJFWrWZ5h64k6N94D6) and put it under directory `Bahnar-TTS\logs\bahnar_exp `
2) You can download HiFi-GAN trained on Bahnar datasets (22kHz) from [here](https://drive.google.com/drive/u/1/folders/1IdvgD1ja0WTYnFDhtoBeaXN9Is-rexUn) and put it under directory `Bahnar-TTS\checkpts\ `

After setup phase, the repo should look like this:

```bash
── Bahnar-TTS
    │
    │
    │
    ├── checkpts
    │     └── hifigan.pt
    │
    └── logs
          └── bahnar_exp
                    └── grad_1344.pt
```


# Voice Conversion setup:
You should create `Bahnar-TTS\checkpts_vc ` directory, under `Bahnar-TTS\checkpts_vc ` you should create 3 sub-directories:
- `Bahnar-TTS\checkpts_vc\spk_encoder `
- `Bahnar-TTS\checkpts_vc\vc `
- `Bahnar-TTS\checkpts_vc\vocoder `
1) You can download pretrained Voice Conversion (22kHz) from [here](https://drive.google.com/drive/u/1/folders/1148vd2twFbmtlsj9RKbjn1I-EnV1ntvH) and put it under directory `Bahnar-TTS\checkpts_vc\vc `
2) You can download pretrained Vocoder (22kHz) and config from [here](https://drive.google.com/drive/u/1/folders/13ZrHBWLtINTzUpXcGOI-mYOvw1lwZPIq) and put it under directory `Bahnar-TTS\checkpts_vc\vocoder `
3) You can download pretrained Encoder (22kHz) from [here](https://drive.google.com/drive/u/1/folders/1nu5al-OZs-jL0o5w2b5YzWDS5SWMJqUJ) and put it under directory `Bahnar-TTS\checkpts_vc\spk_encoder `

After setup phase, the repo should look like this:
```bash
── Bahnar-TTS
    └──checkpts_vc
          ├── spk_encoder
          │         │
          │         └──────  pretrained.pt
          │
          │
          ├── vc
          │    │
          │    └──────────── vc_vctk_wodyn.pt
          │
          │
          └── vocoder
                 │
                 ├──────────── generator
                 └──────────── config.json 
```

# Data setup:
After put necessary all model checkpoints into `checkpts ` folder and `checkpts_vc ` folder. You should create your own data-source. You should create `Bahnar-TTS\document ` to store your own data.
1. Create text file with sentences you want to synthesize like `Bahnar-TTS\document\text\text.txt `
2. Create target audio file you want to converse the voice like `Bahnar-TTS\document\target_sound\0001.wav `

You can dowload the pattern audio file and text file in this [here](https://drive.google.com/drive/u/1/folders/1v40EtocaeHwKeP7j2eUsrTBvUYp5BIwY)

After create the data source, the repo should look like this: 
```bash
── Bahnar-TTS
    └── document
           ├──────── target_sound
           │           └────────── 0001.wav
           │
           └──────── text
                       └────────── text.txt
```
# Inference command:
1) If you want to synthesize only text-to-speech without voice conversion, you should use this command:
    ```bash
    python inference.py -f <your-text-file> -c <grad-tts-checkpoint> -t <number-of-timesteps> -s <speaker-id-if-multispeaker>
    ```
    For example you can run the following command:
    ```bash
    python inference.py -f document/text/text.txt  -c logs\bahnar_exp\grad_1344.pt
    ```
2) If you want to synthesize text-to-speech with voice conversion, you should use this command:
    ```bash
    python inference.py -f <your-text-file> -c <grad-tts-checkpoint> -t <number-of-timesteps> -s <speaker-id-if-multispeaker> -vc <target-speaker-for-voice-conversion>
    ```
    For example you can run the following command:
    ```bash
    python inference.py -f document/text/text.txt  -c logs\bahnar_exp\grad_1344.pt -vc document/target_sound/0001.wav
    ```
3) After inference, check out folder called `out ` for generated audios

# Demo result:
You can check the result of the end-to-end Bahnar Text-To-Speech model and Voice Conversion model [link](https://drive.google.com/drive/u/1/folders/1bLq8fVDlEJFq0Jbdp6FJnpOgmQ1O_IlN)

Male voice is generated by GradTTS and Female Voice is generated by Voice Conversion model.

## References

* HiFi-GAN model is used as vocoder, official github repository: [link](https://github.com/jik876/hifi-gan)
* Monotonic Alignment Search algorithm is used for unsupervised duration modelling, official github repository: [link](https://github.com/jaywalnut310/glow-tts)
* Phonemization utilizes CMUdict, official github repository: [link](https://github.com/cmusphinx/cmudict)
* Voice conversion model, official github repository: [link](https://github.com/huawei-noah/Speech-Backbones/tree/main/DiffVC)
