from main import generator, generator_fm, dct, dct_fm, hifigan, infer, output_sampling_rate
from io import StringIO
from flask import Flask, make_response, request
import io
from scipy.io.wavfile import write
import base64
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def make_audio(y):
    with torch.no_grad():
        audio = hifigan.forward(y).cpu().squeeze().clamp(-1, 1).detach().numpy()

    audio = audio * 4
    bytes_wav = bytes()
    byte_io = io.BytesIO(bytes_wav)
    write(byte_io, output_sampling_rate, audio)
    wav_bytes = byte_io.read()

    audio_data = base64.b64encode(wav_bytes).decode('UTF-8')
    return audio_data

@app.route('/speak', methods=['POST'])
def speak():

    data = request.get_json()
    input_text = data["text"]
    if "gender" in data:
        gender = data["gender"]
    else:
        gender = "both"

    # generate_wav_file should take a file as parameter and write a wav in it
    if gender == "male":
        y = infer(input_text, generator, dct)
    elif gender == "female":
        y = infer(input_text, generator_fm, dct_fm)
    else:
        y = infer(input_text, generator, dct)
        y_fm = infer(input_text, generator_fm, dct_fm)
        
    audio_data = make_audio(y)
    
    if gender == "both":
        audio_data_fm = make_audio(y_fm)
        response = make_response({"speech": audio_data, "speech_fm": audio_data_fm})
        return response

    response = make_response({"speech": audio_data})
    return response