from pydub import AudioSegment
from pydub.playback import play

in_path_1 = "demo/short_2495.wav"
in_path_2 = "demo/medium_2495.wav"
audio_out_file = "demo/concat.wav"

# create 1 sec of silence audio segment
one_sec_segment = AudioSegment.silent(duration=1000)  # duration in milliseconds

# read wav files to audio segments
in1 = AudioSegment.from_wav(in_path_1)
in2 = AudioSegment.from_wav(in_path_2)

# Add above three audio segments
final_song = in1 + one_sec_segment + in2

# Save modified audio
final_song.export(audio_out_file, format="wav")
