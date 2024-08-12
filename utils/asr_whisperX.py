import json
import whisperx
import gc 
import pandas as pd 

device = "cuda" 
# audio_file = "/share/corpus/LTTC2023/IS-1764_sliced/IS-1764_sliced/082031101078_300.wav"
# audio_file = "/share/corpus/LTTC2023/IS-1572_sliced/IS-1572_sliced/472020512015_300.wav"
audio_file = "/share/corpus/LTTC2023/IS-1764_sliced/IS-1764_sliced/652100104081_300.wav"
batch_size = 16 # reduce if low on GPU mem
compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# Intermediate
file_path = f'/share/nas165/peng/thesis_project/SAMAD_06/data_simple/LTTC_Intermediate/Intermediate_All_0520.csv'

# High-Intermediate
file_path = '/share/nas165/peng/thesis_project/ablation_0417/data/LTTC_HS/HI_part3_dataset.csv'

df = pd.read_csv(file_path)
data = {}

for index, row in df.iterrows():
    
    print("**************************************************")
    print(index)
    audio_file = row['wav_path']
    
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size, language="en")

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    data[row['speaker_id']] = result
    print(result["segments"])

# Writing JSON data to a file
with open('High-Intermediate.json', 'w') as file:
    json.dump(data, file, indent=2)

# with open('LTTC_Intermediate.json', 'w') as file:
    # json.dump(data, file, indent=2)