import pandas as pd 
import numpy as np 
import json 
import librosa
from pyAudioAnalysis import ShortTermFeatures
import parselmouth

max_audio_length = 16000 * 90  # 假設音訊的最大長度是16000個樣本點
target_sampling_rate = 16000
def speech_file_to_array_fn(path):
    speech_array, sampling_rate = librosa.load(path, sr=target_sampling_rate)
    mono_waveform = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=target_sampling_rate)
    return mono_waveform, sampling_rate

def save_data(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=2)

file_path_json = '/share/nas165/peng/whisperX/LTTC_Intermediate.json'
# file_path_json = '/share/nas165/peng/whisperX/data.json'
file_path = '/share/nas165/peng/thesis_project/delivery_feat/Delivery_0508.csv'

# open json
fin = open(file_path_json)
data = json.load(fin)
# print(data['112010103002'])
# open origin
df = pd.read_csv(file_path)

new_dict = {}

for index, row in df.iterrows():
    temp_dict = {}
    # if index == 2:
    #     break
    print(index)
    speaker_id = str(row['speaker_id'])
    file_path = row['wav_path']

    x, Fs = speech_file_to_array_fn(file_path)
    strr = ""
    temp_dict['transcription'] = ''
    temp_dict['words'] = []
    for index in range(len(data[speaker_id]['word_segments'])):
        # print(data[speaker_id]['word_segments'][index])
        strr += data[speaker_id]['word_segments'][index]['word'] + " "
        try:
            start_sec = data[speaker_id]['word_segments'][index]['start']
            end_sec = data[speaker_id]['word_segments'][index]['end']

            audio_segment_duration = end_sec - start_sec
            duration = end_sec - start_sec

            if audio_segment_duration <= 0.07:
                tt = (0.08 - audio_segment_duration) / 2
                start_sec -= tt
                end_sec += tt

            start_sample = int(start_sec * Fs)
            end_sample = int(end_sec * Fs)

            # 截取指定时间段的音频数据
            segment = x[start_sample:end_sample]

            # 设置帧大小和步长
            frame_size = int(0.050 * Fs)  # 50ms
            hop_size = int(0.025 * Fs)    # 25ms

            F, feature_names = ShortTermFeatures.feature_extraction(segment, Fs, frame_size, hop_size)
            # 提取特定特征
            zero_crossing_index = feature_names.index('zcr')
            energy_index = feature_names.index('energy')
            energy_entropy_index = feature_names.index('energy_entropy')
            spectral_centroid_index = feature_names.index('spectral_centroid')

            zcr = F[zero_crossing_index, :]
            zero_crossing_rate = np.mean(zcr)
            energy = F[energy_index, :]
            std_dev_energy = np.std(energy)
            energy_entropy = F[energy_entropy_index, :]
            spectral_centroid = F[spectral_centroid_index, :]

            energy_threshold = np.median(energy)  # Threshold for energy
            zcr_threshold = np.median(zcr)        # Threshold for zero-crossing rate

            # Classify frames as voiced or unvoiced
            voiced_frames = (energy > energy_threshold) & (zcr < zcr_threshold)
            unvoiced_frames = (energy <= energy_threshold) | (zcr >= zcr_threshold)

            # Calculate ratios
            voiced_count = np.sum(voiced_frames)
            unvoiced_count = np.sum(unvoiced_frames)
            voiced_to_unvoiced_ratio = voiced_count / unvoiced_count if unvoiced_count != 0 else float('inf')

            # pitch 
            snd = parselmouth.Sound(file_path)
            snd_part = snd.extract_part(from_time=start_sec, to_time=end_sec, preserve_times=True)
            # pitch
            pitch = snd_part.to_pitch()
            pitch_values = pitch.selected_array['frequency']
            pitch_values[pitch_values==0] = np.nan
            mean_pitch = np.nanmean(pitch.selected_array['frequency'])
            # intensity
            intensity = snd_part.to_intensity()
            mean_intensity = intensity.values.mean()
            # local Jitter, local Shimmer, rap jitter
            point_process = parselmouth.praat.call([snd_part, pitch], "To PointProcess (cc)")
            localJitter = parselmouth.praat.call(point_process, "Get jitter (local)", start_sec, end_sec, 0.0001, 0.05, 1.3)
            localShimmer =  parselmouth.praat.call([snd, point_process], "Get shimmer (local)", start_sec, end_sec, 0.0001, 0.05, 1.3, 1.6)
            rapJitter = parselmouth.praat.call(point_process, "Get jitter (rap)", start_sec, end_sec, 0.0001, 0.05, 1.3)

            if np.isnan(localJitter):
                localJitter = 0
            if np.isnan(localShimmer):
                localShimmer = 0
            if np.isnan(rapJitter):
                rapJitter = 0

            # {'speaker_id':'112010103012', 'words': [{'word': 'creek.', 'start': 89.929, 'end': 90.11, 'score': 0.996, 'mean_pitch': 183.60583384821885, 'mean_intensity': 58.034142143289316}]}
            temp_dict['words'].append({**data[speaker_id]['word_segments'][index], 
                                    'duration': duration,
                                    'mean_pitch':mean_pitch,
                                    'mean_intensity':mean_intensity,
                                    'localJitter': localJitter,
                                    'localShimmer': localShimmer,
                                    'rapJitter': rapJitter,
                                    'std_energy': std_dev_energy,
                                    'avg_spectral': np.mean(energy_entropy),
                                    'avg_energy_entropy': np.mean(spectral_centroid),
                                    'zero_cross_rate': zero_crossing_rate,
                                    'v_to_uv_ratio': float(voiced_to_unvoiced_ratio),
                                    'voice_count': float(voiced_count),
                                    'unvoice_count': float(unvoiced_count),                             
                                    })
        except:
            continue
        # print(temp_dict)
    
    
    temp_dict['transcription'] = strr.strip()
    new_dict[speaker_id] = temp_dict

    if (index + 1) % 10 == 0:  # 每10个项目保存一次
        print('Save 10 files')
        save_data(new_dict, 'LTTC_Intermediate_word_level_0509.json')

    # mean_pitch	mean_intensity	duration long_silence	silence	

    # temp_dict['utt_mean_pitch'] = df['mean_pitch']
    # temp_dict['utt_mean_intensity'] = df['mean_intensity']
    # temp_dict['utt_duration'] = df['duration']

    # temp_dict['utt_long_silence'] = df['long_silence']
    # temp_dict['utt_long_silence_num'] = str(df['long_silence_num'])
    # temp_dict['utt_mean_long_silence'] = df['mean_long_silence']

    # temp_dict['utt_silence'] = df['silence']
    # temp_dict['utt_silence_num'] = str(df['silence_num'])
    # temp_dict['utt_mean_silence'] = df['mean_silence']

    # temp_dict['utt_std_energy'] = df['std_energy']
    # temp_dict['utt_avg_spectral'] = df['avg_spectral']
    # temp_dict['utt_avg_energy_entropy'] = df['avg_energy_entropy']
    # temp_dict['utt_zero_cross_num'] = str(df['zero_cross_num'])

    
    # # 输出结果
    # print("Energy Entropy:", type(np.mean(energy_entropy)))
    # print("Standard Deviation of Energy:", std_dev_energy)
    # print("Spectral Centroid:", np.mean(spectral_centroid))
    # print("Zero-Crossing Rate:", np.mean(zero_crossing_rate))
    # print()
    # print(f"Voiced to Unvoiced Ratio: {voiced_to_unvoiced_ratio}")
    # print(f"Total Voiced Frames: {voiced_count}")
    # print(f"Total Unvoiced Frames: {unvoiced_count}")
    # print()
    # print(f"mean pitch: {mean_pitch}")
    # print(f"mean intensity {mean_intensity}")
    # print(f"Local Jitter: {localJitter}")
    # print(f"Local Shimmer: {localShimmer}")
    # print(f"Local Rap Jitter: {rapJitter}")


with open('LTTC_Intermediate_word_level_0509.json',  'w') as file:
    json.dump(new_dict, file, indent=2)

# print(df)
# print(data['112010103002']['segments'])
# print(data['112010103002']['word_segments'][0])#['text'].strip())
# print(data['112010103002']['segments'][1]['text'].strip())
# print(data['112010103002']['segments'][1]['text'].strip())
# speaker_id = '112010103002'
# speaker_id = '112010103012'
# file_path = f'/share/nas165/peng/desktop/ETS_nn/LTTC_Resample/IS-1572_sliced/{speaker_id}_300.wav'

# print(strr) 
# print(data['112010103002']['segments'][0]['words'][0]['end'])

# for index, row in df.iterrows():
#     speaker_id = row['speaker_id']
#     sentence = row[speaker_id]['segments'][0]['text'].strip()
#     sentence_start = row[speaker_id]['segments'][0]['text']['start']
#     end = row[speaker_id]['segments'][0]['text']['end']
#     print(data[speaker_id])