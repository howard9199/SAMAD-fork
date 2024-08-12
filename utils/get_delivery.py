import parselmouth
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from detec_silence import *
import re
from pyAudioAnalysis import ShortTermFeatures
import librosa


max_audio_length = 16000 * 90  # 假設音訊的最大長度是16000個樣本點
target_sampling_rate = 16000
def speech_file_to_array_fn(path):
    speech_array, sampling_rate = librosa.load(path, sr=target_sampling_rate)
    mono_waveform = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=target_sampling_rate)

    return mono_waveform, sampling_rate

# speaker_id = '112030203021' # 2
# speaker_id = '112020503060' # 3
speaker_id = '112050203012' # 3
# speaker_id = '472010112020' # 3
# speaker_id = '472010112055' # 3.5
# speaker_id = '472010112010' # 4
# speaker_id = '472010112012' # 4
# speaker_id = '112030203s029' # 5
# speaker_id = '112010603024' # 5
# file_path = f'/share/nas165/peng/desktop/ETS_nn/LTTC_Resample/IS-1572_sliced/{speaker_id}_300.wav'

file_type = 'fulltest' # test fulltest train
path = f'/share/nas165/peng/thesis_project/ablation_0417/data/LTTC_Intemidiate/Unseen_1964/{file_type}_1964.csv'
outputname = f'./{file_type}_acoustic_0516.csv'
df = pd.read_csv(path)

new_dict = {}
for index, row in df.iterrows():

    # if index == 3:
    #   break

    file_path = row['wav_path']
    print(f'{index}: {file_path}')
    snd = parselmouth.Sound(file_path)
    
    x, Fs = speech_file_to_array_fn(file_path)

    # Intensity
    intensity = snd.to_intensity()
    mean_intensity = intensity.values.mean()
    # print(f"Mean Intensity: {mean_intensity}")


    # Pitch
    pitch = snd.to_pitch()
    mean_pitch = np.nanmean(pitch.selected_array['frequency'])
    # print(f"Mean pitch: {mean_pitch}")

    print(f"Duration: {snd.xmin}, {snd.xmax}")
    duration = snd.xmax - snd.xmin
    point_process = parselmouth.praat.call([snd, pitch], "To PointProcess (cc)")
    localJitter = parselmouth.praat.call(point_process, "Get jitter (local)", snd.xmin, snd.xmax, 0.0001, 0.02, 1.3)
    # print(f"Local Jitter: {localJitter}")

    localShimmer =  parselmouth.praat.call([snd, point_process], "Get shimmer (local)", snd.xmin, snd.xmax, 0.0001, 0.02, 1.3, 1.6)
    # print(f"Local Shimmer: {localShimmer}")

    rapJitter = parselmouth.praat.call(point_process, "Get jitter (rap)", snd.xmin, snd.xmax, 0.0001, 0.02, 1.3)
    # print(f"Local Rap Jitter: {rapJitter}")


    # pause
    long_silences = detect_silences(file_path, 0.495, 0, getSoundFileLength(file_path))
    silences = detect_silences(file_path, 0.145, 0, getSoundFileLength(file_path))
    # print(silences)
    # print(f'silence: {len(silences)}')
    # print(f'Long silence: {len(long_silences)}')
    long_silence = sum(entry['duration'] for entry in long_silences)
    silence = sum(entry['duration'] for entry in silences)
    mean_long_silence = np.mean([entry['duration'] for entry in long_silences])
    mean_silence = np.mean([entry['duration'] for entry in silences])

    print(mean_long_silence)
    print(mean_silence)
    # harmonicity05 = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 500, 0.1, 1.0)
    # hnr05 = parselmouth.praat.call(harmonicity05, "Get mean", snd.xmin, snd.xmax)

    # harmonicity15 = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 1500, 0.1, 1.0)
    # hnr15 = parselmouth.praat.call(harmonicity15, "Get mean", snd.xmin, snd.xmax)

    # harmonicity25 = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 2500, 0.1, 1.0)
    # hnr25 = parselmouth.praat.call(harmonicity25, "Get mean", snd.xmin, snd.xmax)

    # harmonicity05_info = harmonicity05.info()
    # print(harmonicity05_info)
    # sounding05_match   = re.search(r"Number of frames: \d+ \((\d+) sounding\)", harmonicity05_info)
    # median05_match     = re.search(r"Median (\d+\.\d+) dB", harmonicity05_info)
    # minimum05_match    = re.search(r"Minimum: ([-+]?\d+\.\d+) dB", harmonicity05_info)
    # maximum05_match    = re.search(r"Maximum: (\d+\.\d+) dB", harmonicity05_info)
    # averge05_match     = re.search(r"Average: (\d+\.\d+) dB", harmonicity05_info)
    # standard05_dev     = re.search(r"Standard deviation: ([\d.]+) dB", harmonicity05_info)
    

    # harmonicity15_info = harmonicity15.info()
    # sounding15_match = re.search(r"Number of frames: \d+ \((\d+) sounding\)", harmonicity15_info)
    # median15_match = re.search(r"Median (\d+\.\d+) dB", harmonicity15_info)
    # minimum15_match = re.search(r"Minimum: ([-+]?\d+\.\d+) dB", harmonicity15_info)
    # maximum15_match = re.search(r"Maximum: (\d+\.\d+) dB", harmonicity15_info)
    # averge15_match = re.search(r"Average: (\d+\.\d+) dB", harmonicity15_info)
    # standard15_dev = re.search(r"Standard deviation: ([\d.]+) dB", harmonicity15_info)

    # harmonicity25_info = harmonicity25.info()
    # sounding25_match = re.search(r"Number of frames: \d+ \((\d+) sounding\)", harmonicity25_info)
    # median25_match = re.search(r"Median (\d+\.\d+) dB", harmonicity25_info)
    # minimum25_match = re.search(r"Minimum: ([-+]?\d+\.\d+) dB", harmonicity25_info)
    # maximum25_match = re.search(r"Maximum: (\d+\.\d+) dB", harmonicity25_info)
    # averge25_match = re.search(r"Average: (\d+\.\d+) dB", harmonicity25_info)
    # standard25_dev = re.search(r"Standard deviation: ([\d.]+) dB", harmonicity25_info)
    
    # print(f'harmonicity05\n: {harmonicity05}')
    # print(f'\n\nharmonicity15\n: {harmonicity15}')
    # print(f'\n\nharmonicity25\n: {harmonicity25}')
    # print(f'Median: {float(median25_match.group(1))}')
    # print(f'Minimum: {float(minimum25_match.group(1))}')
    # print(f'maximum: {float(maximum25_match.group(1))}')
    # print(f'average: {float(averge25_match.group(1))}')
    # print(f'Standard_dev: {float(standard25_dev.group(1))}')
    # print(f'Number of sounding: {float(sounding25_match.group(1))}')

    # # Set frame size and hop size
    # frame_size = int(0.050 * Fs)  # 50 ms
    # hop_size = int(0.025 * Fs)    # 25 ms

    # # Extract short-term features
    # F, feature_names = ShortTermFeatures.feature_extraction(x, Fs, frame_size, hop_size)

    # # Get indices of the features
    zero_crossing_index = feature_names.index('zcr')
    energy_index = feature_names.index('energy')
    energy_entropy_index = feature_names.index('energy_entropy')
    spectral_centroid_index = feature_names.index('spectral_centroid')

    # # Get specific features
    zcr = F[zero_crossing_index, :]
    energy = F[energy_index, :]
    energy_entropy = F[energy_entropy_index, :]
    spectral_centroid = F[spectral_centroid_index, :]

    # Compute the standard deviation of energy
    std_dev_energy = np.std(energy)

    # The number of zero crossing
    zero_crossings = librosa.zero_crossings(x, pad=False)
    total_zero_crossings = sum(zero_crossings)

    # voice to unvoice ratio
    # Thresholds might need calibration based on your specific audio characteristics
    energy_threshold = np.median(energy)  # Threshold for energy
    zcr_threshold = np.median(zcr)        # Threshold for zero-crossing rate

    # Classify frames as voiced or unvoiced
    voiced_frames = (energy > energy_threshold) & (zcr < zcr_threshold)
    unvoiced_frames = (energy <= energy_threshold) | (zcr >= zcr_threshold)

    # Calculate ratios
    voiced_count = np.sum(voiced_frames)
    unvoiced_count = np.sum(unvoiced_frames)
    voiced_to_unvoiced_ratio = voiced_count / unvoiced_count if unvoiced_count != 0 else float('inf')

    # print(f"Voiced to Unvoiced Ratio: {voiced_to_unvoiced_ratio}")
    # print(f"Total Voiced Frames: {voiced_count}")
    # print(f"Total Unvoiced Frames: {unvoiced_count}")


    # # Output the results
    # print("Energy Entropy (average over frames):", np.mean(energy_entropy))
    # print("Standard Deviation of Energy:", std_dev_energy)
    # print("Average Spectral Centroid:", np.mean(spectral_centroid))
    # print("Total number of zero crossings: ", total_zero_crossings)


    new_dict[row['speaker_id']] = {'mean_pitch':mean_pitch,
                                   'mean_intensity':mean_intensity, 
                                   'duration': duration,
                                   'localJitter': localJitter,
                                   'localShimmer': localShimmer,
                                   'rapJitter': rapJitter,
                                   'long_silence': long_silence,
                                   'silence': silence,

                                   'long_silence_num': len(long_silences),
                                   'silence_num': len(silences),
                                   'std_energy': std_dev_energy,
                                   'avg_spectral': np.mean(energy_entropy),
                                   'avg_energy_entropy': np.mean(spectral_centroid),
                                   'zero_cross_num': total_zero_crossings,
                                   'v_to_uv_ratio': voiced_to_unvoiced_ratio,
                                   'voice_count': voiced_count,
                                   'unvoice_count': unvoiced_count,
                                  # 'mean_long_silence': mean_long_silence,
                                  # 'mean_silence': mean_silence,

                                #    'sounding05_match': float(sounding05_match.group(1)),  
                                #    'median05_match': float(median05_match.group(1)),    
                                #    'minimum05_match': float(minimum05_match.group(1)),   
                                #    'maximum05_match': float(maximum05_match.group(1)),   
                                #    'averge05_match': float(averge05_match.group(1)),    
                                #    'standard05_dev': float(standard05_dev.group(1)),
                                #    'sounding15_match': float(sounding15_match.group(1)),  
                                #    'median15_match': float(median15_match.group(1)),    
                                #    'minimum15_match': float(minimum15_match.group(1)),   
                                #    'maximum15_match': float(maximum15_match.group(1)),   
                                #    'averge15_match': float(averge15_match.group(1)),    
                                #    'standard15_dev': float(standard15_dev.group(1)),
                                #    'sounding25_match': float(sounding25_match.group(1)),  
                                #    'median25_match': float(median25_match.group(1)),    
                                #    'minimum25_match': float(minimum25_match.group(1)),   
                                #    'maximum25_match': float(maximum25_match.group(1)),   
                                #    'averge25_match': float(averge25_match.group(1)),    
                                #    'standard25_dev': float(standard25_dev.group(1)),
                                  }
print(new_dict)
new_data = pd.DataFrame.from_dict(new_dict, orient='index')
new_data = new_data.rename_axis('speaker_id')
# print(new_data)
new_data.to_excel(outputname)

# merged_df = pd.merge(df, new_data, on='speaker_id')
# print(merged_df)