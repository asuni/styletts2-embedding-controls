import numpy as np


import soundfile as sf
import sys, os
import tqdm

from scipy.fft import fft, ifft
from scipy.signal import stft, istft
import scipy.signal

OUT_DIR = "prosody_manipulations/"

def normalize(sig, fname_prefix=""):
    return sig / np.max(np.abs(sig))

    #return adjusted_signal

def adjust_spectral_tilt(input_file, tilt_db_per_octave):
    signal, fs = sf.read(input_file)
    # Perform STFT
    f, t, Zxx = scipy.signal.stft(signal, fs, nperseg=1024)
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)

    # Apply tilt in the frequency domain
    tilt = np.log2(f[1:] / f[1]) * tilt_db_per_octave  # Exclude DC first
    tilt = np.insert(tilt, 0, 0)  # Add 0 tilt for DC at index 0
    tilt = tilt[:, np.newaxis]  # Reshape to match the dimensions of Zxx

    adjusted_magnitude = magnitude * 10 ** (tilt / 20)

    # Reconstruct signal
    adjusted_Zxx = adjusted_magnitude * np.exp(1j * phase)
    _, adjusted_signal = scipy.signal.istft(adjusted_Zxx, fs)

    original_energy = np.sum(signal ** 2)
    modified_energy = np.sum(adjusted_signal ** 2)
    energy_scaling_factor = np.sqrt(original_energy / modified_energy)
    adjusted_signal *= energy_scaling_factor  # Scale to keep energy constant
    sf.write(f'{OUT_DIR}/spec_{tilt_db_per_octave}.wav', adjusted_signal, fs)


def praat_f0_mean(audio_file, f_min=50, f_max=500, f0_shift=1.5):
    from parselmouth import praat, Sound
    sound = Sound(audio_file)
    m_obj = praat.call(sound, "To Manipulation", 1e-3, f_min, f_max)

    pitch_tier = praat.call(m_obj, "Extract pitch tier")
    praat.call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, f0_shift)
    praat.call([pitch_tier, m_obj], "Replace pitch tier")
    tgt_audio = praat.call(m_obj, "Get resynthesis (overlap-add)")
    tgt_audio.save(f'{OUT_DIR}/f0mean_{f0_shift}.wav', "WAV")
    
def praat_f0_variance(audio_file, f_min=50, f_max=500, f0_shift=1.5):
    from parselmouth import praat, Sound
    sound = Sound(audio_file)
    m_obj = praat.call(sound, "To Manipulation", 1e-3, f_min, f_max)
    pitch_tier = praat.call(m_obj, "Extract pitch tier")
    mean_f0 = praat.call(pitch_tier, "Get mean (points)...", 0, 0)
    #mean_f0-=20
    std_f0 = praat.call(pitch_tier, "Get standard deviation (points)...", 0, 0)
    praat.call(pitch_tier, "Formula...", f'{mean_f0} * 2 ** ((( 12 * log2(self / {mean_f0})) * {f0_shift}) / 12);')
    praat.call([pitch_tier, m_obj], "Replace pitch tier")
    tgt_audio = praat.call(m_obj, "Get resynthesis (overlap-add)")
    tgt_audio.save(f'{OUT_DIR}/f0std_{f0_shift}.wav', "WAV")


def praat_stretch(audio_file, f_min=50, f_max=500, stretch=1.5):
    from parselmouth import praat, Sound
    sound = Sound(audio_file)
    m_obj = praat.call(sound, "To Manipulation", 1e-3, f_min, f_max)
    dur_tier = praat.call(m_obj, "Extract duration tier")
    praat.call(dur_tier, "Add point", 0.5, stretch)
    praat.call([dur_tier, m_obj], "Replace duration tier")
    tgt_audio = praat.call(m_obj, "Get resynthesis (overlap-add)")
    tgt_audio.save(f'{OUT_DIR}/rate_{stretch}.wav', "WAV")




    
input_file = sys.argv[1]
os.makedirs(OUT_DIR, exist_ok=True)

f_min = 70
f_max = 450

x, sr = sf.read(input_file)

hop_size = int(sr/100.)

f0_multipliers = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6,1.8, 2.0, 2.2, 2.5]
f0_std_multipliers = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7]
rate_multipliers = f0_multipliers
spec_tilt = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5] 


for val in tqdm.tqdm(f0_multipliers):
    praat_f0_mean(input_file, f_min=f_min, f_max=f_max, f0_shift=val)

for val in tqdm.tqdm(f0_std_multipliers):
    praat_f0_variance(input_file, f_min=f_min, f_max=f_max, f0_shift=val)

for val in tqdm.tqdm(rate_multipliers):
    praat_stretch(input_file, f_min=f_min, f_max=f_max, stretch=val)

for val in tqdm.tqdm(spec_tilt):
    adjust_spectral_tilt(input_file, val)
