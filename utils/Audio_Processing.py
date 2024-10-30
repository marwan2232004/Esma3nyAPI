import librosa
import numpy as np
from utils.Constants import *


def pad_or_trim(array, length=N_SAMPLES, axis=-1, padding=True):
    if array.shape[axis] > length:
        array = array.take(indices=range(length), axis=axis)

    if padding & (array.shape[axis] < length):
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = np.pad(array, pad_widths)

    return array


# Function to load and preprocess audio
def preprocess_audio(audio_data):
    spectrogram = librosa.stft(y=audio_data, n_fft=N_FFT, hop_length=HOP_LENGTH)

    spectrogram_mag, _ = librosa.magphase(spectrogram)

    mel_scale_spectrogram = librosa.feature.melspectrogram(
        S=spectrogram_mag, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )

    mel_spectrogram = librosa.amplitude_to_db(mel_scale_spectrogram, ref=np.min)

    del spectrogram, mel_scale_spectrogram, spectrogram_mag

    return mel_spectrogram
