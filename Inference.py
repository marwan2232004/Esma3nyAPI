import librosa
import numpy as np
from utils.Audio_Processing import preprocess_audio
from utils.Constants import *
from utils.MMS import get_device, MMS, greedyDecoder
from utils.NLP import preprocess_vocab
import torch

############################################################################################


model_path = "./ASR_2_1_300.pth"


############################################################################################


def predict(audio_file):
    device = get_device()

    char2idx, idx2char, vocab_size = preprocess_vocab()

    # load model
    model = MMS(
        vocab_size=vocab_size,
        max_encoder_seq_len=math.ceil(N_FRAMES / 2),
        max_decoder_seq_len=MAX_SEQ_LEN,
        num_encoder_layers=2,
        num_decoder_layers=1,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
    )

    model.load_state_dict(torch.load(model_path, weights_only=False, map_location=device))
    model.to(device)
    model.eval()

    audio_data, _ = librosa.load(audio_file, sr=SAMPLE_RATE)  # Load the audio
    n_chunks = math.ceil(audio_data.shape[0] / N_SAMPLES)  # Get the number of chunks
    # divide the audio into segments of 15 secs
    chunk_size = audio_data.shape[0] if n_chunks == 1 else N_SAMPLES
    audio_segments = [audio_data[i * chunk_size: min(audio_data.shape[0], (i + 1) * chunk_size)]
                      for i in range(n_chunks)]
    result = ""
    for audio_segment in audio_segments:

        mel_spectrogram = preprocess_audio(audio_segment)

        processed_audios = [mel_spectrogram]

        padded_audios = [
            (
                mel_spec.shape[-1],
                np.pad(
                    mel_spec,
                    ((0, 0), (0, N_FRAMES - mel_spec.shape[-1])),
                    mode="constant",
                ),
            )
            for mel_spec in processed_audios
        ]

        result += " " + greedyDecoder(
            model, padded_audios[0][1], padded_audios[0][0], char2idx, idx2char, device
        )

    return result
