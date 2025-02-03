import gc
import librosa
import torch
import numpy as np
from utils.Audio_Processing import preprocess_audio
from utils.Constants import *
from utils.MMS import get_device, MMS, greedyDecoder
from utils.NLP import preprocess_vocab
from transformers import WhisperFeatureExtractor
from peft import PeftModel, PeftConfig
from transformers import WhisperForConditionalGeneration, BitsAndBytesConfig, WhisperProcessor

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


############################################################################################

whisper_model_name = "openai/whisper-large-v3-turbo"
task = "transcribe"
language = "Arabic"


def predict_whisper(audio_file):
    device = get_device()
    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model_name)
    processor = WhisperProcessor.from_pretrained(whisper_model_name, language=language, task=task)
    audio_data, _ = librosa.load(audio_file, sr=SAMPLE_RATE)  # Load the audio
    input_features = feature_extractor(audio_data, sampling_rate=SAMPLE_RATE).input_features  # Extract features
    input_features = [{'input_features': input_features[0]}]
    input_features = processor.feature_extractor.pad(input_features, return_tensors="pt")
    input_features = input_features["input_features"].to(device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    #----------------------------- Load Whisper model --------------------------#
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Use 4-bit quantization
        bnb_4bit_compute_dtype=torch.float16  # Use FP16 for computations
    )
    model_id = "marwan2232004/whisper-turbo-egyptian-arabic-6000steps"
    peft_config = PeftConfig.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path,
        quantization_config=quantization_config,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, model_id)
    model.config.use_cache = True
    # --------------------------------- Inference ---------------------------------#
    with torch.amp.autocast('cuda'):
        with torch.no_grad():
            generated_tokens = (
                model.generate(
                    input_features=input_features.to(device),
                    attention_mask=input_features.ne(processor.tokenizer.pad_token_id),
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=255,
                )
                .cpu()
                .numpy()
            )
            result = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        del generated_tokens
    gc.collect()
    return result[0]
