import io
import sys
import dotenv
import gdown
from fastapi import (
    FastAPI,
    File,
    UploadFile,
)
from typing import Callable
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from Inference import predict, predict_whisper
import tempfile
import os
from utils.Translation import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
app = FastAPI()
dotenv.load_dotenv()


# Function to get the model and tokenizer from Google Drive instead of putting it in the repo
def download_file_from_google_drive(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)


# Helper function to handle file upload and prediction
async def handle_audio_upload(file: UploadFile, predict_function: Callable):
    # Read the uploaded audio file into memory
    contents = await file.read()

    # Get the current working directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}", flush=True)

    # Create a temporary file in the current working directory
    with tempfile.NamedTemporaryFile(
            dir=current_dir, delete=False, suffix=".wav"
    ) as tmp_file:
        tmp_file.write(contents)
        tmp_file_path = tmp_file.name  # Get the path of the temp file

    try:
        # Pass the path of the saved file to the predict function
        print(f"Temporary file created at: {tmp_file_path}", flush=True)
        result = predict_function(tmp_file_path)
    finally:
        # Clean up the temporary file after prediction
        os.remove(tmp_file_path)
        print(f"Temporary file deleted: {tmp_file_path}", flush=True)

    return result


# download_file_from_google_drive(
#     "1wYF0uHMHWdWb6G2XOB6dLQj3LWyz8u5X", "./ASR_2_1_300.pth"
# )
# download_file_from_google_drive(
#     "19hitohi6MgNPpTvsTqvt9fmQLWPxD9ky", "./translate_v1.pth"
# )

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ar")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


class TranslationRequest(BaseModel):
    text: str


@app.post("/translate/auto")
async def translateOpenL(request: TranslationRequest):
    response = translate_openl(request.text)
    return {"translation": response}


@app.post("/translate/en")
async def translate_endpoint(request: TranslationRequest):
    response = translate(request.text)
    return {"translation": response}


@app.post("/translate/en-ar")
async def translate_endpoint(request: TranslationRequest):
    input_text = ">>ar<<" + request.text
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"translation": response}


# Endpoint for generic audio-to-text conversion
@app.post("/audio2text")
async def upload_audio(file: UploadFile = File(...)):
    result = await handle_audio_upload(file, predict)
    return {"text": result}


# Endpoint for Whisper ASR
@app.post("/whisper-asr")
async def whisper_asr(file: UploadFile = File(...)):
    result = await handle_audio_upload(file, predict_whisper)
    return {"text": result}
