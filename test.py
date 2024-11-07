import whisper
import sys
import io

model = whisper.load_model("turbo")
result = model.transcribe("test6.mp3", language="ar")

with open("res.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])
