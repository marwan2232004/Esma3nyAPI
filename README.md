# 🌟 Esma3ny API 🌟

Welcome to the ASR and Translation API! This project focuses on converting spoken Egyptian Arabic into written text and translating English text into Arabic. The architecture is inspired by OpenAI's Whisper model and utilizes a custom Transformer-based implementation.

Frontend repository: [Esma3ny Frontend Repository](https://github.com/marwan2232004/Esma3ny)

## 🚀 Features

- **Automatic Speech Recognition (ASR)**: Converts spoken Egyptian Arabic into written text.
- **Translation**: Translates English text into Arabic using the OpenL Translation API.
- **Frontend**: Built with React Vite.
- **Backend**: Powered by FastAPI.
- **Deployment**: Hosted on Azure App Services and a Virtual Machine.

## 📚 Architecture

- **ASR Component**: Inspired by OpenAI's Whisper model, leveraging a custom Transformer-based implementation.
- **Translation Component**: Integrates the OpenL Translation API for automatic language detection and translation from any language into Arabic.

## 🛠️ Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/marwan2232004/Esma3nyAPI.git
   cd Esma3nyAPI
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
3. **Run the server**:
   ```bash
   uvicorn main:app --reload
## 📚 Deployment

- The API is deployed on Azure App Services and a Virtual Machine. However, deployment is currently suspended due to costs.
## 📄 API Endpoints

- **POST /audio2text**: Converts spoken Egyptian Arabic to text.
  - **Request**: Audio file
  - **Response**: JSON with transcribed text

- **POST /translate/en-ar**: Translates English text to Arabic.
  - **Request**: JSON with English text
  - **Response**: JSON with translated Arabic text
    
- **POST /translate/auto**: Translates from any language to Arabic.
  - **Request**: JSON with text
  - **Response**: JSON with translated Arabic text
 
## ASR Transformer Architecture



![Transformer_Architecture_complete drawio (1)](https://github.com/user-attachments/assets/139ad3f1-9ba0-491d-8dba-b749e8fc4e32)


