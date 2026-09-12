from fastapi import FastAPI, UploadFile, File
import os
import requests

app = FastAPI()

API_URL = "https://router.huggingface.co/hf-inference/models/umm-maybe/AI-image-detector"
HF_TOKEN = os.getenv("HF_TOKEN")

# Hugging Face'in beklediği content-type başlığını ekliyoruz
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/octet-stream"
}

@app.get("/")
def home():
    return {"status": "AI Image Detector API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        response = requests.post(API_URL, headers=HEADERS, data=contents)
        
        if response.status_code != 200:
            return {"error": f"Hugging Face API Error: {response.text}"}

        result = response.json()
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}