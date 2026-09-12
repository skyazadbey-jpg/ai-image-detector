from fastapi import FastAPI, UploadFile, File
import os
from huggingface_hub import InferenceClient

app = FastAPI()

# Hugging Face Inference istemcisini güvenli token ile başlatıyoruz
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(model="umm-maybe/AI-image-detector", token=HF_TOKEN)

@app.get("/")
def home():
    return {"status": "AI Image Detector API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        
        # Resmi InferenceClient üzerinden doğrudan modele gönderiyoruz
        result = client.image_classification(image_bytes)
        
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}