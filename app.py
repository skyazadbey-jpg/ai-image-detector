from fastapi import FastAPI, UploadFile, File
import requests
import io

app = FastAPI()

# Hugging Face Inference API adresi (sunucusuz / serverless)
API_URL = "https://api-inference.huggingface.co/models/umm-maybe/AI-image-detector"
# İstersen daha sonra kendi Hugging Face ücretsiz token'ını buraya yazabilirsin, şimdilik token olmadan da temel istek atabilir
HEADERS = {} 

@app.get("/")
def home():
    return {"status": "AI Image Detector API is running (Lightweight Mode)"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Gelen resmi oku
        contents = await file.read()
        
        # Hugging Face API'sine gönder
        response = requests.post(API_URL, headers=HEADERS, data=contents)
        result = response.json()
        
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}