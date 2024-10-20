from fastapi import FastAPI, File, UploadFile
from model.model import predict

app = FastAPI()

@app.post("/")
async def predict_food(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    predicted_class = predict(image_bytes)
    
    return {
        "predicted_class": predicted_class,
    }
