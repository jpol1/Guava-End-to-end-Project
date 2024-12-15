import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model

from modules.data_preprocessing.load_images import load_image_from_stream

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = load_model("trained_models/model_conv8_16_32_dense64-data_augmentation_2.keras")
class_names = ["Anthracnose", "Healthy Guava", "Fruit Fly"]


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = load_image_from_stream(file.file)
        image = np.expand_dims(image, axis=0)

        predictions = model.predict(image)
        predicted_class = class_names[np.argmax(predictions, axis=1)[0]]

        return JSONResponse(
            content={
                "filename": file.filename,
                "predicted_class": predicted_class,
                "confidence": float(np.max(predictions)),
            }
        )
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
