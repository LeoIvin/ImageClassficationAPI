from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np 
from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
import io


app = FastAPI()

origins = [
    "http://localhost:3000",
    'http://172.20.10.2:3000' # React app
    # other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = ResNet50(weights="imagenet")

@app.get('/')
def home():
    return {"Image Classification:" "API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    
    # preprocess the img
    img = img.resize((224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # make prediction
    predictions = model.predict(img)

    # decode prediction
    prediction_class = decode_predictions(predictions, top=1)[0][0]
    return {
    "Prediction": prediction_class[1],
    "ConfidenceScore": round(prediction_class[2]*100, 2)
}

# use curl to test:
# curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: multipart/form-data" -F "file=@C:\Users\HP\Downloads\lion.jfif"
