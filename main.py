from fastapi import FastAPI, UploadFile, File
import uvicorn  
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf


app = FastAPI()
MODEL = tf.keras.models.load_model("model.h5")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get('/ping')
async def ping():
    return "Hello! I am alive"

def read_file_as_image(data):
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post('/predict')
async def predict(
    file: UploadFile= File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    prediction = MODEL.predict(img_batch)
    return CLASS_NAMES[np.argmax(prediction)]

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port = 8000)