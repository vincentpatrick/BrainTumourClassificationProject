#from ctypes import resize
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image #pil is used to read images in python
import tensorflow as tf

app = FastAPI()
#set an input shape
input_shape = (256, 256)
#load my tensorflow module
MODEL = tf.keras.models.load_model("../models/3")
CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

@app.get("/ping")
#tell the user that the app is still running
async def ping():
    return "hello, I'm alive"

def read_file_as_image(data) -> np.ndarray:
    #image = np.array(Image.open(BytesIO(data)))
    PIL_image = Image.open(BytesIO(data))
    return PIL_image

def preprocess(image: Image.Image):
    #preprocess the image, resize the image to its input shape first
    #then set them as a numpy array
    image = image.resize(input_shape)
    image = np.asfarray(image)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    #pass
    #read the files back
    bytes = await file.read()
    #read the image
    image = read_file_as_image(bytes)
    #preprocess
    image = preprocess(image)
    #set the image into batches
    img_batch = np.expand_dims(image, 0)
    prediction = MODEL.predict(img_batch)
    
    #pick the highest prediction
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return{
        'class':predicted_class,
        'confidence':float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port = 8000)

