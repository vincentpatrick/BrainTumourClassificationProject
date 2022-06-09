from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image #module to import images
import tensorflow as tf

app = FastAPI()

Model = tf.keras.models.load_model("../models/1")
CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
@app.get("/ping")
#tell the user that the app is still running
async def ping():
    return "hello, I'm alive"

def read_file_as_images(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    #read the file
    bytes = await file.read()
    image = read_file_as_images(bytes)
    img_batch = np.expand_dims(image, 0)
    prediction = Model.predict(img_batch)
    #take the label with the highest probability in prediction
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    #CONFIDENCE
    confidence = np.max(prediction[0])
    #return the predicted class and its confidence level
    return{
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port = 8000)