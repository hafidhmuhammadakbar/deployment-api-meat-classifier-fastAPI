from fastapi import FastAPI, HTTPException, Form
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
from io import BytesIO

app = FastAPI()

model = tf.keras.models.load_model('beef_pork_horse_classifier.h5')

@app.get('/')
async def hello_world():
    return {'message': 'Hello, World!'}

@app.get('/models')
async def get_models():
    return {'models': 'MobileNetV3Large', 'framework': 'TensorFlow', 'task': 'Image Classification for Beef, Pork, and Horse', 'accuracy': '97.43%', 'input': 'URL', 'output': 'Predicted class and probabilities', 'model_url': 'https://www.kaggle.com/code/hafidhmuhammadakbar/mobilenetv3large-fix'}

# using form data
# @app.post('/models')
# async def predict(url: str = Form(...)):
#     # Download the image from the URL
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         image = Image.open(BytesIO(response.content))
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f'Failed to download image from URL: {str(e)}')

#     # Resize image
#     image = image.resize((224, 224))

#     # Preprocess image
#     image = np.expand_dims(image, axis=0)

#     # Make prediction
#     predictions = model.predict(image)
#     predicted_label = np.argmax(predictions, axis=1)[0]
#     probabilities = tf.reduce_max(predictions, axis=1) * 100

#     class_names = ['Horse', 'Meat', 'Pork']
#     predicted_class = class_names[predicted_label]
#     probabilities_class = '%.2f' % probabilities.numpy()[0]

#     return {'predicted_class': predicted_class, 'probabilities': probabilities_class}

# using input json
@app.post('/models')
async def predict(data: dict):
    url = data.get('url')
    if not url:
        raise HTTPException(status_code=400, detail='URL field is required')

    # Download the image from the URL
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Failed to download image from URL: {str(e)}')


    # Resize image
    image = image.resize((224, 224))

    # Preprocess image
    image = np.expand_dims(image, axis=0)

    # Make prediction
    predictions = model.predict(image)
    predicted_label = np.argmax(predictions, axis=1)[0]
    probabilities = tf.reduce_max(predictions, axis=1) * 100

    class_names = ['Horse', 'Meat', 'Pork']
    predicted_class = class_names[predicted_label]
    probabilities_class = '%.2f' % probabilities.numpy()[0]

    return {'predicted_class': predicted_class, 'probabilities': probabilities_class}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, port=8000)