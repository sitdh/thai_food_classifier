from pathlib import Path
from io import BytesIO
from fastai.vision.all import *

learn = load_learner('/app/model/thai_food_classifier.pkl')

def preprocess_image(image_bytes):
    img = PILImage.create(BytesIO(image_bytes))
    fastimg = img.convert('RGB')
    return fastimg
    
def predict(image_bytes):
    img = preprocess_image(image_bytes)
    pred, _, probs = learn.predict(img)

    return pred
