from tensorflow import keras
import cv2
from PIL import Image
import numpy as np
import time


class CancerDetector:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)
        self.labels = ['Glioma Tumor', 'No Tumor', 'Meningioma Tumor', 'Pituitary Tumor']
    def predict(self, image_path) -> None:
        img = Image.open(image_path)
        opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = cv2.resize(opencvImage,(150,150))
        img = np.expand_dims(img, 0)
        p = self.model.predict(img)[0]
        max_index = np.argmax(p)
        cancer_type = self.labels[max_index]
        return {'posibilities': p, 'type': cancer_type}

temp_image_path = 'temp'    
def create_random_temp_path(path):  
    filename, extension = path.split('.')
    file_path = f'./api/{temp_image_path}/{filename}{time.time()}.{extension}'
    return file_path