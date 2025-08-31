from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

classes = ['NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']

def predict(img_path):
    model = load_model('models/chest_cnn_model.h5')
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)/255.0
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)
    print(f"Prediction: {classes[np.argmax(pred)]}")

# Example:
# predict("data/test/NORMAL/sample.jpg")
