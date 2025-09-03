import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

classes = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

def predict(img_path):
    model = load_model("saved_model/chestxray_resnet.h5")
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)
    print(f"Prediction: {classes[np.argmax(pred)]} ({pred[0]})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.infer data/person1946_bacteria_4874.jpeg")
    else:
        print("Infer started")
        predict(sys.argv[1])
