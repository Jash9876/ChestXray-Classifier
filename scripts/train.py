from lib.data import get_generators
from lib.model import get_resnet_model
from lib.train import train_model
import os

data_dir = "data"
train_gen, val_gen, test_gen = get_generators(data_dir)

model = get_resnet_model()
history = train_model(model, train_gen, val_gen)

os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/chestxray_resnet.h5")
print("Model saved at saved_model/chestxray_resnet.h5")
