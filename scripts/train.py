from lib.data import get_generators
from lib.model import get_cnn_model
from lib.train import train_model
import os

data_dir = 'data'
train_gen, val_gen, test_gen = get_generators(data_dir)

model = get_cnn_model()
history = train_model(model, train_gen, val_gen, epochs=10)

os.makedirs('models', exist_ok=True)
model.save('models/chest_cnn_model.h5')
print("✅ Model saved at models/chest_cnn_model.h5")
