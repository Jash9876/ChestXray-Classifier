from lib.data import get_generators
from lib.evaluate import evaluate_model
from tensorflow.keras.models import load_model

data_dir = "data"
_, _, test_gen = get_generators(data_dir)

model = load_model("saved_model/chestxray_resnet.h5")
evaluate_model(model, test_gen)
