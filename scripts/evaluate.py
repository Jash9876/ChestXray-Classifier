from lib.data import get_generators
from lib.model import get_cnn_model
from lib.evaluate import evaluate_model

data_dir = 'data'
_, _, test_gen = get_generators(data_dir)

model = get_cnn_model()
model.load_weights('models/chest_cnn_model.h5')

evaluate_model(model, test_gen)
