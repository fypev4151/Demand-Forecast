import tensorflow as tf
from tf.keras.models import load_model
import joblib
import numpy as np

# Load the saved assets
loaded_model = load_model('ev_demand_lstm_model.keras')
loaded_scaler = joblib.load('ev_demand_scaler.pkl')

# Now you can use them exactly like before
# For example, to inverse scale a prediction:
# dummy_data = np.zeros((1, 4))
# dummy_data[0, -1] = loaded_model.predict(new_batch)
# actual_kw = loaded_scaler.inverse_transform(dummy_data)[0, -1]