import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow import keras

model_name = './modelV4Mk1.h5'

custom_objects = {"mish": tfa.activations.mish}

# Load the model
with keras.utils.custom_object_scope(custom_objects):
  model = tf.keras.models.load_model(model_name, compile=False)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float32]
tflite_model = converter.convert()

# Save the model.
with open('./modelV4Mk1.tflite', 'wb') as f:
  f.write(tflite_model) 