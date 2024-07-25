import os
import platform
import warnings
import itertools
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import keras.backend as K
import tensorflow_addons as tfa
from scipy import signal
from tensorflow.keras import layers
from tensorflow.keras.initializers import Constant, GlorotUniform
from tensorflow_addons.optimizers import RectifiedAdam, Lookahead

import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

tf.random.set_seed(9)

X_TRAIN = np.load("direction_x_train.npy", mmap_mode="r")
X_TRAIN_PHI = np.load("direction_x_train_phi.npy", mmap_mode="r")
X_LEVEL_TRAIN = np.load("direction_x_level_train.npy", mmap_mode="r")
Y_TRAIN = np.load("direction_y_train.npy", mmap_mode="r")
whistle_length_train = np.load("whistle_length_train.npy", mmap_mode="r")

X_TEST = np.load("direction_x_test.npy", mmap_mode="r")
X_TEST_PHI = np.load("direction_x_test_phi.npy", mmap_mode="r")
X_LEVEL_TEST = np.load("direction_x_level_test.npy", mmap_mode="r")
Y_TEST = np.load("direction_y_test.npy", mmap_mode="r")
whistle_length_test = np.load("whistle_length_test.npy", mmap_mode="r")

# only first N Samples per whistle
N = 3
np.random.seed(42)

idx = 0
temp_x_train = []
temp_x_level_train = []
temp_x_train_phi = []
temp_y_train = []
if np.sum(whistle_length_train)==np.shape(X_TRAIN)[0]:
 for length in whistle_length_train:
     if length >= N:
         steps = N
     else:
         steps=length
     for step in range(steps):
        temp_x_train.append(X_TRAIN[idx+step])
        temp_x_level_train.append(X_LEVEL_TRAIN[idx+step])
        temp_y_train.append(Y_TRAIN[idx+step])
        temp_x_train_phi.append(X_TRAIN_PHI[idx+step])
     idx+=length

shuffle_mask = np.arange(0, len(temp_x_train), 1)
np.random.shuffle(shuffle_mask)

x_train = np.asarray(temp_x_train)[shuffle_mask]
x_level_train = np.asarray(temp_x_level_train)[shuffle_mask]
x_train_phi = np.asarray(temp_x_train_phi)[shuffle_mask]
y_train = np.asarray(temp_y_train)[shuffle_mask].astype(np.float32)

idx = 0
temp_x_test = []
temp_x_level_test = []
temp_x_test_phi = []
temp_y_test = []
if np.sum(whistle_length_test)==np.shape(X_TEST)[0]:
 for length in whistle_length_test:
     if length >= N:
         steps = N
     else:
         steps=length
     for step in range(steps):
        temp_x_test.append(X_TEST[idx+step])
        temp_x_level_test.append(X_LEVEL_TEST[idx+step])
        temp_y_test.append(Y_TEST[idx+step])
        temp_x_test_phi.append(X_TEST_PHI[idx+step])
     idx+=length

x_test = np.asarray(temp_x_test)
x_level_test = np.asarray(temp_x_level_test)
x_test_phi = np.asarray(temp_x_test_phi)
y_test = np.asarray(temp_y_test).astype(np.float32)

BINS = np.array([2.0, 4.5, 6.75, 9, np.inf])
print(BINS)
print(list(zip(np.unique(y_train[:,1]), np.digitize(np.unique(y_train[:,1]), BINS, right=True))))

BATCH_SIZE = 256
GAUSS = signal.gaussian((2*len(BINS))+1, std=0.56)
GAUSS = (GAUSS - np.min(GAUSS))/(np.max(GAUSS) - np.min(GAUSS))
GAUSS = GAUSS.astype(np.float32)

def get_nearest_bin(data, bins):
    tmp_bins = np.copy(bins)
    tmp_bins = np.hstack([0.0, tmp_bins])
    max_error = np.diff(tmp_bins)/2
    bin_centers = tmp_bins[:-1] + max_error
    max_error[-1] = bin_centers[-1] = np.max(y_train[:,1])
    error = np.abs(tmp_bins - data)
    scale_error = np.abs(bin_centers - data)
    scale = scale_error[np.digitize(data, bins, right=True)] / max_error[np.digitize(data, bins, right=True)]
    if np.min(scale_error) == 0:
        return None, scale
    else:
        return np.argmin(error) - 1, scale

def get_dampening_mask(value, bins):
    i = np.digitize(value,bins, right=True)
    j, scale = get_nearest_bin(value, bins)
    idxs = np.arange(0, len(bins), 1)
    if j is None:
        return idxs != i, scale
    elif i <= j:
        return idxs < i, scale
    else:
        return idxs > i, scale

def discretization(data, bins, tol=0.01):
    idx = np.digitize(data, BINS, right=True)

    encoded_data = np.zeros(shape=BINS.shape)
    encoded_data = np.copy(GAUSS[len(BINS)-idx:(len(BINS)-idx)+len(BINS)])
    mask, scale = get_dampening_mask(data, bins)
    encoded_data[mask] *= (1 + (tol*scale)) - scale
    mask[idx] = True
    mask = np.invert(mask)
    encoded_data[mask] *= (1 - (tol*scale)) + scale
    return encoded_data

def data_generator():
    x = itertools.cycle(x_train)
    x_level = itertools.cycle(x_level_train)
    x_phi = itertools.cycle(x_train_phi)
    y = itertools.cycle(y_train)

    while True:
        X = []
        X_LEVEL = []
        X_PHI = []
        Y = []
        Y_dist = []

        for _ in range(BATCH_SIZE):
            X.append(next(x))
            X_LEVEL.append(next(x_level))
            X_PHI.append(next(x_phi))
            angle, distance = next(y)
            Y.append(angle)
            Y_dist.append(discretization(distance, BINS))
        yield np.asarray(X),np.asarray(X_LEVEL), np.asarray(X_PHI), np.expand_dims(np.asarray(Y), axis=-1), np.asarray(Y_dist)

def test_data_generator():
    x = itertools.cycle(x_test)
    x_level = itertools.cycle(x_level_test)
    x_phi = itertools.cycle(x_test_phi)
    y = itertools.cycle(y_test)

    while True:
        X = []
        X_LEVEL = []
        X_PHI = []
        Y = []
        Y_dist = []

        for _ in range(BATCH_SIZE):
            X.append(next(x))
            X_LEVEL.append(next(x_level))
            X_PHI.append(next(x_phi))
            angle, distance = next(y)
            Y.append(angle)
            Y_dist.append(discretization(distance, BINS))
        yield np.asarray(X), np.asarray(X_LEVEL), np.asarray(X_PHI), np.expand_dims(np.asarray(Y), axis=-1), np.asarray(Y_dist)


def create_attention():
    phase_features = layers.Input(shape=256)
    level_features = layers.Input(shape=256)

    attention = layers.Dense(256, activation="softmax", use_bias=False, kernel_initializer=GlorotUniform)(level_features)
    features = phase_features * attention

    return tf.keras.Model(inputs=[phase_features, level_features], outputs=features)

def create_backbone_phi():
    inputs = layers.Input(shape=(513, 4))
    dropout = layers.SpatialDropout1D(rate=0.2)(inputs)

    conv = layers.Conv1D(filters=64, kernel_size=2, strides=1, padding="same", use_bias=False, kernel_initializer=GlorotUniform)(dropout)
    leakyReLU = layers.LeakyReLU(alpha=0.3)(conv)
    maxpool = layers.MaxPool1D(pool_size=2, strides=2, data_format='channels_first', padding="same")(leakyReLU)
    maxpool = layers.MaxPool1D(pool_size=2, strides=2, data_format='channels_last', padding="valid")(maxpool)

    conv = layers.Conv1D(filters=32, kernel_size=2, strides=1, padding="same", use_bias=False, kernel_initializer=GlorotUniform)(maxpool)
    leakyReLU = layers.LeakyReLU(alpha=0.3)(conv)
    maxpool = layers.MaxPool1D(pool_size=2, strides=2, data_format='channels_first', padding="same")(leakyReLU)
    maxpool = layers.MaxPool1D(pool_size=2, strides=2, data_format='channels_last', padding="valid")(maxpool)

    conv = layers.Conv1D(filters=16, kernel_size=2, strides=1, padding="same", use_bias=False, kernel_initializer=GlorotUniform)(maxpool)
    leakyReLU = layers.LeakyReLU(alpha=0.3)(conv)
    maxpool = layers.MaxPool1D(pool_size=2, strides=2, data_format='channels_first', padding="same")(leakyReLU)
    maxpool = layers.MaxPool1D(pool_size=2, strides=2, data_format='channels_last', padding="valid")(maxpool)

    flatten = layers.Flatten()(maxpool)

    dense = layers.Dense(256, activation=tfa.activations.mish, kernel_regularizer=tf.keras.regularizers.L2(0.001), use_bias=False, kernel_initializer=GlorotUniform)(flatten)
    features = layers.Dropout(rate=0.1)(dense)

    return tf.keras.Model(inputs=inputs, outputs=features)

def create_backbone_level():
    inputs = layers.Input(shape=(1024, 4))
    dropout = layers.SpatialDropout1D(rate=0.2)(inputs)

    conv = layers.Conv1D(filters=64, kernel_size=2, strides=1, padding="same", use_bias=False, kernel_initializer=GlorotUniform)(dropout)
    leakyReLU = layers.LeakyReLU(alpha=0.3)(conv)
    maxpool = layers.MaxPool1D(pool_size=2, strides=2, data_format='channels_first', padding="same")(leakyReLU)
    maxpool = layers.MaxPool1D(pool_size=2, strides=2, data_format='channels_last', padding="valid")(maxpool)

    conv = layers.Conv1D(filters=32, kernel_size=2, strides=1, padding="same", use_bias=False, kernel_initializer=GlorotUniform)(maxpool)
    leakyReLU = layers.LeakyReLU(alpha=0.3)(conv)
    maxpool = layers.MaxPool1D(pool_size=2, strides=2, data_format='channels_first', padding="same")(leakyReLU)
    maxpool = layers.MaxPool1D(pool_size=2, strides=2, data_format='channels_last', padding="valid")(maxpool)

    conv = layers.Conv1D(filters=16, kernel_size=2, strides=1, padding="same", use_bias=False, kernel_initializer=GlorotUniform)(maxpool)
    leakyReLU = layers.LeakyReLU(alpha=0.3)(conv)
    maxpool = layers.MaxPool1D(pool_size=2, strides=2, data_format='channels_first', padding="same")(leakyReLU)
    maxpool = layers.MaxPool1D(pool_size=2, strides=2, data_format='channels_last', padding="valid")(maxpool)

    flatten = layers.Flatten()(maxpool)

    dense = layers.Dense(256, activation=tfa.activations.mish, kernel_regularizer=tf.keras.regularizers.L2(0.001), use_bias=False, kernel_initializer=GlorotUniform)(flatten)
    features = layers.Dropout(rate=0.1)(dense)

    return tf.keras.Model(inputs=inputs, outputs=features)

def create_angle_model():
    inputs_phi = layers.Input(shape=(513, 4))
    inputs_level = layers.Input(shape=(1024, 4))
    backbone_phi = create_backbone_phi()
    features_phi = backbone_phi(inputs_phi)

    backbone_level = create_backbone_level()
    level_differences = backbone_level(inputs_level)
    dropout = layers.Dropout(rate=0.2)(level_differences)


    distribution = layers.Dense(256, activation=tf.keras.activations.softmax, use_bias=False, kernel_initializer=GlorotUniform)(dropout)
    features = distribution * features_phi
    concat = layers.Concatenate()([features, dropout])
    concat = layers.Flatten()(concat)

    dense = layers.Dense(64, activation=tfa.activations.mish, kernel_regularizer=tf.keras.regularizers.L2(0.001), use_bias=False, kernel_initializer=GlorotUniform)(concat)#(features)
    dropout = layers.Dropout(rate=0.2)(dense)
    dense = layers.Dense(32, activation=tfa.activations.mish, kernel_regularizer=tf.keras.regularizers.L2(0.001), use_bias=False, kernel_initializer=GlorotUniform)(dropout)
    dropout = layers.Dropout(rate=0.2)(dense)
    dense = layers.Dense(16, activation=tfa.activations.mish, kernel_regularizer=tf.keras.regularizers.L2(0.001), use_bias=False, kernel_initializer=GlorotUniform)(dropout)
    dropout = layers.Dropout(rate=0.2)(dense)

    # use bias here
    angle = layers.Dense(1, activation=tf.keras.activations.sigmoid, use_bias=True, bias_initializer=Constant(0.4398), kernel_initializer=GlorotUniform)(dropout)
    angle = layers.Lambda(lambda x: x * 360)(angle)

    return tf.keras.Model(inputs=[inputs_phi, inputs_level], outputs=[angle])


tf.random.set_seed(9)
GENERATOR = data_generator()
TEST_GENERATOR = test_data_generator()

angle_model = create_angle_model()

SCALE = 1
A = 10
C = 0.3
lmbda = tf.Variable(1e-5)

CCE = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM)

@tf.function
def ang_loss(y_true, y_pred):
    # Adapted Shrinkage Loss
    true_vec = tf.linalg.normalize(tf.squeeze(
        tf.stack([tf.cos(tf.experimental.numpy.deg2rad(y_true)), tf.sin(tf.experimental.numpy.deg2rad(y_true))],
                 axis=1)), axis=-1)[0]
    pred_vec = tf.linalg.normalize(tf.squeeze(
        tf.stack([tf.cos(tf.experimental.numpy.deg2rad(y_pred)), tf.sin(tf.experimental.numpy.deg2rad(y_pred))],
                 axis=1)), axis=-1)[0]
    l = tf.math.acos(tf.maximum(tf.minimum(tf.reduce_sum(tf.multiply(true_vec, pred_vec), axis=-1), 1. - K.epsilon()),
                                -1. + K.epsilon())) / np.pi
    error = l / (1 + tf.math.exp(A * (C - l)))
    return tf.reduce_mean(error)


@tf.function
def ang_model_loss(y_true, y_pred, y_confidence, training=True):
    # Shrinkage Loss combined with Confidence learning
    BETA = 0.3
    dropout = tf.nn.dropout(tf.ones_like(y_confidence), 0.5) / 2
    y_confidence = tf.math.maximum(dropout, y_confidence)

    y_pred_prime = y_confidence * y_pred + (1.0 - y_confidence) * y_true
    angle_loss_prime = ang_loss(y_true, y_pred_prime)
    c_loss = tf.reduce_mean(-tf.math.log(y_confidence + K.epsilon()))

    if training:
        if c_loss < BETA:
            lmbda.assign(lmbda * 0.95)
        else:
            lmbda.assign(lmbda * 1.05)

    angle_loss = angle_loss_prime + lmbda * c_loss

    return angle_loss

angle_optimizer = tf.optimizers.Nadam(learning_rate=3e-4)

@tf.function
def angle_train_step(freqs, levels, phases, angles, distances):
    with tf.GradientTape() as tape:
        pred_angles = angle_model([phases, levels], training=True)

        model_loss = ang_loss(angles, pred_angles)
        grads = tape.gradient(model_loss, angle_model.trainable_weights)
        angle_optimizer.apply_gradients(zip(grads, angle_model.trainable_weights))

    return model_loss

EPOCHS = 2500
PATIENCE = 500
MIN_DELTA = 0.001
STEPS = x_train.shape[0] / BATCH_SIZE
TEST_STEPS = x_test.shape[0] / BATCH_SIZE
CRITERIA = "median"

mean_epoch_loss = np.inf
mean_train_epoch_losses = []
mean_validate_epoch_losses = []

median_epoch_loss = np.inf
median_train_epoch_losses = []
median_validate_epoch_losses = []

epoch_losses = []
min_validate_loss = np.inf
used_epochs = EPOCHS
no_improvements = 0
current_validation_loss = np.inf

for epoch in range(EPOCHS):
    if len(epoch_losses) > 0:
        mean_epoch_loss = np.mean(epoch_losses)
        mean_train_epoch_losses.append(mean_epoch_loss)

        median_epoch_loss = np.median(epoch_losses)
        median_train_epoch_losses.append(median_epoch_loss)

        validate_losses = []
        validate_ranges = []
        for step, batch in enumerate(TEST_GENERATOR):
            if step >= TEST_STEPS:
                break
            _, levels, phases, angles, _ = batch
            pred_angles= angle_model.predict([phases, levels])
            validate_loss = ang_loss(angles, pred_angles)
            validate_losses.append(validate_loss)
        mean_validate_epoch_losses.append(np.mean(validate_losses))
        median_validate_epoch_losses.append(np.median(validate_losses))

        if CRITERIA == "median":
            current_validation_loss = np.median(validate_losses)
        else:
            current_validation_loss = np.mean(validate_losses)

        if min_validate_loss - current_validation_loss > MIN_DELTA:
            no_improvements = 0
        else:
            no_improvements += 1

        if current_validation_loss < min_validate_loss:
            min_validate_loss = current_validation_loss
            angle_model.save("./angle_ckpt.h5")

        if no_improvements > PATIENCE:
            used_epochs = epoch
            break

    epoch_losses = []
    for step, batch in enumerate(GENERATOR):
        if step >= STEPS:
            break
        freqs, levels, phases, angles, distances = batch
        model_loss = angle_train_step(freqs, levels, phases, angles, distances)
        epoch_losses.append(model_loss)
        print(f"Epoch: {epoch}, Epoch Loss: {round(float(mean_epoch_loss), 4)} | Validation Loss: {round(float(current_validation_loss), 4)} | Training Loss: {round(float(model_loss), 4)}         \r", end="")

if platform.system() == "Windows":
    import winsound

    winsound.PlaySound("SystemHand", winsound.SND_ALIAS)

import matplotlib.pyplot as plt
fig = plt.figure(figsize =(20, 7))

if CRITERIA == "median":
    plt.plot(range(0, len(median_train_epoch_losses)), median_train_epoch_losses, color="blue")
    plt.plot(range(0, len(median_validate_epoch_losses)), median_validate_epoch_losses, color="orange")
    plt.axhline(y = np.min(median_validate_epoch_losses), color = "black", linewidth=0.2, linestyle = "dashed")
else:
    plt.plot(range(0, len(mean_train_epoch_losses)), mean_train_epoch_losses, color="blue")
    plt.plot(range(0, len(mean_validate_epoch_losses)), mean_validate_epoch_losses, color="orange")
    plt.axhline(y = np.min(mean_validate_epoch_losses), color = "black", linewidth=0.2, linestyle = "dashed")
plt.show()

angle_model.save("./angle_model_currentVersion_model.h5")
