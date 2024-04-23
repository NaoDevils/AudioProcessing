import os
import pickle
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import subprocess as sp
import tensorflow as tf
from tensorflow.keras import layers
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from pyflann import *
from tqdm.keras import TqdmCallback
from sklearn.cluster import MiniBatchKMeans
from imblearn.over_sampling import KMeansSMOTE
from tensorflow_addons.optimizers import RectifiedAdam, Lookahead
from sklearn.metrics import confusion_matrix, precision_recall_curve

from custom_loss import WhistleDetectorLoss
from whistle_config import *


def set_seed(seed):
    """
    Set numpy and tensorflow seeds.

    Parameters
    ----------
    seed : int 
        Seed to use.
    """
    # 1. Set `PYTHONHASHSEED` and other environment variables at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    os.environ['GCS_READ_CACHE_DISABLED'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

    # 2. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed)

    # 3. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed)


def cm(ground_truth, predictions, desired_precision=0.99):
    """
    Prints a confusion matrix, the precision, recall and the corresponding 
    whistle recognition threshold to achieve the desired precision.

    Parameters
    ----------
    ground_truth : np.array 
        Ground truth labels.
    predictions : np.array
        Prediction of the labels.
    desired_precision : float
        Desired precision of the whistle recognition.

    Returns
    -------
    float
        Precision
    float
        Recall
    float
        Threshold to achieve the desired precision.
    """

    precisions, _, thresholds = precision_recall_curve(ground_truth, predictions)
    index = np.argmax(precisions > desired_precision)
    if index == len(thresholds):
        index = index - 1
    p = thresholds[index]
    cm = confusion_matrix(ground_truth, predictions >= p)
    print('\t\tTrue Negatives: \t ', cm[0][0])
    print('\t\tFalse Positives: \t ', cm[0][1])
    print('\t\tTrue Positives: \t ', cm[1][1])
    print('\t\tFalse Negatives: \t ', cm[1][0])
    print('\t\tTotal Positives: \t ', np.sum(cm[1]))
    precision = cm[1][1]/(cm[1][1]+cm[0][1]+1e-10)
    recall = cm[1][1]/(cm[1][1]+cm[1][0]+1e-10)
    print('\n\t\tPrecision: \t ', precision)
    print('\t\tRecall: \t ', recall)
    print('\t\tThreshold: \t ', p)

    return (precision, recall, p)


set_seed(1337)

if read_data_from_txt:
    x_train = np.loadtxt(fname=prefix+"_train.txt", delimiter=',')
    y_train = np.loadtxt(fname="y_train.txt", delimiter=',')
    x_test = np.loadtxt(fname=prefix+"_test.txt", delimiter=',')
    y_test = np.loadtxt(fname="y_test.txt", delimiter=',')

    if use_oversampling:
        print("Over-Sampling started")
        kmeans = MiniBatchKMeans(   n_clusters=n_clusters,
                                    init="k-means++", 
                                    max_iter=100, 
                                    batch_size=4096, 
                                    random_state=42)
        kmsm = KMeansSMOTE( sampling_strategy="minority", 
                            random_state=42, 
                            k_neighbors=2, 
                            n_jobs=-1, # Use all CPU-Cores
                            kmeans_estimator=kmeans, 
                            cluster_balance_threshold="auto",
                            density_exponent="auto")
        x_train, y_train = kmsm.fit_resample(x_train, y_train)
        print("Over-Sampling done")

    if use_undersampling: # Fast implentation of Edited Nearest Neighbours
        print("Under-Sampling started")
        flann = FLANN()
        print("0%\r", end="")
        nn, _ = flann.nn(x_train, x_train, n_neighbours + 1)
        idx_to_remove = []
        for idx, neighbour in enumerate(nn):
            percentage = idx/len(nn)
            print(f"{int(percentage*100)}%\r", end="")
            unique_neigbours = np.unique(y_train[neighbour[1:n_neighbours + 1]])
            if len(unique_neigbours) == 1:
                if neighbour[0] != unique_neigbours[0]:
                    idx_to_remove.append(neighbour[0])
        mask = np.ones(len(x_train), bool)
        mask[idx_to_remove] = 0
        x_train = x_train[mask]
        y_train = y_train[mask]
        print("\nUnder-Sampling done")

    with open(prefix+"_train.pkl", 'wb') as f:
        pickle.dump(x_train, f)
    with open("y_train.pkl", 'wb') as f:
        pickle.dump(y_train, f)
    with open(prefix+"_test.pkl", 'wb') as f:
        pickle.dump(x_test, f)
    with open("y_test.pkl", 'wb') as f:
        pickle.dump(y_test, f)
else:
    #read data from pickle file
    x_train = pickle.load(open(prefix+"_train.pkl", "rb"))
    y_train = pickle.load(open("y_train.pkl", "rb"))
    x_test = pickle.load(open(prefix+"_test.pkl", "rb"))
    y_test = pickle.load(open("y_test.pkl", "rb"))

x_train = np.expand_dims(x_train, axis=-1)
y_train = np.expand_dims(y_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
y_test = np.expand_dims(y_test, axis=-1)


if load_model:
    model = tf.keras.models.load_model(model_name, compile=False)
else:
    print(f"Whistle samples:\t{int(y_train.sum())}")
    print(f"Noise samples:\t\t{int(y_train.shape[0]-y_train.sum())}")

    dense_params = {#parameter dense layer
    }

    conv_params = { # parameter conv layer
        "padding": "same",
        "use_bias": True,
    }

    # model design
    inputs = layers.Input(shape=(513, 1))

    conv1D = layers.Conv1D(filters=32, kernel_size=5, strides=2, **conv_params)(inputs)
    batchNorm = layers.BatchNormalization()(conv1D)
    leakyReLU = layers.LeakyReLU(alpha=0.1)(batchNorm)
    maxpool = layers.MaxPooling1D(pool_size=2, strides=2, padding="same")(leakyReLU)
    dropout = layers.Dropout(rate=0.3)(maxpool)

    conv1D = layers.Conv1D(filters=64, kernel_size=5, strides=2, **conv_params)(dropout)
    batchNorm = layers.BatchNormalization()(conv1D)
    leakyReLU = layers.LeakyReLU(alpha=0.1)(batchNorm)
    maxpool = layers.MaxPooling1D(pool_size=2, strides=2, padding="same")(leakyReLU)
    dropout = layers.Dropout(rate=0.3)(maxpool)

    conv1D = layers.Conv1D(filters=128, kernel_size=5, strides=2, **conv_params)(dropout)
    batchNorm = layers.BatchNormalization()(conv1D)
    leakyReLU = layers.LeakyReLU(alpha=0.1)(batchNorm)
    maxpool = layers.MaxPooling1D(pool_size=2, strides=2, padding="same")(leakyReLU)
    dropout = layers.Dropout(rate=0.3)(maxpool)

    conv1D = layers.Conv1D(filters=64, kernel_size=5, strides=1, activation=tf.keras.activations.elu, **conv_params)(dropout)
    flatten = layers.Flatten()(conv1D)
    outputs = layers.Dense(1, activation=tf.keras.activations.sigmoid, **dense_params)(flatten)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

if show_model_summary or load_model:
    model.summary()

whistle_detector_loss = WhistleDetectorLoss()
metrics = [whistle_detector_loss.accuracy, whistle_detector_loss.fscore, whistle_detector_loss.combi_score]
whistle_loss = whistle_detector_loss.whistle_loss

radam = RectifiedAdam(
    learning_rate=1e-3, # learning-rate
    total_steps=20000,
    warmup_proportion=0.25,
    min_lr=1e-6,
)
ranger = Lookahead(radam, sync_period=6, slow_step_size=0.5)

if clear_checkpoints and not load_model:
    for root, dirs, files in os.walk("checkpoints", topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(  "checkpoints/{epoch:02d}_whistle_{val_fscore:.2f}_{val_combi_score:.2f}.h5",
                                                        monitor="val_fscore",
                                                        verbose=0,
                                                        save_best_only=True,
                                                        save_weights_only=False,
                                                        mode="max",
                                                        save_freq="epoch",
                                                        options=None)

early_stopping = tf.keras.callbacks.EarlyStopping(  monitor='val_fscore',
                                                    min_delta=0.001,
                                                    patience=20,
                                                    verbose=1,
                                                    mode='max',
                                                    restore_best_weights=True)

print(f"Batch size:\t\t{batch_size}")

# Training of the model
model.compile(optimizer=ranger, loss=whistle_loss, metrics=metrics, run_eagerly=False)

if not load_model:
    if pycharm:
        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=max_epochs, batch_size=batch_size,
                    callbacks=[model_checkpoint, early_stopping, TqdmCallback(position=0, leave=True, verbose=1)], verbose=0)
    else:
        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=max_epochs, batch_size=batch_size,
                    callbacks=[model_checkpoint, early_stopping, TqdmCallback(verbose=1)], verbose=0)

# validation
print("Validation:")

print("\t x_train")
train_predictions_baseline = model.predict(x_train, batch_size=batch_size)
cm(y_train, train_predictions_baseline, desired_precision=desired_precision)

print("\t x_test")
test_predictions_baseline = model.predict(x_test, batch_size=batch_size)
precision, recall, threshold = cm(y_test, test_predictions_baseline, desired_precision=desired_precision)

print("Validation2: ")
model.evaluate(x_test, y_test, batch_size=batch_size)

if not load_model:
    model.save(f"WhistleNet_precision_{precision:.4f}_recall_{recall:.4f}_threshold_{threshold:.2f}.h5")
