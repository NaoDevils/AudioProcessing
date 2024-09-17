import os
import sys
import glob
import pickle
import random
import warnings
import sounddevice as sd
from tqdm import tqdm
from pathlib import Path
warnings.filterwarnings("ignore", category=DeprecationWarning) 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button
import matplotlib.style as mplstyle
mplstyle.use('fast')

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

MICS = ["rear left", "rear right", "front left", "front right"]
FOLDER = "AnnotationData"
MODEL_NAME = "../PreTrainedNetworks/WhistleDetection/WhistleNetMk16.h5"
WINDOW_SIZE = 1024
THRESHOLD = 0.65
NEGATIVE_THERSHOLD = 0.01
ATTENTION_MULTIPLIER = 2.0
MIN_WHISTLE_FRAMES = 1
MAX_WHISTLE_FRAMES = np.inf
MAX_ATTENTION_LENGTH = 3

def calculate_whistle_data(dataset = None, folder = FOLDER, model_name = MODEL_NAME, window_size = WINDOW_SIZE, threshold = THRESHOLD, attention_multiplier = ATTENTION_MULTIPLIER, data = None, samplerate = None):
    global whistle_cluster
    global whistle_cutout
    global whistle_center
    global whistle_idx_range
    global labeled_whistles
    global non_whistle_label
    global x_min_limit
    global x_max_limit
    global min_num_of_detected_whistles
    global max_num_of_detected_whistles
    global fft_windowed_data
    global fft_windowed_data_log10

    if dataset is None or model_name is None:
        return None
    
    path = f"../WhistleDetector/{folder}/{dataset}.wav"
    if not Path(path).exists():
        print(f"ERROR: No such file: {Path(path).stem}.wav")
        return

    model = tf.keras.models.load_model(model_name, compile=False)

    if data is None or samplerate is None:
        samplerate, data = wavfile.read(path)
    channels = data.shape[1]
    hamming = np.hamming(window_size)

    windowed_data = np.lib.stride_tricks.sliding_window_view(data, window_size, axis=0)[0::window_size//2, :] * hamming
    fft_windowed_data_log10 = np.reshape(20 * np.log10(np.abs(np.fft.fft(windowed_data)[:, :, 0:window_size//2 + 1])),
                                (windowed_data.shape[0], channels, window_size//2 + 1, 1))

    fft_windowed_data = np.reshape(np.fft.fft(windowed_data)[:, :, 0:window_size//2 + 1],
                                (windowed_data.shape[0], channels, window_size//2 + 1, 1))

    # Shape => Channels, Number of Windows, FFT Data, 1
    fft_windowed_data_log10 = np.reshape(np.asarray(np.split(fft_windowed_data_log10, channels, axis=1)),
                                (channels, windowed_data.shape[0], window_size//2 + 1, 1))

    fft_windowed_data = np.reshape(np.asarray(np.split(fft_windowed_data, channels, axis=1)),
                                (channels, windowed_data.shape[0], window_size//2 + 1, 1))

    label = []
    for fft_data_windows in fft_windowed_data_log10:
        channel_label = []
        for fft_data in np.array_split(fft_data_windows, np.ceil(fft_data_windows.shape[0]/2048)):
            channel_label.extend(model.predict(fft_data))
        label.append(channel_label)
    label = np.asarray(label)
    negativ_label = np.copy(label)

    label[label > threshold] = window_size//4
    label[label <= threshold] = 0.0

    negativ_label[negativ_label > NEGATIVE_THERSHOLD] = window_size//4
    negativ_label[negativ_label <= NEGATIVE_THERSHOLD] = 0.0
    negativ_label[label > 0.0] = 0.0
    non_whistle_label = np.sum(negativ_label, axis=0).flatten()
    non_whistle_label[non_whistle_label > 0] = window_size//4

    longest_sequence = 0
    for channel in range(channels):
        last_label = label[channel][0]
        sequence = 0
        for whistle_label in label[channel]:
            if last_label > 0 and whistle_label == last_label:
                sequence += 1
            else:
                if sequence > longest_sequence:
                    longest_sequence = sequence
                sequence = 0
            last_label = whistle_label
    attention_length = longest_sequence * attention_multiplier

    if MAX_ATTENTION_LENGTH is not None and attention_length > MAX_ATTENTION_LENGTH:
        attention_length = MAX_ATTENTION_LENGTH

    whistle_idx_range = []
    detected_whistles = []
    for channel in range(channels):
        perv_whistle_label = 0.0
        min_idx = 0
        max_idx = 0
        tmp_whistle_idx_range = []
        attention = 0
        active_attention = False
        for idx, whistle_label in enumerate(label[channel]):
            if not active_attention and perv_whistle_label == 0.0 and whistle_label == window_size//4:
                min_idx = idx
                active_attention = True
            elif perv_whistle_label == window_size//4 and whistle_label == 0.0:
                max_idx = idx - 1

            if active_attention and attention > attention_length and np.abs(max_idx - min_idx) + 1 > MIN_WHISTLE_FRAMES and np.abs(max_idx - min_idx) + 1 < MAX_WHISTLE_FRAMES:
                tmp_whistle_idx_range.append([min_idx, max_idx])
                attention = 0
                active_attention = False

            if active_attention:
                attention += 1

            perv_whistle_label = whistle_label
        if not tmp_whistle_idx_range:
            print(f"WARNING: No whistle detected: mic at {MICS[channel]} possibly broken or no whistle was recorded!")
        else:
            detected_whistles.append(len(tmp_whistle_idx_range))
            whistle_idx_range.append(tmp_whistle_idx_range)
    whistle_idx_range = np.asarray(whistle_idx_range, dtype=object)
    if detected_whistles != []:
        max_num_of_detected_whistles = np.max(detected_whistles)
        min_num_of_detected_whistles = np.min(detected_whistles)
    else:
        max_num_of_detected_whistles = min_num_of_detected_whistles = 0

    whistle_cluster = np.zeros(fft_windowed_data_log10[0].T.shape[2])
    x_min_limit = 0
    x_max_limit = 0
    if len(np.unique(detected_whistles)) == 1:
        whistle_idx_range = np.hstack((np.expand_dims(np.min(np.min(whistle_idx_range, axis=0), axis=1), axis=-1), np.expand_dims(np.max(np.max(whistle_idx_range, axis=0), axis=1), axis=-1)))
        whistle_idx_range = whistle_idx_range.astype(np.int64)

        whistle_idx_gap = []
        last_upper_bound = None
        x_min_limit = np.min(whistle_idx_range)
        x_max_limit = np.max(whistle_idx_range)
        whistle_cluster = np.zeros(fft_windowed_data_log10[0].T.shape[2])
        whistle_cutout = np.zeros(fft_windowed_data_log10[0].T.shape[2])
        for lower_bound, upper_bound in whistle_idx_range:
            if last_upper_bound is None:
                last_upper_bound = upper_bound
            else:
                whistle_idx_gap.append(lower_bound - last_upper_bound)
                last_upper_bound = upper_bound
            for idx in range(lower_bound, upper_bound+1):
                whistle_cluster[idx] = window_size//2 - (window_size / 100 * 10)
            for idx in range(np.max((0, lower_bound-100)), np.min((upper_bound+100, fft_windowed_data_log10[0].T.shape[2]-1))):
                whistle_cutout[idx] = window_size//2 - (window_size / 100 * 10)

        whistle_center = ((whistle_idx_range[:,1] - whistle_idx_range[:,0])//2) + whistle_idx_range[:,0]
    elif detected_whistles != []:
        whistle_idx_range = np.asarray(whistle_idx_range[np.argmax(detected_whistles)], dtype=np.int64)

        whistle_idx_gap = []
        last_upper_bound = None
        x_min_limit = np.min(whistle_idx_range)
        x_max_limit = np.max(whistle_idx_range)
        whistle_cluster = np.zeros(fft_windowed_data_log10[0].T.shape[2])
        whistle_cutout = np.zeros(fft_windowed_data_log10[0].T.shape[2])
        for lower_bound, upper_bound in whistle_idx_range:
            if last_upper_bound is None:
                last_upper_bound = upper_bound
            else:
                whistle_idx_gap.append(lower_bound - last_upper_bound)
                last_upper_bound = upper_bound
            for idx in range(lower_bound, upper_bound+1):
                whistle_cluster[idx] = window_size//2 - (window_size / 100 * 10)
            for idx in range(np.max((0, lower_bound-100)), np.min((upper_bound+100, fft_windowed_data_log10[0].T.shape[2]-1))):
                whistle_cutout[idx] = window_size//2 - (window_size / 100 * 10)

        whistle_center = ((whistle_idx_range[:,1] - whistle_idx_range[:,0])//2) + whistle_idx_range[:,0]
    else:
        whistle_cluster = non_whistle_label
        whistle_center = np.argwhere(whistle_cluster)
        whistle_cutout = np.zeros(fft_windowed_data_log10[0].T.shape[2])
        for center in whistle_center:
            for idx in range(np.max((0, int(center)-100)), np.min((int(center)+100, fft_windowed_data_log10[0].T.shape[2]-1))):
                whistle_cutout[idx] = window_size//2 - (window_size / 100 * 10)

    return whistle_cluster, whistle_center, whistle_idx_range, x_min_limit, x_max_limit, min_num_of_detected_whistles, max_num_of_detected_whistles, fft_windowed_data, fft_windowed_data_log10, data, samplerate


def update_plot(fig, axs, data, dataset, threshold, samplerate, whistle_cluster, fft_windowed_data_log10, min_num_of_detected_whistles, max_num_of_detected_whistles, x_min_limit, labeled_whistles, window_size):
    global active_labeling
    global whistle_cutout

    labeled_whistle_cluster = np.zeros(fft_windowed_data_log10[0].T.shape[2])
    for lower_bound, upper_bound in labeled_whistles:
        for idx in range(lower_bound, upper_bound+1):
            labeled_whistle_cluster[idx] = window_size//2 - (window_size / 100 * 10)

    fig.clf()
    fig.add_axes(axs[0])
    fig.add_axes(axs[1])
    fig.add_axes(axs[2])
    fig.add_axes(axs[3])
    fig.canvas.manager.set_window_title("Whistle-Autolabeltool: " + dataset + f" ({len(data)//samplerate} sec.)")
    axs[0].cla()
    axs[0].set_ylabel("Rear Left")
    axs[0].plot(whistle_cluster[whistle_cutout > 0], color="white", label="whistledata")
    if active_labeling:
        axs[0].plot(labeled_whistle_cluster[whistle_cutout > 0], color="green", label="labeled whistledata")
    axs[0].imshow(np.reshape(fft_windowed_data_log10[0].T, (fft_windowed_data_log10[0].T.shape[1], fft_windowed_data_log10[0].T.shape[2], fft_windowed_data_log10[0].T.shape[0]))[:,whistle_cutout > 0], cmap="gnuplot2", interpolation="nearest", vmin=-60, vmax=0)
    axs[1].cla()
    axs[1].set_ylabel("Rear Right")
    axs[1].plot(whistle_cluster[whistle_cutout > 0], color="white")
    if active_labeling:
        axs[1].plot(labeled_whistle_cluster[whistle_cutout > 0], color="green")
    axs[1].imshow(np.reshape(fft_windowed_data_log10[1].T, (fft_windowed_data_log10[1].T.shape[1], fft_windowed_data_log10[1].T.shape[2], fft_windowed_data_log10[1].T.shape[0]))[:,whistle_cutout > 0], cmap="gnuplot2", interpolation="nearest", vmin=-60, vmax=0)
    axs[2].cla()
    axs[2].set_ylabel("Front Left")
    axs[2].plot(whistle_cluster[whistle_cutout > 0], color="white")
    if active_labeling:
        axs[2].plot(labeled_whistle_cluster[whistle_cutout > 0], color="green")
    axs[2].imshow(np.reshape(fft_windowed_data_log10[2].T, (fft_windowed_data_log10[2].T.shape[1], fft_windowed_data_log10[2].T.shape[2], fft_windowed_data_log10[2].T.shape[0]))[:,whistle_cutout > 0], cmap="gnuplot2", interpolation="nearest", vmin=-60, vmax=0)
    axs[3].cla()
    axs[3].set_ylabel("Front Right")
    axs[3].plot(whistle_cluster[whistle_cutout > 0], color="white")
    if active_labeling:
        axs[3].plot(labeled_whistle_cluster[whistle_cutout > 0], color="green")
    axs[3].imshow(np.reshape(fft_windowed_data_log10[3].T, (fft_windowed_data_log10[3].T.shape[1], fft_windowed_data_log10[3].T.shape[2], fft_windowed_data_log10[3].T.shape[0]))[:,whistle_cutout > 0], cmap="gnuplot2", interpolation="nearest", vmin=-60, vmax=0)
    fig.legend(facecolor="grey", title=f"Threshold: {threshold}")
    fig.suptitle(f"Detected whistles: {max_num_of_detected_whistles} | Labeled whistles: {len(labeled_whistles)}", fontsize=12, fontweight='normal')

    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0, wspace=0)
    limit = len(whistle_cluster[whistle_cutout > 0]) if len(whistle_cluster[whistle_cutout > 0]) < 3950 else 3950
    plt.xlim(0, limit)
    plt.gca().invert_yaxis()
    plt.show()

def save_whistle_label(dataset):
    global whistle_cluster
    global labeled_whistles
    global non_whistle_label
    global fft_windowed_data_log10
    print("save")

    whistle_idx = []
    for whistle_range in labeled_whistles:
        range_min, range_max = whistle_range
        for idx in range(range_min, range_max + 1):
            whistle_idx.append(idx)
    
    all_detected_whistle_idx = []
    for idx, label in enumerate(whistle_cluster):
        if label > 0.0:
            all_detected_whistle_idx.append(idx)

    non_whistle_idx = list(set(all_detected_whistle_idx).difference(set(whistle_idx)))

    non_whistle_idx_candidates = []
    for idx, label in enumerate(non_whistle_label):
        if label > 0.0 and idx not in whistle_idx and idx not in non_whistle_idx:
            non_whistle_idx_candidates.append(idx)
    random.shuffle(non_whistle_idx_candidates)

    i = 0
    while len(non_whistle_idx) < len(whistle_idx) and i < len(non_whistle_idx_candidates):
        non_whistle_idx.append(non_whistle_idx_candidates[i])
        i += 1

    x = []
    y = []

    for idx in whistle_idx:
        for channel in range(4):
            x.append(fft_windowed_data_log10[channel][idx])
            y.append(1)
    
    for idx in non_whistle_idx:
        for channel in range(4):
            x.append(fft_windowed_data_log10[channel][idx])
            y.append(0)
    
    x = np.asarray(x)
    y = np.asarray(y)

    with open(f"../whistle_x_{dataset}.npy", 'wb') as f:
        np.save(f, x, allow_pickle=False, fix_imports=True)
    with open(f"../whistle_y_{dataset}.npy", 'wb') as f:
        np.save(f, y, allow_pickle=False, fix_imports=True)
    
    print(f"Files: whistle_x_{dataset}.npy and whistle_y_{dataset}.npy saved.")


def label_dataset(dataset = None, folder = FOLDER, model_name = MODEL_NAME, window_size = WINDOW_SIZE, threshold = THRESHOLD, attention_multiplier = ATTENTION_MULTIPLIER):
    global data
    global samplerate
    global whistle_cluster
    global whistle_center
    global whistle_idx_range
    global labeled_whistles
    global x_min_limit
    global x_max_limit
    global min_num_of_detected_whistles
    global max_num_of_detected_whistles
    global fft_windowed_data
    global fft_windowed_data_log10

    for key in plt.rcParams['keymap.save']:
        plt.rcParams['keymap.save'].remove(key)

    fig, axs = plt.subplots(4, 1, sharex=True, sharey=True)

    whistle_cluster, whistle_center, whistle_idx_range, x_min_limit, x_max_limit, min_num_of_detected_whistles, max_num_of_detected_whistles, fft_windowed_data, fft_windowed_data_log10, data, samplerate = calculate_whistle_data(dataset, folder, model_name, window_size, threshold, attention_multiplier)

    def on_click(event):
        global data
        global samplerate
        global active_labeling
        global whistle_cluster
        global whistle_cutout
        global whistle_center
        global whistle_idx_range
        global labeled_whistles
        global x_min_limit
        global x_max_limit
        global min_num_of_detected_whistles
        global max_num_of_detected_whistles
        global fft_windowed_data
        global fft_windowed_data_log10

        if event.dblclick:
            frame = np.argwhere(whistle_cutout).flatten()[np.uint64(np.floor(event.xdata))]
            x, y = fig.transFigure.inverted().transform((event.x, event.y))
            rear_left_bb = axs[0].get_position()
            rear_right_bb = axs[1].get_position()
            front_left_bb = axs[2].get_position()
            front_right_bb = axs[3].get_position()
            if rear_left_bb.contains(x, y):
                sd.play(data[np.int64(np.floor(frame*(window_size//2))):np.int64(np.floor(frame*(window_size//2))) + 2*samplerate,0], samplerate)
            elif rear_right_bb.contains(x, y):
                sd.play(data[np.int64(np.floor(frame*(window_size//2))):np.int64(np.floor(frame*(window_size//2))) + 2*samplerate,1], samplerate)
            elif front_left_bb.contains(x, y):
                sd.play(data[np.int64(np.floor(frame*(window_size//2))):np.int64(np.floor(frame*(window_size//2))) + 2*samplerate,2], samplerate)
            elif front_right_bb.contains(x, y):
                sd.play(data[np.int64(np.floor(frame*(window_size//2))):np.int64(np.floor(frame*(window_size//2))) + 2*samplerate,3], samplerate)
        

        if not active_labeling and event.button is MouseButton.RIGHT:
            frame = np.argwhere(whistle_cutout).flatten()[np.uint64(np.floor(event.xdata))]
            x, y = fig.transFigure.inverted().transform((event.x, event.y))
            rear_left_bb = axs[0].get_position()
            rear_right_bb = axs[1].get_position()
            front_left_bb = axs[2].get_position()
            front_right_bb = axs[3].get_position()
            if rear_left_bb.contains(x, y) or rear_right_bb.contains(x, y) or front_left_bb.contains(x, y) or front_right_bb.contains(x, y):
                mask = np.ones(data.shape[0])
                mask[list(range(np.int64(np.floor(frame*(window_size//2))), np.int64(np.floor(frame*(window_size//2))) + 2*samplerate))] = 0
                data = np.copy(data[mask.astype(np.bool_)])
                labeled_whistles = []
                whistle_cluster, whistle_center, whistle_idx_range, x_min_limit, x_max_limit, min_num_of_detected_whistles, max_num_of_detected_whistles, fft_windowed_data, fft_windowed_data_log10, data, samplerate = calculate_whistle_data(dataset, folder, model_name, window_size, threshold, attention_multiplier, data, samplerate)
                update_plot(fig, axs, data, dataset, threshold, samplerate, whistle_cluster, fft_windowed_data_log10, min_num_of_detected_whistles, max_num_of_detected_whistles, x_min_limit, labeled_whistles, window_size)
        elif active_labeling and event.button is MouseButton.RIGHT and whistle_idx_range != []:
            frame = np.argwhere(whistle_cutout).flatten()[np.uint64(np.floor(event.xdata))]
            labeled_whistles.append(tuple(whistle_idx_range[np.argmin(np.abs(frame - whistle_center))])) if tuple(whistle_idx_range[np.argmin(np.abs(frame - whistle_center))]) not in labeled_whistles and np.min(np.abs(frame - whistle_center)) < 50 else labeled_whistles
            update_plot(fig, axs, data, dataset, threshold, samplerate, whistle_cluster, fft_windowed_data_log10, min_num_of_detected_whistles, max_num_of_detected_whistles, x_min_limit, labeled_whistles, window_size)
      

    def on_press(event):
        global data
        global samplerate
        global active_labeling
        global whistle_cluster
        global whistle_center
        global whistle_idx_range
        global labeled_whistles
        global x_min_limit
        global x_max_limit
        global min_num_of_detected_whistles
        global max_num_of_detected_whistles
        global fft_windowed_data
        global fft_windowed_data_log10

        if event.key == "s":
            save_whistle_label(dataset)
        if event.key == "l":
            if active_labeling:
                active_labeling = False
            else:
                active_labeling = True
            update_plot(fig, axs, data, dataset, threshold, samplerate, whistle_cluster, fft_windowed_data_log10, min_num_of_detected_whistles, max_num_of_detected_whistles, x_min_limit, labeled_whistles, window_size)
        if event.key == "c":
            wavfile.write(f"{dataset}_edit.wav", samplerate, data)
        if event.key == "r":
            whistle_cluster, whistle_center, whistle_idx_range, x_min_limit, x_max_limit, min_num_of_detected_whistles, max_num_of_detected_whistles, fft_windowed_data, fft_windowed_data_log10, data, samplerate = calculate_whistle_data(dataset, folder, model_name, window_size, threshold, attention_multiplier)
            update_plot(fig, axs, data, dataset, threshold, samplerate, whistle_cluster, fft_windowed_data_log10, min_num_of_detected_whistles, max_num_of_detected_whistles, x_min_limit, labeled_whistles, window_size)

    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    pid = fig.canvas.mpl_connect("key_press_event", on_press)

    update_plot(fig, axs, data, dataset, threshold, samplerate, whistle_cluster, fft_windowed_data_log10, min_num_of_detected_whistles, max_num_of_detected_whistles, x_min_limit, labeled_whistles, window_size)

    fig.canvas.mpl_disconnect(cid)
    fig.canvas.mpl_disconnect(pid)

data = None
samplerate = None
active_labeling = True
labeled_whistles = []
non_whistle_label = None
whistle_cutout = None
whistle_cluster = None
whistle_center = None
whistle_idx_range = None
x_min_limit = None
x_max_limit = None
min_num_of_detected_whistles = None
max_num_of_detected_whistles = None
fft_windowed_data = None
fft_windowed_data_log10 = None

label_dataset(dataset = "Demo", folder = "AnnotationData")
