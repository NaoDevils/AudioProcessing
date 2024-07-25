import os
import sys
import glob
import pickle
import warnings
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.io import wavfile
from scipy.signal import correlate
from matplotlib.backend_bases import MouseButton

WINDOW_SIZE = 1024
CHRIP_SOUND_RATE, CHIRP_SOUND = wavfile.read(f"./Utils/chirp.wav")
CHIRP_LENGTH = len(CHIRP_SOUND)//CHRIP_SOUND_RATE

RIR_LENGTH = 2 #seconds
RIR_CUT = 45 #percent
RIR_FRONT_MARGIN = 50 #percent
RIR_BACK_MARGIN = 225 #percent

def wiener_deconvolution(sound_a, sound_b):
    padding = np.zeros_like(sound_a)
    padding[0:len(sound_b)] = sound_b
    sound_b = padding

    fft_b = np.fft.fft(sound_b.flatten())

    l = 1e-1

    s = np.max(np.abs(fft_b)) * l
    conj = np.conj(fft_b)
    div = np.abs(fft_b)**2 + s**2
    
    fft_a = np.fft.fft(sound_a.flatten())
    W = (fft_a * conj) / div

    return np.real(np.fft.ifft(W))

def update_plot(fig, axs, dataset = None, window_size = WINDOW_SIZE, load_data = True):
    global data
    global room_impulse_response
    global samplerate
    
    path = f"./WhistleDirectionDistance/RoomImpulseResponse/{dataset}.wav"
    if not Path(path).exists():
        print(f"ERROR: No such file: {Path(path).stem}.wav")
        return

    if load_data:
        samplerate, data = wavfile.read(path)

    chirp_center_position = np.array([np.argmax(correlate(data[:,0], CHIRP_SOUND, "same")), np.argmax(correlate(data[:,1], CHIRP_SOUND, "same")), np.argmax(correlate(data[:,2], CHIRP_SOUND, "same")), np.argmax(correlate(data[:,3], CHIRP_SOUND, "same"))])
    chirp_position_range = np.array([chirp_center_position - (1 + RIR_FRONT_MARGIN/100) * samplerate * (CHIRP_LENGTH/2), chirp_center_position + ( 1 + RIR_BACK_MARGIN/100) * samplerate * (CHIRP_LENGTH/2)], dtype=np.int64)
    chirp_position_min = np.min(chirp_position_range)
    chirp_position_max = np.max(chirp_position_range)

    data = data[chirp_position_min:chirp_position_max,:]
    
    channels = data.shape[1]
    hamming = np.hamming(window_size)

    windowed_data = np.lib.stride_tricks.sliding_window_view(data, window_size, axis=0)[0::window_size//2, :] * hamming
    fft_windowed_data_log10 = np.reshape(20 * np.log10(np.abs(np.fft.fft(windowed_data)[:, :, 0:window_size//2 + 1])),
                                (windowed_data.shape[0], channels, window_size//2 + 1, 1))
    
    # Shape => Channels, Number of Windows, FFT Data, 1
    fft_windowed_data_log10 = np.reshape(np.asarray(np.split(fft_windowed_data_log10, channels, axis=1)),
                                (channels, windowed_data.shape[0], window_size//2 + 1, 1))
    
    tmp_rir = []
    tmp_rir.append(wiener_deconvolution(data[:,0], CHIRP_SOUND))
    tmp_rir.append(wiener_deconvolution(data[:,1], CHIRP_SOUND))
    tmp_rir.append(wiener_deconvolution(data[:,2], CHIRP_SOUND))
    tmp_rir.append(wiener_deconvolution(data[:,3], CHIRP_SOUND))
    tmp_rir = np.array(tmp_rir)
    room_impulse_response = np.zeros((RIR_LENGTH*samplerate, tmp_rir.shape[0]))
    room_impulse_response[:RIR_LENGTH*samplerate,0] = tmp_rir[0,(int)(samplerate*(RIR_CUT/100)):(RIR_LENGTH*samplerate) + (int)(samplerate*(RIR_CUT/100))]
    room_impulse_response[:RIR_LENGTH*samplerate,1] = tmp_rir[1,(int)(samplerate*(RIR_CUT/100)):(RIR_LENGTH*samplerate) + (int)(samplerate*(RIR_CUT/100))]
    room_impulse_response[:RIR_LENGTH*samplerate,2] = tmp_rir[2,(int)(samplerate*(RIR_CUT/100)):(RIR_LENGTH*samplerate) + (int)(samplerate*(RIR_CUT/100))]
    room_impulse_response[:RIR_LENGTH*samplerate,3] = tmp_rir[3,(int)(samplerate*(RIR_CUT/100)):(RIR_LENGTH*samplerate) + (int)(samplerate*(RIR_CUT/100))]
    room_impulse_response = room_impulse_response.astype(np.float32)
    
    min_whistle_freq = 2000
    max_whistle_freq = 4000
    reverb_T60 = {}
    for whistle_freq in range(min_whistle_freq, max_whistle_freq+1):
        idx = np.argmax(np.squeeze(fft_windowed_data_log10[0,:,(int)((window_size/samplerate)*whistle_freq)]))
        max_amp = np.max(np.squeeze(fft_windowed_data_log10[0,:,(int)((window_size/samplerate)*whistle_freq)]))
        current_amp = np.max(np.squeeze(fft_windowed_data_log10[0,:,(int)((window_size/samplerate)*whistle_freq)]))
        count = 0
        reverb_T60[whistle_freq] = -1
        while np.abs(max_amp - current_amp) < 60 and idx < fft_windowed_data_log10.shape[1]:
            current_amp = np.squeeze(fft_windowed_data_log10[0,idx,(int)((window_size/samplerate)*whistle_freq)])
            idx += 1
            count += 1
        if count > reverb_T60[whistle_freq] and np.abs(max_amp - current_amp) >= 60:
            reverb_T60[whistle_freq] = count
        else:
            reverb_T60.pop(whistle_freq)
    print(np.median(list(reverb_T60.values())))

    direct = {}
    for whistle_freq in range(min_whistle_freq, max_whistle_freq+1):
        idx = np.argmax(np.squeeze(fft_windowed_data_log10[0,:,(int)((window_size/samplerate)*whistle_freq)]))
        current_amp = np.max(np.squeeze(fft_windowed_data_log10[0,:,(int)((window_size/samplerate)*whistle_freq)]))
        count = 0
        direct[whistle_freq] = -1
        while current_amp > 0 and idx > 0:
            current_amp = np.squeeze(fft_windowed_data_log10[0,idx,(int)((window_size/samplerate)*whistle_freq)])
            idx -= 1
            count += 1
        if count > direct[whistle_freq] and current_amp <= 0:
            direct[whistle_freq] = count
        else:
            direct.pop(whistle_freq)
    print(np.median(list(direct.values())))

    fig.clf()
    fig.add_axes(axs[0,0])
    fig.add_axes(axs[0,1])
    fig.add_axes(axs[1,0])
    fig.add_axes(axs[1,1])
    fig.add_axes(axs[2,0])
    fig.add_axes(axs[2,1])
    fig.add_axes(axs[3,0])
    fig.add_axes(axs[3,1])
    fig.canvas.manager.set_window_title("Room-Impulse-Response-Debug: " + dataset + f" ({len(room_impulse_response)//samplerate} sec.)")
    axs[0,0].cla()
    axs[0,0].set_ylabel("Rear Left")
    axs[0,0].imshow(np.reshape(fft_windowed_data_log10[0].T, (fft_windowed_data_log10[0].T.shape[1], fft_windowed_data_log10[0].T.shape[2], fft_windowed_data_log10[0].T.shape[0])), cmap="gnuplot2", interpolation="nearest", vmin=-60, vmax=0)
    axs[0,0].invert_yaxis()
    axs[0,1].plot(range(0, len(room_impulse_response[:,0])), room_impulse_response[:,0].astype(np.float32))
    axs[1,0].cla()
    axs[1,0].set_ylabel("Rear Right")
    axs[1,0].imshow(np.reshape(fft_windowed_data_log10[1].T, (fft_windowed_data_log10[1].T.shape[1], fft_windowed_data_log10[1].T.shape[2], fft_windowed_data_log10[1].T.shape[0])), cmap="gnuplot2", interpolation="nearest", vmin=-60, vmax=0)
    axs[1,0].invert_yaxis()
    axs[1,1].plot(range(0, len(room_impulse_response[:,1])), room_impulse_response[:,1].astype(np.float32))
    axs[2,0].cla()
    axs[2,0].set_ylabel("Front Left")
    axs[2,0].imshow(np.reshape(fft_windowed_data_log10[2].T, (fft_windowed_data_log10[2].T.shape[1], fft_windowed_data_log10[2].T.shape[2], fft_windowed_data_log10[2].T.shape[0])), cmap="gnuplot2", interpolation="nearest", vmin=-60, vmax=0)
    axs[2,0].invert_yaxis()
    axs[2,1].plot(range(0, len(room_impulse_response[:,2])), room_impulse_response[:,2].astype(np.float32))
    axs[3,0].cla()
    axs[3,0].set_ylabel("Front Right")
    axs[3,0].imshow(np.reshape(fft_windowed_data_log10[3].T, (fft_windowed_data_log10[3].T.shape[1], fft_windowed_data_log10[3].T.shape[2], fft_windowed_data_log10[3].T.shape[0])), cmap="gnuplot2", interpolation="nearest", vmin=-60, vmax=0)
    axs[3,0].invert_yaxis()
    axs[3,1].plot(range(0, len(room_impulse_response[:,3])), room_impulse_response[:,3].astype(np.float32))

    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0, wspace=0)
    plt.show()

def debug_room_impulse_response(dataset = None, window_size = WINDOW_SIZE):
    fig, axs = plt.subplots(4, 2)

    def on_click(event):
        global data
        global samplerate
        global global_window_size
        global global_load_data
        global room_impulse_response

        if event.dblclick:
            x, y = fig.transFigure.inverted().transform((event.x, event.y))
            rear_left_bb = axs[0,0].get_position()
            rear_right_bb = axs[1,0].get_position()
            front_left_bb = axs[2,0].get_position()
            front_right_bb = axs[3,0].get_position()
            if rear_left_bb.contains(x, y):
                sd.play(data[:,0], samplerate)
            elif rear_right_bb.contains(x, y):
                sd.play(data[:,1], samplerate)
            elif front_left_bb.contains(x, y):
                sd.play(data[:,2], samplerate)
            elif front_right_bb.contains(x, y):
                sd.play(data[:,3], samplerate)

            rear_left_bb = axs[0,1].get_position()
            rear_right_bb = axs[1,1].get_position()
            front_left_bb = axs[2,1].get_position()
            front_right_bb = axs[3,1].get_position()
            if rear_left_bb.contains(x, y):
                sd.play(room_impulse_response[:,0] * 30, samplerate)
            elif rear_right_bb.contains(x, y):
                sd.play(room_impulse_response[:,1] * 30, samplerate)
            elif front_left_bb.contains(x, y):
                sd.play(room_impulse_response[:,2] * 30, samplerate)
            elif front_right_bb.contains(x, y):
                sd.play(room_impulse_response[:,3] * 30, samplerate)

    def on_press(event):
        global room_impulse_response
        global global_load_data
        global samplerate
        global data

        if event.key == "w":
            wavfile.write(f"{dataset}_room_impulse_response.wav", samplerate, room_impulse_response)
        if event.key == "r":
            global_load_data = True
            update_plot(fig, axs,  dataset, window_size, global_load_data)

    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    pid = fig.canvas.mpl_connect("key_press_event", on_press)

    update_plot(fig, axs, dataset, window_size)

    fig.canvas.mpl_disconnect(cid)
    fig.canvas.mpl_disconnect(pid)

data = None
room_impulse_response = None
samplerate = None
global_window_size = WINDOW_SIZE
global_load_data = True

debug_room_impulse_response(dataset="Arena_3_1m")
print(samplerate)