import os
import sys
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write

def get_ste(sr, frame):
    win = signal.get_window("hamming", len(frame))
    filtered_frame = frame * win
    stave = np.sum(filtered_frame**2) / float(len(frame))
    return stave

def get_zcr(sr, frame):
    frame_duration = len(frame) / float(sr)
    zero_crossings = np.where(np.diff(np.sign(frame)))[0]
    zcr = len(zero_crossings)/frame_duration
    return zcr

def isSilence(sr, frame):
    zcr = get_zcr(sr, frame)
    ste = get_ste(sr, frame)
    ratio=0
    result = False
    if ste == 0:
        result = True
    if zcr == 0:
        result = True
    else:
        ratio = ste / zcr
  
    return zcr, ste, ratio, result

def silent_part(sr, audio):
    ratio_list = []
    sil_list = []
    non_sil_list = []
    assert len(audio.shape)==1
    assert sr==16000
    base = 0
    frame_size = int(0.025 * sr)
    frame_shift = int(0.025 * sr)
    while base + frame_size <= len(audio):
        frame = audio[base:base+frame_size]
        zcr, ste, ratio, result = isSilence(sr, frame)
        if(ratio<0.86):
            sil_list.append(frame)
        else :
            non_sil_list.append(frame)
        ratio_list.append(ratio)
        base += frame_shift

    sil_arr = np.array(sil_list)
    sil_arr = np.ravel(sil_arr)
    non_sil_arr = np.array(non_sil_list)
    non_sil_arr = np.ravel(non_sil_arr)

    return sil_arr, non_sil_arr

if __name__ == '__main__':
    inpfile = sys.argv[1]
    
    sr, audio = read(inpfile)
    arr1, arr2 = silent_part(sr, audio)
    write('silent.wav', sr, arr1)
    print(np.mean(np.array(ratio_list)))




