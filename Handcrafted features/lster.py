import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read,write
import scipy.signal as sgnl

def get_ste(sr, frame):
    win = sgnl.get_window("hamming",0.025*sr)
    """
    plt.subplot(2, 2, 1)
    plt.plot(win)
    plt.subplot(2, 2, 2)
    plt.plot(frame)
    filtered_frame = sgnl.convolve(frame, win, mode='same')
    plt.subplot(2, 2, 3)
    plt.plot(filtered_frame)
    plt.subplot(2, 2, 4)
    plt.plot(frame * win)
    plt.show()

    plt.subplot(1, 3, 1)
    plt.plot(np.log(np.abs(np.fft.rfft(frame))))

    plt.subplot(1, 3, 2)
    plt.plot(np.log(np.abs(np.fft.rfft(filtered_frame))))


    plt.subplot(1, 3, 3)
    plt.plot(np.log(np.abs(np.fft.rfft(frame*win))))
    plt.show()
    """
    filtered_frame = frame * win
    stave = np.sum(filtered_frame**2)# / float(len(frame))
    return stave

def get_lster(sr, audio):
    
    """
    for 1 sec audio
    """
    #assert sr==len(audio)
    frame_size=0.025*sr
    frame_shift=0.025*sr
    frame_count=0
    base = 0
    ste_list = []

    while base+frame_size <= len(audio):
        part = audio[base : base+frame_shift]
        ste = get_ste(sr, part)
        frame_count +=1
        ste_list.append(ste)
        base +=frame_shift

    ste_arr = np.array(ste_list)
    avste = np.mean(ste_arr)
    lster = np.sum(np.sign(0.5*avste - ste_arr) + 1)/(2.0*frame_count)
    return lster

def plot_lster(sr, audio):
    
    """    
    lster_list = []
    base = 0
    while base+sr <= len(audio):
        part = audio[base : base+sr]
        lster = get_lster(sr, part)
        lster_list.append(lster)
        base += sr

    lster_arr=np.array(lster_list)
    plt.plot(lster_arr)
    plt.show()
    """
    #sr = 0.5*sr
    base = 0
    window_size = 2.0 * sr
    window_shift = 1.0 * sr
    lster_per_sample = np.zeros_like(audio).astype(np.float)
    n_times = np.zeros_like(audio).astype(np.float)
    pure_music_threshold = 0.12

    while base + window_size <= len(audio):
        onesec = audio[base : base+window_size]
        lster = get_lster(sr, onesec)
        lster_per_sample[base : base+window_size] += np.ones(window_size) * lster
        n_times[base : base+window_size] += 1.0
        base += window_shift
    avlster_per_sample = lster_per_sample / n_times

    music = audio[avlster_per_sample < pure_music_threshold]
    nonmusic = audio[avlster_per_sample >= pure_music_threshold]

    #write('music.wav', sr, music)
    #write('nonmusic.wav', sr, nonmusic)

    plt.plot(avlster_per_sample)
    return avlster_per_sample
    #plt.show()

if __name__ == '__main__':
    inpfile = sys.argv[1]
    sr, audio = read(inpfile)
    #audio = audio - np.mean(audio)
    #audio = audio / np.var(audio)
    part=audio[:]
    plot_lster(sr,part)
    plt.show()            
