import numpy as np
import sys
import os
from scipy.io.wavfile import read,write
import matplotlib.pyplot as plt
    
    
def get_framevalue(sr, frame1, frame2):
    delta=np.finfo(float).eps    
    fourier1 = np.log(np.abs(np.fft.rfft(frame1)) + delta)
    fourier2 = np.log(np.abs(np.fft.rfft(frame2)) + delta)
    num = (fourier2 - fourier1)**2
    result = np.sum(num)/(len(num)-1)
    return result

def get_specflux(sr, audio):
    
    """
    for 1 sec window
    """
    frame_size = int(0.025*sr)
    frame_shift = int(0.025*sr)
    base = 0
    frame_list = []

    while(base + frame_shift + frame_size <= len(audio)):
        frame1 = audio[base : base+frame_size]
        frame2 = audio[base+frame_shift : base+frame_shift+frame_size]
        framevalue = get_framevalue(sr, frame1, frame2)
        frame_list.append(framevalue)
        base += frame_shift

    frame_arr=np.array(frame_list)
    result = np.sum(frame_arr)/(len(frame_arr)-1)

    return result

def get_specflux_per_sec(sr, audio):

    #sr = 0.5*sr
    base = 0
    window_size = 2.0 * sr
    window_shift = 1.0 * sr
    spflux_per_sample = np.zeros_like(audio).astype(np.float)
    n_times = np.zeros_like(audio).astype(np.float)
    pure_music_threshold = 0.12

    while base + window_size <= len(audio):
        onesec = audio[base : base+window_size]
        spflux = get_specflux(sr, onesec)
        spflux_per_sample[base : base+window_size] += np.ones(window_size) * spflux
        n_times[base : base+window_size] += 1.0
        base += window_shift
    avspflux_per_sample = spflux_per_sample / n_times

    music = audio[avspflux_per_sample < pure_music_threshold]
    nonmusic = audio[avspflux_per_sample >= pure_music_threshold]

    #write('music.wav', sr, music)
    #write('nonmusic.wav', sr, nonmusic)

    plt.plot(avspflux_per_sample)
    return avspflux_per_sample
    #plt.show()

    """
    win_shift = sr
    win_size = sr
    base = 0
    specflux_list = []

    while(base + win_size <= len(audio)):
        part = audio[base : base+win_size]
        value=get_specflux(sr, part)
        specflux_list.append(value)
        #else:
         #   specflux_list.append(0)
        base += win_shift

    print specflux_list
    specflux_arr = np.array(specflux_list)
    #plt.plot(specflux_arr)
    #plt.show()

    return specflux_arr
    """

if __name__ == '__main__':
    inpfile = sys.argv[1]
    sr, audio = read(inpfile)
    #assert sr == 16000
    assert len(audio.shape)==1
    part=audio[:]
    get_specflux_per_sec(sr, part)
    plt.show()
    print(np.finfo(float).eps)
