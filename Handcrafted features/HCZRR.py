import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write

def get_zcr(sr, frame):
    frame_duration = len(frame) / float(sr)
    zero_crossings = np.where(np.diff(np.sign(frame)))[0]
    zcr = len(zero_crossings)/frame_duration
    return zcr


def get_hczrr(sr, onesec):
    """
    Calculates HCZRR for 1 sec audio clip
    """
    #assert sr==len(onesec)
    base = 0
    frame_size = 0.025 * sr
    frame_shift = 0.025 * sr
    frame_count = 0
    zcr_list = []
    while base + frame_size <= len(onesec):
        frame = onesec[base : base+frame_size]
        zcr = get_zcr(sr, frame)
        zcr_list.append(zcr)
        base += frame_shift
        frame_count += 1
    #print(zcr_list)
    zcr_arr = np.array(zcr_list)
    avzcr = np.mean(zcr_arr)
    hzcrr = np.sum(np.sign(zcr_arr - 1.5*avzcr) + 1) / (2.0 * frame_count)
    return hzcrr

def classify_and_store_music(sr, audio, music_dir, nonmusic_dir):
    base = 0
    results = []
    base_dir = ''
    pure_music_threshold = 0.15
    while base + sr <= len(audio):
        onesec = audio[base:base+sr]
        result = get_hczrr(sr, onesec)  
        if result < pure_music_threshold:
           base_dir = music_dir 
        else:
            base_dir = nonmusic_dir
        outfilename = os.path.join(base_dir, str(int(base/sr)) + '.wav')
        write(outfilename, sr, onesec)
        results.append(result)
        base += sr
    plt.plot(results)
    plt.show()


def classify_and_store_nonmusic_as_single_file(sr, audio):
    base = 0
    results = []
    classification_results = {}
    pure_music_threshold = 0.15
    while base + sr <= len(audio):
        onesec = audio[base:base+sr]
        result = get_hczrr(sr, onesec)  
        if result >= pure_music_threshold:
            if 'nonmusic' not in classification_results.keys():
                classification_results['nonmusic'] = onesec
            else:
                classification_results['nonmusic'] = np.concatenate((classification_results['nonmusic'], onesec))
        else:
            if 'music' not in classification_results.keys():
                classification_results['music'] = onesec
            else:
                classification_results['music'] = np.concatenate((classification_results['music'], onesec))
        results.append(result)
        base += sr
    for category in classification_results.keys():
        fname = os.path.join('outputs', category + '.wav')
        print classification_results[category]
        write(fname, sr, classification_results[category])
    plt.plot(results)
    plt.show()


if __name__ == '__main__':
    inpfile = sys.argv[1]
    sr, audio = read(inpfile)
    classify_and_store_nonmusic_as_single_file2(sr, audio)
    plt.show()

