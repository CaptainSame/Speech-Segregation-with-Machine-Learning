import numpy as np
from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
import sys
import os

def store_dataset(sr, audio1, audio2, audio3, speech_dir, music_dir, speech_music_dir):
    base = 0
    base_dir = ''

    while base + sr <= len(audio1):
        onesec = audio1[base:base+sr]  
        base_dir = speech_dir 
        outfilename1 = os.path.join(base_dir, str(int(base/sr)) + '.wav')
        write(outfilename1, sr, onesec)
        base += sr
    
    base=0
    while base + sr <= len(audio2):
        onesec = audio2[base:base+sr]  
        base_dir = music_dir 
        outfilename2 = os.path.join(base_dir, str(int(base/sr)) + '.wav')
        write(outfilename2, sr, onesec)
        base += sr

    base=0
    while base + sr <= len(audio3):
        onesec = audio3[base:base+sr]  
        base_dir = speech_music_dir 
        outfilename3 = os.path.join(base_dir, str(int(base/sr)) + '.wav')
        write(outfilename3, sr, onesec)
        base += sr

if __name__ == '__main__':

    speech_dir = sys.argv[1]
    sr, audio1 = read(sys.argv[2])
    music_dir = sys.argv[3]
    sr, audio2 = read(sys.argv[4])
    speech_music_dir = sys.argv[5]
    sr, audio3 = read(sys.argv[6])
    
    store_dataset(sr, audio1, audio2, audio3, speech_dir, music_dir, speech_music_dir)
