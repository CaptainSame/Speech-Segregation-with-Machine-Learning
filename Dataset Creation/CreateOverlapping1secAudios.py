import numpy as np
from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
import sys
import os

def store_dataset(sr, audio, test_dir):

    """
    separates the input audio into segments and stores in input directory. 
    """
    base = 0
    base_dir = ''

    name = 0
    while base + sr <= len(audio):
        onesec = audio[base:base+sr]  
        base_dir = test_dir 
        outfilename = os.path.join(base_dir, str(name) + '.wav')
        write(outfilename, sr, onesec)
        base += sr*0.3
        name += 1

if __name__ == '__main__':

    test_dir = sys.argv[1]
    sr, audio = read(sys.argv[2])
    
    store_dataset(sr, audio, test_dir)
