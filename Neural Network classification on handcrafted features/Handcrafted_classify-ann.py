import os
import sys
import numpy as np
from lster import get_lster
from HCZRR import get_hczrr
from sflux import get_specflux
from scipy.io.wavfile import read, write

from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def getFeatureVector(audio_desc):
    sr, onesec = audio_desc
    feature1 = get_hczrr(sr, onesec)
    feature2 = get_lster(sr, onesec)
    feature3 = get_specflux(sr, onesec)
    return np.array([feature1, feature2, feature3])
    

if __name__ == '__main__':
   
    speech_dir = 'pure_speech'
    music_dir = 'pure_music'
    mixed_dir = 'speech_music'

    speech_files = [os.path.join(speech_dir, f) for f in os.listdir(speech_dir)]
    music_files = [os.path.join(music_dir, f) for f in os.listdir(music_dir)]
    mixed_files = [os.path.join(mixed_dir, f) for f in os.listdir(mixed_dir)]
    
    nonspeech_files = music_files + mixed_files

    # randomize to ensure unbiased sampling
    np.random.shuffle(speech_files)
    np.random.shuffle(nonspeech_files)

    # convert to features and labels
    speech_feature_vectors = np.array([getFeatureVector(read(f)) for f in speech_files])
    nonspeech_feature_vectors = np.array([getFeatureVector(read(f)) for f in nonspeech_files])

    # feature vectors
    X = np.vstack((speech_feature_vectors, nonspeech_feature_vectors))
    # labels
    y = np.array([1] * speech_feature_vectors.shape[0] + [0] * nonspeech_feature_vectors.shape[0])
    
    # convert y to one-hot representation
    y = np.eye(2)[y]

    # split into training and test set
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5)

    from keras.models import Sequential
    from keras.layers import Dense

    model = Sequential()
    model.add(Dense(10, input_dim=3, activation='sigmoid'))
    model.add(Dense(20, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    print model.summary()
    
    model.fit(Xtrain, ytrain, nb_epoch=25000, batch_size=100, verbose=True, validation_data=(Xtest, ytest))
