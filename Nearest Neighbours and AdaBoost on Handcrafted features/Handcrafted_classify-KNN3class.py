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
    

    # randomize to ensure unbiased sampling
    np.random.shuffle(speech_files)
    np.random.shuffle(music_files)
    np.random.shuffle(mixed_files)

    # convert to features and labels
    speech_feature_vectors = np.array([getFeatureVector(read(f)) for f in speech_files])[:199]
    music_feature_vectors = np.array([getFeatureVector(read(f)) for f in music_files])[:199]
    mixed_feature_vectors = np.array([getFeatureVector(read(f)) for f in mixed_files])[:199]

    print speech_feature_vectors.shape, music_feature_vectors.shape, mixed_feature_vectors.shape
    
    # feature vectors
    X = np.vstack((speech_feature_vectors, music_feature_vectors, mixed_feature_vectors))
    # labels
    y = np.array([0] * speech_feature_vectors.shape[0] + [1] * music_feature_vectors.shape[0] + [2] * mixed_feature_vectors.shape[0])

    # split into training and test set
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    """
    ax.scatter(Xtrain[ytrain==0,0], Xtrain[ytrain==0,1], Xtrain[ytrain==0,2], c='b')
    ax.scatter(Xtrain[ytrain==1,0], Xtrain[ytrain==1,1], Xtrain[ytrain==1,2], c='r')
    ax.scatter(Xtrain[ytrain==2,0], Xtrain[ytrain==2,1], Xtrain[ytrain==2,2], c='g')
    plt.show()
    """
    
    from sklearn.neighbors import KNeighborsClassifier as KNC
    from sklearn.ensemble import AdaBoostClassifier

    knc = KNC(5)
    #knc = AdaBoostClassifier()
    knc.fit(Xtrain, ytrain)
    
    ypred = knc.predict(Xtest)

    from sklearn.metrics import accuracy_score

    score = accuracy_score(ytest, ypred)
    print score
