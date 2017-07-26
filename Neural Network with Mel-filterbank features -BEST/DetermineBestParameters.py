import os
import sys
import numpy as np
from features import logfbank
from scipy.io.wavfile import read, write

from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import random

random.seed(0)

np.random.seed(0)

def getFeatureVector(audio_desc):
    sr, onesec = audio_desc
    feature_mx = logfbank(onesec, sr, nfilt=40)
    #print 'logfbank ' , feature_mx.shape
    feature_vec = feature_mx.flatten()
    return feature_vec
    

if __name__ == '__main__':
   
    speech_dir = 'pure_speech'
    music_dir = 'pure_music'
    mixed_dir = 'speech_music'

    speech_files = [os.path.join(speech_dir, f) for f in os.listdir(speech_dir)]
    music_files = [os.path.join(music_dir, f) for f in os.listdir(music_dir)]
    mixed_files = [os.path.join(mixed_dir, f) for f in os.listdir(mixed_dir)]
    
    speech_files = speech_files + mixed_files
    nonspeech_files = music_files #+ mixed_files

    # randomize to ensure unbiased sampling
    np.random.shuffle(speech_files)
    np.random.shuffle(nonspeech_files)
    min_class_size = min(len(speech_files), len(nonspeech_files))
    speech_files = speech_files[:min_class_size]
    nonspeech_files = nonspeech_files[:min_class_size]

    # convert to features and labels
    speech_feature_vectors = np.array([getFeatureVector(read(f)) for f in speech_files])
    print speech_feature_vectors.shape
    nonspeech_feature_vectors = np.array([getFeatureVector(read(f)) for f in nonspeech_files])
    print nonspeech_feature_vectors.shape

    # feature vectors
    X = np.vstack((speech_feature_vectors, nonspeech_feature_vectors))
    # labels
    y = np.array([1] * speech_feature_vectors.shape[0] + [0] * nonspeech_feature_vectors.shape[0])
    
    # convert y to one-hot representation
    y = np.eye(2)[y]

    # split into training and test set
    from keras.models import Sequential
    from keras.regularizers import l2
    from keras.layers import Dense
    from keras.optimizers import SGD

    
    max_acc_list = []
    alpha_list = []
    t_size_list = []

    t_size = 0.20
    while (t_size <= 0.50):
        print ' + Test size:', t_size

        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = t_size, random_state=0)
        print ' + Consistency check:', Xtrain[0, :4], ytrain[0]

        from sklearn.decomposition import PCA
        pca = PCA(400)
        pca.fit(Xtrain)
        reduced_Xtrain = pca.transform(Xtrain)
        reduced_Xtest = pca.transform(Xtest)

        #print 'reduced shapes', reduced_Xtrain.shape, reduced_Xtest.shape
        #print 'ys', ytrain.shape, ytest.shape

        alpha = 0.0
        while(alpha <= 0.9):
            print '   + alpha:', alpha

            np.random.seed(0)
            random.seed(0)

            model = Sequential()
            model.add(Dense(5, input_dim=reduced_Xtrain.shape[1], activation='sigmoid', W_regularizer=l2(alpha)))
            model.add(Dense(2, activation='softmax'))

            sgd = SGD(lr=0.03, momentum=0.0, decay=0.0)

            model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

            sys.stdout = open(os.devnull, "w")

            hist = model.fit(reduced_Xtrain, ytrain, nb_epoch=100, batch_size=100, verbose=True, validation_data=(reduced_Xtest, ytest))
    
            sys.stdout = sys.__stdout__
            #loss_and_metrics = model.evaluate(reduced_Xtest, ytest, batch_size=100)

            arr = hist.history['val_acc']
            max_acc_list.append(max(arr))
            alpha_list.append(alpha)
            t_size_list.append(t_size)
            print '    + max of arr:', max(arr)
            print ''

            alpha = alpha + 0.05
        print '\n'

        t_size = t_size + 0.05

    index_of_best_max = np.argmax(max_acc_list)
    best_acc = max_acc_list[index_of_best_max]
    best_alpha = alpha_list[index_of_best_max]
    best_t_size = t_size_list[index_of_best_max]

    print 'Best acc:', best_acc
    print 'Best alpha:', best_alpha
    print 'Best test size:', best_t_size

    

    
    
    

