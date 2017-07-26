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

    """
    computes (NUMFRAMES * NUMFILTERS)-dimensional feature vector for 1 sec audio.
    """

    sr, onesec = audio_desc
    feature_mx = logfbank(onesec, sr, nfilt=40)
    feature_vec = feature_mx.flatten()
    return feature_vec


def create_features_labels(speech_files, nonspeech_files, test_files, val_size):

    """
    creates feature vector arrays of data and labels the training and validation data.
    """

    speech_feature_vectors = np.array([getFeatureVector(read(f)) for f in speech_files])
    nonspeech_feature_vectors = np.array([getFeatureVector(read(f)) for f in nonspeech_files])
    test_feature_vectors = np.array([getFeatureVector(read(f)) for f in test_files])

    # feature vectors
    x = np.vstack((speech_feature_vectors, nonspeech_feature_vectors))
    # labels
    y = np.array([1] * speech_feature_vectors.shape[0] + [0] * nonspeech_feature_vectors.shape[0])
    
    # convert y to one-hot representation
    y = np.eye(2)[y]

    xtrain, xval, ytrain, yval = train_test_split(x, y, test_size = val_size, random_state=0)
    return xtrain, ytrain, xval, yval, test_feature_vectors


def dim_reduce(xtrain, xval, xtest):

    """
    reduces the dimensions of xtrain, xval and xtest.
    """

    from sklearn.decomposition import PCA
    pca = PCA(800)
    pca.fit(xtrain)
    reduced_xtrain = pca.transform(xtrain)
    reduced_xval = pca.transform(xval)
    reduced_xtest = pca.transform(xtest)

    return reduced_xtrain, reduced_xval, reduced_xtest

    
def ANN_model(xtrain, ytrain, xval, yval, xtest, alpha, neurons):

    """
    creates an ANN model and returns loss and accuracy statistics and predicted values.
    """

    from keras.models import Sequential
    from keras.regularizers import l2
    from keras.layers import Dense
    from keras.optimizers import SGD

    random.seed(0)
    np.random.seed(0)

    model = Sequential()
    model.add(Dense(neurons, input_dim = xtrain.shape[1], activation='sigmoid', W_regularizer=l2(alpha)))
    #model.add(Dense(neurons, activation='sigmoid'))
    #model.add(Dense(neurons, activation='sigmoid'))
    #model.add(Dense(neurons, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))

    #sgd = SGD(lr=0.03, momentum=0.0, decay=0.0)
    
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    hist = model.fit(xtrain, ytrain, nb_epoch=100, batch_size=100, verbose=False, validation_data = (xval, yval))
    ypred_label = model.predict_classes(xtest)
    ypred_prob = model.predict(xtest)

    return hist, ypred_label, ypred_prob


def loss_acc_plots(hist):

    """
    plots the loss and accuracy graphs for training and validation data.
    """
    
    acc = hist.history['acc']
    loss = hist.history['loss']
    val_acc = hist.history['val_acc']
    val_loss = hist.history['val_loss']
    
    plt.subplot(2, 1, 1)
    plt.plot(acc,'goldenrod', label = 'Train acc')
    plt.plot(val_acc,'forestgreen', label = 'Val acc')
    plt.legend(loc='upper right')    

    plt.subplot(2, 1, 2)
    plt.plot(loss, 'goldenrod', label = 'Train loss')
    plt.plot(val_loss, 'forestgreen', label = 'Val loss')
    plt.legend(loc='upper right')

    print max(val_acc)
    plt.show()


def get_avprob_per_sample(test_audio, ypred_prob) :

    """
    returns array of probability values for each category for each sample.
    """

    base = 0
    count = 0
    window_size = sr
    window_shift = 0.3*sr
    prob_per_sample = np.zeros((len(test_audio), 2))
    n_times = np.zeros((len(test_audio), 2))

    while base + window_size <= len(test_audio):
        prob_per_sample[base : base+window_size, :] += np.repeat(ypred_prob[count : count+1, :], [window_size], axis = 0)
        n_times[base : base+window_size, :] += 1.0
        base += window_shift
        count += 1
    avprob_per_sample = prob_per_sample / n_times

    return avprob_per_sample


def write_result(sr, test_audio, avprob_per_sample, result_dir):

    """
    writes the speech and nonspeech part in two different files in result_dir.
    """
    count = 0
    speech_list = []
    nonspeech_list = []
    while count < avprob_per_sample.shape[0] : 
        if (np.argmax(avprob_per_sample[count, :]) == 1):
            speech_list.append(test_audio[count])
        else :
            nonspeech_list.append(test_audio[count])

        count += 1

    speech_result = np.array(speech_list)
    nonspeech_result = np.array(nonspeech_list)
    
    outfilename1 = os.path.join(result_dir, 'speech.wav')
    outfilename2 = os.path.join(result_dir, 'nonspeech.wav')
    
    write(outfilename1, sr, speech_result.astype(np.int16))
    write(outfilename2, sr, nonspeech_result.astype(np.int16))   

    

if __name__ == '__main__':
   
    val_size = 0.4
    alpha = 0.1
    neurons = 85

    sr, test_audio = read(sys.argv[1])    

    speech_dir = 'pure_speech'
    music_dir = 'pure_music'
    mixed_dir = 'speech_music'
    test_dir = 'test_data'
    result_dir = 'results'

    from dataset import store_dataset
    store_dataset(sr, test_audio, test_dir)
    
    speech_files = [os.path.join(speech_dir, f) for f in os.listdir(speech_dir)]
    music_files = [os.path.join(music_dir, f) for f in os.listdir(music_dir)]
    mixed_files = [os.path.join(mixed_dir, f) for f in os.listdir(mixed_dir)]
    
    test_files = [os.path.join(test_dir, str(i)+'.wav') for i in range(len(os.listdir(test_dir)))]
    
    speech_files = speech_files + mixed_files
    nonspeech_files = music_files #+ mixed_files

    np.random.shuffle(speech_files)
    np.random.shuffle(nonspeech_files)

    min_class_size = min(len(speech_files), len(nonspeech_files))
    speech_files = speech_files[:min_class_size]
    nonspeech_files = nonspeech_files[:min_class_size]
        
    xtrain, ytrain, xval, yval, xtest = create_features_labels(speech_files, nonspeech_files, test_files, val_size)

    reduced_xtrain, reduced_xval, reduced_xtest = dim_reduce(xtrain, xval, xtest)
    
    hist, ypred_label, ypred_prob = ANN_model(reduced_xtrain, ytrain, reduced_xval, yval, reduced_xtest, alpha, neurons)

    loss_acc_plots(hist)

    avprob_per_sample = get_avprob_per_sample(test_audio, ypred_prob)

    write_result(sr, test_audio, avprob_per_sample, result_dir)

        
