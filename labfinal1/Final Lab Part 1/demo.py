import os
import cv2
import numpy as np
import pickle
import argparse

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC as LinearSVM
from sklearn.metrics import average_precision_score as AP

from utils import *
from stl10_input import read_all_images, read_labels, keep_relevant_images


plt.rcParams.update({'font.size': 6})


if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(
        description='Demo for the Part 1 of the Final Project.'
    )
    
    parser.add_argument(
        '--datapath', type=str, default='./data/stl10_binary/',
        help='datapath for the STL10 dataset',
    )
    parser.add_argument(
        '--n_samples_feat', type=int, default=200,
        help='number of samples to build the visual vocabulary',
    )
    parser.add_argument(
        '--n_samples_svm', type=int, default=300,
        help='number of samples to train the SVMs',
    )
    parser.add_argument(
        '--method', type=str, choices=['sift', 'hog'], default='sift',
        help='choice of feature descriptor (SIFT / HoG)',
    )
    parser.add_argument(
        '--codewords', type=int, default=500,
        help='number of codewords in the  visual vocabulary',
    )

    args = parser.parse_args()

    # read train and test data
    print('Reading STL10 images...')

    train_images, train_labels = keep_relevant_images(
        read_all_images(args.datapath + 'train_X.bin'),
        read_labels(args.datapath + 'train_y.bin'),
        relevant_labels = [1, 2, 9, 7, 3], # airplanes, birds, ships, horses, cars 
    )

    test_images, test_labels = keep_relevant_images(
        read_all_images(args.datapath + 'test_X.bin'),
        read_labels(args.datapath + 'test_y.bin'),
        relevant_labels = [1, 2, 9, 7, 3], # airplanes, birds, ships, horses, cars
    )

    train_labels = remap_classes(train_labels)
    test_labels = remap_classes(test_labels)

    N_train, N_test = len(train_labels), len(test_labels)
    classes = ['airplanes', 'birds', 'ships', 'horses', 'cars']

    # 1. Feature Extraction and Description
    plot_sift_features(train_images, train_labels, classes)

    # 2. Building Visual Vocabulary
    sift = cv2.SIFT_create()

    if os.path.exists(f'kmeans_{args.method}_{args.codewords}.pkl'):
        print('Loading codebook...')

        with open(f'kmeans_{args.method}_{args.codewords}.pkl', 'rb') as f:
            kmeans, samples_idx = pickle.load(f).values()

        if len(samples_idx) != 5 * args.n_samples_feat:
            raise RuntimeError('Your training set split does not match pretrained k-means')

        _, training_idx = split_train_samples(samples_idx, args.n_samples_svm, train_labels)

    else:
        print('Constructing codebook...')

        samples_idx, training_idx = split_train_samples(args.n_samples_feat, args.n_samples_svm, train_labels)

        features = []
        for i in samples_idx:
            des = descriptor(train_images[i], args.method)
            features += des.tolist()
        features = np.array(features)

        kmeans = KMeans(n_clusters=args.codewords).fit(features)

        with open(f'kmeans_{args.method}_{args.codewords}.pkl', 'wb') as f:
            pickle.dump({'kmeans': kmeans, 'samples_idx': samples_idx}, f)

    # 3 & 4. Encoding Features Using Visual Vocabulary and Representing images by frequencies of visual words
    print('Converting images to BoW...')

    X_train = np.zeros((len(training_idx), args.codewords))
    y_train = np.zeros(len(training_idx), dtype=int)
    X_test = np.zeros((N_test, args.codewords))
    y_test = np.zeros(N_test, dtype=int)

    for i, j in enumerate(training_idx):
        X_train[i, :] = codeword(train_images[j], args.method, kmeans, args.codewords)
        y_train[i] = train_labels[j].astype(int)

    for i in range(N_test):
        X_test[i, :] = codeword(test_images[i], args.method, kmeans, args.codewords)
        y_test[i] = test_labels[i].astype(int)

    fig, ax = plt.subplots(1, 5, sharey=True, sharex=True, num='Class histograms')
    for c, class_name in enumerate(classes):
        hist = X_train[(y_train == c), :].sum(axis=0)
        hist /= hist.sum()

        ax[c].bar(range(args.codewords), hist)
        ax[c].set_title(class_name)
        ax[c].set_box_aspect(1)

    fig.savefig(f'figures/histogram_{args.method}_{args.codewords}.eps', format='eps', bbox_inches='tight')

    # 5. Classification
    print('Training binary SVM classifiers...')

    svm = LinearSVM().fit(X_train, y_train)
    
    y_test = np.eye(5)[y_test] # convert to one-hot vectors
    y_score = svm.decision_function(X_test)

    print()
    for c, class_name in enumerate(classes):
        plot_retrieved(test_images, y_score[:, c], classes[c], args.method, args.codewords)
        print(f'Average Precision for class {class_name}: {100*AP(y_test[:, c], y_score[:, c]):.2f}')
    print()
    
    print('*'*44)
    print(f'mAP = {100*AP(y_test, y_score):.2f}')
    print('*'*44)

    #plt.tight_layout()
    #plt.show()
