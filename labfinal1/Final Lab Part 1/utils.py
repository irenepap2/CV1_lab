import cv2
import numpy as np

from PIL import Image, ImageOps
from typing import Callable, Union
from matplotlib import pyplot as plt
from skimage.feature import hog


def remap_classes(labels: np.ndarray) -> np.ndarray:
    d = {1: 0, 2: 1, 9: 2, 7: 3, 3: 4}

    return np.array([d[label] for label in labels])


def plot_sift_features(images: np.ndarray, labels: np.ndarray, classes: list):
    samples = []
    for c in np.unique(labels):
        available_idx = np.where(labels == c)[0]
        samples += np.random.choice(available_idx, size = 2, replace=False).tolist()
    
    fig, ax = plt.subplots(2, 5, num='SIFT features', figsize=(6, 2.4))
    sift = cv2.SIFT_create()
    for j, sample in enumerate(samples):
        img = cv2.cvtColor(images[sample], cv2.COLOR_RGB2GRAY)
        kp = sift.detect(img, None)
        img = cv2.drawKeypoints(images[sample], kp, img, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        ax[j%2, j//2].imshow(img)
        ax[j%2, j//2].set_axis_off()
        if j%2 == 0:
            ax[j%2, j//2].set_title(f'{classes[labels[sample]]}')
    
    fig.savefig('figures/sift.eps', format='eps', bbox_inches='tight')


def split_train_samples(feat_samples: Union[int, np.ndarray], svm_samples: int, labels: np.ndarray):
    if type(feat_samples) == int:
        samples_idx = []
        for c in np.unique(labels):
            available_idx = np.where(labels == c)[0]

            samples_idx += np.random.choice(
                available_idx,
                size = feat_samples,
                replace = False,
            ).tolist()
    else:
        samples_idx = feat_samples

    training_idx = []
    for c in np.unique(labels):
        available_idx = np.setdiff1d(
            np.where(labels == c)[0], samples_idx, assume_unique=True,
        )

        training_idx += np.random.choice(
            available_idx, 
            size = svm_samples,
            replace = False,
        ).tolist()

    return np.array(samples_idx), np.array(training_idx)


def counter(iter: np.ndarray, N: int) -> np.ndarray:
    arr = np.zeros(N, dtype=int)
    for element in iter:
        arr[element] += 1

    return arr


def descriptor(img: np.ndarray, method: str) -> np.ndarray:
    if method == 'sift':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create()
        _, des = sift.detectAndCompute(img, None)
    elif method == 'hog':
        des = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), multichannel=(img.ndim == 3)).reshape(6*6, 8)
    else:
        raise NotImplemented(f'Descriptor {method} is not supported.')

    return des


def codeword(img: np.ndarray, method: str, kmeans: Callable, n_codewords: int) -> np.ndarray:
    des = descriptor(img, method)
    codewords = kmeans.predict(des.astype(np.float64))
    histogram = counter(codewords, N = n_codewords)
        
    return histogram / histogram.sum()


def plot_retrieved(images, score, class_name, method, codewords):
    top_idx = np.argpartition(score, -5)[-5:]
    bottom_idx = np.argpartition(score, 5)[:5]

    fig, ax = plt.subplots(2, 5, num=f'Top & bottom ranked images for class {class_name}', figsize=(6, 2.4))
    for i, idx in enumerate(top_idx):
        img = Image.fromarray(images[idx], 'RGB')
        img = ImageOps.expand(img, border=5, fill='green')
        ax[0, i].imshow(img)
        ax[0, i].set_axis_off()

    for i, idx in enumerate(bottom_idx):
        img = Image.fromarray(images[idx], 'RGB')
        img = ImageOps.expand(img, border=5, fill='red')
        ax[1, i].imshow(img)
        ax[1, i].set_axis_off()
    
    fig.savefig(f'figures/top_bottom_{class_name}_{method}_{codewords}.eps', format='eps', bbox_inches='tight')
