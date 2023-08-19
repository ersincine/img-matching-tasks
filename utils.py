import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def read_image(path, is_grayscale=False):
    if is_grayscale:
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    else:
        img = cv.imread(path, cv.IMREAD_COLOR)
    assert img is not None, f'Failed to read image from {path}'
    return img
    

def show_image(img):
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        assert len(img.shape) == 3 and img.shape[2] == 3
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()


def detect_keypoints(img, detector=cv.SIFT_create(contrastThreshold=-10000, edgeThreshold=-10000), order=None):
    # order=lambda x: -x.size is useful.
    kp = list(detector.detect(img))
    if order is not None:
        kp.sort(key=order)
    assert len(kp) > 0
    return kp


def compute_descriptors(img, keypoints, extractor=cv.SIFT_create(contrastThreshold=-10000, edgeThreshold=-10000), root=True):
    # describe keypoints
    _, descriptors = extractor.compute(img, keypoints)
    assert len(descriptors) == len(keypoints)
    if root:
        descriptors /= descriptors.sum(axis=1, keepdims=True) + 1e-7
        descriptors = np.sqrt(descriptors)
    return descriptors


def euclidean_distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def average_corner_error(img_height: int, img_width: int, H_true: np.ndarray, H_estimated: np.ndarray) -> float:
    pts = np.array([[0, 0], [img_width, 0], [img_width, img_height], [0, img_height]], dtype=np.float32).reshape(1, 4, 2)
    pts_true = cv.perspectiveTransform(pts, H_true).reshape(4, 2)
    pts_estimated = cv.perspectiveTransform(pts, H_estimated).reshape(4, 2)
    error = 0
    for pt_true, pt_estimated in zip(pts_true, pts_estimated):
        error += euclidean_distance(*pt_true, *pt_estimated)
    return error / 4
