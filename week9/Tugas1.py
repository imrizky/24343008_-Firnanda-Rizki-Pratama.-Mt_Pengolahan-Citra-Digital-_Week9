import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from skimage.filters import threshold_otsu
from skimage.filters import prewitt
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from skimage.metrics import adapted_rand_error

from scipy import ndimage as ndi
from sklearn.metrics import accuracy_score, precision_score, recall_score


# =========================================================
# LOAD IMAGE
# =========================================================

# Ganti dengan gambar milikmu
img1 = cv2.imread("week9/bimodial.jpg", 0)
img2 = cv2.imread("week9/illumination.jpg", 0)
img3 = cv2.imread("week9/overlap.jpg", 0)
images = [
    ("Bimodal", img1),
    ("Iluminasi Tidak Merata", img2),
    ("Overlapping", img3)
]


# =========================================================
# GROUND TRUTH SEDERHANA
# =========================================================

def create_ground_truth(img):
    _, gt = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return gt


# =========================================================
# METRIC EVALUATION
# =========================================================

def evaluate_metrics(gt, pred):

    gt = gt.flatten() > 0
    pred = pred.flatten() > 0

    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()

    iou = intersection / union if union != 0 else 0

    dice = (2 * intersection) / (gt.sum() + pred.sum())

    accuracy = accuracy_score(gt, pred)
    precision = precision_score(gt, pred, zero_division=0)
    recall = recall_score(gt, pred, zero_division=0)

    return {
        "IoU": iou,
        "Dice": dice,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall
    }


# =========================================================
# REGION GROWING
# =========================================================

def region_growing(img, seed, threshold=10):

    h, w = img.shape
    segmented = np.zeros((h, w), np.uint8)

    seed_value = img[seed]

    stack = [seed]

    while stack:
        x, y = stack.pop()

        if segmented[x, y] == 0:

            segmented[x, y] = 255

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:

                    nx = x + dx
                    ny = y + dy

                    if 0 <= nx < h and 0 <= ny < w:

                        if segmented[nx, ny] == 0:
                            if abs(int(img[nx, ny]) - int(seed_value)) < threshold:
                                stack.append((nx, ny))

    return segmented


# =========================================================
# PROCESSING
# =========================================================

for title, img in images:

    gt = create_ground_truth(img)

    methods = {}

    # -----------------------------------------------------
    # GLOBAL THRESHOLD
    # -----------------------------------------------------

    start = time.time()

    _, global_thresh = cv2.threshold(
        img, 127, 255, cv2.THRESH_BINARY
    )

    methods["Global Threshold"] = (
        global_thresh,
        time.time() - start
    )

    # -----------------------------------------------------
    # OTSU
    # -----------------------------------------------------

    start = time.time()

    _, otsu = cv2.threshold(
        img, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    methods["Otsu"] = (
        otsu,
        time.time() - start
    )

    # -----------------------------------------------------
    # ADAPTIVE MEAN
    # -----------------------------------------------------

    start = time.time()

    adaptive_mean = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    methods["Adaptive Mean"] = (
        adaptive_mean,
        time.time() - start
    )

    # -----------------------------------------------------
    # ADAPTIVE GAUSSIAN
    # -----------------------------------------------------

    start = time.time()

    adaptive_gaussian = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    methods["Adaptive Gaussian"] = (
        adaptive_gaussian,
        time.time() - start
    )

    # -----------------------------------------------------
    # SOBEL
    # -----------------------------------------------------

    start = time.time()

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(sobelx, sobely)
    magnitude = np.uint8(magnitude)

    methods["Sobel"] = (
        magnitude,
        time.time() - start
    )

    # -----------------------------------------------------
    # PREWITT
    # -----------------------------------------------------

    start = time.time()

    prewitt_img = prewitt(img)
    prewitt_img = (prewitt_img * 255).astype(np.uint8)

    methods["Prewitt"] = (
        prewitt_img,
        time.time() - start
    )

    # -----------------------------------------------------
    # CANNY
    # -----------------------------------------------------

    start = time.time()

    canny = cv2.Canny(img, 100, 200)

    methods["Canny"] = (
        canny,
        time.time() - start
    )

    # -----------------------------------------------------
    # REGION GROWING
    # -----------------------------------------------------

    start = time.time()

    seed = (img.shape[0] // 2, img.shape[1] // 2)

    rg = region_growing(img, seed)

    methods["Region Growing"] = (
        rg,
        time.time() - start
    )

    # -----------------------------------------------------
    # CONNECTED COMPONENT
    # -----------------------------------------------------

    start = time.time()

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    num_labels, labels = cv2.connectedComponents(binary)

    cc = (labels > 0).astype(np.uint8) * 255

    methods["Connected Components"] = (
        cc,
        time.time() - start
    )

    # -----------------------------------------------------
    # WATERSHED
    # -----------------------------------------------------

    start = time.time()

    ret, thresh = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(
        opening,
        cv2.DIST_L2,
        5
    )

    ret, sure_fg = cv2.threshold(
        dist_transform,
        0.7 * dist_transform.max(),
        255,
        0
    )

    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1

    markers[unknown == 255] = 0

    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    markers = cv2.watershed(color, markers)

    watershed_result = np.zeros(img.shape, dtype=np.uint8)
    watershed_result[markers > 1] = 255

    methods["Watershed"] = (
        watershed_result,
        time.time() - start
    )

    # =====================================================
    # VISUALIZATION
    # =====================================================

    plt.figure(figsize=(18, 12))

    plt.subplot(3, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Original\n{title}")
    plt.axis('off')

    plt.subplot(3, 4, 2)
    plt.imshow(gt, cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')

    idx = 3

    print("\n================================================")
    print(f"HASIL EVALUASI : {title}")
    print("================================================")

    for method_name, (result, comp_time) in methods.items():

        metrics = evaluate_metrics(gt, result)

        print(f"\n{method_name}")
        print(f"IoU       : {metrics['IoU']:.4f}")
        print(f"Dice      : {metrics['Dice']:.4f}")
        print(f"Accuracy  : {metrics['Accuracy']:.4f}")
        print(f"Precision : {metrics['Precision']:.4f}")
        print(f"Recall    : {metrics['Recall']:.4f}")
        print(f"Waktu     : {comp_time:.4f} detik")

        if idx <= 12:
            plt.subplot(3, 4, idx)
            plt.imshow(result, cmap='gray')
            plt.title(method_name)
            plt.axis('off')
            idx += 1

    plt.tight_layout()
    plt.show()