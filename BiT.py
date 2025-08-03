import numpy as np
import cv2

def bio_taxo(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(image)
    std = np.std(image)
    mini = np.min(image)
    maxi = np.max(image)
    median = np.median(image)
    hist = cv2.calcHist([image], [0], None, [5], [0, 256]).flatten()
    vecteur = [mean, std, mini, maxi, median] + hist.tolist()
    return vecteur[:10]
import numpy as np
import cv2

def bio_taxo(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(image)
    std = np.std(image)
    mini = np.min(image)
    maxi = np.max(image)
    median = np.median(image)
    hist = cv2.calcHist([image], [0], None, [5], [0, 256]).flatten()
    vecteur = [mean, std, mini, maxi, median] + hist.tolist()
    return vecteur[:10]
