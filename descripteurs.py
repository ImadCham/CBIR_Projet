import cv2
from skimage.feature import graycomatrix, graycoprops
from BiT import bio_taxo
from mahotas.features import haralick
import numpy as np

def glcm_RGB(chemin):
    data = cv2.imread(chemin)
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    list_carac = []
    for i in range(3):
        canal = data[:, :, i]
        co_matrice = graycomatrix(canal, [1], [3*np.pi/2], symmetric=False, normed=True)
        features = [graycoprops(co_matrice, p)[0, 0] for p in ['contrast', 'dissimilarity', 'homogeneity', 'correlation', 'energy', 'ASM']]
        list_carac.extend([float(x) for x in features])
    return list_carac

def haralick_feat_RGB(chemin):
    data = cv2.imread(chemin)
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    list_carac = []
    for i in range(3):
        canal = data[:, :, i]
        features = haralick(canal).mean(0).tolist()
        list_carac.extend([float(x) for x in features])
    return list_carac

def bitdesc_feat_RGB(chemin):
    data = cv2.imread(chemin)
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    list_carac = []
    for i in range(3):
        canal = data[:, :, i]
        features = bio_taxo(canal)
        list_carac.extend([float(x) for x in features])
    return list_carac

def concatenation_RGB(chemin):
    return glcm_RGB(chemin) + haralick_feat_RGB(chemin) + bitdesc_feat_RGB(chemin)
