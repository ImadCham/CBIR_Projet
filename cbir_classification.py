import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import cv2
import os
from descripteurs import glcm_RGB, haralick_feat_RGB, bitdesc_feat_RGB, concatenation_RGB

@st.cache_data
def load_signatures(descripteur):
    return np.load(f'Signatures_{descripteur}.npy')

def distance(v1, v2, methode):
    if methode == 'Euclidienne':
        return np.linalg.norm(v1 - v2)
    elif methode == 'Manhattan':
        return np.sum(np.abs(v1 - v2))
    elif methode == 'Tchebychev':
        return np.max(np.abs(v1 - v2))
    elif methode == 'Canberra':
        return np.sum(np.abs(v1 - v2) / (np.abs(v1) + np.abs(v2) + 1e-8))
    return 0

def extraire_features(image_path, type_desc):
    if type_desc == 'GLCM':
        return glcm_RGB(image_path)
    elif type_desc == 'Haralick':
        return haralick_feat_RGB(image_path)
    elif type_desc == 'BiT':
        return bitdesc_feat_RGB(image_path)
    elif type_desc == 'Concaténation':
        return concatenation_RGB(image_path)

st.title("🔍 Recherche d’images par le contenu (CBIR)")
image_query = st.file_uploader("Téléversez une image", type=["jpg", "png", "jpeg"])
nb_images = st.slider("Nombre d’images similaires à afficher", 1, 10, 5)
type_desc = st.selectbox("Choix du descripteur", ["GLCM", "Haralick", "BiT", "Concaténation"])
methode_dist = st.selectbox("Méthode de distance", ["Euclidienne", "Manhattan", "Tchebychev", "Canberra"])

if image_query:
    file_bytes = np.asarray(bytearray(image_query.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    cv2.imwrite("temp.jpg", img)
    feature_image = np.array(extraire_features("temp.jpg", type_desc))

    base = load_signatures(type_desc)
    x = base[:, :-1].astype('float')
    y = base[:, -1]

    scores = []
    for i in range(len(x)):
        d = distance(feature_image, x[i], methode_dist)
        scores.append((d, int(y[i]), i))

    scores.sort()
    st.image("temp.jpg", caption="Image requête", width=150)
    st.subheader("🔎 Images les plus similaires :")

    img_folder = "./Iris"
    count = 0
    for _, classe, idx in scores[:nb_images]:
        for root, _, files in os.walk(img_folder):
            for file in files:
                chemin = os.path.join(root, file)
                if file.lower().endswith(('jpg', 'jpeg', 'png')):
                    vect = extraire_features(chemin, type_desc)
                    if np.allclose(vect, x[idx]):
                        st.image(chemin, width=150, caption=f"Classe : {int(classe)}")
                        count += 1
                        break
            if count >= nb_images:
                break
