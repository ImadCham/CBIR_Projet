
# 🔍 CBIR - Content-Based Image Retrieval (IA2 - Vision Artificielle)

Ce projet est une application Web simple de **recherche d’images par le contenu (CBIR)** développée dans le cadre du cours **420-1AB-TT - IA2**.

L'application permet de comparer une image requête avec une base d'images à l'aide de descripteurs visuels et d'afficher les images les plus similaires.

---

## 📁 Structure du projet

```
CBIR_Projet/
├── Iris/                           # Images classées en 3 catégories
│   ├── iris-setosa/
│   ├── iris-versicolour/
│   └── iris-virginica/
├── cbir_classification.py         # Application principale Streamlit
├── descripteurs.py                # Fonctions pour GLCM, Haralick, BiT...
├── BiT.py                         # Descripteur Bio-Inspiré
├── extraction_signatures.py       # Script pour générer les fichiers .npy
├── Signatures_GLCM.npy
├── Signatures_Haralick.npy
├── Signatures_BiT.npy
├── Signatures_Concaténation.npy
```

---

## ⚙️ Fonctionnalités

- 📤 Téléversement d’une image via une interface Streamlit
- 🧠 Choix du descripteur :
  - GLCM
  - Haralick
  - BiT
  - Concaténation des 3
- 📏 Choix de la distance de similarité :
  - Euclidienne
  - Manhattan
  - Tchebychev
  - Canberra
- 🖼️ Affichage des N images les plus proches visuellement

---

## 🚀 Lancer l'application

### 1. Installer les dépendances

```bash
pip install streamlit opencv-python mahotas scikit-image
```

### 2. Générer les fichiers `.npy` si ce n’est pas encore fait

```bash
python extraction_signatures.py
```

> Tu peux modifier `extraction_signatures.py` pour générer d'autres descripteurs (GLCM, Haralick, etc.)

### 3. Lancer Streamlit

```bash
streamlit run cbir_classification.py
```

---

## 📸 Exemple d’utilisation

- Upload une image de fleur
- Choisis `Haralick` et `Distance Euclidienne`
- L’application t’affiche les fleurs les plus similaires de la base

---

## 👨‍💻 Réalisé par

Imad Chamoumi  
Projet IA2 - Été 2025  
Institut Teccart – Cours Vision Artificielle & Reconnaissance de formes
