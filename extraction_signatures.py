from descripteurs import glcm_RGB, haralick_feat_RGB, bitdesc_feat_RGB, concatenation_RGB
import os
import numpy as np

dict_class = {'iris-setosa': 0, 'iris-versicolour': 1, 'iris-virginica': 2}

def extraction_signatures(repertoire, descripteur='GLCM'):
    signatures = []
    for root, _, files in os.walk(repertoire):
        for f in files:
            if f.lower().endswith(('.jpg', '.png')):
                chemin = os.path.join(root, f)
                classe = os.path.basename(root)
                if descripteur == 'GLCM':
                    vecteur = glcm_RGB(chemin)
                elif descripteur == 'Haralick':
                    vecteur = haralick_feat_RGB(chemin)
                elif descripteur == 'BiT':
                    vecteur = bitdesc_feat_RGB(chemin)
                elif descripteur == 'Concaténation':
                    vecteur = concatenation_RGB(chemin)
                vecteur.append(dict_class[classe])
                signatures.append(vecteur)
    np.save(f'Signatures_{descripteur}.npy', np.array(signatures))

if __name__ == '__main__':
    extraction_signatures('./Iris', 'Concaténation')
