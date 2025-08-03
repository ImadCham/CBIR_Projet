
Instructions pour exécuter le projet CBIR (Cours 10):

1. Placez les images d'entraînement dans les sous-dossiers de 'Iris/' :
   - Iris/iris-setosa/
   - Iris/iris-versicolour/
   - Iris/iris-virginica/

2. Exécute extraction_signatures.py pour générer les fichiers Signatures_X.npy :
   python extraction_signatures.py

3. Exécutez l'application CBIR :
   streamlit run cbir_classification.py

4. L'interface Streamlit vous permettra de :
   - Uploader une image requête
   - Choisir un descripteur et une distance
   - Voir les images les plus similaires
