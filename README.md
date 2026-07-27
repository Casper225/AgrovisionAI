# 🌱 AgroVision - Plant Disease Classifier

Classificateur d'images pour détecter les maladies des plantes basé sur le dataset PlantVillage.

## 📋 Prérequis

- Python 3.8+
- Fichier modèle: `agrovision_model.h5`

## 🚀 Installation Locale

### 1. Cloner/Préparer les fichiers

```bash
# Créer un répertoire pour le projet
mkdir agrovision
cd agrovision

# Placer ces fichiers dans le répertoire:
# - streamlit_app.py
# - requirements.txt
# - agrovision_model.h5
# - .streamlit/config.toml (optionnel)
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Lancer l'application

```bash
streamlit run streamlit_app.py
```

L'application s'ouvrira à `http://localhost:8501`

## 🌐 Déploiement sur Streamlit Cloud

### Étape 1: Préparer sur GitHub

```bash
# Initialiser git (si ce n'est pas déjà fait)
git init
git add .
git commit -m "Deploy AgroVision app"
git branch -M main
git remote add origin https://github.com/VOTRE_USERNAME/agrovision.git
git push -u origin main
```

**Fichiers à inclure:**
- `streamlit_app.py`
- `requirements.txt`
- `agrovision_model.h5`
- `.streamlit/config.toml`
- `README.md`

### Étape 2: Déployer sur Streamlit Cloud

1. Aller sur https://streamlit.io/cloud
2. Cliquer sur "New app"
3. Sélectionner votre repository GitHub
4. Spécifier:
   - Repository: `agrovision`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
5. Cliquer "Deploy"

## 🔧 Dépannage

### Erreur: "FileNotFoundError: agrovision_model.h5"

**Solution:** Assurez-vous que le fichier `agrovision_model.h5` est dans le même répertoire que `streamlit_app.py`

```bash
ls -la agrovision_model.h5  # Vérifier le fichier existe
```

### Erreur: TypeError/Incompatibilité TensorFlow

**Solution:** Vérifier les versions compatibles

```bash
# Réinstaller avec les bonnes versions
pip install --upgrade -r requirements.txt
```

### L'app est lente au démarrage

**Raison:** TensorFlow prend du temps à charger

**Solution:** C'est normal pour la première fois. Les suivantes seront plus rapides grâce au cache Streamlit.

### Erreur de mémoire au déploiement

**Solution:** Réduire la taille du modèle
- Vérifier que `agrovision_model.h5` fait < 500MB
- Sur Streamlit Cloud, le limite est ~500MB par app

```bash
# Vérifier la taille
du -h agrovision_model.h5
```

## 📊 Structure de l'Application

```
agrovision/
├── streamlit_app.py          # Application principale
├── requirements.txt          # Dépendances Python
├── agrovision_model.h5       # Modèle entraîné
├── .streamlit/
│   └── config.toml          # Configuration Streamlit
└── README.md                 # Ce fichier
```

## 🎯 Features

✅ Upload d'images (JPG, JPEG, PNG, BMP)
✅ Capture d'images via caméra
✅ Prédictions en temps réel
✅ Visualisation des probabilités
✅ Top 10 des classes prédites
✅ Affichage détaillé de toutes les classes
✅ Guide d'utilisation intégré
✅ Cache du modèle pour performance optimale

## 📚 Dataset

- **Dataset:** PlantVillage
- **Classes:** 38 (plantes + maladies)
- **Input:** 224x224 pixels (RGB)
- **Prétraitement:** Rescale 1/255

## 🔬 Architecture du Modèle

```
Sequential Model:
├── Conv2D(32) → MaxPooling2D → Dropout(0.2)
├── Conv2D(64) → MaxPooling2D → Dropout(0.2)
├── Conv2D(128) → MaxPooling2D → Dropout(0.2)
├── Flatten
├── Dense(128) → Dropout(0.5)
└── Dense(38, softmax)

Optimizer: Adam
Loss: Categorical Crossentropy
```

## ⚠️ Limitations

- Fonctionne mieux avec des images claires et bien éclairées
- Optimisé pour les plantes du dataset PlantVillage
- Nécessite une feuille entière visible dans l'image

## 📧 Support

Pour toute question: `kouakouericlionel@gmail.com`

## 📄 License

All Rights Reserved © 2024

---

**Dernière mise à jour:** 2024
