# -*- coding: utf-8 -*-
"""
AgroVision - Plant Disease Classifier
Streamlit App optimisé pour déploiement avec modèle .keras
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import warnings
import os
from tensorflow import keras
import keras

# Supprimer les avertissements
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# ============================================================================
# CONFIGURATION PAGE
# ============================================================================

st.set_page_config(
    page_title="AgroVision",
    page_icon="🍃",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CHARGEMENT MODÈLE (Cache Resource)
# ============================================================================

@st.cache_resource
def load_model():
    """Charge le modèle .keras avec gestion d'erreur robuste"""
    try:
        model = tf.keras.models.load_model("agrovision_model.keras")
        return model, True, None
    except FileNotFoundError:
        return None, False, "Modèle non trouvé"
    except Exception as e:
        return None, False, str(e)

# Charger le modèle
model, model_loaded, error_msg = load_model()

# ============================================================================
# LISTES DE CLASSES
# ============================================================================

CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites_Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# ============================================================================
# CSS PERSONNALISÉ
# ============================================================================

st.markdown("""
    <style>
    body {
        background-color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stApp {
        background-color: #ffffff;
    }
    .stButton > button {
        width: 100%;
        height: 50px;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        font-size: 16px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .prediction-box {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        margin: 20px 0;
        text-align: center;
    }
    .confidence-high {
        color: #4CAF50;
        font-weight: bold;
        font-size: 18px;
    }
    h1 {
        text-align: center;
        color: #2d5016;
    }
    h2 {
        color: #4CAF50;
        border-bottom: 2px solid #4CAF50;
        padding-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)


# 1. Utiliser le cache pour éviter les rechargements inutiles
@st.cache_resource
def load_my_model():
    # Remplacez par le chemin exact de votre fichier
    return keras.models.load_model("agrovision_model.keras")

# 2. Appeler la fonction pour récupérer le modèle
model = load_my_model()

st.title("AgrovisionAI - Analyse d'images")
st.write("Modèle Keras chargé avec succès !")

# ============================================================================
# VÉRIFICATION MODÈLE
# ============================================================================

if not model_loaded:
    st.error("⚠️ Le modèle n'est pas disponible. Veuillez réessayer.")
    st.stop()

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.subheader("📚 À Propos")
    st.markdown("""
    **AgroVision** utilise l'intelligence artificielle pour détecter les maladies des plantes.
    
    Uploadez une photo de feuille et recevez un diagnostic instantané !
    """)
    
    st.divider()
    
    st.subheader("🌾 Plantes Supportées")
    plants = sorted(list(set([cls.split('___')[0] for cls in CLASS_NAMES])))
    with st.expander(f"Voir {len(plants)} plantes"):
        for plant in plants:
            st.text(f"• {plant}")
    
    st.divider()
    st.markdown("**Contact:** lovibenedite@gmail.com")

# ============================================================================
# INTERFACE PRINCIPALE
# ============================================================================

st.title("🍃 AgroVision")
st.write("### Identifiez les maladies des plantes en un instant")

# Uploader
uploaded_file = st.file_uploader(
    "📤 Téléchargez une image de feuille (JPG, PNG)",
    type=['jpg', 'jpeg', 'png'],
    help="Assurez-vous que la feuille est bien visible"
)

if uploaded_file:
    # Charger et redimensionner l'image
    try:
        img = Image.open(uploaded_file)
        
        # Afficher l'image
        st.image(img, caption="Image chargée", use_column_width=True)
        
        # Redimensionner à 224x224
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prédiction
        with st.spinner("🔍 Analyse en cours..."):
            prediction = model.predict(img_array, verbose=0)[0]
        
        # Résultats
        top_idx = np.argmax(prediction)
        result = CLASS_NAMES[top_idx]
        confidence = prediction[top_idx]
        
        # Afficher le résultat principal
        st.markdown(f"""
            <div class='prediction-box'>
                <h3>🎯 Diagnostic</h3>
                <h2 style='margin: 0; color: white;'>{result.replace('_', ' ')}</h2>
                <p class='confidence-high' style='color: white; font-size: 20px;'>
                    Confiance: {confidence:.1%}
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Top 10 probabilités
        st.subheader("📊 Probabilités (Top 10)")
        top_indices = np.argsort(prediction)[-10:][::-1]
        top_classes = [CLASS_NAMES[i].replace('_', ' ') for i in top_indices]
        top_probs = prediction[top_indices]
        
        chart_df = {cls: prob for cls, prob in zip(top_classes, top_probs)}
        st.bar_chart(chart_df)
        
        # Voir toutes les prédictions
        with st.expander("📋 Voir toutes les prédictions (38 classes)"):
            all_indices = np.argsort(prediction)[::-1]
            all_classes = [CLASS_NAMES[i].replace('_', ' ') for i in all_indices]
            all_probs = prediction[all_indices]
            all_percent = (all_probs * 100).round(2)
            
            df_results = {
                "Classe": all_classes,
                "Probabilité": [f"{p:.4f}" for p in all_probs],
                "Pourcentage": [f"{p:.2f}%" for p in all_percent]
            }
            st.dataframe(df_results, use_container_width=True)
        
    except Exception as e:
        st.error(f"Erreur lors du traitement de l'image: {str(e)}")

# ============================================================================
# ONGLET GUIDE
# ============================================================================

st.divider()
st.subheader("📖 Guide d'Utilisation")

with st.expander("💡 Comment utiliser AgroVision ?", expanded=False):
    st.markdown("""
    ### 📸 Conseils pour une bonne photo
    
    ✅ **À faire:**
    - Prendre une photo claire et bien éclairée
    - Centrer la feuille dans le cadre
    - Utiliser un arrière-plan neutre (blanc, gris)
    - Photographier une feuille entière
    
    ❌ **À éviter:**
    - Photos floues ou mal éclairées
    - Arrière-plan chaotique ou coloré
    - Fragments de feuille
    - Ombres sur la feuille
    
    ### 🎯 Comment lire les résultats
    
    - **Diagnostic**: La prédiction principale du modèle
    - **Confiance**: Pourcentage de certitude (plus haut = mieux)
    - **Top 10**: Les classes les plus probables
    - **Toutes les classes**: La distribution complète sur les 38 classes
    """)

st.divider()
st.markdown("""
<div style='text-align: center; color: #999; font-size: 0.9em;'>
    🌱 AgroVision - Plant Disease Classification System | lovibenedite@gmail.com
</div>
""", unsafe_allow_html=True)
