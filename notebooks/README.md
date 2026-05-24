# 🌿 Détection de Maladies des Plantes par Deep Learning

Système de diagnostic automatique des maladies de plantes à partir d'une photo de feuille.
À partir d'une image, le modèle prédit **le type de plante, son état (saine / malade) et la maladie**, avec un **score de confiance**.

Projet réalisé dans le cadre du module **Deep Learning** (encadré par M. Abdallah Khemais).

---

## 📋 Description du projet

Il s'agit d'un problème de **classification d'images multi-classes** (15 classes). À partir d'une feuille photographiée, le système identifie la combinaison *plante + état + maladie* comme une classe unique. L'objectif est de fournir un outil d'aide à la décision pour les agriculteurs.

## 🗂️ Jeu de données

Nous utilisons le **PlantVillage Dataset** (Kaggle : `emmarex/plantdisease`), en gardant **3 plantes** : Tomate, Poivron et Pomme de terre.

- **Nombre total d'images :** ~20 638
- **Nombre de classes :** 15

<details>
<summary>Liste des 15 classes</summary>

| # | Classe | Plante | État |
|---|--------|--------|------|
| 1 | Pepper__bell___Bacterial_spot | Poivron | Malade (tache bactérienne) |
| 2 | Pepper__bell___healthy | Poivron | Saine |
| 3 | Potato___Early_blight | Pomme de terre | Malade (brûlure précoce) |
| 4 | Potato___Late_blight | Pomme de terre | Malade (mildiou) |
| 5 | Potato___healthy | Pomme de terre | Saine |
| 6 | Tomato_Bacterial_spot | Tomate | Malade (tache bactérienne) |
| 7 | Tomato_Early_blight | Tomate | Malade (brûlure précoce) |
| 8 | Tomato_Late_blight | Tomate | Malade (mildiou) |
| 9 | Tomato_Leaf_Mold | Tomate | Malade (moisissure) |
| 10 | Tomato_Septoria_leaf_spot | Tomate | Malade (tache septorienne) |
| 11 | Tomato_Spider_mites_Two_spotted_spider_mite | Tomate | Malade (acariens) |
| 12 | Tomato__Target_Spot | Tomate | Malade (tache cible) |
| 13 | Tomato__Tomato_YellowLeaf__Curl_Virus | Tomate | Malade (virus feuilles jaunes) |
| 14 | Tomato__Tomato_mosaic_virus | Tomate | Malade (virus mosaïque) |
| 15 | Tomato_healthy | Tomate | Saine |

</details>

**Découpage stratifié** : 70 % entraînement · 15 % validation · 15 % test (chaque classe représentée proportionnellement dans les trois jeux).

## 🏗️ Architecture du modèle

Nous comparons **deux architectures** par *transfer learning* (poids pré-entraînés sur ImageNet), puis nous **déployons automatiquement la meilleure** :

| Modèle | Base pré-entraînée | Paramètres |
|--------|--------------------|------------|
| **EfficientNetB0** | ImageNet | ~5,3 M |
| **ResNet50V2** | ImageNet | ~25,6 M |

**Tête de classification commune :**
`GlobalAveragePooling2D → BatchNorm → Dropout(0.4) → Dense(256, swish) → BatchNorm → Dropout(0.3) → Dense(15, softmax)`

**Entraînement en deux phases :**
- **Phase 1** — base gelée, on entraîne seulement la tête (`lr = 1e-3`, 10 époques max).
- **Phase 2** — *fine-tuning*, base dégelée (`lr = 1e-5`, 8 époques max).

**Techniques de régularisation :** data augmentation (flip, rotation, zoom, contraste), Dropout, BatchNormalization, Label Smoothing, EarlyStopping, ReduceLROnPlateau.

## 📊 Performances obtenues

> ⚠️ À compléter avec **vos chiffres réels** après avoir exécuté le notebook corrigé (blocs 1 → 13).

| Modèle | Accuracy (test) |
|--------|-----------------|
| EfficientNetB0 | `__ %` |
| ResNet50V2 | `__ %` |
| **Meilleur modèle déployé** | **`______`** |

- **F1-score (macro)** : `____`
- **Kappa de Cohen** : `____`
- **AUC macro (One-vs-Rest)** : `____`

La matrice de confusion et les courbes ROC par classe sont générées dans le notebook (Bloc 12).

## 🚀 Comment exécuter le code

### 1. Cloner le dépôt
```bash
git clone https://github.com/VOTRE_USER/Plant_Disease_Detection.git
cd Plant_Disease_Detection
```

### 2. Ouvrir le notebook final dans Google Colab
Importez `notebooks/06_final_pipeline_COMPLET.ipynb` dans [Google Colab](https://colab.research.google.com), puis activez le GPU :
`Exécution ▸ Modifier le type d'exécution ▸ GPU (T4)`.

### 3. Entraînement (première fois)
Exécutez les **blocs 1 → 13** dans l'ordre (`Exécution ▸ Tout exécuter`). Le dataset se télécharge automatiquement depuis Kaggle ; le meilleur modèle est sauvegardé sur votre Google Drive.

### 4. Déploiement / test (sans ré-entraîner)
Une fois le modèle entraîné et sauvegardé, exécutez seulement les **blocs 1, 2, 4, 14 et 15** : le modèle est rechargé depuis Drive et l'**interface Gradio** s'ouvre avec un lien public. Vous chargez une photo → diagnostic + score de confiance.

## 📁 Structure du dépôt

```
Plant_Disease_Detection/
├── README.md
├── presentation.pptx                       # Diaporama de soutenance (7-8 slides)
├── notebooks/
│   ├── 01_draft_brouillon.ipynb            # Expérimentations (brouillon)
│   └── 06_final_pipeline_COMPLET.ipynb     # Notebook final, propre et commenté
├── models/
│   └── (best_model.keras — généré à l'exécution, sauvegardé sur Drive)
└── data/
    └── (téléchargé automatiquement depuis Kaggle par le notebook)
```

## 🛠️ Technologies

Python · TensorFlow / Keras · scikit-learn · Gradio · Matplotlib · Seaborn · Google Colab
