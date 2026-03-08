# 🎨 Générateur de Coloriage Mystère

Un générateur Python qui transforme automatiquement des images colorées en **coloriages mystère** personnalisés avec difficulté ajustable. Utilise K-Means pour la quantification des couleurs et la tessellation de Voronoï pour créer un effet mosaïque sophistiqué.

## 🌟 Fonctionnalités

- **Quantification intelligente des couleurs** via K-Means clustering (jusqu'à 256 couleurs)
- **Tessellation de Voronoï contrainte** pour subdiviser les zones selon la difficulté
- **Placement optimal des symboles** avec l'algorithme Pole of Inaccessibility
- **Niveaux de difficulté** de 1 (facile) à 10 (expert - 1000+ pièces)
- **Symboles personnalisables** : nombres, lettres, ou symboles personnalisés
- **Légende automatique** avec mapping couleur ↔ symbole
- **Export haute qualité** en PNG ou PDF

## 📋 Prérequis

- Python 3.8 ou supérieur
- Système d'exploitation : Windows, macOS, ou Linux

## 🚀 Installation

### 1. Cloner ou télécharger le projet

```powershell
cd c:\Users\Stephen\Documents\colo_generation
```

### 2. Créer un environnement virtuel (recommandé)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Installer les dépendances

```powershell
pip install -r requirements.txt
```

### Dépendances principales :
- **opencv-python** : Traitement d'image et détection de contours
- **numpy** : Calculs matriciels
- **scikit-learn** : K-Means clustering
- **scipy** : Diagrammes de Voronoï
- **Pillow** : Rendu de texte et export
- **matplotlib** : Visualisation (optionnel)

## 📖 Utilisation

### API FastAPI pour Render

Cette app peut maintenant etre deployee comme API sur Render avec `FastAPI`.
L'endpoint principal accepte une image en entree et renvoie une archive ZIP contenant :

- `<nom>_combined.png`
- `<nom>_preview.png`

Fichiers ajoutes pour le deploiement :

- `api.py`
- `render.yaml`
- `runtime.txt`

#### Lancer l'API en local

```powershell
uvicorn api:app --reload
```

Puis ouvrir `http://127.0.0.1:8000/docs`.

#### Tester l'endpoint

```bash
curl -X POST "http://127.0.0.1:8000/generate" \
  -F "image=@input/mon_image.jpg" \
  -F "output_name=mon_coloriage" \
  -F "colors=16" \
  -F "difficulty=5" \
  -F "symbols=numbers" \
  --output mon_coloriage_outputs.zip
```

Le ZIP contient les deux fichiers demandes : `_combined` et `_preview`.

#### Deploiement sur Render

1. Pousser le projet sur GitHub
2. Creer un nouveau `Blueprint` ou `Web Service` sur Render
3. Laisser Render lire `render.yaml`
4. Une fois deploye, utiliser `https://votre-app.onrender.com/docs`

#### Parametres de l'API

L'endpoint `POST /generate` accepte les champs `multipart/form-data` suivants :

- `image` : fichier image requis
- `output_name` : nom de base optionnel
- `colors` : nombre de couleurs
- `difficulty` : niveau 1 a 10
- `symbols` : `numbers`, `letters`, `custom`
- `min_area` : aire minimale d'une region
- `resolution` : dimension max de travail
- `symbol_size` : ratio de taille du symbole
- `prefill_dark` : seuil de pre-remplissage des zones sombres
- `mode_filter` : taille du filtre de nettoyage
- `no_bilateral` : desactive le filtre bilateral
- `force_colors` : format `R,G,B;R,G,B`

### Interface en ligne de commande

#### Utilisation basique :

```powershell
python main.py input/mon_image.jpg
```

#### Avec options personnalisées :

```powershell
python main.py input/mon_image.jpg -d 7 -c 12 -s numbers --min-area 100
```

### Paramètres disponibles :

| Paramètre | Description | Valeurs | Défaut |
|-----------|-------------|---------|--------|
| `input` | Chemin de l'image source | chemin fichier | - |
| `-o, --output` | Nom de base des fichiers sortie | texte | nom de l'image |
| `-c, --colors` | Nombre de couleurs | 2-256 | 16 |
| `-d, --difficulty` | Niveau de difficulté | 1-10 | 5 |
| `-s, --symbols` | Type de symboles | numbers/letters/custom | numbers |
| `-m, --min-area` | Aire minimale (px²) - régions plus petites ignorées | entier > 0 | 50 |
| `-r, --resolution` | Dimension max de l'image (px) | entier > 0 | 1400 |
| `--symbol-size` | Taille des symboles (ratio) | 0.1-1.0 | 0.5 |
| `--prefill-dark` | Seuil pré-remplissage zones noires (px²) | entier ≥ 0 | 500 |
| `--legend` | Position de la légende | bottom/separate | bottom |

### Exemples pratiques :

```powershell
# Coloriage facile avec 10 couleurs
python main.py input/chat.png -d 2 -c 10

# Coloriage expert avec lettres
python main.py input/paysage.jpg -d 9 -s letters -m 30

# Haute résolution pour des traits plus fins
python main.py input/portrait.jpg -r 2500 -d 5

# Très haute qualité (traits ultra-fins)
python main.py input/image.png -r 3500 -d 7

# Symboles plus petits (discrets)
python main.py input/image.jpg --symbol-size 0.3

# Symboles plus grands (lisibles)
python main.py input/image.jpg --symbol-size 0.7

# Ignorer les petites régions (moins de clutter)
python main.py input/image.jpg -m 100 -d 5

# Ignorer les très petites régions (image complexe, moins de cafouillage)
python main.py input/image.jpg -m 200 -r 2200

# Pré-remplir les petites zones noires (< 500px²)
python main.py input/image.jpg --prefill-dark 500

# Pré-remplir les zones noires moyennes (< 1000px²)
python main.py input/image.jpg --prefill-dark 1000

# Désactiver le pré-remplissage
python main.py input/image.jpg --prefill-dark 0

# Légende sur page séparée
python main.py input/fleur.png --legend separate -o ma_fleur
```

## 🔧 Utilisation via script Python

```python
from main import MysteryColoringGenerator
from config import Config

# Configuration personnalisée
config = Config()
config.num_colors = 12
config.difficulty_level = 7
config.symbol_type = "numbers"
config.min_region_area = 80
config.line_thickness_main = 3
config.line_thickness_sub = 1

# Créer le générateur
generator = MysteryColoringGenerator(config)

# Générer le coloriage
coloring_path, legend_path = generator.generate(
    "input/mon_image.jpg",
    output_name="mon_coloriage"
)

print(f"Coloriage créé : {coloring_path}")
```

## 🎯 Guide des niveaux de difficulté

| Niveau | Description | Régions typiques | Usage recommandé |
|--------|-------------|------------------|------------------|
| **1** | Très facile | Aucune subdivision | Jeunes enfants (4-6 ans) |
| **2-3** | Facile | 50-100 régions | Enfants (7-9 ans) |
| **4-6** | Moyen | 200-500 régions | Adolescents/adultes débutants |
| **7-8** | Difficile | 500-1000 régions | Adultes expérimentés |
| **9-10** | Expert | 1000+ régions | Passionnés extrêmes |

## 📁 Structure du projet

```
colo_generation/
├── main.py                    # Point d'entrée principal
├── config.py                  # Configuration centralisée
├── color_quantization.py      # Module 1: K-Means clustering
├── voronoi_tessellation.py    # Module 2: Tessellation Voronoï
├── symbol_placement.py        # Module 3: Placement des symboles
├── renderer.py                # Module 4 & 5: Rendu et légende
├── requirements.txt           # Dépendances Python
├── README.md                  # Documentation
├── input/                     # Dossier pour images sources
└── output/                    # Dossier pour coloriages générés
```

## 🧪 Pipeline technique

### Module 1 : Quantification des couleurs (K-Means)
1. Charge l'image source
2. Applique K-Means clustering pour réduire à N couleurs dominantes
3. Crée une matrice d'indexation (chaque pixel = ID couleur 0 à N-1)
4. Nettoie les artefacts d'anti-aliasing avec opérations morphologiques

### Module 2 : Tessellation de Voronoï
1. Isole chaque zone de couleur
2. Génère des points aléatoires proportionnels à la difficulté et la surface
   - Formule : `nb_points = (surface_px² / 1000) × (difficulté / 5) × 5`
3. Applique un diagramme de Voronoï
4. Clippe les cellules avec le masque original (contrainte)

### Module 3 : Placement des symboles
1. Pour chaque sous-région, calcule le **Pole of Inaccessibility**
   - Point le plus éloigné de tous les bords (meilleur que centroïde)
   - Utilise `cv2.distanceTransform` et `cv2.pointPolygonTest`
2. Détermine la taille du symbole proportionnellement au rayon inscrit
3. Filtre les régions trop petites (< `min_region_area`)

### Module 4 & 5 : Rendu graphique
1. Dessine les contours noirs (épaisseur variable)
2. Place les symboles aux positions optimales avec PIL/ImageDraw
3. Génère la légende (couleur ↔ symbole)
4. Export en PNG haute résolution

## 🎨 Conseils pour de meilleurs résultats

### Préparation de l'image source :
- ✅ Utiliser des images avec **zones de couleurs distinctes**
- ✅ Préférer des images **simplifiées** ou stylisées (dessins, illustrations)
- ✅ Résolution recommandée : **1000-2000px** de largeur
- ❌ Éviter les photos très détaillées ou avec dégradés complexes

### Ajustement de la résolution et des traits :
- **Résolution standard** (1400px) : Bon équilibre vitesse/qualité
- **Haute résolution** (2000-2500px) : Traits plus fins et précis, recommandé pour impression
- **Très haute résolution** (3000-3500px) : Traits ultra-fins, qualité maximale, mais plus lent
- **Note** : Une résolution plus élevée permet d'obtenir des traits noirs plus fins et plus nets

**Exemples :**
```powershell
# Standard - traits normaux
python main.py image.jpg -r 1400

# Haute qualité - traits fins
python main.py image.jpg -r 2500

# Qualité maximale - traits ultra-fins
python main.py image.jpg -r 3500
```

### Ajustement des paramètres :
- **Peu de couleurs** (5-10) → coloriages plus simples et rapides
- **Difficulté basse** (1-3) → grandes zones, idéal enfants
- **Difficulté haute** (7-10) → effet puzzle, très long à colorier
- **Min area élevé** (100-200) → ignore les petites régions, réduit l'encombrement
- **Min area faible** (20-50) → inclut même les petites régions, plus de détail
- **Prefill-dark** (500-1000) → pré-remplit les petites zones noires automatiquement, évite de colorier des zones fastidieuses

## 🐛 Dépannage

### Erreur "Could not load image"
- Vérifiez que le chemin est correct
- Formats supportés : JPG, PNG, BMP, TIFF

### Police de caractères non trouvée
- Le script utilise Arial par défaut (Windows)
- Modifiez `config.font_name` pour spécifier une autre police

### Régions trop petites/symboles illisibles
- Augmentez `min_region_area` (ex: 100 ou 200)
- Réduisez la difficulté
- Augmentez `font_size_ratio` (défaut: 0.5)

### Temps de génération très long
- Réduisez la résolution de l'image source
- Diminuez le nombre de couleurs
- Baissez le niveau de difficulté

## 📊 Configuration avancée

Modifiez [config.py](config.py) pour accéder à tous les paramètres :

```python
class Config:
    # Quantification
    num_colors = 16                    # Nombre de couleurs K-Means
    
    # Difficulté
    difficulty_level = 5               # 1-10
    points_per_1000px2 = 5.0          # Densité des points Voronoï
    
    # Symboles
    min_region_area = 50              # Aire minimale (px²)
    symbol_type = "numbers"           # numbers/letters/custom
    custom_symbols = ["@", "#", "★"]  # Liste personnalisée
    
    # Rendu
    line_thickness_main = 3           # Contours principaux
    line_thickness_sub = 1            # Subdivisions Voronoï
    font_name = "arial.ttf"           # Police
    font_size_ratio = 0.5             # Ratio taille symbole
    
    # Export
    output_format = "A4"              # Format page
    dpi = 300                         # Résolution
    legend_position = "bottom"        # bottom/separate
    legend_columns = 8                # Colonnes légende
```

## 🤝 Contribution

Ce projet est conçu selon le cahier des charges fourni. Pour toute amélioration :
1. Testez vos modifications
2. Documentez les changements
3. Assurez la compatibilité avec tous les modules

## 📄 Licence

Projet personnel - Usage libre pour applications non commerciales.

## 📧 Support

Pour questions ou problèmes :
1. Vérifiez d'abord la section Dépannage
2. Consultez les exemples d'utilisation
3. Testez avec une image simple avant complexe

---

**Créé avec ❤️ par le Générateur de Coloriage Mystère v1.0**

*Transformez vos images en heures de plaisir créatif !* 🎨✨
