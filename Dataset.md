# 🌿 New Plant Diseases Dataset

> **Source:** [Kaggle – vipoooool/new-plant-diseases-dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

---

## 📌 Overview

This dataset is a **recreated version** of the original PlantVillage Dataset using offline augmentation techniques. It is widely used for training deep learning models to detect and classify plant leaf diseases.

| Property | Details |
|---|---|
| **Created by** | vipoooool |
| **Source** | Kaggle |
| **License** | GPL 2 |
| **Total Images** | ~87,000 (RGB) |
| **Classes** | 38 (disease + healthy) |
| **Plants Covered** | 14 unique plant species |
| **Train Split** | 80% (~70,295 images) |
| **Validation Split** | 20% (~17,572 images) |

---

## 🌱 Supported Plant Species

| # | Plant |
|---|-------|
| 1 | Apple |
| 2 | Blueberry |
| 3 | Cherry (including sour) |
| 4 | Corn (Maize) |
| 5 | Grape |
| 6 | Orange |
| 7 | Peach |
| 8 | Bell Pepper |
| 9 | Potato |
| 10 | Raspberry |
| 11 | Soybean |
| 12 | Squash |
| 13 | Strawberry |
| 14 | Tomato |

---

## 🦠 Disease Classes (38 Total)

### 🍎 Apple
| Class | Label |
|---|---|
| Apple Scab | `Apple___Apple_scab` |
| Black Rot | `Apple___Black_rot` |
| Cedar Apple Rust | `Apple___Cedar_apple_rust` |
| Healthy | `Apple___healthy` |

### 🫐 Blueberry
| Class | Label |
|---|---|
| Healthy | `Blueberry___healthy` |

### 🍒 Cherry
| Class | Label |
|---|---|
| Powdery Mildew | `Cherry_(including_sour)___Powdery_mildew` |
| Healthy | `Cherry_(including_sour)___healthy` |

### 🌽 Corn (Maize)
| Class | Label |
|---|---|
| Cercospora Leaf Spot / Gray Leaf Spot | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` |
| Common Rust | `Corn_(maize)___Common_rust_` |
| Northern Leaf Blight | `Corn_(maize)___Northern_Leaf_Blight` |
| Healthy | `Corn_(maize)___healthy` |

### 🍇 Grape
| Class | Label |
|---|---|
| Black Rot | `Grape___Black_rot` |
| Esca (Black Measles) | `Grape___Esca_(Black_Measles)` |
| Leaf Blight (Isariopsis Leaf Spot) | `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` |
| Healthy | `Grape___healthy` |

### 🍊 Orange
| Class | Label |
|---|---|
| Haunglongbing (Citrus Greening) | `Orange___Haunglongbing_(Citrus_greening)` |

### 🍑 Peach
| Class | Label |
|---|---|
| Bacterial Spot | `Peach___Bacterial_spot` |
| Healthy | `Peach___healthy` |

### 🫑 Bell Pepper
| Class | Label |
|---|---|
| Bacterial Spot | `Pepper,_bell___Bacterial_spot` |
| Healthy | `Pepper,_bell___healthy` |

### 🥔 Potato
| Class | Label |
|---|---|
| Early Blight | `Potato___Early_blight` |
| Late Blight | `Potato___Late_blight` |
| Healthy | `Potato___healthy` |

### 🫐 Raspberry
| Class | Label |
|---|---|
| Healthy | `Raspberry___healthy` |

### 🫘 Soybean
| Class | Label |
|---|---|
| Healthy | `Soybean___healthy` |

### 🎃 Squash
| Class | Label |
|---|---|
| Powdery Mildew | `Squash___Powdery_mildew` |

### 🍓 Strawberry
| Class | Label |
|---|---|
| Leaf Scorch | `Strawberry___Leaf_scorch` |
| Healthy | `Strawberry___healthy` |

### 🍅 Tomato
| Class | Label |
|---|---|
| Bacterial Spot | `Tomato___Bacterial_spot` |
| Early Blight | `Tomato___Early_blight` |
| Late Blight | `Tomato___Late_blight` |
| Leaf Mold | `Tomato___Leaf_Mold` |
| Septoria Leaf Spot | `Tomato___Septoria_leaf_spot` |
| Spider Mites / Two-spotted Spider Mite | `Tomato___Spider_mites Two-spotted_spider_mite` |
| Target Spot | `Tomato___Target_Spot` |
| Tomato Mosaic Virus | `Tomato___Tomato_mosaic_virus` |
| Yellow Leaf Curl Virus | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` |
| Healthy | `Tomato___healthy` |

---

## 📁 Dataset Structure

```
New Plant Diseases Dataset/
│
├── train/                  # Training images (~70,295)
│   ├── Apple___Apple_scab/
│   ├── Apple___Black_rot/
│   ├── Apple___healthy/
│   └── ... (38 folders)
│
└── valid/                  # Validation images (~17,572)
    ├── Apple___Apple_scab/
    ├── Apple___Black_rot/
    ├── Apple___healthy/
    └── ... (38 folders)
```

---

## 🤖 Usage in This Project

This dataset was used to train a **MobileNetV2** model with transfer learning for the PlantCare AI application.

```python
# Class labels used in prediction
class_names = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
    "Apple___healthy", "Blueberry___healthy",
    "Cherry_(including_sour)___healthy", "Cherry_(including_sour)___Powdery_mildew",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy", "Corn_(maize)___Northern_Leaf_Blight",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___healthy", "Potato___Late_blight",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___healthy", "Strawberry___Leaf_scorch",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]
```

---

## 📊 Model Training Details

| Parameter | Value |
|---|---|
| **Model Architecture** | MobileNetV2 (Transfer Learning) |
| **Input Image Size** | 224 × 224 × 3 |
| **Number of Classes** | 38 |
| **Training Accuracy** | ~95% |
| **Preprocessing** | `mobilenet_v2.preprocess_input` |

---

## 📎 Citation

```
@misc{new-plant-diseases-dataset,
  author    = {vipoooool},
  title     = {New Plant Diseases Dataset},
  year      = {2020},
  publisher = {Kaggle},
  url       = {https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset}
}
```

---

## 🔗 Links

- 📦 [Download Dataset on Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- 🌿 [PlantCare AI App](https://Macmacmacmacmacmac-plant-disease-app.hf.space)

---

*Dataset used for educational and research purposes under GPL 2 License.*
