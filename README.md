# ☕ Coffee Plant Nutritional Deficiencies Classification using Transfer Learning

This project applies **deep learning** and **transfer learning** techniques to detect and classify **nutritional deficiencies** in coffee plant leaves using image data. The goal is to support **smallholder farmers** with early detection tools to optimize crop nutrition and improve coffee yield and quality.

---

## 📌 Project Overview

Nutritional deficiencies in coffee plants often go unnoticed, overshadowed by more visible pests and diseases. However, lack of nutrients like **Boron, Calcium, Iron, Magnesium**, and others significantly affects productivity. Using **Convolutional Neural Networks (CNNs)** and pre-trained models, this project builds an automated image classification system.

### 🔍 Objective

- Identify 10 categories of nutritional deficiency (including **Healthy** class).
- Use transfer learning to overcome the challenge of limited labeled data.
- Compare performance across five CNN architectures.

---

## 📊 Dataset

- **CoLeaf Dataset** from [Mendeley Data](https://data.mendeley.com/datasets/brfgw46wzb/2)
- **1,290 images** across 10 classes:
  - Nutrients: B, Ca, Fe, Mg, Mn, N, P, K, Other
  - Healthy Leaves

---

## 🧠 Models Used

| Model         | Accuracy (Test Set) |
|---------------|---------------------|
| ResNet-50     | **94.23%** ⭐        |
| VGG-16        | 94.05%              |
| DenseNet-121  | 91.06%              |
| MobileNet-V2  | 85.29%              |
| Inception-V3  | 83.24%              |

**Best model**: 🏆 **ResNet-50**, fine-tuned to achieve 99.83% training accuracy.

---

## 🧪 Techniques

- **Transfer Learning**: VGG-16, ResNet-50, DenseNet-121, MobileNet-V2, Inception-V3
- **Data Augmentation**: Rotation, Shearing, Zooming, Flipping, Brightness Adjustment
- **Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

---

## 📈 Results

- Fine-tuning significantly improved generalization.
- Most deficiencies classified with precision >90%.
- **Healthy**, **Potassium**, and **Phosphorus** classes had the highest precision and recall.
- Imbalance in class samples posed minor challenges.

---

## 🔧 Tools & Libraries

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib / Seaborn
- Jupyter Notebook

---

## 📂 Structure

📁 data/           # Coffee leaf images
📁 notebooks/      # Model training and evaluation
📁 models/         # Saved model weights
📁 results/        # Accuracy, loss plots, confusion matrices
README.md          # You're here!


---

## 💡 Future Work

- Address dataset imbalance with advanced augmentation or sampling.
- Deploy as a **mobile app** for field use.
- Explore explainable AI (XAI) techniques for visualizing model focus areas (e.g., Grad-CAM).

---

## 🧑‍💻 Author

**Reyna Vargas Antonio**  
MSc in Data Analytics, National College of Ireland  
**Supervisor**: Paul Stynes
