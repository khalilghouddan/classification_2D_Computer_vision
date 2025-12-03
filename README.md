# 🔮 2D Points Prediction & Hyperparameter Analysis

A machine learning project focused on predicting 2D points and analyzing how different **hyperparameters** impact model performance. The repository includes visualizations, comparison diagrams, and accuracy curves to help understand model behavior.

---

## 🚀 Project Overview

This project develops a predictive model capable of estimating 2D coordinates from training data. Its main objective is to evaluate how tuning key hyperparameters influences accuracy and performance.

You will find:

* A working model for 2D point prediction
* Hyperparameter testing scripts
* Visual results showing prediction quality
* Comparison diagrams
* Accuracy evolution curves

---

## 🎯 Goals

* Build and train a predictive model
* Experiment with multiple hyperparameters
* Visualize the impact of parameter changes
* Compare results using diagrams and accuracy metrics

---

## 🖼 Example Outputs

Include your generated images here:

1. **Predicted Points vs Real Points**

   ![Predictions vs Real](img/placeholder1.png)

2. **Hyperparameter Comparison (Batch Size, Recognition Rate, Iterations)**

   ![Hyperparameter Diagram](img/placeholder2.png)

3. **Accuracy Curve Over Training Epochs**

   ![Accuracy Curve](img/placeholder3.png)

---

## ⚙️ Methodology

### 1️⃣ Data Preparation

* Normalization of coordinates
* Train/Test split

### 2️⃣ Model Architecture

* Dense Neural Network (MLP) or CNN depending on dataset
* Loss Function: MSE or Cross Entropy
* Optimizer: Adam

### 3️⃣ Training Process

Hyperparameters tested:

* **Batch Size**: 8, 16, 32, 64
* **Epochs**: 20 → 200
* **Learning Rate**: 1e-2, 1e-3, 1e-4

### 4️⃣ Evaluation & Visualization

* Performance comparison diagrams
* Accuracy curves
* Predicted vs Real point plots

---

## 📊 Key Results (example)

* Smaller batch sizes often produce more stable training
* High learning rates lead to poor accuracy
* More epochs improve accuracy until reaching a plateau

(Replace with your real results)

---

## 🛠 Installation

```bash
git clone <your-repository-url>
cd prediction-2d
pip install -r requirements.txt
```

---

## ▶️ Running the Project

```bash
python train.py
python evaluate.py
python plot_results.py
```

All generated graphs are saved in:

```
outputs/plots/
```

---

## 🏁 Conclusion

This project offers a practical exploration of how hyperparameters affect the performance of a 2D prediction model. Diagrams and accuracy curves help clearly visualize the impact of each parameter.

---

## 👤 Author

**Khalil Ghouddan**

---

## 📜 License

Choose any license you want (MIT recommended).
