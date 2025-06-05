# 🧬 Breast Cancer Diagnosis: Data Analysis and Prediction

This project applies statistical analysis and machine learning techniques to classify breast tumors as **benign** or **malignant** using diagnostic features. Developed in R, it provides a full pipeline from data cleaning and visualization to predictive modeling.

## 📌 Project Summary

Breast cancer remains one of the most common cancers worldwide. Accurate early diagnosis can significantly improve patient outcomes. This project explores the use of machine learning models like **Support Vector Machines (SVM)** and **K-Nearest Neighbors (KNN)** on a labeled dataset of tumor measurements to automate classification.

The goal is to:
- Preprocess and normalize the data
- Identify significant features using statistical tests
- Train and evaluate machine learning classifiers
- Visualize insights and model performance

---

## 📂 Project Structure

```
.
├── code.R             # Main R script for full analysis
├── code.Rmd           # R Markdown version of the script
├── code.html          # Rendered HTML output of the R Markdown
├── Breast_cancer_data.csv  # Dataset used for training/testing
├── Breast Cancer Diagnosis.pptx  # Project summary slides
├── *.jpg              # Visuals used for reporting
```

---

## ⚙️ How to Run the Project

### ✅ Prerequisites

You’ll need R installed (version 4.0+ recommended), along with the following packages:

```r
install.packages(c("tidyverse", "caret", "e1071", "pROC", "ggpubr", "corrplot"))
```

### ▶️ Running the R Script

1. **Clone this repository** to your local machine:
   ```bash
   git clone https://github.com/your-username/breast-cancer-diagnosis.git
   cd breast-cancer-diagnosis
   ```

2. **Open RStudio** or any R environment.

3. **Run `code.R`**:
   ```r
   source("code.R")
   ```

   This script will:
   - Load and normalize the data
   - Perform ANOVA and visualize key features
   - Train multiple classifiers (SVM, KNN, etc.)
   - Evaluate models using confusion matrices and ROC curves

4. (Optional) You can also open and **knit `code.Rmd`** for a report-style HTML output.

---

## 📊 Dataset Information

The dataset used is a version of the **Breast Cancer Wisconsin Diagnostic Data Set**, containing attributes like:
- Mean radius, texture, perimeter, area
- Concavity, symmetry, fractal dimension
- Diagnosis (B = benign, M = malignant)

---

## 📈 Results

The models are evaluated using:
- Accuracy
- Confusion matrix
- ROC-AUC scores

Preliminary results show high accuracy for SVM and Logistic Regression models in detecting malignant tumors.

---

## 📋 License

This project is for educational and research purposes. Dataset used is publicly available for academic use.

---

## 👩‍💻 Author

**Glizzy**

