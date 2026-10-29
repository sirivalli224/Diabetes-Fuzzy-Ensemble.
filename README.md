Diabetes Prediction Using Fuzzy Ensemble Learning

📌 Project Overview

This project implements a diabetes prediction system using Machine Learning and Fuzzy Logic. It combines predictions from multiple machine learning models using a fuzzy logic-based fusion approach to classify individuals as diabetic or non-diabetic.

The project is developed as a Soft Computing Case Study inspired by research on integrating machine learning and fuzzy logic for diabetes prediction.

🎯 Objectives

- Predict diabetes using patient health and symptom-related information.
- Apply multiple machine learning algorithms.
- Combine model predictions using fuzzy logic.
- Evaluate model performance using standard classification metrics.
- Visualize and analyze the prediction results.

🧠 Algorithms Used

The project uses the following machine learning models:

1. Support Vector Machine (SVM)
2. Artificial Neural Network (ANN) using a Multi-Layer Perceptron (MLP)
3. Random Forest
4. Gradient Boosting

A fuzzy logic-based fusion system combines the models' predicted probabilities to generate the final classification.

📊 Dataset

Dataset: UCI Early Stage Diabetes Risk Prediction Dataset

- Number of instances: 520
- Number of features: 16
- Target classes: Positive and Negative diabetes risk

Dataset source: "UCI Machine Learning Repository" (https://archive.ics.uci.edu/dataset/529/early)

⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Scikit-fuzzy
- Matplotlib
- Jupyter Notebook / Google Colab

🔄 Project Workflow

1. Data Collection: Load the diabetes dataset from the UCI Machine Learning Repository.
2. Data Preprocessing: Handle categorical features, encode variables, and scale numerical features.
3. Data Splitting: Divide the dataset into training and testing sets.
4. Model Training: Train SVM, ANN, Random Forest, and Gradient Boosting models.
5. Fuzzy Logic Integration: Combine model prediction probabilities using a fuzzy fusion system.
6. Model Evaluation: Calculate accuracy, precision, recall, F1-score, and confusion matrix.
7. Visualization: Compare model performance and visualize classification results.

📁 Project Structure

Diabetes-Fuzzy-Ensemble/
│
├── Diabetes_Fuzzy_Ensemble_Case_Study.ipynb
├── Diabetes_Fuzzy_Ensemble_Case_Study.py
└── README.md

🚀 How to Run the Project

Option 1: Google Colab

1. Download the ".ipynb" notebook from this repository.
2. Open "Google Colab" (https://colab.research.google.com/).
3. Upload the notebook.
4. Run the cells sequentially or select Runtime → Run all.
5. Allow the required libraries to install and the dataset to load.

Option 2: Run Locally

Step 1: Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL

Step 2: Navigate to the project folder

cd Diabetes-Fuzzy-Ensemble

Step 3: Install the required libraries

pip install pandas numpy scikit-learn scikit-fuzzy matplotlib ucimlrepo jupyter

Step 4: Launch Jupyter Notebook

jupyter notebook

Open "Diabetes_Fuzzy_Ensemble_Case_Study.ipynb" and run the cells.

📈 Evaluation Metrics

The system evaluates model performance using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The actual performance values are generated when the notebook is executed.

📚 Research Reference

Kathiria et al. (2026). Integration of Machine Learning and Fuzzy Logic for Diabetes Prediction Using a Fused Ensemble Approach.

Published in: Discover Artificial Intelligence.

Research paper: https://link.springer.com/article/10.1007/s44163-026-01648-8

Note: This project is an educational implementation inspired by the research paper. It does not claim to reproduce the paper's exact implementation or reported performance.

🔮 Future Enhancements

- Improve fuzzy membership functions and inference rules.
- Perform hyperparameter tuning for individual models.
- Add cross-validation and more robust evaluation.
- Develop a web interface for interactive predictions.
- Explore explainable AI techniques for understanding model decisions.

⚠️ Disclaimer

This project is intended for educational and research purposes only. Its predictions must not be used as a substitute for professional medical advice, diagnosis, or treatment.

👩‍💻 Author

Developed as part of a Soft Computing Case Study on diabetes prediction using machine learning and fuzzy logic.
