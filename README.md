# 🦀 Regression with Artificial Neural Networks (ANN) – Crab Age Prediction

This project applies Artificial Neural Networks (ANNs) to predict the age of crabs using biological measurements such as length, diameter, weight, and morphological features. The goal is to understand how neural networks can model nonlinear relationships in biological datasets and generate accurate age predictions.

# 📂 Dataset

The project uses the Crab Age Prediction Dataset available on Kaggle.

# Dataset Includes:

Sex (M / F / I)

Length, Diameter, Height

Whole weight, Shucked weight, Viscera weight, Shell weight

Target: Age (rings → converted to years)

# 🚀 Project Workflow
1️⃣ Data Understanding

1. Loaded dataset and inspected first few rows

2. Identified input features and target variable (Crab Age)

3. Checked datatypes of numerical & categorical variables

4. Handled missing values and removed duplicates

5. Converted rings into actual age (rings + 1.5 years)

2️⃣ Exploratory Data Analysis (EDA)

1. Distribution plots for age and all continuous features

2. Species-wise comparisons using Sex column (M/F/I)

3. Feature vs Age relationships

4. Boxplots to detect outliers

5. Scatterplots to observe biological growth patterns

6. Correlation heatmap to identify important predictors

3️⃣ Data Preprocessing

1. Label encoding for categorical feature Sex

2. Scaling numerical features using StandardScaler

3. Split dataset into training & testing sets

4. Checked for data imbalance and feature variance

4️⃣ ANN Model Building (Baseline Model)

Built a baseline ANN with:

1. Input layer

2. One hidden layer

3. Output neuron for regression

4. Trained model using MSE loss

5. Evaluated using MSE, MAE, RMSE, R² Score

5️⃣ Model Optimization & Tuning

Experimented with:

1. Additional hidden layers & neurons

2. ReLU activation tuning

3. Dropout layers for regularization

4. Adam optimizer with adjusted learning rates

5. Batch size & epoch variations

6. EarlyStopping to reduce overfitting

6️⃣ Model Evaluation

1. Compared Training vs Validation performance

2. Visualized loss curves

3. Predicted vs Actual Age scatter comparison

4. Highlighted underfitting/overfitting patterns

5. Identified most influential biological features

# 📊 Results & Insights

1. ANN was effective in predicting crab age with improved accuracy after tuning

2. Scaling & label encoding were essential for stable learning

3. Dropout + EarlyStopping significantly reduced overfitting

4. Feature analysis showed that shell weight, whole weight, and length were highly predictive

5. Biological measurements showed clear relationship trends with age

# 📝 Deliverables

Jupyter Notebook:
Full workflow including EDA → Preprocessing → ANN Modeling → Optimization → Evaluation

Short Summary Report:
Key insights from EDA, model performance comparison, and challenges resolved

# 🔑 Key Learnings

Working with biological regression datasets

Building ANN models for continuous output prediction

Importance of preprocessing in neural network training

Understanding feature–age biological relationships

Applying performance metrics like MAE, RMSE & R²

# 🛠️ Tech Stack

1. Python

2. TensorFlow / Keras

3. Pandas, NumPy

4. Matplotlib, Seaborn

5. Scikit-learn (preprocessing & metrics)

# 👤 Author

Sowjanya U
Data Science & Machine Learning Enthusiast

📧 Email: usowjanyakiran@gmail.com

🌐 GitHub: https://github.com/SowjanyaKiran/
