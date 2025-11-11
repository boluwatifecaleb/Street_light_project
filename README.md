💡 Streetlight Status Predictor: A Machine Learning Deployment Project This repository hosts a machine learning application designed to predict the optimal operational state of a smart streetlight. The core objective is to move a pre-trained model from development (Google Colab) to a live, interactive web application using Streamlit and GitHub.

🚀 Overview The project uses a trained Decision Tree Classifier model to categorize the required streetlight status as either On, Dim, or Off. This is a functional demonstration of a complete MLOps (Machine Learning Operations) pipeline, covering data preprocessing, model training, hyperparameter tuning, and containerized deployment.

✨ Key Features Prediction Model: Utilizes a scikit-learn Pipeline featuring a ColumnTransformer for robust preprocessing of mixed data types (categorical and numerical).

Data Types Handled:

Categorical Features: Time, Weather, Battery, Motion, and Traffic are handled using OneHotEncoder.

Numerical Features: AmbientLight and SolarOutput are scaled using StandardScaler.

Deployment: The Python application is containerized and hosted live via Streamlit Community Cloud, using this GitHub repository as the source.

Model Diagnostics: The Streamlit app includes a performance dashboard displaying the Confusion Matrix and Feature Importances for transparency and validation.
