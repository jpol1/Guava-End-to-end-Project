# 🍇 Image Classification of Guava Fruits

Author: **Jakub Połęć**  
GitHub: [![GitHub](https://img.shields.io/badge/GitHub-%2312100E.svg?&style=flat-square&logo=github&logoColor=white)](https://github.com/jpol1)  
LinkedIn: https://www.linkedin.com/in/jakub-po%C5%82e%C4%87-1955a0317/

---

## 🔍 Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Technologies Used](#technologies-used)
4. [Key Features](#key-features)
5. [How to Get Started](#how-to-get-started)
6. [How to Use](#how-to-use)
7. [Test Coverage](#test-coverage)
8. [Project Structure](#project-structure)
9. [Report](#report)
10. [Final Notes](#final-notes)
11. [License](#license)
12. [Acknowledgments](#acknowledgments)

---

## 1. 🌎 Introduction

This project is a final AI engineering solution focused on **image classification of guava fruits** to identify diseases affecting them. The goal is to automatically classify guava fruit images into categories such as:

- **Anthracnose**
- **Fruit Fly**
- **Healthy Guava**

This model is designed to:

🎨 Automate **disease classification** using deep learning techniques.  
📊 Demonstrate a complete machine learning pipeline.

The pipeline includes:
- Data preprocessing
- Model training
- API deployment
- Test coverage

---

## 2. 📚 Dataset

The dataset used is sourced from Kaggle: **Guava Disease Dataset**.  

License: [Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

### Categories:
- Anthracnose
- Fruit Fly
- Healthy Guava

The dataset is split into **training** and **test** sets to evaluate model performance.

---

## 3. 📊 Technologies Used

This project uses a combination of powerful tools and libraries:

- **Programming Language**: Python 3.11
- **Deep Learning Framework**: TensorFlow & Keras
- **API Development**: FastAPI
- **Testing**: Pytest
- **Model Tracking**: MLflow
- **Visualization**: Matplotlib & Pandas

---

## 4. 🛠️ Key Features

This project offers:

- **Preprocessing**: Automated loading, resizing, and normalization of images.
- **Model Creation**: Customizable neural network architectures.
- **API Integration**: Real-time predictions via FastAPI.
- **Model Tracking**: Monitor training metrics with MLflow.
- **Visualization**: Loss/accuracy charts, class distributions.
- **Testing**: Comprehensive Pytest coverage.
- **Reusable Components**: Modular and scalable structure.

---

## 5. 🚀 How to Get Started

Follow these steps to set up and run the project:

### 5.1 Prerequisites

Ensure you have the following installed:
- Python 3.11
- Git
- pip (Python package manager)
- Virtual Environment tool (e.g., `venv`)

### 5.2 Clone the Repository

```bash
git clone https://github.com/CodecoolGlobal/AIEngineerFinalProject-python-jpol1.git
cd AIEngineerFinalProject-python-jpol1
```

### 5.3 Set Up the Environment

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 5.4 Configure Environment Variables

Edit a `.env` file in the root directory with the necessary environment variables.
Important change for mlflow to work correctly on the selected host :
   ```bash
    SET_TRACKING_URI="<your_tracking_uri>"
   ```

### 5.5 Run the Project

1. **Start the MLflow Server**:

    ```bash
    mlflow server --host localhost --port 5000
    ```

2. **Launch the FastAPI Server**:

    Do not change location and write in terminal:

    ```bash
    uvicorn fast_api_model:app --reload
    ```

3. **Serve the HTML Page**:

    ```bash
    python -m http.server 3000
    ```

---

## 6. 🗃️ How to Use

### 6.1 Testing the Model

1. Open the `model_page.html` file in the `templates` folder.
2. Upload a test image from the following directory:

```
attrib/GuavaDiseaseDataset/test_files/
```

3. The model will classify the uploaded image into one of the categories:
   - Anthracnose
   - Fruit Fly
   - Healthy Guava

### 6.2 Retrain the Model (Optional)

To retrain the model using the dataset:

```bash
python main.py
```

The updated model will be saved in the `trained_models` folder.

---

## 7. 🔒 Test Coverage

The project ensures reliable code with **89% test coverage**:

```plaintext
Name                                                  Stmts   Miss  Cover
-------------------------------------------------------------------------
modules\creating_model\model_callbacks.py                 7      0   100%
modules\creating_model\model_skeleton.py                 24      1    96%
modules\data_preprocessing\data_visualization.py         31      0   100%
... more rows ...
-------------------------------------------------------------------------
TOTAL                                                   170     18    89%
```

Run tests locally using:

```bash
pytest tests/
```

---

## 8. 📁 Project Structure

```plaintext
image-classification-guava/
├── attrib/                      # Dataset and test images
│   └── GuavaDiseaseDataset/
│       ├── Anthracnose/
│       ├── fruit_fly/
│       ├── healthy_guava/
│       └── test_files/
├── charts/                      # Training charts
├── mlruns/                      # MLflow tracking directory
├── modules/                     # Core modules
│   ├── creating_model/          # Model architecture
│   ├── data_preprocessing/      # Data preprocessing scripts
│   ├── folder_management/       # Folder management utilities
│   ├── model_report/            # Generate model reports
│   └── tracking_models/         # MLflow integration
├── reports/                     # Images used in Report.ipynb
├── templates/                   # HTML templates
├── tests/                       # Unit tests
├── trained_models/              # Saved models
├── fast_api_model.py            # Turn on FastAPI Model
├── main.py                      # Script to train models
├── README.md                    # Project documentation
├── Report.ipynb                 # Report 
└── requirements.txt             # Required Python libraries
```

---

## 9. 🔄 Report

A detailed project report is available in:

```plaintext
Report.ipynb
```

This report includes:
- Model performance metrics
- Confusion matrices
- Training and validation accuracy/loss
- Visualizations of class distributions

---

## 10. ✅ Final Notes

This project showcases a complete pipeline for **image classification** in agriculture.  
It combines **deep learning**, **deployment**, and **testing** to address real-world problems.  

Feel free to contribute, explore, or retrain the model to improve its performance! 🚀

---

## 11. 📃 License

This project is licensed under the **Attribution 4.0 International (CC BY 4.0)** license. You are free to share, copy, and adapt the work with proper credit.

---

## 12. 💕 Acknowledgments

Special thanks to:

- **Kaggle** for the Guava Disease Dataset.
- The open-source community for amazing tools and libraries.

---

## 🙋‍♂️ Invitation for Feedback

I invite everyone to test the project, explore its features, and provide suggestions or improvements! 

Feel free to open an **issue** or **pull request** on GitHub. Your feedback is greatly appreciated! 

Let's work together to make this project better! 🚀
