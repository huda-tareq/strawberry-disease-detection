# 🍓 Strawberry Disease Detection

AI-based system to detect strawberry plant diseases from images and suggest treatments.

## 📌 Project Idea

This project uses Machine Learning to classify strawberry diseases based on image features such as color, texture, and edges.

## 🧠 Model Approach

* Feature Extraction:

  * RGB & HSV color features
  * LBP (texture)
  * Edge detection
  * Color histogram
  * White pixel ratio (for powdery mildew)
* Model: Random Forest Classifier
* Data preprocessing and scaling applied

## 🦠 Supported Classes

* Healthy
* Leaf Spot
* Gray Mold
* Blossom Blight
* Anthracnose Fruit Rot
* Angular Leafspot
* Powdery Mildew Fruit
* Powdery Mildew Leaf

## 💊 Output

* Predicted disease
* Suggested treatment

## 🚀 How to Run

### 1. Install requirements

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

## 📂 Project Structure

```
├── app.py
├── model.pkl
├── scaler.pkl
├── label_map.pkl
├── notebook.ipynb
├── requirements.txt
└── README.md
```

## 📊 Example

Upload an image of a strawberry leaf or fruit and the system will predict the disease and show treatment.

## 👩‍💻 Author

Huda Tareq

## 📎 Notes

This project is built as a graduation project using Machine Learning and Computer Vision techniques.

#dataset from here:
[https://universe.roboflow.com/strawberry-disease-detection-tzjua/strawberry-disease-detection-dataset-b2y4y/dataset/1]
[https://universe.roboflow.com/thesis-okplj/strawberry-healthy/dataset/1]


