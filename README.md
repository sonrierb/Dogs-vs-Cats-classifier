# Cat vs Dog Image Classifier

A deep learning CNN model trained to classify **cats** and **dogs**, with:

- ✔️ Streamlit UI
- ✔️ FastAPI backend
- ✔️ TensorFlow CNN model
- ✔️ Dataset folder structure (train, test, validation)

---

## Project Structure

cat-dog-classifier/
│
├── model.py
├── image.py
├── main.py
├── app.py
├── requirements.txt
├── README.md
│
├── saved_model/
│ └── model.h5
│
└── data/
├── train/cat/
├── train/dog/
├── test/cat/
├── test/dog/
├── validation/cat/
└── validation/dog/


---
## Installation
1. Clone the repo:
   ```bash
   git clone https://github.com/sonrierb/Dogs-vs-Cats-classifier.git
   cd Dogs-vs-Cats-classifier

---

##  Install Dependencies
pip install -r requirements.txt

---

## Run Streamlit App
streamlit run app.py

---

## Run FastAPI Server
uvicorn main:app --reload


API Endpoint:
POST /predict
Make sure FastAPI server is running before using Streamlit.

---

## Model Training
Training code is inside `model.py`.  
It loads images from:
/data/train/

Change path if needed.

---

## Contribute
Feel free to improve the UI or backend!

---

## Contact
Created by Muskan (GenAI Engineer)






