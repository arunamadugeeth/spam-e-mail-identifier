# Spam Mail Detection – Machine Learning Practice Project

This project uses **Machine Learning** to classify emails as **Spam** or **Not Spam** based on their message content.  
It’s a **practice project** to learn how text data can be processed, converted into numerical form, and used to train an ML model for classification.

---

## Project Overview
- **Goal:** Detect whether a given email message is spam or not  
- **Algorithm Used:** Logistic Regression  
- **Dataset:** SMS Spam Collection Dataset  
- **Language:** Python  
- **Libraries Used:**  
  - `pandas` – for dataset loading and manipulation  
  - `numpy` – for numerical operations  
  - `scikit-learn` – for model building and evaluation  
  - `TfidfVectorizer` – to convert text into numerical feature vectors  

---

## Steps and Workflow

### 1. Import Libraries
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
```

### 2. Load and Clean Dataset
```python
raw_mail_dataset = pd.read_csv('/content/mail_data.csv')
mail_data = raw_mail_dataset.where((pd.notnull(raw_mail_dataset)), '')
```
Missing values are replaced with empty strings for cleaner data.

---

### 3. Encode Labels
- Spam → `0`  
- Not Spam (ham) → `1`

```python
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1
```

---

### 4. Split Data
Separate messages and labels, then split them into training and test sets:
```python
mails = mail_data['Message']
result = mail_data['Category']
train_mails, test_mails, train_result, test_result = train_test_split(mails, result, test_size=0.2, random_state=3)
```

---

### 5. Text to Numeric Conversion (TF-IDF)
Convert email text into numeric vectors using **TF-IDF Vectorizer**, removing common stop words.
```python
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
x_train_features = feature_extraction.fit_transform(train_mails)
x_test_features = feature_extraction.transform(test_mails)
```

---

### 6. Model Training and Evaluation
Train the model using **Logistic Regression**:
```python
model = LogisticRegression()
model.fit(x_train_features, train_result.astype('int'))
```

Evaluate accuracy:
```python
model_predict_data = model.predict(x_test_features)
print("Accuracy:", accuracy_score(model_predict_data, test_result.astype('int')))
```
**Accuracy:** ~96.6%

---

### 7. Prediction Example
```python
input_mail = ['WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!']
input_features = feature_extraction.transform(input_mail)
prediction = model.predict(input_features)

if prediction[0] == 0:
  print("Spam Mail")
else:
  print("Not Spam Mail")
```

**Output:**
```
Spam Mail
```

---

## What I Learned
- How to handle text data using Pandas  
- Encoding categorical data  
- Using **TF-IDF Vectorizer** to convert text into numerical features  
- Building and testing an ML model using Logistic Regression  
- Creating a simple prediction system  

---

## Future Improvements
- Add a web or GUI interface  
- Try deep learning models (e.g., LSTM, BERT)  
- Use other classifiers like SVM or Naive Bayes  
- Perform better text preprocessing (like stemming and tokenization)

---

### ✅ Author
**Aruna Madugeeth**  
Machine Learning Practice Project | Text Classification | Logistic Regression | Python | NLP
