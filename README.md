# 📊 Logistic Regression from Scratch in Python

This project demonstrates a full implementation of **Logistic Regression** from scratch using only **NumPy**, without relying on machine learning libraries like Scikit-learn for model creation. We then test our model using a real-world dataset: the Breast Cancer dataset.

---

## 📂 Files Structure

- **`LogisticModel.py`**
  - Custom `LogisticRegression` class implementation.
  - Contains `fit()` and `predict()` methods.

- **`logisticmodel.py`**
  - Loads dataset.
  - Splits into training and testing sets.
  - Trains and tests the Logistic Regression model.
  - Calculates and prints accuracy.

- **`main.py`**
  - (Can be used for running or extending the project.)

---

## 🔄 Project Flow

### 1. Dataset
- We use the **Breast Cancer** dataset from Scikit-learn:
  - `data` contains 30 numerical features.
  - `target` contains binary labels (malignant = 0, benign = 1).

### 2. Model Training
- The model is initialized with:
  - `alpha` (learning rate) = 0.001
  - `iter` (number of iterations) = 1000

- During training (`fit` method):
  - Computes linear predictions.
  - Applies **Sigmoid function** to map outputs between 0 and 1.
  - Uses **Gradient Descent** to update weights and bias.

### 3. Prediction
- After training, predictions are made on the test set.
- Probabilities above 0.5 are classified as **1**, otherwise **0**.

### 4. Accuracy Calculation
- Calculates the ratio of correct predictions over total samples.

```python
accuracy = np.sum(predicted == test) / len(test)
```

---

## 📊 Example Output

```bash
0.9473684210526315
```

(Meaning ~94.7% accuracy on the Breast Cancer test dataset.)

---

## 💡 Requirements

- Python 3.x
- Libraries:
  - `numpy`
  - `scikit-learn`

Install the required libraries with:

```bash
pip install numpy scikit-learn
```

---

## 🚀 How to Run

```bash
python logisticmodel.py
```

---

## 📘 Notes

- **Why Logistic Regression?**
  - Best suited for binary classification problems.
  - Outputs probabilities.

- **Sigmoid Function:**
  - Maps any real value to (0, 1), perfect for classification.

- **Gradient Descent:**
  - Optimization technique to minimize the cost function iteratively.

---

## 📈 Future Improvements

- Add regularization (L1 or L2).
- Implement multiclass logistic regression.
- Plot decision boundaries.

---

> Built from scratch to fully understand how Logistic Regression works under the hood. 🚀
