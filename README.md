# First we are create a virtual envoriment .
    Create a virtual envoriment simply follow two steps:
    1. Create a Envoriment :-
    
    Windows: python -m venv .venv
    macOS/Linux: python3 -m venv .venv
    
    2.Activate the envoriment  
    
    Windows: .venv\Scripts\activate
    macOS/Linux: source .venv/bin/activate
### Install requirments.txt file 

    pip install -r requirements.txt

## 📁 Project Structure



## Project Structure

```
ode_solver_project/
│
├── main.py                        # Entry point — run this
│
├── config/
│   ├── __init__.py
│   └── settings.py                # Global constants: paths, step size, threshold
│
├── src/
│   └── ode_solver/
│       ├── __init__.py
│       ├── problems.py            # 60+ ODE problem definitions
│       ├── methods.py             # RK step functions (Euler, Midpoint, Heun, RK4)
│       └── solver.py              # Core solver logic + DataFrame utilities
│
├── tests/
│   ├── __init__.py
│   ├── test_methods.py            # Unit tests for RK step functions
│   └── test_solver.py             # Unit tests for solver utilities
│
├── data/                          # Auto-generated CSV files (git-ignored)
│   ├── final_ode_Problem2.csv
│   └── final_ode_Problem_results2.csv
│
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py
```

This will:
1. Build the input CSV in `data/`
2. Run all four RK methods on every ODE problem
3. Select the best method per problem (accuracy + speed)
4. Save results to `data/final_ode_Problem_results2.csv`
5. Print a summary of which method won most often


# FNN-Based Selection of Optimal Runge–Kutta Method

## 📌 Overview
This project aims to automatically select the best Runge–Kutta (RK) method for solving first-order Initial Value Problems (IVPs) using a Feedforward Neural Network (FNN).

The objective is to balance:
- Accuracy (error)
- Computational efficiency (time)

---

## ⚙️ Dataset

- Total samples: **7390**

### Best Method Distribution:
| Method    | Count |
|----------|------|
| RK4      | 3001 |
| HEUN     | 1786 |
| EULER    | 1684 |
| MIDPOINT |  919 |

👉 The dataset is **imbalanced**, with RK4 dominating.

---

## 💡 Approach

1. Solve ODE using multiple RK methods:
   - Euler
   - Heun
   - Midpoint
   - RK4

2. Use:
   - Exact solution (if available)
   - Otherwise RK10 as reference

3. Compute:
   - Error
   - Time

4. Select best method

5. Train a Feedforward Neural Network (FNN)

---

## 🧠 Model Details

- Architecture: Fully Connected Neural Network
- Activation:
  - ReLU (hidden layers)
  - Softmax (output layer)
- Loss Function:
  - Sparse Categorical Crossentropy
- Optimizer:
  - Adam

---

## 📊 Results

### ✅ Test Accuracy
**0.8187**

---

### 🔍 Random Prediction Example

Predicted : RK4 (99.99% confidence)
Actual : RK4
Match : ✅ Correct
### 📊 Class-wise Probabilities

EULER 0.0001
HEUN 0.0000
MIDPOINT 0.0000
RK4 0.9999
---


---

### 📉 Confusion Matrix

| Actual \ Predicted | Euler | Heun | Midpoint | RK4 |
|------------------|------|------|----------|-----|
| Euler            | 223  | 90   | 9        | 15  |
| Heun             | 28   | 304  | 18       | 8   |
| Midpoint         | 2    | 52   | 129      | 0   |
| RK4              | 21   | 25   | 3        | 551 |

---

### 📈 Classification Report

| Class     | Precision | Recall | F1-score |
|----------|----------|--------|----------|
| Euler    | 0.81     | 0.66   | 0.73     |
| Heun     | 0.65     | 0.85   | 0.73     |
| Midpoint | 0.81     | 0.70   | 0.75     |
| RK4      | 0.96     | 0.92   | 0.94     |

- 
- **Weighted Avg F1**: 0.82  

---

## ⚠️ Imbalanced Data Issue

The dataset is imbalanced:
- RK4 has significantly more samples than others

### 🔴 Problem:
- Accuracy may be misleading
- Model can bias towards RK4

---

## ✅ Solution

To handle imbalance:

### ✔️ 1. Class Weights
Assign higher importance to minority classes during training.

### ✔️ 2. SMOTE (Used)
Generate synthetic samples for minority classes.

### ✔️ 3. Better Evaluation Metrics
Instead of relying only on accuracy, use:
- **Weighted F1-score**
- **Macro F1-score**

👉 These give a more realistic performance measure.

---

## 🔥 Key Insights

- RK4 performs best overall due to higher accuracy
- Heun has good recall but lower precision
- Model successfully learns method selection patterns
- Weighted metrics provide better evaluation than accuracy

---


---

## 🔮 Future Work

- Extend to systems of ODEs
- Include adaptive RK methods
- Improve model architecture
- Apply to real-world datasets

---

## 👨‍🎓 Author

Priyanshu Gupta  
M.Sc. Mathematics, IIT Gandhinagar

---

## Acknowledgement

Supervisor: Dr. Abhinav Jha (IIT GANDHINAGAR)