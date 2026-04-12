# ODE Solver Project

This project solves many first-order ordinary differential equations (ODEs) with multiple Runge-Kutta methods, compares their accuracy and runtime, and then trains a neural network to predict which RK method is the best choice for a given ODE.

## What The Project Does

The pipeline has two connected parts:

1. Numerical analysis
   - Builds a dataset of ODE initial value problems
   - Solves each problem with `Euler`, `Midpoint`, `Heun`, and `RK4`
   - Computes error and runtime for every method
   - Selects the best RK method for each problem

2. Machine learning
   - Converts each ODE expression into numeric features
   - Uses the selected best method as the target label
   - Trains a feedforward neural network classifier
   - Reports accuracy, confusion matrix, and classification metrics

## Why This Project Is Useful

Different Runge-Kutta methods have different tradeoffs:

- `Euler` is simple and fast, but less accurate
- `Midpoint` and `Heun` are better than Euler in many cases
- `RK4` is usually the most accurate, but can cost more computation

This project tries to automate method selection instead of always choosing the same solver.

## Project Structure

```text
ode_solver_project/
├── main.py
├── README.md
├── requirements.txt
├── config/
│   ├── __init__.py
│   └── settings.py
├── data/
│   ├── final_ode_Problem.csv
│   ├── final_ode_Problem_results.csv
│   ├── function_value.csv
│   └── model_ode.csv
├── model/
│   ├── encoder.joblib
│   ├── model.joblib
│   └── scaler.joblib
├── src/
│   └── ode_solver/
│       ├── __init__.py
│       ├── generate_dataset.py
│       ├── methods.py
│       ├── model_RK.py
│       ├── problem_generater.py
│       ├── problems.py
│       └── solver.py
└── tests/
    ├── __init__.py
    ├── test_methods.py
    └── test_solver.py
```

## Main Files

- `main.py`
  Runs the complete workflow from dataset creation to model training.

- `src/ode_solver/problems.py`
  Contains the ODE problem definitions.

- `src/ode_solver/methods.py`
  Implements the one-step solvers:
  - Euler
  - Midpoint
  - Heun
  - RK4
  - RK10 reference method

- `src/ode_solver/solver.py`
  Contains the core numerical workflow:
  - integrates each ODE
  - computes L2 error
  - measures runtime
  - chooses the best method

- `src/ode_solver/generate_dataset.py`
  Converts symbolic ODE expressions into machine learning features.

- `src/ode_solver/model_RK.py`
  Trains and evaluates the neural network model.

- `config/settings.py`
  Stores paths, step size, threshold, and result column names.

## Installation

### 1. Create a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## How To Run

Run the complete pipeline:

```bash
python main.py
```

## Run The Local Web App

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

The web app lets you:

- enter a first-order ODE in terms of `t` and `y`
- provide `t0`, `y0`, `tf`, and `step_size`
- optionally provide an exact solution
- get the predicted best RK method from the trained model
- see the computed numerical solution for that method
- compare all methods by error and runtime

### API endpoint

You can also call the JSON API directly:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "f_expression": "t + y",
    "t0": 0,
    "y0": 1,
    "tf": 1,
    "step_size": 0.01,
    "exact_solution": "2*exp(t) - t - 1"
  }'
```

Example response fields:

- `predicted_best_method`
- `benchmark_best_method`
- `class_probabilities`
- `method_metrics`
- `solution.t_values`
- `solution.y_values`

## What Happens When You Run `main.py`

The script performs these steps:

1. Builds an input table of ODE problems
2. Saves that table to `data/final_ode_Problem.csv`
3. Solves each ODE using all RK methods
4. Computes:
   - L2 error
   - CPU time
   - best RK method
5. Saves solver results to `data/final_ode_Problem_results.csv`
6. Creates a reduced file for model labels in `data/model_ode.csv`
7. Generates feature values from each ODE expression
8. Saves those features to `data/function_value.csv`
9. Trains the neural network
10. Prints:
   - test accuracy
   - a random prediction example
   - class probabilities
   - confusion matrix
   - classification report
11. Saves trained artifacts in `model/`

## Numerical Methods Used

The project compares these methods:

- `Euler`
  First-order method. Fastest and simplest, but least accurate.

- `Midpoint`
  Second-order method. Better accuracy than Euler.

- `Heun`
  Second-order predictor-corrector method.

- `RK4`
  Classical fourth-order Runge-Kutta method. Usually the strongest general-purpose method here.

- `RK10`
  Used only as a high-accuracy reference when an exact solution is not available or cannot be evaluated.

## How Error Is Computed

The code uses a discrete L2 error:

```text
sqrt( h * sum( (y_exact - y_num)^2 ) )
```

Where:

- `h` = step size
- `y_exact` = exact or reference solution
- `y_num` = numerical solution from a method

## How The Best Method Is Chosen

The project uses a two-stage rule:

1. Among methods with L2 error below the threshold, choose the fastest one
2. If no method is below the threshold, choose the most accurate one

Current threshold from `config/settings.py`:

```python
THRESHOLD = 1e-5
```

This makes the project balance both accuracy and computational cost.

## Machine Learning Pipeline

After the solver labels each ODE with its best RK method, the project trains a classifier.

### Input features

`generate_dataset.py` evaluates each `f_expression` numerically and creates:

- sampled values from the function
- mean
- standard deviation
- minimum
- maximum
- absolute mean
- absolute maximum

It also keeps useful metadata such as:

- `ode_type`
- `t0`
- `y0`
- `tf`
- `step_size_h`
- `num_steps`
- per-method error and runtime

### Target label

- `best_rk_method`

### Model details

The classifier in `model_RK.py` uses:

- a feedforward neural network
- dense layers
- batch normalization
- LeakyReLU activations
- dropout
- softmax output layer
- Adam optimizer

### Data preparation

Before training, the code:

- converts features to numeric values
- fills invalid values
- encodes labels
- splits train/test data
- standardizes features
- applies `SMOTE` to balance minority classes
- computes class weights

These are important because the dataset is imbalanced.

## Important Output Files

- `data/final_ode_Problem.csv`
  Raw ODE input dataset.

- `data/final_ode_Problem_results.csv`
  Full solver comparison results including errors, runtimes, and best method.

- `data/model_ode.csv`
  Reduced file containing `f_expression` and `best_rk_method`.

- `data/function_value.csv`
  Feature dataset used for model training.

- `model/model.joblib`
  Saved trained model.

- `model/model.keras`
  Native Keras model format used by the FastAPI app when available.

- `model/scaler.joblib`
  Saved feature scaler.

- `model/encoder.joblib`
  Saved label encoder.

## Important Evaluation Metrics

These are the most important outputs from the classification stage.

### 1. Test Accuracy

Accuracy tells you how many predictions are correct overall.

Example:

```text
Test Accuracy = 0.9836
```

This means about 98.36% of test examples were classified correctly.

Important note:

- Accuracy alone is not enough when classes are imbalanced
- A model can get high accuracy by favoring the dominant class

That is why the confusion matrix and classification report are also important.

### 2. Confusion Matrix

The confusion matrix shows how predictions are distributed across classes.

Example:

| Actual \ Predicted | Euler | Heun | Midpoint | RK4 |
|--------------------|------:|-----:|---------:|----:|
| Euler              | 200   | 0    |  0       | 0   |
| Heun               | 1     | 40   |  8       | 0   |
| Midpoint           | 0     | 14   | 47       | 0   |
| RK4                | 0     | 0    |  0       | 1090|

How to read it:

- Rows are the true classes
- Columns are the predicted classes
- Larger diagonal values mean more correct predictions
- Off-diagonal values show where the model is confusing one method with another

What this matrix tells you:

- `RK4` is recognized very well
- `Heun` is also learned fairly well
- `Midpoint` is sometimes confused with `Heun`
- `Euler` is sometimes misclassified as `Heun`

This is one of the most important parts of the README because it shows model behavior, not just final accuracy.

### 3. Classification Report

Example:

| Class     | Precision | Recall | F1-score | Support |
|-----------|----------:|-------:|---------:| ------: |
| Euler     | 1.00      | 1.00   | 1.00     |   200   |
| Heun      | 0.74      | 0.82   | 0.78     |    49   |
| Midpoint  | 0.85      | 0.77   | 0.81     |    61   |
| RK4       | 1.00      | 1.00   | 1.00     |   1090  |

Why these matter:

- `Precision`
  Out of all times the model predicted a class, how often it was correct.

- `Recall`
  Out of all true examples of a class, how many the model found.

- `F1-score`
  Balance between precision and recall.

Most important interpretation here:

- `RK4` has the strongest overall performance
- `Heun` has strong recall, meaning the model often finds true Heun cases
- `Midpoint` is harder than `RK4`
- `Euler` is reasonable, but there is still confusion with other methods

### 4. Why Weighted Metrics Matter

The class distribution is not perfectly balanced. Because of that:

- weighted F1-score is important
- macro F1-score is also useful
- confusion matrix should always be checked beside accuracy

This gives a more realistic evaluation than accuracy alone.

## Example Results

The project documentation currently reports these example model results:

- Total samples: `7390`
- Train samples: `5912`
- Test samples: `1478`
- Test accuracy: `0.9835714101791382`

Best-method distribution example:

| Method   | Count |
|----------|------:|
| RK4      | 5448  |
| HEUN     |  999  |
| EULER    |  306  |
| MIDPOINT |  247  |

These values may change if the dataset or model setup changes.

## Why SMOTE And Class Weights Are Important

The project uses both:

- `SMOTE`
  Creates synthetic training samples for minority classes

- `class_weight`
  Makes the loss function pay more attention to underrepresented classes

These two steps are important because without them the model can become biased toward the majority class such as `RK4`.

## Testing

Run the unit tests with:

```bash
pytest
```

The tests check:

- numerical method correctness
- expected RK accuracy ordering on a simple ODE
- L2 error computation
- best-method selection logic
- DataFrame construction

## Key Takeaways

- This is both a numerical solver project and a machine learning project
- The solver produces labels by comparing RK methods on the same ODE
- The neural network learns to predict the best RK method from generated features
- The FastAPI app exposes the model as a local website and JSON API
- Confusion matrix and classification report are essential for understanding model quality
- Accuracy is useful, but it is not enough by itself

## Future Improvements

- Add more nonlinear and challenging ODE families
- Extend to systems of ODEs
- Compare against adaptive RK methods
- Add plots for error and runtime comparisons
- Store training history and visualizations
- Add reproducibility notes for random seeds and dataset generation

## Author

Priyanshu Gupta  
M.Sc. Mathematics, IIT Gandhinagar

## Acknowledgement

Supervisor: Dr. Abhinav Jha, IIT Gandhinagar
