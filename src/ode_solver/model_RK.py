import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras
import joblib

np.random.seed(42)
tf.random.set_seed(42)

def model_RK_(data):
    data = data.drop(columns=["f_expression"])

    X = data.drop(columns=["best_rk_method"])
    y = data["best_rk_method"]

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu',input_shape=(X.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(4, activation='softmax')
    ])

    optimizer = keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=8,
        validation_data=(X_test, y_test)
    )

    loss, acc = model.evaluate(X_test, y_test)

    idx = np.random.randint(0, len(X_test))
    sample = X_test[idx].reshape(1, -1)

    pred = model.predict(sample)
    pred_class = np.argmax(pred)

    print("\nRandom Test Example:")
    print("Predicted:", label_encoder.inverse_transform([pred_class])[0])
    print("Actual:", label_encoder.inverse_transform([y_test[idx]])[0])

    joblib.dump(model, "model/model.joblib")
    
    return acc