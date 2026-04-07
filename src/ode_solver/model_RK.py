# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# import tensorflow as tf
# from tensorflow import keras
# import joblib
# from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.utils.class_weight import compute_class_weight
# # np.random.seed(42)
# # tf.random.set_seed(42)

# def model_RK_(data):
#     data = data.drop(columns=["f_expression"])

#     X = data.drop(columns=["best_rk_method"])
#     y = data["best_rk_method"]

#     X = X.apply(pd.to_numeric, errors="coerce")
#     X = X.replace([np.inf, -np.inf], np.nan)
#     X = X.fillna(0)

#     label_encoder = LabelEncoder()
#     y = label_encoder.fit_transform(y)

#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     # model = keras.Sequential([
#     #     keras.layers.Dense(128, activation='relu',input_shape=(X.shape[1],)),
#     #     keras.layers.Dense(64, activation='relu'),
        
#     #     keras.layers.Dense(64, activation='relu'),
#     #     keras.layers.Dense(4, activation='softmax')
#     # ])
#     # model = keras.Sequential([
#     #     keras.layers.Dense(256, activation='relu', input_shape=(X.shape[1],)),
#     #     keras.layers.Dropout(0.3),

#     #     keras.layers.Dense(128, activation='relu'),
#     #     keras.layers.Dropout(0.3),

#     #     keras.layers.Dense(64, activation='relu'),
#     #     keras.layers.Dropout(0.3),

#     #     keras.layers.Dense(4, activation='softmax')
#     # ])


    
#     model = keras.Sequential([
#         keras.layers.Dense(256, input_shape=(X.shape[1],)),
#         keras.layers.BatchNormalization(),
#         keras.layers.Activation('relu'),
#         keras.layers.Dropout(0.2),

#         keras.layers.Dense(128),
#         keras.layers.BatchNormalization(),
#         keras.layers.Activation('relu'),
#         keras.layers.Dropout(0.2),

#         keras.layers.Dense(64),
#         keras.layers.BatchNormalization(),
#         keras.layers.Activation('relu'),
#         keras.layers.Dropout(0.2),

#         keras.layers.Dense(32),
#         keras.layers.BatchNormalization(),
#         keras.layers.Activation('relu'),
#         keras.layers.Dropout(0.2),


#         keras.layers.Dense(4, activation='softmax')
#     ])
#     optimizer = keras.optimizers.Adam(
#         learning_rate=0.0005,
#         clipnorm=1.0
#     )

#     model.compile(
#         optimizer=optimizer,
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"]
#     )

#     ##$$##
#     classes = np.unique(y_train)
#     weights = compute_class_weight(
#         class_weight='balanced',
#         classes=classes,
#         y=y_train
#     )

#     class_weights = dict(zip(classes, weights))

#     early_stop = keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     patience=10,
#     restore_best_weights=True)

#     history = model.fit(
#         X_train,
#         y_train,
#         epochs=100,
#         batch_size=32,
#         validation_data=(X_test, y_test),
#         class_weight=class_weights,
#         callbacks = [early_stop]   

#     )

#     loss, acc = model.evaluate(X_test, y_test)

#     idx = np.random.randint(0, len(X_test))
#     sample = X_test[idx].reshape(1, -1)

#     # pred = model.predict(sample)
#     # pred_class = np.argmax(pred)

#     # print("\nRandom Test Example:")
#     # print("Predicted:", label_encoder.inverse_transform([pred_class])[0])
#     # print("Actual:", label_encoder.inverse_transform([y_test[idx]])[0])




#     pred = model.predict(sample)
#     pred_class = np.argmax(pred)

#     # Confidence of predicted class
#     confidence = np.max(pred)

#     print("\nRandom Test Example:")
#     print("Predicted:", label_encoder.inverse_transform([pred_class])[0])
#     print("Confidence:", round(confidence * 100, 2), "%")
#     print("Actual:", label_encoder.inverse_transform([y_test[idx]])[0])

#     # Full probability distribution (VERY useful)
#     print("\nClass-wise Probabilities:")
#     for i, prob in enumerate(pred[0]):
#         print(f"{label_encoder.inverse_transform([i])[0]}: {prob:.4f}")

#     # Predict on test data
#     y_pred_probs = model.predict(X_test)
#     y_pred = np.argmax(y_pred_probs, axis=1)

#     # Confusion Matrix
#     cm = confusion_matrix(y_test, y_pred)

#     print("\nConfusion Matrix:")
#     print(cm)

#     # Optional: Pretty display with labels
#     labels = label_encoder.classes_
#     cm_df = pd.DataFrame(cm, index=labels, columns=labels)

#     print("\nConfusion Matrix (with labels):")
#     print(cm_df)

#     # Classification Report (very useful)
#     print("\nClassification Report:")
#     print(classification_report(y_test, y_pred, target_names=labels))


#     joblib.dump(model, "model/model.joblib")
    
#     return acc

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
import joblib

def model_RK_(data):
    data = data.drop(columns=["f_expression"])

    X = data.drop(columns=["best_rk_method"])
    y = data["best_rk_method"]

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Stratified split to preserve class ratios
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # SMOTE to oversample minority classes
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    # Class weights as safety net after SMOTE
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    # Model architecture
    inputs = keras.Input(shape=(X_scaled.shape[1],))

    # x = keras.layers.Dense(256)(inputs)
    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.Activation('relu')(x)
    # x = keras.layers.Dropout(0.3)(x)

    # x = keras.layers.Dense(128)(x)
    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.Activation('relu')(x)
    # x = keras.layers.Dropout(0.2)(x)

    # x = keras.layers.Dense(64)(x)
    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dense(256)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.01)(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Dense(128)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.01)(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(64)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.01)(x)

    outputs = keras.layers.Dense(4, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    # FIX: Plain float LR — compatible with ReduceLROnPlateau
    optimizer = keras.optimizers.Adam(
        learning_rate=0.001,   # plain float, NOT a schedule object
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        min_delta=0.001
    )

    # ✅ FIX: Now works because LR is a plain float
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=32,
        validation_data=(X_test, y_test),
        class_weight=class_weights,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )


    loss, acc = model.evaluate(X_test, y_test)
    print(f"\n✅ Test Accuracy: {acc:.4f}")

    # ✅ Random single example check
    idx = np.random.randint(0, len(X_test))
    sample = X_test[idx].reshape(1, -1)
    pred_probs = model.predict(sample, verbose=0)
    pred_class = np.argmax(pred_probs)
    confidence = np.max(pred_probs) * 100

    print("\n--- Random Test Example ---")
    print(f"Predicted : {label_encoder.inverse_transform([pred_class])[0]}  ({confidence:.2f}% confidence)")
    print(f"Actual    : {label_encoder.inverse_transform([y_test[idx]])[0]}")
    print(f"Match     : {'✅ Correct' if pred_class == y_test[idx] else '❌ Wrong'}")

    print("\nClass-wise Probabilities:")
    for i, prob in enumerate(pred_probs[0]):
        bar = '█' * int(prob * 30)
        print(f"  {label_encoder.inverse_transform([i])[0]:<10} {prob:.4f}  {bar}")


    # Evaluation
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    labels = label_encoder.classes_

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(cm, index=labels, columns=labels))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))

    joblib.dump(model, "model/model.joblib")
    joblib.dump(scaler, "model/scaler.joblib")
    joblib.dump(label_encoder, "model/encoder.joblib")

    return acc

