import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
#from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from tensorflow.keras.utils import plot_model


path = "/cloud/project/Deep_learning_R/cancer.csv"
df = pd.read_csv(path)

X = df.drop(columns=["diagnosis(1=m, 0=b)"]).values

y = df["diagnosis(1=m, 0=b)"].astype(int).values

# --- Split & scale ---
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

# --- Build model ---
p = x_train.shape[1]
p

model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(p,)),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy", keras.metrics.AUC(name="auc")])

# --- Train ---
model.fit(x_train, y_train, epochs=30, batch_size=32,
          validation_split=0.2,
          callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
          verbose=2)

# --- Evaluate ---
loss, acc, auc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test — Loss: {loss:.4f} | Acc: {acc:.4f} | AUC: {auc:.4f}")

prob = model.predict(x_test).ravel()
pred = (prob >= 0.5).astype(int)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=["benign","malignant"]))
print("ROC AUC (sklearn):", roc_auc_score(y_test, prob))





RocCurveDisplay.from_predictions(y_test, prob)
plt.title("ROC Curve — Malignant vs Benign")
plt.show()


# Assuming you've already built your model (named 'model' in the previous example)
plot_model(
    model,
    to_file='model_architecture.png',
    show_shapes=True,
    show_layer_names=True,
    dpi=144,
    rankdir='LR'  # Optional: 'LR' for horizontal layout, 'TB' for vertical (default)
)


