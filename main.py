# @title MAIN
import pandas as pd
import numpy as np
import tensorflow as tf
import time, warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, PReLU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW

warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

df = pd.read_csv("MSK-MetTropism.csv")
X = df.drop(columns=["CANCER_TYPE"])
y = df["CANCER_TYPE"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

model = Sequential([
    Dense(1024, input_shape=(X_train.shape[1],)),
    PReLU(), BatchNormalization(), Dropout(0.4),
    Dense(512), PReLU(), BatchNormalization(), Dropout(0.4),
    Dense(256), PReLU(), BatchNormalization(), Dropout(0.3),
    Dense(128), PReLU(), BatchNormalization(), Dropout(0.2),
    Dense(len(le.classes_), activation='softmax')
])

y_train_cat = to_categorical(y_train_final, len(le.classes_))
y_val_cat = to_categorical(y_val, len(le.classes_))

model.compile(
    optimizer=AdamW(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

start_time = time.time()
history = model.fit(
    X_train_final, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=100,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)
end_time = time.time()

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

accuracy = np.mean(y_pred == y_test)
top3_preds = np.argsort(y_pred_probs, axis=1)[:, -3:]
correct_top3 = sum([y_test[i] in top3_preds[i] for i in range(len(y_test))])
top3_accuracy = correct_top3 / len(y_test)

print(f"Top-1 Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Top-3 Accuracy: {top3_accuracy:.4f} ({top3_accuracy * 100:.2f}%)")
