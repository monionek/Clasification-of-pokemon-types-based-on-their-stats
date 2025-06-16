import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

# Wczytanie danych
df = pd.read_csv('./data/Pokemon.csv')

# 🔀 Połącz typy
df['Type 2'] = df['Type 2'].fillna('None')  # brak typu zamieniamy na tekst "None"
df['CombinedType'] = df['Type 1'] + "_" + df['Type 2']

# 🔍 Top N kombinacji typów
TOP_K_TYPES = 20  # ← change this number
top_types = df['CombinedType'].value_counts().nlargest(TOP_K_TYPES).index.tolist()
df = df[df['CombinedType'].isin(top_types)]

# Cechy i etykiety
features = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
X = df[features]
y = df['CombinedType'].values

# Kodowanie etykiet
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

# Obliczanie wag klas
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_encoded),
    y=y_encoded
)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# Skalowanie
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Podział na dane
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model = Sequential([
    Dense(243, activation='elu', input_shape=(7,)),
    Dropout(0.5),
    Dense(81, activation='elu'),
    Dropout(0.5),
    Dense(27, activation='elu'),
    Dropout(0.4),
    Dense(9, activation='elu'),
    Dropout(0.3),
    Dense(3, activation='elu'),
    Dense(num_classes, activation='softmax')
])


# Kompilacja
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Trening
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=8,
    validation_split=0.1,
    callbacks=[early_stop],
    class_weight=class_weight_dict
)

# Predykcje
y_pred_train = np.argmax(model.predict(X_train), axis=1)
y_pred_test = np.argmax(model.predict(X_test), axis=1)

# Ocena
print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))

# Wykresy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("Acc.png")

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig("Loss.png")

# Macierz błędów
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("Confusion_matrix.png")