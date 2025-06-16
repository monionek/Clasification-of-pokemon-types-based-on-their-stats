# Pokémon Type Classifier

Projekt polega na stworzeniu modelu sieci neuronowej do klasyfikacji typu Pokémona na podstawie jego bazowych statystyk (takich jak HP, Attack, Defense, Sp. Atk, Sp. Def, Speed, Total).

Celem było sprawdzenie, czy typ Pokémona (lub jego kombinacja) może być przewidziany wyłącznie na podstawie tych cech liczbowych, przy użyciu różnych funkcji aktywacji oraz konfiguracji modelu.

## Użyte biblioteki

W projekcie wykorzystano następujące biblioteki:

- `pandas` – wczytywanie i przetwarzanie danych
- `numpy` – operacje numeryczne i przekształcenia
- `matplotlib`, `seaborn` – tworzenie wykresów i wizualizacji
- `scikit-learn` – preprocessing danych, podział na zbiory, metryki, standaryzacja
- `tensorflow.keras` – budowanie i trenowanie modeli sieci neuronowych
- `sklearn.utils.class_weight` – wyważenie klas przy nierównomiernej liczności

Model trenowano z użyciem mechanizmu `EarlyStopping`, funkcji strat `categorical_crossentropy` i metryk opartych o `accuracy`.


