import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Wczytaj dane
df = pd.read_csv('./data/Pokemon.csv')

# ============
# 🔹 Wykres 1: Type 1 → liczba Pokémonów
# ============
type1_counts = df['Type 1'].value_counts()

plt.figure(figsize=(10, 6))
type1_counts.plot(kind='bar', color='skyblue')
plt.title('Liczba Pokémonów według Type 1')
plt.xlabel('Type 1')
plt.ylabel('Liczba Pokémonów')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y')
plt.savefig("pokemon_nubmers2.png")

# ============
# 🔸 Wykres 2: Wszystkie typy (Type 1 + Type 2)
# ============
# Uzupełnij NaN w Type 2 pustym stringiem, potem zlicz
all_types = df['Type 1'].tolist() + df['Type 2'].dropna().tolist()
type_all_counts = dict(Counter(all_types))

# Konwersja do Series dla sortowania i rysowania
type_all_series = pd.Series(type_all_counts).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
type_all_series.plot(kind='bar', color='orange')
plt.title('Łączna liczba wystąpień typów (Type 1 + Type 2)')
plt.xlabel('Typ')
plt.ylabel('Liczba wystąpień')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y')
plt.savefig("pokemon_numbers.png")
