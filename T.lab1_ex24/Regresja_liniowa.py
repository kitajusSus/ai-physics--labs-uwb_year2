import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 0. Otwieranie danych
file_path = 'lab_1_dane.xlsx'
# Sprawdzenie, czy plik istnieje
if os.path.exists(file_path):
    # odczytanie danych z arkusza 
    df = pd.read_excel(file_path, sheet_name='Arkusz1')
    print(df.head())
else:
    print(f"Plik {file_path} nie istnieje. Upewnij się, że plik znajduje się w odpowiednim folderze.")


# 1. Przygotowanie danych
# Przykładowe dane: temperatury (T) i odpowiadające im zmiany objętości (delta_V)
delta_volumes = df['X'].values  # Zmiana objętości w m^3
temperatures = df['Y'].values  # Temperatura w Kelvinach


# Konwersja danych do tensorów PyTorch
temperatures = torch.tensor(temperatures, dtype=torch.float32).reshape(-1, 1)  # reshaping na kolumnę
delta_volumes = torch.tensor(delta_volumes, dtype=torch.float32).reshape(-1, 1)

# Konwersja tensorów do numpy arrays dla sklearn
Y = temperatures.numpy()
X = delta_volumes.numpy()

# Inicjalizacja modelu regresji liniowej
model = LinearRegression()

# Trenowanie modelu na danych
model.fit(X, Y)

# Predykcja wartości Y dla danych X
Y_pred = model.predict(X)

# Obliczanie współczynników regresji
A = model.coef_[0]
Delta_A = np.sqrt(np.mean((model.predict(X) - Y) ** 2)) / np.sqrt(len(X)) / np.std(X, ddof=1)
B = model.intercept_
Delta_B = np.sqrt(np.mean((model.predict(X) - Y) ** 2)) * np.sqrt(1 / len(X) + np.mean(X ** 2)) / np.std(X, ddof=1)

# Wyświetlanie współczynników regresji
print("Współczynniki regresji:")
print("Współczynnik nachylenia (A):", A)
print("Niepewność współczynnika nachylenia A (Delta_A):", Delta_A)
print("Wyraz wolny (B):", B)
print("Niepewność wyrazu wolnego B (Delta_B):", Delta_B)

# Zakładamy, że Delta_X i Delta_Y są obliczone wcześniej
Delta_X = np.full_like(X, 0.1)  # Przykładowa niepewność dla X
Delta_Y = np.full_like(Y, 0.2)  # Przykładowa niepewność dla Y

# Rysowanie wykresu z niepewnościami
plt.errorbar(X.flatten(), Y, xerr=Delta_X, yerr=Delta_Y, fmt='o', color='blue', label='Dane z niepewnościami')
plt.plot(X.flatten(), Y_pred, color='red', linewidth=2, label='Regresja liniowa')
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.title('Regresja liniowa z niepewnościami')
plt.legend()
plt.show()