import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 0. Otwieranie danych
file_path = 'daneex6.xlsx'
# Sprawdzenie, czy plik istnieje
if os.path.exists(file_path):
    # odczytanie danych z arkusza 
    df = pd.read_excel(file_path, sheet_name='Arkusz1')
    print(df.head())
else:
    print(f"Plik {file_path} nie istnieje. Upewnij się, że plik znajduje się w odpowiednim folderze.")

# 1. Przygotowanie danych
# Przykładowe dane: temperatury (T) i odpowiadające im zmiany objętości (delta_V)
y = df['U(V)'].values  
x = df['X'].values

# Konwersja danych do tensorów PyTorch
y_ = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)  # reshaping na kolumnę
x_ = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)

# Konwersja tensorów do numpy arrays dla sklearn
Y = y_.numpy()
X = x_.numpy()

# 2. Obliczanie niepewności (przykładowe wartości)
Delta_X = np.full(X.shape[0], 0.0)  # Przykładowa niepewność dla X
Delta_Y = np.full(Y.shape[0], 0.0)  # Przykładowa niepewność dla Y

# 3. Regresja liniowa
model = LinearRegression()
model.fit(X, Y)
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

# 4. Rysowanie wykresu z niepewnościami
plt.errorbar(X.flatten(), Y.flatten(), xerr=Delta_X, yerr=Delta_Y, fmt='o', color='blue', label='Dane z niepewnościami')
plt.plot(X.flatten(), Y_pred, color='red', linewidth=2, label='Regresja liniowa')
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.title('Regresja liniowa z niepewnościami')
plt.legend()
plt.show()
