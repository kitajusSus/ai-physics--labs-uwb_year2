'''
import numpy as np
import pandas as pd

# Generowanie losowych danych
np.random.seed(42)  # Ustawienie ziarna losowości
num_samples = 50 # Liczba próbek

# Generowanie losowych temperatur (270-400K) i zmian objętości (0.1-1.0)
temperatures = np.random.uniform(270, 400, num_samples)
delta_volumes = np.random.uniform(0.1, 1.0, num_samples)

# Tworzenie DataFrame z danymi
data = pd.DataFrame({
    'temperature': temperatures,
    'delta_volume': delta_volumes
})

# Zapisanie danych do pliku Excel
file_path = 'lab_1_dane.xlsx'
data.to_excel(file_path, index=False)

print(f'Dane zapisane do {file_path}')
'''