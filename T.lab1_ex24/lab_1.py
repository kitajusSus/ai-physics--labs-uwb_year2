#lab_1 zadanie 24
import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
"""
Classes:
- NeuralNet: Defines the structure of the neural network model.
Variables:
- file_path: Path to the Excel file containing the data.
- df: DataFrame containing the loaded data.
- temperatures: Tensor containing temperature data.
- delta_volumes: Tensor containing volume change data.
- model: Instance of the NeuralNet class.
- criterion: Loss function (MSE).
- optimizer: Adam optimizer.
- num_epochs: Number of training epochs.
- new_temperatures: Tensor containing new temperature values for prediction.
- predicted_volumes: Predicted volume changes for the new temperatures.
- errors: Prediction errors for the training data.
"""
# 0. Otwieranie danych
file_path = 'lab_1_dane.xlsx'
# Sprawdzenie, czy plik istnieje
if os.path.exists(file_path):
    # odczytanie danych z arkusza 
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    print(df.head())
else:
    print(f"Plik {file_path} nie istnieje. Upewnij się, że plik znajduje się w odpowiednim folderze.")


# 1. Przygotowanie danych
# Przykładowe dane: temperatury (T) i odpowiadające im zmiany objętości (delta_V)
temperatures = df['temperature'].values  # Temperatura w Kelvinach
delta_volumes = df['delta_volume'].values  # Zmiana objętości

# Konwersja danych do tensorów PyTorch
temperatures = torch.tensor(temperatures, dtype=torch.float32).reshape(-1, 1)  # reshaping na kolumnę
delta_volumes = torch.tensor(delta_volumes, dtype=torch.float32).reshape(-1, 1)

# 2. Definicja modelu sieci neuronowej
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.hidden = nn.Linear(1, 21)  # Warstwa ukryta z 10 neuronami
        self.relu = nn.ReLU()  # Funkcja aktywacji ReLU
        self.output = nn.Linear(21, 1)  # Warstwa wyjściowa

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

# Inicjalizacja modelu
model = NeuralNet()

# 3. Definicja funkcji straty i optymalizatora
criterion = nn.MSELoss()  # Funkcja straty - błąd średniokwadratowy (MSE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Optymalizator Adam

# 4. Trenowanie modelu
num_epochs = 500
for epoch in range(num_epochs):
    # Forward pass (przekazywanie sygnału)
    outputs = model(temperatures)
    loss = criterion(outputs, delta_volumes)
    
    # Backward pass i optymalizacja
    optimizer.zero_grad()  # Zerowanie gradientów
    loss.backward()  # Obliczanie gradientów
    optimizer.step()  # Aktualizacja wag

    # Co 50 epok wyświetlamy stratę
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. Predykcja dla nowych temperatur
new_temperatures = torch.tensor([360, 370, 380], dtype=torch.float32).reshape(-1, 1)
predicted_volumes = model(new_temperatures).detach().numpy()

# Wyświetlenie wyników
for i, temp in enumerate(new_temperatures):
    print(f"Przewidywana zmiana objętości dla {temp.item()}K: {predicted_volumes[i][0]:.4f}")

# 6. Wykres wyników
import matplotlib.pyplot as plt

# Wykres danych treningowych i predykcji
plt.figure(figsize=(10, 5))
plt.scatter(temperatures.numpy(), delta_volumes.numpy(), label='Dane treningowe', color='blue')
plt.plot(new_temperatures.numpy(), predicted_volumes, label='Predykcje', color='red')
plt.xlabel('Temperatura (K)')
plt.ylabel('Zmiana objętości')
plt.title('Predykcja zmiany objętości w zależności od temperatury')
plt.legend()
plt.show()

# Obliczanie błędów predykcji
errors = delta_volumes - model(temperatures).detach()
errors = errors.numpy()

# Wykres danych treningowych z błędami
plt.figure(figsize=(10, 5))
plt.scatter(temperatures.numpy(), delta_volumes.numpy(), label='Dane treningowe', color='blue')
plt.plot(new_temperatures.numpy(), predicted_volumes, label='Predykcje', color='red')

# Dodanie błędów do wykresu
for i in range(len(temperatures)):
    color = 'green' if errors[i] >= 0 else 'red'
    plt.plot([temperatures[i], temperatures[i]], [delta_volumes[i], delta_volumes[i] - errors[i]], color=color)

plt.xlabel('Temperatura (K)')
plt.ylabel('Zmiana objętości')
plt.title('Predykcja zmiany objętości z błędami w zależności od temperatury')
plt.legend()
plt.show()
