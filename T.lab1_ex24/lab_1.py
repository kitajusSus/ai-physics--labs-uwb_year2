#lab_1 zadanie 24
import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns

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
        self.hidden = nn.Linear(1, 10)  # Warstwa ukryta z 10 neuronami
        self.relu = nn.ReLU()  # Funkcja aktywacji ReLU
        self.output = nn.Linear(10, 1)  # Warstwa wyjściowa

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

# Wykres danych treningowych
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

# Wykres błędów
plt.figure(figsize=(10, 5))
plt.scatter(temperatures.numpy(), errors, label='Błędy predykcji', color='green')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Temperatura (K)')
plt.ylabel('Błąd predykcji')
plt.title('Błędy predykcji w zależności od temperatury')
plt.legend()
plt.show()

# Mapa cieplna błędów

plt.figure(figsize=(10, 5))
sns.heatmap(errors.reshape(-1, 1), annot=True, cmap='coolwarm', cbar=True)
plt.xlabel('Błąd predykcji')
plt.ylabel('Indeks próbki')
plt.title('Mapa cieplna błędów predykcji')
plt.show()