# Labolatoria 1 Zadanie 24


Eksperyment polega na zbadaniu, jak objętość powietrza zmienia się pod wpływem temperatury przy stałym ciśnieniu. Wykorzystuje się do tego układ z naczyniem zanurzonym w łaźni wodnej, połączonym z cylindrem pomiarowym, w którym odczytuje się zmiany objętości powietrza.

## Główne elementy eksperymentu:
- Układ pomiarowy: Składa się z naczynia z powietrzem umieszczonego w wodzie oraz **cylindra pomiarowego**, w którym odczytywane są zmiany objętości gazu.
- Proces pomiarowy: Odczytywana jest temperatura wody oraz zmiany objętości powietrza w cylindrze przy zmieniającej się temperaturze. Kluczowe jest utrzymanie **stałego** ciśnienia w układzie.
- Równania: Wykorzystuje się równania gazu doskonałego (**Clapeyrona**), aby opisać zależność między temperaturą a objętością gazu.
- Analiza danych: Zmiany objętości są analizowane w odniesieniu do zmieniającej się temperatury. Wyniki są przedstawiane na wykresie, gdzie osie reprezentują względne zmiany objętości ($\frac{ΔV}{V}$) i temperatury ($\frac{T0}{T}$). Otrzymany wykres powinien dawać liniową zależność, co umożliwia wyznaczenie współczynnika rozszerzalności.

*Celem jest wyznaczenie współczynnika rozszerzalności powietrza oraz analiza uzyskanych wyników pod kątem ich zgodności z teorią gazu doskonałego*

# Budowanie modelu

## Poszczególne kroki. 
- Import bibliotek: PyTorch oraz NumPy do zarządzania danymi.
- Przygotowanie danych: Wprowadzamy dane wejściowe (temperatury) i dane wyjściowe (zmiany objętości).
- Tworzenie modelu: Zbudujemy sieć neuronową z warstwą wejściową, jedną ukrytą warstwą i warstwą wyjściową.
- Trenowanie modelu: Model uczy się zależności między temperaturą a objętością.
- Predykcje: Używamy modelu do przewidywania nowych wyników.