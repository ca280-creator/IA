import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Incarcare si pregatirea datelor
data = pd.read_csv("/content/sample_data/earthquake_data_tsunami.csv")
data = data.dropna(subset=['tsunami'])  # eliminam randurile cu etichete lipsa
X = data.drop("tsunami", axis=1).values
y = data["tsunami"].values

# Impartire in set de antrenare si set de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizare min-max
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Metoda standard - Cross-validation simpla pentru k
import warnings
warnings.filterwarnings('ignore')

start_time_std = time.time()

k_values = range(1, 31)
cv_scores = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

best_k = k_values[np.argmax(cv_scores)]
best_cv_score_std = max(cv_scores)

# Model final standard
model_standard = KNeighborsClassifier(n_neighbors=best_k)
model_standard.fit(X_train, y_train)
acc_standard = model_standard.score(X_test, y_test)

time_std = time.time() - start_time_std

# Grafic evolutie precizie vs k (metoda standard)
plt.figure(figsize=(8,5))
plt.plot(k_values, cv_scores, marker='o', color='blue')
plt.title("Figura 1. Determinarea optima a valorii lui k (Metoda standard)")
plt.xlabel("Valoarea lui k")
plt.ylabel("Precizia medie (Cross-validation)")
plt.grid(True)
plt.show()

print(f"--- Rezultate metoda standard ---")
print(f"Valoare optima k: {best_k}")
print(f"Precizie cross-val medie: {best_cv_score_std:.4f}")
print(f"Precizie test finala: {acc_standard:.4f}")
print(f"Durata totala: {time_std:.2f} secunde\n")

# Metoda alternativa - Grid Search (optimizare completa)
start_time_opt = time.time()

param_grid = {
    'n_neighbors': np.arange(1, 31),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

best_params = grid.best_params_
best_score_opt = grid.best_score_

best_model = grid.best_estimator_
acc_optimized = best_model.score(X_test, y_test)

time_opt = time.time() - start_time_opt

print("--- Rezultate metoda alternativa (optimizata) ---")
print("Parametrii optimi:", best_params)
print(f"Precizie medie (GridSearchCV): {best_score_opt:.4f}")
print(f"Precizie test finala: {acc_optimized:.4f}")
print(f"Durata totala: {time_opt:.2f} secunde\n")

# Grafic evolutie precizie vs k (metoda optimizata)
grid_results = pd.DataFrame(grid.cv_results_)
mean_scores_per_k = grid_results.groupby('param_n_neighbors')['mean_test_score'].max()

plt.figure(figsize=(8,5))
plt.plot(mean_scores_per_k.index.astype(int), mean_scores_per_k.values, marker='o', color='green')
plt.title("Figura 1b. Determinarea optima a valorii lui k (Metoda optimizata)")
plt.xlabel("Valoarea lui k")
plt.ylabel("Precizia medie (GridSearchCV)")
plt.grid(True)
plt.show()

# Comparatie detaliata (tabel + figuri)
results = pd.DataFrame({
    "Metoda": ["Standard", "Optimizata"],
    "Precizie CV": [best_cv_score_std, best_score_opt],
    "Precizie Test": [acc_standard, acc_optimized],
    "Durata (s)": [time_std, time_opt],
    "Parametri evaluati": [len(k_values),
                           len(param_grid['n_neighbors']) * len(param_grid['weights']) * len(param_grid['metric'])]
})

print("COMPARATIE FINALA")
print(results)

# Bar chart pentru acuratete
plt.figure(figsize=(8,5))
plt.bar(results["Metoda"], results["Precizie Test"], color=["gray", "green"])
plt.title("Figura 2. Comparatie a preciziei intre metode")
plt.ylabel("Precizie pe setul de test")
plt.ylim(0.0, 1.0)
plt.grid(axis="y")
plt.show()

# Bar chart pentru durata de executie
plt.figure(figsize=(8,5))
plt.bar(results["Metoda"], results["Durata (s)"], color=["gray", "orange"])
plt.title("Figura 3. Durata de executie a celor doua metode")
plt.ylabel("Timp (secunde)")
plt.grid(axis="y")
plt.show()

# Matrice de confuzie pentru modelul optimizat
y_pred = best_model.predict(X_test)
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(cmap="Blues")
plt.title("Figura 4. Matricea de confuzie - model optimizat")
plt.show()

# Raport detaliat pentru modelul optimizat
print("\nRAPORT COMPLET PENTRU MODELUL OPTIMIZAT")
print(classification_report(y_test, y_pred))
