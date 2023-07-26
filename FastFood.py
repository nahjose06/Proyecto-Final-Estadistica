import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.stats.anova as anova
import scipy.stats as stats

# Leer el archivo CSV
df = pd.read_csv('Datos/fastfood.csv')

# Calcular histograma
plt.hist(df['calories'], bins='auto')
plt.xlabel('Calorías')
plt.ylabel('Frecuencia')
plt.title('Histograma de Frecuencia')
plt.show()

# Calcular histograma de frecuencia relativa
plt.hist(df['calories'], bins='auto', density=True)
plt.xlabel('Calorías')
plt.ylabel('Frecuencia Relativa')
plt.title('Histograma de Frecuencia Relativa')
plt.show()

# Calcular frecuencia acumulativa
frecuencia_acumulativa = df['calories'].value_counts().sort_index().cumsum()

# Calcular frecuencia relativa acumulativa
frecuencia_relativa_acumulativa = frecuencia_acumulativa / len(df)

# Gráfico de frecuencia acumulativa
plt.bar(frecuencia_acumulativa.index, frecuencia_acumulativa.values)
plt.xlabel('Calorías')
plt.ylabel('Frecuencia Acumulativa')
plt.title('Frecuencia Acumulativa')
plt.show()

# Gráfico de frecuencia relativa acumulativa
plt.bar(frecuencia_relativa_acumulativa.index, frecuencia_relativa_acumulativa.values)
plt.xlabel('Calorías')
plt.ylabel('Frecuencia Relativa Acumulativa')
plt.title('Frecuencia Relativa Acumulativa')
plt.show()

# Diagrama de Pareto 
calories_counts = df['calories'].value_counts()
relative_freq = calories_counts / len(df)

sorted_calories = relative_freq.sort_values(ascending=True)
cumulative_freq = sorted_calories.cumsum()

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(cumulative_freq) + 1), cumulative_freq, color='b', alpha=0.7)
plt.plot(range(1, len(cumulative_freq) + 1), cumulative_freq, color='r', marker='o')
plt.xlabel('Calorías (calories)')
plt.ylabel('Frecuencia Relativa Acumulativa')
plt.title('Diagrama de Pareto: Calorías (calories) (Frecuencia Relativa Acumulativa)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Calcular media
media = df['calories'].mean()
print('Media:', media)

# Calcular varianza
varianza = df['calories'].var()
print('Varianza:', varianza)

# Calcular desviación estándar
desviacion_estandar = df['calories'].std()
print('Desviación Estándar:', desviacion_estandar)

# Calcular el coeficiente de correlación de Pearson entre "calories" y "cholesterol"
correlation_coef = df['calories'].corr(df['cholesterol'])

# Mostrar el coeficiente de correlación
print("Coeficiente de correlación entre calories y cholesterol:", correlation_coef)

# Regresion lineal
# Definir las variables dependiente e independiente
y = df['cholesterol']
X = df['calories']

# Ajustar el modelo de regresión lineal
model = sm.ols('cholesterol ~ calories', data=df).fit()

# Obtener los resultados del modelo
results = model.summary()

# Mostrar los resultados
# print(results)

# Gráfica de dispersión de puntos con la línea de regresión
plt.scatter(df['calories'], df['cholesterol'], label='Datos')
plt.plot(df['calories'], model.fittedvalues, color='red', label='Regresión')
plt.xlabel('calories')
plt.ylabel('cholesterol')
plt.legend()
plt.title('Regresión Lineal: cholesterol vs calories')
plt.show()

# Gráfica de residuales
residuals = model.resid
plt.scatter(df['calories'], residuals)
plt.axhline(y=0, color='red', linestyle='dashed')
plt.xlabel('calories')
plt.ylabel('Residuales')
plt.title('Gráfica de Residuales')
plt.show()

# Gráfica de probabilidad normal de los residuales
stats.probplot(residuals, plot=plt)
plt.title('Gráfica de Probabilidad Normal de Residuales')
plt.show()

# ANOVA
anova_table = anova.anova_lm(model)
print("Tabla ANOVA:")
print(anova_table)

# R cuadrado
print("R cuadrado:", model.rsquared)

# Pruebas de t y F
print("Prueba de t para el intercepto:")
print(model.t_test("Intercept = 1"))  # Para el intercepto

print("Prueba de t para la pendiente:")
print(model.t_test("calories = 1"))  # Para la pendiente

print("Prueba de F para el modelo:")
print(model.f_test("calories = 0"))  # Para el modelo

# Intervalos de confianza para el intercepto y la pendiente
conf_int = model.conf_int(alpha=0.05)
print("Intervalo de confianza para el intercepto y la pendiente:")
print(conf_int)