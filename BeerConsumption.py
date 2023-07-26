import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.stats.anova as anova
import scipy.stats as stats

# Leer el archivo CSV
df = pd.read_csv('Datos/BeerConsumption.csv')

# Calcular Histograma
plt.hist(df['AvgTemp'], bins='auto')
plt.xlabel('Temperatura Media')
plt.ylabel('Frecuencia')
plt.title('Histograma de Frecuencia')
plt.show()

# Calcular Histograma de frecuencia relativa
plt.hist(df['AvgTemp'], bins='auto', density=True)
plt.xlabel('Temperatura Media')
plt.ylabel('Frecuencia Relativa')
plt.title('Histograma de Frecuencia Relativa')
plt.show()

# Calcular frecuencia acumulativa
frecuencia_acumulativa = df['AvgTemp'].value_counts().sort_index().cumsum()

# Calcular frecuencia relativa acumulativa
frecuencia_relativa_acumulativa = frecuencia_acumulativa / len(df)

# Gráfico de frecuencia acumulativa
plt.bar(frecuencia_acumulativa.index, frecuencia_acumulativa.values)
plt.xlabel('Temperatura Media')
plt.ylabel('Frecuencia Acumulativa')
plt.title('Frecuencia Acumulativa')
plt.show()

# Gráfico de frecuencia relativa acumulativa
plt.bar(frecuencia_relativa_acumulativa.index, frecuencia_relativa_acumulativa.values)
plt.xlabel('Temperatura Media')
plt.ylabel('Frecuencia Relativa Acumulativa')
plt.title('Frecuencia Relativa Acumulativa')
plt.show()

# Diagrama de Pareto 
AvgTemp_counts = df['AvgTemp'].value_counts()
relative_freq = AvgTemp_counts / len(df)

sorted_AvgTemp = relative_freq.sort_values(ascending=True)
cumulative_freq = sorted_AvgTemp.cumsum()

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(cumulative_freq) + 1), cumulative_freq, color='b', alpha=0.7)
plt.plot(range(1, len(cumulative_freq) + 1), cumulative_freq, color='r', marker='o')
plt.xlabel('Temperatura Media (AvgTemp)')
plt.ylabel('Frecuencia Relativa Acumulativa')
plt.title('Diagrama de Pareto: Temperatura Media (AvgTemp) (Frecuencia Relativa Acumulativa)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Calcular media
media = df['AvgTemp'].mean()
print('Media:', media)

# Calcular varianza
varianza = df['AvgTemp'].var()
print('Varianza:', varianza)

# Calcular desviación estándar
desviacion_estandar = df['AvgTemp'].std()
print('Desviación Estándar:', desviacion_estandar)

# Calcular el coeficiente de correlación de Pearson entre "AvgTemp" y "Consumption"
correlation_coef = df['AvgTemp'].corr(df['Consumption'])

# Mostrar el coeficiente de correlación
print("Coeficiente de correlación entre AvgTemp y Consumption:", correlation_coef)

# Regresion lineal
# Definir las variables dependiente e independiente
y = df['Consumption']
X = df['AvgTemp']

# Ajustar el modelo de regresión lineal
model = sm.ols('Consumption ~ AvgTemp', data=df).fit()

# Obtener los resultados del modelo
results = model.summary()

# Mostrar los resultados
# print(results)

# Gráfica de dispersión de puntos con la línea de regresión
plt.scatter(df['AvgTemp'], df['Consumption'], label='Datos')
plt.plot(df['AvgTemp'], model.fittedvalues, color='red', label='Regresión')
plt.xlabel('AvgTemp')
plt.ylabel('Consumption')
plt.legend()
plt.title('Regresión Lineal: Consumption vs AvgTemp')
plt.show()

# Gráfica de residuales
residuals = model.resid
plt.scatter(df['AvgTemp'], residuals)
plt.axhline(y=0, color='red', linestyle='dashed')
plt.xlabel('AvgTemp')
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
print(model.t_test("AvgTemp = 1"))  # Para la pendiente

print("Prueba de F para el modelo:")
print(model.f_test("AvgTemp = 0"))  # Para el modelo

# Intervalos de confianza para el intercepto y la pendiente
conf_int = model.conf_int(alpha=0.05)
print("Intervalo de confianza para el intercepto y la pendiente:")
print(conf_int)