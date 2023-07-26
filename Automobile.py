import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.stats.anova as anova
import scipy.stats as stats


# Leer el archivo CSV
df = pd.read_csv('Datos/Automobile.csv')

# Calcular histograma de frecuencia
plt.hist(df['mpg'], bins='auto')
plt.xlabel('Consumo de combustible (mpg)')
plt.ylabel('Frecuencia')
plt.title('Histograma de Frecuencia')
plt.show()

# Calcular histograma de frecuencia relativa
frecuencia_relativa = df['mpg'].value_counts(normalize=True)
plt.bar(frecuencia_relativa.index, frecuencia_relativa.values)
plt.xlabel('Consumo de combustible (mpg)')
plt.ylabel('Frecuencia Relativa')
plt.title('Histograma de Frecuencia Relativa')
plt.show()

# Calcular frecuencia acumulativa
frecuencia_acumulativa = df['mpg'].value_counts().sort_index().cumsum()
plt.bar(frecuencia_acumulativa.index, frecuencia_acumulativa.values)
plt.xlabel('Consumo de combustible (mpg)')
plt.ylabel('Frecuencia Acumulativa')
plt.title('Histograma de Frecuencia Acumulativa')
plt.show()

# Calcular frecuencia relativa acumulativa
frecuencia_relativa_acumulativa = frecuencia_acumulativa / len(df)
plt.bar(frecuencia_relativa_acumulativa.index, frecuencia_relativa_acumulativa.values)
plt.xlabel('Consumo de combustible (mpg)')
plt.ylabel('Frecuencia Relativa Acumulativa')
plt.title('Histograma de Frecuencia Relativa Acumulativa')
plt.show()

# Diagrama de Pareto 
mpg_counts = df['mpg'].value_counts()
relative_freq = mpg_counts / len(df)

sorted_mpg = relative_freq.sort_values(ascending=True)
cumulative_freq = sorted_mpg.cumsum()

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(cumulative_freq) + 1), cumulative_freq, color='b', alpha=0.7)
plt.plot(range(1, len(cumulative_freq) + 1), cumulative_freq, color='r', marker='o')
plt.xlabel('Consumo de combustible (mpg)')
plt.ylabel('Frecuencia Relativa Acumulativa')
plt.title('Diagrama de Pareto: Consumo de Combustible (mpg) (Frecuencia Relativa Acumulativa)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Calcular media
media = df['mpg'].mean()
print('Media:', media)

# Calcular varianza
varianza = df['mpg'].var()
print('Varianza:', varianza)

# Calcular desviación estándar
desviacion_estandar = df['mpg'].std()
print('Desviación Estándar:', desviacion_estandar)

# Calcular el coeficiente de correlación de Pearson entre "mpg" y "displacement"
correlation_coef = df['mpg'].corr(df['displacement'])

# Mostrar el coeficiente de correlación
print("Coeficiente de correlación entre mpg y displacement:", correlation_coef)

# Regresion lineal
# Definir las variables dependiente e independiente
y = df['mpg']
X = df['displacement']

# Ajustar el modelo de regresión lineal
model = sm.ols('mpg ~ displacement', data=df).fit()

# Obtener los resultados del modelo
results = model.summary()

# Mostrar los resultados
print(results)

# Gráfica de dispersión de puntos con la línea de regresión
plt.scatter(df['displacement'], df['mpg'], label='Datos')
plt.plot(df['displacement'], model.fittedvalues, color='red', label='Regresión')
plt.xlabel('Displacement')
plt.ylabel('MPG')
plt.legend()
plt.title('Regresión Lineal: MPG vs Displacement')
plt.show()

# Gráfica de residuales
residuals = model.resid
plt.scatter(df['displacement'], residuals)
plt.axhline(y=0, color='red', linestyle='dashed')
plt.xlabel('Displacement')
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
print(model.t_test("Intercept = 1"))

print("Prueba de t para la pendiente:")
print(model.t_test("displacement = 1"))

print("Prueba de F para el modelo:")
print(model.f_test("displacement = 0"))

# Intervalos de confianza para el intercepto y la pendiente
conf_int = model.conf_int(alpha=0.05)
print("Intervalo de confianza para el intercepto y la pendiente:")
print(conf_int)