import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.stats.anova as anova
import scipy.stats as stats

# Leer el archivo CSV
df = pd.read_csv('Datos/Malls.csv')

# Calcular Histograma
plt.hist(df['GLA'], bins='auto')
plt.xlabel('Área del Centro Comercial')
plt.ylabel('Frecuencia')
plt.title('Histograma de Frecuencia')
plt.show()

# Calcular Histograma de frecuencia relativa
plt.hist(df['GLA'], bins='auto', density=True)
plt.xlabel('Área del Centro Comercial')
plt.ylabel('Frecuencia Relativa')
plt.title('Histograma de Frecuencia Relativa')
plt.show()

# Calcular frecuencia acumulativa
frecuencia_acumulativa = df['GLA'].value_counts().sort_index().cumsum()

# Calcular frecuencia relativa acumulativa
frecuencia_relativa_acumulativa = frecuencia_acumulativa / len(df)

hist_values, hist_edges = np.histogram(df['GLA'], bins='auto')

# Calcular la frecuencia acumulativa
cumulative_freq = np.cumsum(hist_values)
cumulative_rel_freq = cumulative_freq / len(df)


# Gráfico de frecuencia acumulativa en forma de barra
plt.bar(hist_edges[:-1], cumulative_freq, width=np.diff(hist_edges), align='edge')
plt.xlabel('Área del Centro Comercial')
plt.ylabel('Frecuencia Acumulativa')
plt.title('Frecuencia Acumulativa')
plt.show()

# Gráfico de frecuencia relativa acumulativa
plt.bar(hist_edges[:-1], cumulative_rel_freq, width=np.diff(hist_edges), align='edge')
plt.xlabel('Área del Centro Comercial')
plt.ylabel('Frecuencia Relativa Acumulativa')
plt.title('Frecuencia Relativa Acumulativa')
plt.show()

# Diagrama de Pareto 
GLA_counts = df['GLA'].value_counts()
relative_freq = GLA_counts / len(df)

sorted_GLA = relative_freq.sort_values(ascending=True)
cumulative_freq = sorted_GLA.cumsum()

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(cumulative_freq) + 1), cumulative_freq, color='b', alpha=0.7)
plt.plot(range(1, len(cumulative_freq) + 1), cumulative_freq, color='r', marker='o')
plt.xlabel('Área del Centro Comercial (GLA)')
plt.ylabel('Frecuencia Relativa Acumulativa')
plt.title('Diagrama de Pareto: Área del Centro Comercial (GLA) (Frecuencia Relativa Acumulativa)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Calcular media
media = df['GLA'].mean()
print('Media:', media)

# Calcular varianza
varianza = df['GLA'].var()
print('Varianza:', varianza)

# Calcular desviación estándar
desviacion_estandar = df['GLA'].std()
print('Desviación Estándar:', desviacion_estandar)

# Calcular el coeficiente de correlación de Pearson entre "GLA" y "Shops"
correlation_coef = df['GLA'].corr(df['Shops'])

# Mostrar el coeficiente de correlación
print("Coeficiente de correlación entre GLA y Shops:", correlation_coef)

# Regresion lineal
# Definir las variables dependiente e independiente
y = df['Shops']
X = df['GLA']

# Ajustar el modelo de regresión lineal
model = sm.ols('Shops ~ GLA', data=df).fit()

# Obtener los resultados del modelo
results = model.summary()

# Mostrar los resultados
# print(results)

# Gráfica de dispersión de puntos con la línea de regresión
plt.scatter(df['GLA'], df['Shops'], label='Datos')
plt.plot(df['GLA'], model.fittedvalues, color='red', label='Regresión')
plt.xlabel('GLA')
plt.ylabel('Shops')
plt.legend()
plt.title('Regresión Lineal: Shops vs GLA')
plt.show()

# Gráfica de residuales
residuals = model.resid
plt.scatter(df['GLA'], residuals)
plt.axhline(y=0, color='red', linestyle='dashed')
plt.xlabel('GLA')
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
print(model.t_test("GLA = 1"))  # Para la pendiente

print("Prueba de F para el modelo:")
print(model.f_test("GLA = 0"))  # Para el modelo

# Intervalos de confianza para el intercepto y la pendiente
conf_int = model.conf_int(alpha=0.05)
print("Intervalo de confianza para el intercepto y la pendiente:")
print(conf_int)