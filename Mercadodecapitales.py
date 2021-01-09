print("hello")
import yfinance as yf
import pandas as pd 
import statsmodels.api as sm

from matplotlib import pyplot as plot 
from matplotlib import pyplot as plt
import numpy as np

# precios = yf.download('FB',period ='6mo' )
# intervalo 
precios = yf.download('FB', period ='6mo')
# descripcion de datos
precios.info()
# grafica 
plot = precios['Adj Close'].plot(figsize=(10, 8))

# Aplicando el filtro Hodrick-Prescott para separar en tendencia y 
# componente ciclico.
precios_ciclo, precios_tend = sm.tsa.filters.hpfilter(precios['Adj Close'])
precios['tend'] = precios_tend

# graficando la variacion del precio real con la tendencia.
precios[['Adj Close', 'tend']].plot(figsize=(10, 8), fontsize=12);
legend = plt.legend()
legend.prop.set_size(14)

# graficando rendimiento diario
variacion_diaria = precios['Adj Close'] / precios['Adj Close'].shift(1) - 1
precios['var_diaria'] = variacion_diaria
precios['var_diaria'][:5]
plot = precios['var_diaria'].plot(figsize=(10, 8))
# Pronostico 
# Modelo ARIMA sobre el valor de cierre de la acción.
modelo = sm.tsa.ARIMA(precios['Adj Close'].iloc[1:], order=(1, 0, 0))  
resultados = modelo.fit(disp=-1)  
precios['pronostico'] = resultados.fittedvalues  
plot = precios[['Adj Close', 'pronostico']].plot(figsize=(10, 8))

# histograma 
plt.hist(precios['Close'],20,color = "blue")

# Estadisticas descriptiva 
plt.boxplot(precios['Close'])
# Descomposion de la serie
descomposicion = sm.tsa.seasonal_decompose(precios['Adj Close'],
                                                  model='additive', freq=30)  
fig = descomposicion.plot()
