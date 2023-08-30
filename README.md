# bayesian-estimation
This repository is for projects solving the practices assigned
at the bayesian estimation class of the Master in Computer Science.

## Practice 1

The objective of this practice is to compare the behavior of
mean and standard deviation when the dataset is known
completely and when the data is arriving dynamically.

Whe the data set is known from the beginning the procedure is 
quite straight forward. 

```
# Mean when the complete dataset is known
full_mean = sum(data) / n
```

```
# Standard Deviation when the complete dataset is known
temp = [(x - full_mean) ** 2 for x in data]
full_stand_des = math.sqrt(sum(temp) / n)
```

Pero cuando los datos van llegando a medida que pasa el tiempo
entonces es necesario utilizar otro método para calcular la media
y la desviación estandar con los datos conocidos hasta el momento.

```
# Formulas that calculates the next mean based on the actual known mean
mean_k_next = (k * known_mean + data[i + 1]) / (k + 1)
```

```
# Formulas that calculates the next variation based on the actual known mean
variation_k_next = ((k * known_variation) + ((data[i + 1] - mean_k_next) ** 2)) / (k + 1)
```

En este caso a medida que llega un nuevo dato se utiliza la media
y la desviación conocidas hasta el momento para calcular los nuevos
valores de media y desviación estándar. Todos los valores encontrados
se almacenan para luego poder graficarlos y saber si los dos métodos 
llegan a la misma solución.



