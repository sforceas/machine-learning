# Curso práctico de Machine Learning

# 1. Definiciones de machine learning

**Aprendizaje supervisado** H
Hace uso un información de entrada y de sus etiquetas correspondientes para cada conjunto de información.
**Aprendizaje no supervisado** 
Solo tiene acceso a información, no se asocia una etiqueta. el algoritmo aprende haciendo asociación de los datos de entrada.
**Aprendizaje profundo** 
Se basa en redes neuronales.

**Ventajas del uso de machine learning**
Se le provee la capacidad a los algoritmos para aprender.
Permite orientar los problemas propuestos.
Permite construir un modelo a partir de datos históricos para poder realizar clasificaciones con nuevos datos.

Ejemplos: 
 - Optimizar rutas (UBER)
 - Clasificación de imagenes (Google Images)
 - Sistemas de recomendación de contenidos (Netflix)
 
 # 2. Fundamentos de Numpy
 
 Numpy es la principal libreria de python para el manejo de arreglos.
 Es sencilla, adecuada para el manejo de arreglos y rapida.
 Ver repositorio de Collab: [Algoritmos_python/ML_Numpy.ipynb](https://github.com/sforceas/machine-learning/blob/master/ML_Numpy.ipynb)

 # 3. Fundamentos de Pandas
 
 Numpy es la principal libreria de python para el manejo de datos y la provisión de datos para machine learning.
 Es muy copmleta, adecuada para el manejo de tablas, conjuntamente con numpy.
 Ver repositorio de Collab: [Algoritmos_python/ML_Numpy.ipynb](https://github.com/sforceas/machine-learning/blob/master/ML_Pandas.ipynb)

# 4. Scikit Learn para ML
Scikit-Learn es una de estas librerías gratuitas para Python. Cuenta con algoritmos de clasificación, regresión, clustering y reducción de dimensionalidad. Además, presenta la compatibilidad con otras librerías de Python como NumPy, SciPy y matplotlib.

La gran variedad de algoritmos y utilidades de Scikit-learn la convierten en la herramienta básica para empezar a programar y estructurar los sistemas de análisis datos y modelado estadístico. Los algoritmos de Scikit-Learn se combinan y depuran con otras estructuras de datos y aplicaciones externas como Pandas o PyBrain.

La ventaja de la programación en Python, y Scikit-Learn en concreto, es la variedad de módulos y algoritmos que facilitan el aprendizaje y trabajo del científico de datos en las primeras fases de su desarrollo. La formación de un Máster en Data Science hace hincapié en estas ventajas, pero también prepara a sus alumnos para trabajar en otros lenguajes. La versatilidad y formación es la clave en el campo tecnológico.

**Comandos basicos par SK Learn

* Importar biblioteca:
```from sklearn import [modulo]```

* División del conjunto de datos para entrenamiento y pruebas:
```X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)```

* Entrenar modelo:
```[modelo].fit(X_train, y_train)```

* Predicción del modelo:
```Y_pred = [modelo].predict(X_test)```

* Matriz de confusión:
```metrics.confusion_matrix(y_test, y_pred)```

* Calcular la exactitud:
```metrics.accuracy_score(y_test, y_pred)```

## 4.1. Predicción de datos

El análisis predictivo agrupa una variedad de técnicas estadísticas de modelización, aprendizaje automático y minería de datos que analiza los datos actuales e históricos reales para hacer predicciones acerca del futuro o acontecimientos no conocidos.

Existen algoritmos que se definen como “clasificadores” y que identifican a qué conjunto de categorías pertenecen los datos.
Para entrenar estos algoritmos:

Es importante comprender el problema que se quiere solucionar o que es lo que se quiere aplicar.
Obtener un conjunto de datos para entrenar el modelo.
Cuando entrenamos un modelo para llevar a cabo una prueba, es importante cuidar la información que se le suministra, es decir, debemos verificar si existen datos no validos o nulos, si las series de datos esta completa, etc.

Podemos entrenar un modelo con la información historica de cierta cantidad de estudiantes y sus calificaciones en diferentes cursos, un modelo bien entrenado con estos datos debería ser capas de hacer una predicción de que tan bien le irá a un estudiante nuevo en algun tipo de curso al evaluar sus carácteristicas.

**Sobreajuste o subajuste**
Sobreajunte (overfiting): Es cuando intentamos obligar a nuestro algoritmo a que se ajuste demasiado a todos los datos posibles. Es muy importante proveer con información abundante a nuestro modelo pero también esta debe ser lo suficientemente variada para que nuestro algoritmo pueda generalizar lo aprendido.

Subajuste (underfiting): Es cuando le suministramo a nuestro modelo un conjunto de datos es muy pequeño, en este caso nuestro modelo no sera capas de aprender lo suficiente ya que tiene muy poca infomación. La recomendación cuando se tienen muy pocos datos es usar el 70% de los datos para que el algoritmo aprenda y usar el resto para entrenamiento.

![](https://i0.wp.com/www.aprendemachinelearning.com/wp-content/uploads/2017/12/overfitting-underfitting-machine-learning.png?w=800&ssl=1)

## 4.2. Regresión lineal con Scikit Learn

El algoritmo de regresión lineal nos ayuda a conseguir tendencia en los datos, este es un algoritmo de tipo supervisado ya que debemos de usar datos previamente etiquetados.

En la regresión lineal generamos, a partir de los datos, una recta y es a partir de esta que podremos encontrar la tendencia o predicción.
Generalmente es importante tener en cuenta varias dimensiones o variables al considerar los datos que estamos suministrando al modelo, recordando siempre cuidar este set de sobreajuste o subajuste.

Cuando nuestro modelo considera más de dos variables el algoritmo de regresión que usamos se conoce como Regresión Lineal Múltiple y este trabaja sobre un sistema de referencia conocido como hiperplano. Los algoritmos de regresión, tanto lineal como múltiple trabajan únicamente con datos de tipo cuantitativos.

![](https://www.youtube.com/watch?v=k964_uNn3l0)
