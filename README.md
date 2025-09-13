**Clasificación de Imágenes: Perros vs Gatos con Red Neuronal Convolucional (CNN) , Red Neuronal Completamente Conectada (FCNN) en Keras/TensorFlow**
- Descripción del Proyecto
Este proyecto implementa una solución de clasificación binaria de imágenes utilizando una Red Neuronal Convolucional (CNN) construida con la API funcional de Keras, sobre el backend de TensorFlow. El objetivo es clasificar imágenes de perros y gatos a partir de un dataset público. El proyecto aborda las etapas clave del ciclo de vida de un modelo de Machine Learning en visión artificial: preparación de datos, definición de arquitectura, entrenamiento, evaluación y mejora del modelo.

Se destacan las técnicas de preprocesamiento de datos y aumento de datos (Data Augmentation) para mejorar el rendimiento y la robustez del modelo, así como la superación de desafíos comunes en el manejo de datasets de imágenes del mundo real, como la presencia de archivos corruptos o con formatos inconsistentes.

- Objetivos
Implementar una Red Neuronal Convolucional para clasificación binaria.
Preparar y limpiar un conjunto de datos de imágenes a gran escala.
Aplicar técnicas de Aumento de Datos para regularización y mejora del rendimiento.
Entrenar y evaluar el modelo utilizando los conjuntos de datos de entrenamiento, validación y prueba.
Analizar y mejorar los resultados obtenidos mediante la iteración en la arquitectura del modelo y la configuración del entrenamiento.
- Dataset
El dataset utilizado es el famoso "Cats vs Dogs" de Kaggle, que contiene miles de imágenes de perros y gatos. El acceso al dataset se realiza mediante descarga directa y posterior almacenamiento en Google Drive para persistencia.

Ruta del Dataset: /content/drive/MyDrive/Red_Neuronal_Artificial_modelo/PetImages

El dataset original contiene imágenes en diversos formatos y con posibles inconsistencias que requieren limpieza.

- Preparación y Limpieza de Datos
Una de las etapas cruciales y desafiantes de este proyecto fue la preparación y limpieza del dataset. Los problemas encontrados incluyeron:

Archivos con formatos incorrectos o corruptos: Algunas imágenes no eran archivos JPEG, PNG, GIF o BMP válidos, o estaban incompletas/dañadas.
Imágenes con número de canales inesperado: Se identificaron imágenes con 2 canales de color, lo cual no es un formato estándar (se esperan 1, 3 o 4 canales).

Superación de Inconvenientes:
Para abordar estos problemas, se implementó un proceso de limpieza manual robusto:

Se iteró sobre cada archivo en las carpetas 'Cat' y 'Dog'.
Se intentó leer y decodificar cada imagen utilizando tf.io.read_file y tf.image.decode_image.
Si la decodificación fallaba debido a un formato inválido o si la imagen decodificada presentaba un número de canales diferente a 1, 3 o 4, el archivo se identificaba como corrupto o inválido.
Los archivos inválidos fueron eliminados físicamente del sistema de archivos utilizando os.remove().
Este paso de limpieza fue fundamental para asegurar que los conjuntos de datos cargados por keras.utils.image_dataset_from_directory solo contuvieran imágenes válidas, evitando errores durante las etapas de entrenamiento y evaluación.

Posteriormente, el dataset limpio se dividió en conjuntos de entrenamiento (aprox. 80%), validación (aprox. 10%) y prueba (aprox. 10%) utilizando image_dataset_from_directory y la manipulación de los objetos tf.data.Dataset.

Aumento de Datos (Data Augmentation)
Para aumentar la diversidad del conjunto de entrenamiento y mejorar la capacidad de generalización del modelo, se aplicaron técnicas de Aumento de Datos al conjunto de entrenamiento. Se utilizaron las siguientes transformaciones aleatorias:

Volteo Horizontal Aleatorio (layers.RandomFlip("horizontal"))
Rotación Aleatoria (layers.RandomRotation(0.1))
Estas transformaciones se aplicaron al dataset de entrenamiento utilizando el método .map() de tf.data.Dataset con paralelización (num_parallel_calls=tf.data.AUTOTUNE), lo que permite generar variaciones de las imágenes "sobre la marcha" durante el entrenamiento sin necesidad de almacenar todas las imágenes aumentadas en disco.

- **Arquitectura del Modelo**
Inicialmente, se exploró una Red Neuronal Completamente Conectada (FCNN), pero debido a la naturaleza de los datos de imagen y la alta dimensionalidad de los datos aplanados, no mostró un rendimiento óptimo.

Para mejorar los resultados, se adoptó una arquitectura de Red Neuronal Convolucional (CNN) inspirada en principios de arquitecturas eficientes para clasificación de imágenes. Se definió un modelo utilizando la API Funcional de Keras, lo que permite una mayor flexibilidad en la construcción de modelos complejos.

- La arquitectura implementada incluye:

Capa de entrada con las dimensiones esperadas de la imagen (180x180x3).
Capa de reescalado para normalizar los valores de píxel entre 0 y 1.
Bloques convolucionales con capas Conv2D, SeparableConv2D, BatchNormalization y Activation('relu').
Capas de MaxPooling2D para reducción de dimensionalidad.
Conexiones residuales (layers.add) inspiradas en arquitecturas avanzadas como ResNet o Xception, para facilitar el flujo de gradientes y entrenar redes más profundas.
Capa GlobalAveragePooling2D para aplanar las características antes de la capa densa final.
Capa Dropout para regularización y prevención del overfitting.
Capa densa de salida con activación sigmoid para la clasificación binaria.
Esta arquitectura convolucional es mucho más adecuada para extraer características espaciales de las imágenes en comparación con una FCNN simple.

- **Configuración y Entrenamiento**
El modelo se compiló con la siguiente configuración:

Función de Pérdida (loss): binary_crossentropy, adecuada para problemas de clasificación binaria.
Optimizador (optimizer): Adam con una tasa de aprendizaje inicial de 1e-3. Adam es un optimizador popular y eficiente que adapta dinámicamente la tasa de aprendizaje.
Métricas (metrics): accuracy, para monitorear la proporción de predicciones correctas durante el entrenamiento y la evaluación.
El entrenamiento se realizó utilizando el método model.fit(), especificando el conjunto de entrenamiento aumentado (train_ds), el número de épocas y el conjunto de validación (val_ds) para monitorear el rendimiento en datos no vistos durante cada época.

Evaluación y Resultados
El modelo entrenado se evaluó en el conjunto de datos de prueba (test_ds), que contiene imágenes que el modelo no ha visto en absoluto durante el entrenamiento o la validación.

Los resultados de la evaluación (pérdida y precisión) proporcionan una estimación del rendimiento del modelo en datos nuevos y no sesgados.

Se realizó una visualización de las predicciones en un lote de imágenes del conjunto de prueba, mostrando la probabilidad predicha para cada clase ('Cat' y 'Dog'), lo que permite una inspección cualitativa del comportamiento del modelo.

Tecnologías Utilizadas
Python 3.x
Google Colab: Entorno de desarrollo basado en la nube con acceso a GPUs (GPU T4 utilizada en este proyecto).
TensorFlow: Framework de código abierto para Machine Learning.
Keras: API de alto nivel para construir y entrenar modelos de Machine Learning, utilizada con el backend de TensorFlow. (Se usó keras-core y la integración from tensorflow import keras).
NumPy: Librería fundamental para computación numérica en Python.
Matplotlib: Librería para la creación de visualizaciones estáticas, interactivas y animadas en Python.
os: Módulo para interactuar con el sistema operativo, utilizado en la limpieza de archivos.
Cómo Ejecutar el Proyecto
Clonar este repositorio de GitHub en tu máquina local o abrir el notebook directamente en Google Colab.
Si usas Colab, montar Google Drive para acceder a la ruta donde se almacenará el dataset.
Ejecutar las celdas del notebook secuencialmente:
Instalación de keras-core.
Verificación del uso de GPU.
Descarga y descompresión del dataset en la ruta especificada en Google Drive.
Ejecutar la función de limpieza de datos para eliminar archivos inválidos.
Cargar y dividir el dataset limpio en conjuntos de entrenamiento, validación y prueba utilizando image_dataset_from_directory.
Visualizar ejemplos del dataset.
Definir y aplicar la secuencia de Aumento de Datos.
Definir la arquitectura del modelo CNN utilizando la API funcional de Keras.
Compilar el modelo.
Entrenar el modelo con el conjunto de entrenamiento aumentado y el conjunto de validación.
Guardar el modelo entrenado.
Cargar el modelo (opcional, para verificación).
Evaluar el modelo con el conjunto de prueba.
Visualizar predicciones en ejemplos de prueba.
Conclusiones y Próximos Pasos
Este proyecto demuestra un pipeline completo para la clasificación de imágenes, destacando la importancia de la preparación de datos robusta y el uso de arquitecturas de CNN adecuadas. La superación del desafío de los archivos corruptos y con canales inconsistentes fue clave para el éxito del entrenamiento.

Posibles mejoras futuras incluyen:

Experimentar con diferentes arquitecturas de CNN (por ejemplo, modelos pre-entrenados como VGG, ResNet o Xception con Transfer Learning).
Ajustar los hiperparámetros del modelo y el proceso de entrenamiento (tasa de aprendizaje, número de épocas, tamaño de lote, etc.).
Implementar técnicas de regularización adicionales.
Explorar otras técnicas de Aumento de Datos.
Realizar un análisis más detallado de los errores del modelo (por ejemplo, matriz de confusión).
Este proyecto sirve como una base sólida para futuros trabajos en tareas de visión artificial y demuestra la capacidad de manejar desafíos prácticos en datasets del mundo real.
