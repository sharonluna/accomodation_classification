# Prueba Técnica

Author: Sharon Trejo

Este proyecto está enfocado en el Análisis Exploratorio de Datos (EDA) y el desarrollo de modelos. Incluye un notebook de Jupyter para realizar el análisis de datos y el entrenamiento de modelos, junto con un script en Python que contiene funciones auxiliares utilizadas a lo largo del notebook.

# Versión Python
`3.13.0`

## Estructura del Proyecto
```js
project-root/ 
├── pycache/ 
├── input/ 
├── requirements.txt
├── exercise.ipynb 
└── funcs.py
```


- `__pycache__/`: Directorio para archivos cacheados de Python, generado automáticamente.
- `requirements.txt`: Librerias para instalar 
- `input/`: Directorio para almacenar archivos de datos de entrada para el análisis y modelado.
- `exercise.ipynb`: Notebook de Jupyter que contiene el código principal para realizar el EDA y el desarrollo del modelo.
- `funcs.py`: Script en Python con funciones auxiliares utilizadas en `exercise.ipynb` para optimizar los procesos de análisis y modelado.

# Instala las dependencias:

Instala los paquetes necesarios ejecutando:
```bash
pip install -r requirements.txt
```


# Uso
`exercise.ipynb`: Este notebook contiene todos los pasos para el análisis de datos y el desarrollo del modelo, incluyendo:

* Carga y Limpieza de Datos
* Análisis Exploratorio de Datos (EDA)
* Entrenamiento y Evaluación del Modelo

`funcs.py`: Contiene funciones auxiliares utilizadas en `exercise.ipynb` para hacer que el análisis sea más modular y organizado. 

## Requisitos
* Python 3.8 o superior
* Jupyter Notebook
* Paquetes de Python necesarios listados en requirements.txt
