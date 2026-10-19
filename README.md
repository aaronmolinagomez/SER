# SER - Speech Emotion Recognition (Tarea 2)

Repositorio oficial del codigo y scripts de entrenamiento para el modelo final de Reconocimiento de Emociones en la Voz (Speech Emotion Recognition), basado en la combinacion de representaciones auto-supervisadas WavLM y arquitecturas TDNN (Time-Delay Neural Networks).

## Estructura del Repositorio

* train.py: Script principal para el entrenamiento de los modelos de clasificacion de emociones.
* prepare_features.py: Script encargado de la extraccion, procesamiento previo y almacenamiento en cache de caracteristicas acusticas (como embeddings de WavLM).
* ensemble.py: Script para combinar (ensamblar) las predicciones de multiples modelos y mejorar el rendimiento de inferencia.
* ser_lib.py: Libreria de funciones auxiliares, manejo de datasets, bloques de red y utilidades generales.

## Detalles del Entrenamiento (WavLM + TDNN)

El pipeline de procesamiento y entrenamiento implementa los siguientes componentes clave:
* Extraccion de Caracteristicas: Uso de capas congeladas y ajustadas del modelo WavLM para capturar representaciones foneticas y acusticas ricas de las señales de voz.
* Modelado Temporal: Arquitectura basada en TDNN (estilo x-vectors) para modelar dinamicas temporales de largo alcance sobre los embeddings extraidos.
* Estrategia de Cache: Los directorios emociones_cache* almacenan las caracteristicas pre-calculadas para acelerar de forma drastica las iteraciones de entrenamiento y validacion.
* Ensembling: Post-procesamiento de probabilidades mediante tecnicas de ensamble (ensemble_test_preds.npy y su version calibrada) para robustecer la decision final.
