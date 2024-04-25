# 🎵 Reconocimiento Automatizado de Sonidos de Instrumentos y Más Allá 🎧
Este repositorio trata de resolver el challenge de NUWE -- > https://jobs.nuwe.io/coding-challenges

### Categoría   ➡️   Datos

### Subcategoría   ➡️   Ciencia de Datos

### Dificultad   ➡️   (Avanzado)

## 🌍 Contexto

En nuestro mundo moderno, la música y los sonidos nos rodean en todo momento, desde el suave rasgueo de una guitarra hasta el pegadizo ritmo de un beat contemporáneo. A medida que la tecnología avanza, el procesamiento automatizado de estos sonidos se ha convertido en un desafío crucial. Presentamos a "WaveTech", un líder innovador en el ámbito de la tecnología del sonido, que busca enfrentar el reto de identificar automáticamente esta amplia gama de sonidos.

Con la vasta diversidad de sonidos disponibles, desde instrumentos musicales hasta ruidos ambientales naturales, la clasificación manual se ha vuelto impracticable. WaveTech ha acumulado una rica colección de muestras de audio, pero la identificación automatizada de estos sonidos es la próxima frontera esencial.

Tu papel como investigador de IA es primordial. Al entrenar una red neuronal con estas muestras de audio, puedes ayudar a WaveTech a clasificarlas con precisión, allanando el camino para la próxima generación de herramientas de audio.

![Imagen](https://cdn.nuwe.io/infojobs-data/__images/AIR_AudioClassification.png)

## 🎯 Objetivos

Tu misión es diseñar y entrenar una red neuronal capaz de escuchar y distinguir varios sonidos, que van desde instrumentos como la trompeta, la guitarra, el piano y el violín hasta sonidos naturales y ritmos contemporáneos. Esta clasificación precisa será invaluable para WaveTech en su búsqueda por revolucionar la industria del sonido.

## 📁 Dataset

Se te proporcionarán archivos de audio para entrenar tu modelo. Estos audios representan las diversas categorías que necesitan ser reconocidas:

- 🎺 Trompeta
- 🎸 Guitarra
- 🎹 Piano
- 🎻 Violín
- 🍃 Sonidos naturales
- 🎛️ Beats (ritmos electrónicos)

### Enlaces de Descarga:
- Para el conjunto de datos de entrenamiento: [Descargar train.zip](https://cdn.nuwe.io/NUWE_Data_Hackathon(March2024)/Train&Test/train.zip)
- Para el conjunto de datos de prueba: [Descargar test.zip](https://cdn.nuwe.io/NUWE_Data_Hackathon(March2024)/Train&Test/test.zip)

## 🗄️ Estructura del repositorio:

Se proporciona la estructura del repositorio, que debe respetarse estrictamente:

```
|__README.md
|__requirements.txt
|
|__data
|  |__labels_paths_train.csv
|  |__labels_tests.csv
|
|__src
|  |__data_processing.py
|  |__model_training.py 
|  |__model_prediction.py
|  |__utils.py
|
|__models
|  |__model.pkl
|
|__scripts
|  |__run_pipeline.sh
|
|__predictions
   |__example_predictions.json
   |__predictions.json
```


The `predictions` folder should contain the `predictions.json` file with your model's predicted insect categories.


## 🎯 Tareas:

Diseñar y entrenar una red neuronal para escuchar y distinguir varios sonidos, contribuyendo a los innovadores esfuerzos de WaveTech en la identificación de sonidos.

## 📊 Procesamiento de Datos:

Se debe aplicar un preprocesamiento de datos para normalizar o escalar las variables continuas de los sensores.

## 🤖 Modelo:

Seleccionar y entrenar una red neuronal adecuada que pueda clasificar eficazmente diferentes sonidos. Puedes explorar diversas arquitecturas, como redes neuronales convolucionales (CNNs) o redes neuronales recurrentes (RNNs), para encontrar la solución óptima.

## 📤 Entrega

Enviar un archivo predictions.json que contenga la clasificación de las muestras de audio por el modelo. Asegúrate de que el archivo esté correctamente formateado, con el identificador del archivo de audio como clave y la categoría predicha como valor.
predictions.json:
```json
{
    "target": {
        "1": 0,
        "2": 3,
        "3": 8,
        "4": 5,
        "5": 2,
        "6": 7,
        "7": 4,
        "8": 1,
        "9": 6,
        "10": 3,
        ...
    }
}
```

## 📊 Evaluación

El rendimiento se medirá utilizando el F1 score para evaluar la precisión y la exhaustividad, ofreciendo una visión equilibrada de la precisión y robustez del modelo. Tus predicciones serán rigurosamente probadas con muestras de audio no vistas para determinar el F1 score.

**⚠️ Nota importante**:
Todas las entregas pasarán por un proceso de revisión manual del código para asegurar que el trabajo se haya realizado de manera honesta y se adhiera a los más altos estándares de integridad académica. Cualquier forma de deshonestidad o mala conducta será abordada seriamente y puede llevar a la descalificación del desafío.

## ❓ FAQs

## P1: ¿Cuál es el objetivo del Desafío SonicWave?
R1: El objetivo es desarrollar un modelo que pueda identificar automáticamente varios sonidos, desde instrumentos musicales hasta ritmos naturales y artificiales.

## P2: ¿Con qué tipo de datos trabajaré?
R2: Trabajarás con un conjunto seleccionado de archivos de audio categorizados en instrumentos, sonidos naturales y ritmos.

## P3: ¿Qué arquitecturas de redes neuronales se recomiendan?
R3: Aunque no hay un requisito estricto, las CNNs y RNNs son prometedoras para tareas de clasificación de audio. Sin embargo, se te anima a explorar y seleccionar la arquitectura que ofrezca los mejores resultados.

## P4: ¿Cómo se evaluará el rendimiento del modelo?
R4: El F1 score, considerando tanto la precisión como la exhaustividad, será la métrica principal para evaluar el rendimiento de tu modelo en la tarea de clasificación de sonidos.