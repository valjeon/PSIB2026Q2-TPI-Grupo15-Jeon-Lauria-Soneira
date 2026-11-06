# PSIB - Grupo 15: Detección de PCOS en ecografías
Trabajo final de Procesamiento de Señales e Imágenes Biomédicas (ITBA). Pipeline de procesamiento de imágenes, sin aprendizaje automático, que detecta y cuenta folículos en ecografías ováricas, extrae características de los folículos y del estroma y analiza si distinguen las imágenes con SOP de las sanas, separando por fuente de datos para detectar sesgos del dataset.

## Datasets
Los datasets utilizados en el proyecto son los siguientes:
[PCOS detection using ultrasound images](https://www.kaggle.com/datasets/anaghachoudhari/pcos-detection-using-ultrasound-images)
(Kaggle). 
[Polycystic Ovary Ultrasound Images Dataset](https://dataverse.telkomuniversity.ac.id/dataset.xhtml%3Bjsessionid%3Daaebf699ef92d0fd585a6ae61f15?fileAccess=&fileSortField=date&fileTypeGroupFacet=%22Image%22&persistentId=doi%3A10.34820%2FFK2%2FQVCP6V&q=&version=&utm_source=chatgpt.com)
Se tienen que descargar las imágenes y poner la carpeta descomprimida en data/raw. 

## Instalación
    python -m venv venv
    source venv/bin/activate          # en Windows: venv\Scripts\activate
    pip install -r requirements.txt


## Estructura
    ```
    data/raw/                  es donde se guardan los datasets descargados
    data/all/                  se unifican los datasets y se dividen en clases (se genera después de limpieza.py unir) 
    data/clean/                imágenes sin ilegibles, vacías ni duplicados (se genera después de limpieza.py limpiar)
    procesamiento/             contiene los 4 módulos que forman el pipeline
    reports/                   tablas, resúmenes y figuras generados

    Archivos generado:
    rois.csv                   bloques homogéneos de 16×16 px marcados a mano (para el ENL)
    mascaras.csv, mascaras/    folículos dibujados a mano (referencia para el Dice)
    splits.csv, splits2.csv    grupos de imágenes de una misma ecografía y particiones
    excluidas.csv              imágenes excluidas y el motivo
    ```

## Código (`procesamiento/`)

Son cuatro archivos, cada uno tiene comandos (`python procesamiento/<archivo>.py <comando> -h`):

| Archivo | Utilidad | Comandos |
|---|---|---|
| `limpieza.py` |Unifica los datasets, elimina imágenes ilegibles, vacías y duplicadas, audita sesgos de adquisición (tamaño, brillo, marcas) y agrupa las imágenes de una misma ecografía| `unir`, `limpiar`, `auditar`, `splits`, `duplicados` |
| `preprocesamiento.py` |Quita marcas, recorta el área útil, lleva a 256 × 256 px y compara filtros de speckle con el ENL y el índice β.| `comparar`, `anotar-rois`, `elegir-filtro`, `procesar` |
| `segmentacion.py` |Delimita el abanico ecográfico, umbrala los folículos y evalúa las combinaciones de método, filtro y contraste contra las máscaras manuales (Dice, detección por folículo, error del conteo).| `comparar`, `ver-casos`, `anotar-mascaras` |
| `texturas.py` |Calcula las características de folículos y estroma y las compara entre clases por fuente (AUC y test de permutaciones).| (sin comandos)|

## Orden en que se corren
Todos los comandos se corren desde la raíz del repositorio y en este orden.
```
# 1. Limpieza
python procesamiento/limpieza.py unir --extra <carpeta_dataverse> --fuente dv
python procesamiento/limpieza.py limpiar
python procesamiento/limpieza.py auditar
python procesamiento/limpieza.py splits
python procesamiento/limpieza.py duplicados            # genera splits2.csv

# 2. Preprocesamiento y elección del filtro
python procesamiento/preprocesamiento.py comparar
python procesamiento/preprocesamiento.py elegir-filtro # usa rois.csv
python procesamiento/preprocesamiento.py procesar

# 3. Segmentación (compara contra las máscaras manuales)
python procesamiento/segmentacion.py comparar --mascaras mascaras.csv --splits splits2.csv --conjunto ajuste
python procesamiento/segmentacion.py comparar --mascaras mascaras.csv --splits splits2.csv --conjunto reporte \
    --metodos multiotsu --filtros wiener3 --contrastes ninguno

# 4. Características y comparación entre clases
python procesamiento/texturas.py --filtro wiener3 --contraste ninguno

# 5. Figuras
python procesamiento/figuras.py
```

unir sin --extra usa solo el dataset de Kaggle de data/raw/. Si se agrega otro dataset, sus archivos se copian a data/all/<clase>/<fuente>_<nombre>.

### Anotaciones manuales
Si no se quiere utilizar las que ya están en el repo:
```
python procesamiento/preprocesamiento.py anotar-rois     # bloques homogéneos (para el ENL)
python procesamiento/segmentacion.py anotar-mascaras     # contornos de folículos (para el Dice)
````

## Resultados
Los resultados (tablas, resúmenes y figuras) se escriben en `reports/`. Los principales:
| Archivo | Contenido |
|`resumen_limpieza.txt`,`resumen_auditoria.txt`|Limpieza, sesgos de adquisición y grupos de ecografías|
|`enl_resumen_manual.txt`|Ranking de filtros por ENL y β|
|`resumen_dice.txt`, `seg_finalistas.csv`|Evaluación de la segmentación y combinaciones finalistas|
|`caracteristicas_comparacion.csv`, `caracteristicas_resumen.txt`|AUC, p de permutación y sentido de cada característica por fuente|

