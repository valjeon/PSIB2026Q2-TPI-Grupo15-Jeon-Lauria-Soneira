# PSIB - Grupo 15: Detección de PCOS en ecografías

## Datasets
Los datasets utilizados en el proyecto son los siguientes:
[PCOS detection using ultrasound images](https://www.kaggle.com/datasets/anaghachoudhari/pcos-detection-using-ultrasound-images)
(Kaggle). 
[Polycystic Ovary Ultrasound Images Dataset](https://dataverse.telkomuniversity.ac.id/dataset.xhtml%3Bjsessionid%3Daaebf699ef92d0fd585a6ae61f15?fileAccess=&fileSortField=date&fileTypeGroupFacet=%22Image%22&persistentId=doi%3A10.34820%2FFK2%2FQVCP6V&q=&version=&utm_source=chatgpt.com)
Se tienen que descargar las imágenes y poner la carpeta descomprimida en data/raw. 

## Estructura

- `data/raw/`: dataset original, tal cual se descargó
- `data/all/`: train y test unificados (se genera con el script)
- `procesamiento`

## Código (`procesamiento/`)

Cuatro archivos, cada uno con comandos (`python procesamiento/<archivo>.py <comando> -h`):

| Archivo | Comandos |
|---|---|
| `limpieza.py` | `unir`, `limpiar`, `auditar`, `splits`, `duplicados` |
| `preprocesamiento.py` | `comparar`, `anotar-rois`, `elegir-filtro`, `procesar` |
| `segmentacion.py` | `comparar`, `ver-casos`, `anotar-mascaras` |
| `texturas.py` | (sin comandos) características de folículos y estroma |

## Orden en que se corren

```
python procesamiento/limpieza.py unir     # con otro dataset: --extra data/raw_dataverse --fuente dv
python procesamiento/limpieza.py limpiar
python procesamiento/limpieza.py auditar
python procesamiento/limpieza.py splits
python procesamiento/limpieza.py duplicados
python procesamiento/preprocesamiento.py comparar
python procesamiento/preprocesamiento.py procesar
python procesamiento/segmentacion.py comparar --splits splits2.csv
python procesamiento/texturas.py --imagenes 3
```