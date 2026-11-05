"""Limpieza y partición del dataset (módulo de A).

    python procesamiento/limpieza.py unir         # data/raw (train+test de Kaggle) [+ otro dataset] -> data/all
    python procesamiento/limpieza.py limpiar      # data/all -> data/clean (ilegibles, vacías, duplicadas)
    python procesamiento/limpieza.py auditar      # tamaños, marcas y control de atajos por metadatos
    python procesamiento/limpieza.py splits       # grupos pHash + hold-out + folds -> splits.csv
    python procesamiento/limpieza.py duplicados   # grupos ORB+RANSAC (rotadas/recortadas) -> splits2.csv

Orden: unir -> limpiar -> auditar -> splits -> duplicados.
"""
from collections import Counter, defaultdict, namedtuple
import argparse
import csv
import random
import shutil
import statistics as st
import sys
import time
import cv2
import imagehash
import numpy as np
from PIL import Image, ImageStat
from sklearn.model_selection import StratifiedGroupKFold #importa de la librería machine learning pero no se utiliza con ese proposito, solo particiona en folds y hold-out
from scipy import stats
from preprocesamiento import a_gris, leer, recortar_util
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if ROOT.name == "procesamiento":  # funciona dentro de procesamiento/ o en la raíz del repo
    ROOT = ROOT.parent
CLEAN = ROOT / "data" / "clean"
REPORTES = ROOT / "reports"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Unimos los datasets en data/all/ para que limpieza.py pueda procesarlos todos juntos.
# Junta todo en una sola carpeta por clase, sin la división train/test
RAW = ROOT / "data" / "raw"
ALL = ROOT / "data" / "all"


# Creamos función que lee los argumentos y unifique las carpetas de datasets en data/all/
def cmd_unir(argv=None):
    ap = argparse.ArgumentParser(description="Une los datasets en data/all/") 
    ap.add_argument("--extra", default=None, help="carpeta de otro dataset (imágenes mezcladas)") 
    ap.add_argument("--fuente", default=None, help="prefijo corto de esa fuente, sin guiones bajos (ej. dv)")
    ap.add_argument("--prefijo-pos", default="pco", help="los nombres que empiezan así son PCOS (infected)")
    ap.add_argument("--prefijo-neg", default="normal", help="los nombres que empiezan así son sanas (notinfected)")
    ap.add_argument("--sin-kaggle", action="store_true", help="solo sumar --extra, sin tocar lo de Kaggle")
    args = ap.parse_args(argv)
    if args.extra and (not args.fuente or "_" in args.fuente):
        raise SystemExit("Con --extra hace falta --fuente (un prefijo corto sin guiones bajos, ej. dv).")
    if args.sin_kaggle and not args.extra:
        raise SystemExit("--sin-kaggle solo tiene sentido con --extra.")

    cuenta = Counter() # con un Counter imprime cuántas imágenes aportó cada fuente a cada clase. Si no copió nada, corta con error.
    if not args.sin_kaggle:
        if not RAW.exists():
            raise SystemExit(f"No existe {RAW}. Poné ahí el dataset de Kaggle (train/ y test/).")
        for split in ("train", "test"):
            for cls_dir in sorted((RAW / split).iterdir()) if (RAW / split).exists() else []:
                if not cls_dir.is_dir():
                    continue
                out = ALL / cls_dir.name
                out.mkdir(parents=True, exist_ok=True)
                for img in sorted(cls_dir.iterdir()):
                    if img.is_file() and img.suffix.lower() in EXT:
                        shutil.copy2(img, out / f"{split}_{img.name}")
                        cuenta[(f"kaggle/{split}", cls_dir.name)] += 1
    if args.extra:
        carpeta = Path(args.extra)
        carpeta = carpeta if carpeta.is_absolute() else ROOT / carpeta
        if not carpeta.exists():
            raise SystemExit(f"No existe {carpeta}.")
        sin_clase = []
        for img in sorted(carpeta.rglob("*")):
            if not img.is_file() or img.suffix.lower() not in EXT:
                continue
            nombre = img.name.lower()
            if nombre.startswith(args.prefijo_pos.lower()):
                clase = "infected"
            elif nombre.startswith(args.prefijo_neg.lower()):
                clase = "notinfected"
            else:
                sin_clase.append(img.name)
                continue
            (ALL / clase).mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, ALL / clase / f"{args.fuente}_{img.name}")
            cuenta[(args.fuente, clase)] += 1
        if sin_clase:
            print(f"Sin clase (no empiezan con '{args.prefijo_pos}' ni '{args.prefijo_neg}'): "
                  f"{len(sin_clase)}, ej. {sin_clase[:3]}")
    if not cuenta:
        raise SystemExit("No se copió ninguna imagen: revisá las carpetas de origen.")
    for (fte, cl), n in sorted(cuenta.items()):
        print(f"  {fte:14s}{cl:13s}{n:6d}")
    print(f"Total en {ALL}: {sum(1 for _ in ALL.rglob('*') if _.is_file())} archivos")



SRC = ROOT / "data" / "all"
DST = ROOT / "data" / "clean"
UMBRAL = 4      # distancia de Hamming máxima entre pHash para considerar "duplicada"
STD_MIN = 2.0   # desvío estándar de grises por debajo del cual la imagen es casi vacía

Img = namedtuple("Img", "area clase ruta hash size mode")
# es una tupla con nombre (área, clase, ruta, hash, tamaño, modo) para no manejar diccionarios


# función que cuenta cuántos bits difieren entre dos hashes
# Es la distancia de Hamming. Cuanto más baja, más parecidas son las imágenes.
def distancia(a, b):
    return bin(a ^ b).count("1")


# Toma lo que está antes del primer _ y devuelve train o test
def origen(nombre):
    pref = nombre.split("_", 1)[0]
    return pref if pref in ("train", "test") else "?"


# Arma el diccionario de una fila de excluidas.csv. 
def fila(clase, archivo, motivo, grupo="", dist="", o_grupo=""):
    return {"clase": clase, "archivo": archivo, "motivo": motivo, "grupo": grupo,
            "distancia": dist, "origen": origen(archivo), "origen_grupo": o_grupo}


def cmd_limpiar(argv=None):
    excluidas = []
    candidatas = []

    # Primero realizamos una lectura del archivo
    for clase_dir in sorted(SRC.iterdir()):
        if not clase_dir.is_dir():
            continue
        for ruta in sorted(clase_dir.iterdir()):
            try:
                with Image.open(ruta) as img:
                    img.load()  # abre cada imagen y fuerza la lectura completa, así detecta archivos truncados
                    gris = img.convert("L")
                    desvio = ImageStat.Stat(gris).stddev[0]
                    h = int(str(imagehash.phash(img)), 16)
                    size, mode = img.size, img.mode
            except Exception:
                excluidas.append(fila(clase_dir.name, ruta.name, "no se pudo abrir")) 
                # si no se puede abribr la imagen se excluye
                continue
            if desvio < STD_MIN:
                excluidas.append(fila(clase_dir.name, ruta.name, "casi vacía"))
                # si el desvío de grises es menor que STD_MIN se considera casi vacía y también se excluye
                continue
            candidatas.append(Img(size[0] * size[1], clase_dir.name, ruta, h, size, mode))
            # si pasa calcula su pHash y la guarda como candidata.

    # Agrupamos de mayor a menor la resolución de cada imagen
    # se asigna al representante más cercano si está a <= UMBRAL
    # sino se crea un grupo nuevo
    candidatas.sort(key=lambda c: (-c.area, c.ruta.name))
    grupos = []  # cada grupo: [representante, [(miembro, distancia), ...]]
    for c in candidatas:
        mejor, d_mejor = None, UMBRAL + 1 # el representante de cada grupo es la copia de mayor resolución.
        for g in grupos:
            d = distancia(g[0].hash, c.hash)
            if d < d_mejor:
                mejor, d_mejor = g, d
        if mejor is None:
            grupos.append([c, []])
        else:
            mejor[1].append((c, d_mejor))

    # Decisión por grupo
    conservadas = []
    n_conflictos = 0
    for rep, miembros in grupos:
        clases = {rep.clase} | {m.clase for m, _ in miembros}
        if len(clases) > 1:
            # Si un grupo tiene imágenes de dos clases, es la misma imagen con etiquetas contradictorias
            # no se sabe cual es correcta, se excluyen todas y se reporta el conflicto
            n_conflictos += 1
            excluidas.append(fila(rep.clase, rep.ruta.name, "conflicto de clase",
                                  rep.ruta.name, 0, origen(rep.ruta.name)))
            for m, d in miembros:
                excluidas.append(fila(m.clase, m.ruta.name, "conflicto de clase",
                                      rep.ruta.name, d, origen(rep.ruta.name)))
            continue
        # si no hay conflicto de clase se conserva el representante 
        # los demás miembros se excluyen como duplicados
        conservadas.append(rep)
        for m, d in miembros:
            excluidas.append(fila(m.clase, m.ruta.name, "duplicada",
                                  rep.ruta.name, d, origen(rep.ruta.name)))

    # borra data/clean y lo regenera desde cero, para que no queden restos de una corrida anterior
    if DST.exists():
        shutil.rmtree(DST)
    for c in conservadas: 
        (DST / c.clase).mkdir(parents=True, exist_ok=True)
        shutil.copy2(c.ruta, DST / c.clase / c.ruta.name)
    # copia solo las conservadas y guarda excluidas.csv
    REPORTES.mkdir(exist_ok=True)
    campos = ["clase", "archivo", "motivo", "grupo", "distancia", "origen", "origen_grupo"]
    with open(REPORTES / "excluidas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(excluidas)

    # se realiza un resumen que cuenta:
    # las exclusiones por motivo
    # los duplicados según origen (cuántos son copias train–test, por ejemplo)
    # las imágenes por clase
    # los modos de color 
    # los tamaños más frecuentes
    motivos = Counter(e["motivo"] for e in excluidas)
    dup_origen = Counter("-".join(sorted((e["origen"], e["origen_grupo"])))
                         for e in excluidas if e["motivo"] == "duplicada")
    total = len(candidatas) + motivos["no se pudo abrir"] + motivos["casi vacía"]

    lineas = [
        f"Umbral pHash: {UMBRAL} | desvío mínimo: {STD_MIN}",
        f"Imágenes en {SRC}: {total}",
        f"Excluidas: {len(excluidas)}",
        f"  no se pudo abrir: {motivos['no se pudo abrir']}",
        f"  casi vacías: {motivos['casi vacía']}",
        f"  duplicadas: {motivos['duplicada']}",
        "    por origen (copia - original conservado):",
        *[f"      {k}: {v}" for k, v in sorted(dup_origen.items())],
        f"  conflicto de clase: {motivos['conflicto de clase']} imágenes en {n_conflictos} grupos",
        f"Conservadas: {len(conservadas)}",
        *[f"  {cl}: {n}" for cl, n in sorted(Counter(c.clase for c in conservadas).items())],
        "  por origen: " + ", ".join(f"{k}={v}" for k, v in
                                      sorted(Counter(origen(c.ruta.name) for c in conservadas).items())),
        "Modos de color: " + ", ".join(f"{k}={v}" for k, v in
                                        Counter(c.mode for c in conservadas).most_common()),
        "Tamaños más frecuentes: " + ", ".join(f"{s[0]}x{s[1]}={v}" for s, v in
                                                Counter(c.size for c in conservadas).most_common(5)),
        f"Tamaños distintos: {len({c.size for c in conservadas})}",
    ]
    # Imprime todo y lo guarda en resumen_limpieza.txt.
    texto = "\n".join(lineas)
    print(texto)
    (REPORTES / "resumen_limpieza.txt").write_text(texto + "\n", encoding="utf-8")


# función que calcula los metadatos de una imagen
def medir(ruta, sin_color=False):
    with Image.open(ruta) as im: # ancho, alto, modo de color y peso en bytes
        ancho, alto = im.size
        modo = im.mode
        gris = np.asarray(im.convert("L"), dtype=np.float32)
        rgb = np.asarray(im.convert("RGB"), dtype=np.int16)
    color = (rgb.max(axis=2) - rgb.min(axis=2)) > 40  # una máscara de píxeles con color
    # aquellos donde la diferencia entre el canal R, G o B más alto y el más bajo supera 4
    # el umbral de 40 deja pasar el ruido normal de una imagen gris y marca solo lo que tiene color de verdad
    # sirve para sacar el color en caso de que lo haya (anotaciones amarillas, felchas, marcas de escala, etc.)
    g = gris[~color] if (sin_color and (~color).any()) else gris.ravel()
    return {
        "path": ruta.relative_to(ROOT).as_posix(),
        "clase": ruta.parent.name,
        "origen": origen(ruta.name),
        "ancho": ancho,
        "alto": alto,
        "modo": modo,
        "bytes": ruta.stat().st_size,
        "brillo": float(g.mean()), # brillo medio y desvío.
        "desvio": float(g.std()),
        "frac_sat": float((g >= 250).mean()), # fracción de píxeles casi blancos (≥ 250)
        # suelen ser texto, calibres o escalas
        "frac_color": float(color.mean()), # fracción de píxeles de color 
        # píxeles con color marcado: anotaciones amarillas/verdes sobre fondo gris
    }


# función que achica la imagen sin deformarla y la centra sobre un cuadrado gris oscuro.
def miniatura(ruta, lado):
    with Image.open(ruta) as im:
        im = im.convert("RGB")
        im.thumbnail((lado, lado))
        tile = Image.new("RGB", (lado, lado), (40, 40, 40))
        tile.paste(im, ((lado - im.width) // 2, (lado - im.height) // 2))
    return tile


# función que pega varias miniaturas en una grilla y la guarda como JPG para revisarla a ojo
# arma una sola imagen grande con muchas imágenes chiquitas en grilla, como una hoja de contactos de fotos
# en lugar de abrir 50 ecografías una por una, abrís un JPG y las ves todas juntas
def montaje(rutas, destino, cols=8, lado=180): 
    # ruta: lista de imágenes a mostrar
    # destino: dónde se guarda el JPG resultante
    # cols: cuantas columnas (8)
    # lado: tamaño de cada cuadradito en pixeles (180)
    if not rutas: # Si no hay imágenes, no hace nada.
        return
    filas = -(-len(rutas) // cols) 
    hoja = Image.new("RGB", (cols * lado, filas * lado), (40, 40, 40)) 
    # Calcula cuántas filas hacen falta y crea una imagen nueva del tamaño adecuado, con fondo gris oscuro
    for k, r in enumerate(rutas):
        hoja.paste(miniatura(r, lado), ((k % cols) * lado, (k // cols) * lado))
    hoja.save(destino, quality=90)


# funcion que calcula el AUC de Mann-Whitney para una sola variable, sin entrenar ningún modelo
# mide cuánto separa las clases una sola variable
# es la probabilidad de que un caso positivo al azar tenga un valor mayor que uno negativo
# 0,5 = no separa, 1 = separa perfecto
def auc_mann_whitney(x, y):
    x = np.asarray(x, dtype=float) # ordena todos los valores y les asigna rangos 
    y = np.asarray(y, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    rangos = stats.rankdata(x) # los empates comparten rango
    u = rangos[y].sum() - n1 * (n1 + 1) / 2 # con la suma de rangos de los positivos calcula el estadístico U
    a = u / (n1 * n0) # es la probabilidad de que un positivo al azar tenga un valor mayor que un negativo al azar
    return float(max(a, 1 - a)) # lo simetriza


# función que arma una matriz con 8 metadatos
# ancho, alto, aspecto, bytes, brillo, desvío, frac_sat, frac_color
def sesgo_por_metadatos(filas, excluir=()):
    clases = sorted({f["clase"] for f in filas})
    if len(clases) != 2 or len(filas) < 50:
        return None
    nombres = ["ancho", "alto", "aspecto", "bytes", "brillo", "desvio", "frac_sat", "frac_color"]
    X = np.array([[f["ancho"], f["alto"], f["ancho"] / f["alto"], f["bytes"], f["brillo"],
                   f["desvio"], f["frac_sat"], f["frac_color"]] for f in filas])
    y = np.array([f["clase"] == clases[1] for f in filas])
    return {nom: auc_mann_whitney(X[:, k], y) for k, nom in enumerate(nombres) if nom not in excluir}
    # calcula el AUC de cada uno por separado
    # ninguna de estas características debería tener relación con la enfermedad
    # si alguna separa bien las clases es un sesgo del dataset que el modelo puede explotar
    # si el mejor AUC llega a 0,65, se imprime la alerta
    # el modelo podría aprender ese atajo en lugar de la patología


# función que busca asegurarse de que el clasificador que van a entrenar aprenda
# a reconocer PCOS mirando el ovario
# y no por algún detalle de cómo se sacaron o guardaron las imágenes.
def cmd_auditar(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-azar", type=int, default=50, help="imágenes al azar por clase")
    ap.add_argument("--n-marcas", type=int, default=40, help="imágenes 'sospechosas' por clase")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sin-color", action="store_true",
                    help="estadísticas de gris sin los píxeles con color; excluye frac_color del chequeo")
    args = ap.parse_args(argv)
    sufijo = "_sin_color" if args.sin_color else ""

    rutas = sorted(p for p in CLEAN.rglob("*") if p.is_file() and p.suffix.lower() in EXT)
    if not rutas:
        raise SystemExit(f"No hay imágenes en {CLEAN}. ¿Corriste limpieza.py limpiar?")
    filas = [medir(r, args.sin_color) for r in rutas]
    # busca todas las imágenes de data/clean, les aplica medir() a cada una y escribe una tabla con una fila por imagen
    REPORTES.mkdir(exist_ok=True) # crea la carpeta reports/
    with open(REPORTES / f"audit{sufijo}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    clases = sorted({f["clase"] for f in filas}) 
    # se arma un conjunto, que elimina repetidos
    # quedan solo las filas de {"infected", "notinfected"}, y sorted las ordena
    rng = random.Random(args.seed)
    # L es la lista de líneas del resumen
    # se le van agregando líneas y al final se juntan en un solo texto
    # el "" es una línea en blanco
    L = [f"Imágenes auditadas: {len(filas)}", ""]
    for cl in clases:
        fs = [f for f in filas if f["clase"] == cl]
        tam = Counter((f["ancho"], f["alto"]) for f in fs)
        (t_top, n_top), = tam.most_common(1)
        modos = Counter(f["modo"] for f in fs) # Counter cuenta cuántas veces aparece cada tamaño
        L += [
            f"[{cl}]  n = {len(fs)}  (train={sum(f['origen'] == 'train' for f in fs)}, "
            f"test={sum(f['origen'] == 'test' for f in fs)})",
            f"  tamaños distintos: {len(tam)} | más frecuente: {t_top[0]}x{t_top[1]} "
            f"({100 * n_top / len(fs):.0f} %)",
            f"  ancho mediano: {st.median(f['ancho'] for f in fs):.0f} | "
            f"alto mediano: {st.median(f['alto'] for f in fs):.0f} | "
            f"bytes medianos: {st.median(f['bytes'] for f in fs):.0f}",
            "  modos: " + ", ".join(f"{k}={v}" for k, v in modos.most_common()),
            f"  brillo medio: {st.mean(f['brillo'] for f in fs):.1f} | "
            f"desvío medio: {st.mean(f['desvio'] for f in fs):.1f}",
            f"  con >1 % de píxeles saturados: "
            f"{100 * sum(f['frac_sat'] > 0.01 for f in fs) / len(fs):.0f} % | "
            f"con color (>0,1 % de píxeles): "
            f"{100 * sum(f['frac_color'] > 0.001 for f in fs) / len(fs):.0f} %",
            "",
        ]
        al = rng.sample(fs, min(args.n_azar, len(fs)))
        montaje([ROOT / f["path"] for f in al], REPORTES / f"muestra_azar_{cl}.jpg")
        sosp = sorted(fs, key=lambda f: f["frac_sat"] + f["frac_color"], reverse=True)
        montaje([ROOT / f["path"] for f in sosp[:args.n_marcas]],
                REPORTES / f"muestra_marcas_{cl}.jpg")

    if args.sin_color:
        L.insert(1, "MODO --sin-color: brillo, desvío y saturación sin píxeles coloreados; "
                    "frac_color fuera del chequeo de atajos.")
    solas = sesgo_por_metadatos(filas, excluir=("frac_color",) if args.sin_color else ())
    if solas is not None:
        mejor, auc = max(solas.items(), key=lambda kv: kv[1])
        veredicto = ("ALERTA: la clase se separa sin mirar el ovario; hay sesgo de adquisición"
                     if auc >= 0.65 else "sin señal fuerte de sesgo")
        L += [f"Chequeo de atajos: mejor metadato solo = {mejor} (AUC = {auc:.3f}) -> {veredicto}",
              "  (AUC = U de Mann-Whitney normalizado; imágenes sin agrupar, los near-duplicados pesan.)",
              "  AUC de cada metadato solo (0,5 = no separa): " +
              ", ".join(f"{k}={v:.2f}" for k, v in sorted(solas.items(), key=lambda kv: -kv[1])), ""]
    L += ["Revisión visual pendiente (a mano):",
          f"  - {REPORTES}/muestra_azar_<clase>.jpg y muestra_marcas_<clase>.jpg",
          "  - anotar en el informe: ¿hay texto, calibres, logos? ¿en qué proporción por clase?"]
    texto = "\n".join(L)
    print(texto)
    (REPORTES / f"resumen_auditoria{sufijo}.txt").write_text(texto + "\n", encoding="utf-8")


CLASE_POSITIVA = "infected"


# devuelve el pHash de una imagen como entero
def phash_int(ruta):
    with Image.open(ruta) as im:
        return int(str(imagehash.phash(im)), 16)


# la idea es juntar cortes parecidos, probablemente del mismo paciente, no solo copias exactas.
def componentes(hashes, umbral):
    n = len(hashes)
    padre = list(range(n)) # Se usa una lista padre, donde padre[i] dice a quién apunta la imagen i
    #Si una imagen se apunta a sí misma, es la jefa (raíz) de su grupo

    # función que busca encontrar la jefa del grupo
    def raiz(i): # Sigue las flechas hasta llegar a alguien que se apunta a sí mismo
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    for i in range(n):
        hi = hashes[i]
        for j in range(i + 1, n):
            if bin(hi ^ hashes[j]).count("1") <= umbral:
                ri, rj = raiz(i), raiz(j)
                if ri != rj:
                    padre[rj] = ri
    return [raiz(i) for i in range(n)]
# Dos imágenes están en el mismo grupo si tienen la misma jefa


# función que separa las imágenes en 2:
# 1. folds: son las imagenes que se utilizan para probar ideas, elegir parámetros
# 2. hold_out: son las imágenes que se dejan aparte y se utilizan para validar las métricas
def cmd_splits(argv=None):
    ap = argparse.ArgumentParser() 
    # primero lista las imágenes y arma las etiquetas
    ap.add_argument("--umbral-grupo", type=int, default=8,
                    help="distancia pHash máxima para agrupar near-duplicados")
    # qué tan parecidas tienen que ser dos imágenes para ir al mismo grupo
    ap.add_argument("--holdout", type=float, default=0.2) # 20% de las imagenes van al hold-out
    ap.add_argument("--folds", type=int, default=5) # en cuantos folds se divide el resto
    ap.add_argument("--seed", type=int, default=42) #semilla al azar para que la división salga siempre igual
    ap.add_argument("--salida", default="splits.csv",
                    help="archivo de salida en la raíz del repo (usar otro nombre para experimentar)")
    ap.add_argument("--clase-positiva", default=CLASE_POSITIVA,
                    help="nombre de la carpeta de la clase PCOS/anormal (label = 1)") # qué carpeta es PCOS
    args = ap.parse_args(argv)

    rutas = sorted(p for p in CLEAN.rglob("*") if p.is_file() and p.suffix.lower() in EXT)
    if not rutas:
        raise SystemExit(f"No hay imágenes en {CLEAN}. ¿Corriste limpieza.py limpiar?")
    n = len(rutas)
    clases = np.array([r.parent.name for r in rutas])
    y = (clases == args.clase_positiva).astype(int) # compara elemento por elemento
    if y.sum() == 0 or y.sum() == n: # los modelos y las métricas trabajan con 0 y 1
        # 0 = ningún PCOS
        # n = todos PCOS
        raise SystemExit(f"Se esperan dos clases y una es '{args.clase_positiva}'. "
                         f"Encontré: {set(clases)}. Usá --clase-positiva.")
    orig = np.array([origen(r.name) for r in rutas])

    print(f"Calculando pHash de {n} imágenes...")
    hashes = [phash_int(r) for r in rutas] # calcula el pHash de cada imagen y aplica el union-find
    raices = componentes(hashes, args.umbral_grupo) # raices queda con la “jefa” de cada imagen
    ids = {r: k for k, r in enumerate(sorted(set(raices)))} # se ordenan las jefas y se les asigna un id único
    grupos = np.array([ids[r] for r in raices])

    # se cuentan la cantidad de imagenes en cada grupo
    tam = Counter(grupos)
    multi = [g for g, c in tam.items() if c > 1] # muestra los grupos con más de una imagen (los que realmente juntaron imágenes parecidas)
    mixtos_clase = [g for g in multi if len(set(y[grupos == g])) > 1]
    # si hay etiquetas distintas en el mismo grupo es un error
    # puede ser un error de etiqueta o dos imágenes que se agruparon por encadenamiento
    mixtos_origen = [g for g in multi if len(set(orig[grupos == g])) > 1]
    # si hay imágenes de train y test en el mismo grupo, es un error de la partición original
    mayor = max(tam.values())
    # el tamaño del grupo más grande, para detectar encadenamiento

    # hold-out y folds
    # imágenes parecidas (el mismo ovario, copias) tienen que ir juntas al mismo fold.
    n_ho = max(2, round(1 / args.holdout))
    sgkf_ho = StratifiedGroupKFold(n_splits=n_ho, shuffle=True, random_state=args.seed)
    idx_cv, idx_ho = next(sgkf_ho.split(np.zeros(n), y, grupos))
    # idx_cv son los índices de las imágenes que van a los folds, idx_ho las que van al hold-out

    fold = np.full(n, -1, dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    # cada fold tiene que tener más o menos la misma proporción de PCOS y sanas.
    # la función divide en partes respetando grupos y manteniendo la proporción de clases
    for k, (_, val) in enumerate(sgkf.split(np.zeros(len(idx_cv)), y[idx_cv], grupos[idx_cv])):
        fold[idx_cv[val]] = k
    split = np.where(fold >= 0, "cv", "holdout")
    # genera 5 divisiones posibles (cada uno 0,2) y se le aplica la primera parte al hold_out

    # verificaciones: ningún grupo repartido en más de un fold / hold-out 
    por_grupo = defaultdict(set) # Arma un diccionario que dice en qué folds aparece cada grupo
    for g, f in zip(grupos, fold):
        por_grupo[g].add(f)
    fugas = [g for g, fs in por_grupo.items() if len(fs) > 1]
    # primera verificación: si algún grupo aparece en más de un fold es fuga
    assert not fugas, f"Grupos repartidos en más de un fold: {fugas[:5]}"
    assert (fold[idx_ho] == -1).all() and (fold[idx_cv] >= 0).all()
    # segunda verificación: los índices de hold-out y folds no se mezclan
    # las del hold-out quedaron todas en -1 y las de CV todas en un fold válido
    # es redundante pero si algún cambio futuro en el código rompe algo, el programa frena en lugar de generar un splits.csv con fuga sin avisar.

    # se escribe splits.csv con las columnas: path, label, fold, split, clase, grupo, origen_kaggle
    with open(ROOT / args.salida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "fold", "split", "clase", "grupo", "origen_kaggle"])
        for i, r in enumerate(rutas):
            w.writerow([r.relative_to(ROOT).as_posix(), int(y[i]), int(fold[i]), split[i],
                        clases[i], int(grupos[i]), orig[i]])

    # también se escribe un resumen con la distribución de clases por fold
    L = [
        f"Imágenes: {n} | infected: {int(y.sum())} | notinfected: {int((1 - y).sum())}",
        f"Umbral de agrupado (pHash): {args.umbral_grupo}",
        f"Grupos: {len(tam)} | grupos con >1 imagen: {len(multi)} "
        f"({sum(tam[g] for g in multi)} imágenes) | grupo más grande: {mayor}",
        f"Grupos con clases mezcladas: {len(mixtos_clase)}",
        f"Grupos con imágenes de train Y test de Kaggle: {len(mixtos_origen)} "
        f"({sum(tam[g] for g in mixtos_origen)} imágenes)  <- evidencia de fuga en la partición original",
    ]
    if mayor > max(20, 0.02 * n):
        L.append(f"ATENCIÓN: un grupo de {mayor} imágenes ({100 * mayor / n:.1f} %) sugiere "
                 f"encadenamiento de similitud. Probá --umbral-grupo más bajo (6, 5...).")
    L += ["", f"{'split':8s}{'fold':>5s}{'n':>7s}{'infected':>10s}{'notinf.':>9s}{'% inf.':>8s}"]
    for s, fo in [("cv", k) for k in range(args.folds)] + [("holdout", -1)]:
        m = fold == fo
        pos, tot = int(y[m].sum()), int(m.sum())
        L.append(f"{s:8s}{fo:>5d}{tot:>7d}{pos:>10d}{tot - pos:>9d}{100 * pos / max(tot, 1):>8.1f}")
    L += ["", "Verificación: ningún grupo cruza folds ni hold-out: OK"]
    texto = "\n".join(L)
    print(texto)
    REPORTES.mkdir(exist_ok=True)
    (REPORTES / f"resumen_{Path(args.salida).stem}.txt").write_text(texto + "\n", encoding="utf-8")


# ahora chequeamos duplicados: copias rotadas, recortadas o espejadas
# pHash no reconoce una imagen rotada 30°, así que esta etapa usa puntos característicos

# función que:
# lee la imagen
# la pasa a gris
# la recorta al área útil
# la escala para que el lado mayor mida tam píxeles, sin deformarla.
def preparar(ruta, tam=320):
    g = recortar_util(a_gris(leer(ruta)))
    s = tam / max(g.shape)
    return cv2.resize(g, (max(1, round(g.shape[1] * s)), max(1, round(g.shape[0] * s))),
                      interpolation=cv2.INTER_AREA)


# histograma con solo el tejido
# se descarta el fondo negro (≤ 12) y el blanco (≥ 250)
def histograma(g):
    v = g[(g > 12) & (g < 250)].astype(np.float32)
    if v.size < 100:
        return np.full(64, 1 / 64, np.float32)
    lo, hi = np.percentile(v, (2, 98)) 
    # estira el contraste entre los percentiles 2 y 98
    # para que una copia más clara u oscura dé casi el mismo histograma
    # similar a la ecualización
    v = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 0.999) * 64
    h = np.bincount(v.astype(int), minlength=64).astype(np.float32)
    h = cv2.GaussianBlur(h.reshape(1, -1), (1, 5), 0).ravel() # Después lo suaviza y lo normaliza
    return h / h.sum()
# si hay muy poco tejido, devuelve un histograma plano.


# función que detecta puntos ORB en el tejido
# ORB busca en la imagen puntos característicos
def orb_desc(orb, g):
    # máscara: solo tejido, para no tomar puntos en el borde del relleno
    m = ((g > 12) & (g < 250)).astype(np.uint8)
    m = cv2.erode(m, np.ones((7, 7), np.uint8)) 
    # la máscara se erosiona para que no aparezcan puntos falsos en el borde del relleno
    kp, des = orb.detectAndCompute(g, m)
    if des is None or len(kp) < 10:
        return None # devuelve None si encuentra menos de 10 puntos
    return np.float32([k.pt for k in kp]), des
    # devuelve las coordenadas y los descriptores


# función que permite hacer luego el filtrado rápido
# crea las categorías que permiten resumir cada imagen en una lista corta
# para descartar rápido los millones de pares que obviamente no son copias
def vocabulario(descs, k=256, muestra=60000, seed=0):
    todos = np.vstack([d for d in descs if d is not None])
    # vstack apila las matrices de todas las imágenes en una sola
    # las imágenes donde ORB no encontró puntos (None) se saltean
    # con aprox 3800 imágenes y unos 800 puntos cada una, quedan cerca de 3 millones de filas
    rng = np.random.default_rng(seed) # semilla al azar
    if len(todos) > muestra:
        todos = todos[rng.choice(len(todos), muestra, replace=False)]
    # agrupar 3 millones de descriptores tardaría mucho
    # se toman 60 000 al azar (sin repetir) como muestra representativa
    bits = np.unpackbits(todos, axis=1).astype(np.float32) # paso de bytes a bits
    # k-means
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5) # criterio de cuando parar
    # después de 20 vueltas, o cuando los centros se muevan menos de 0,5
    _, _, centros = cv2.kmeans(bits, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    return np.packbits((centros > 0.5).astype(np.uint8), axis=1)


# función que convierte cada imagen en su resumen
# asigna cada punto ORB de la imagen al grupo de k-means más cercano
# y cuenta cuántas veces aparece cada palabra
def bolsa(des_list, centros, bf):
    v = np.zeros(len(centros), np.float32)
    for des in des_list:
        if des is None:
            continue
        for m in bf.match(des, centros):
            v[m.trainIdx] += 1
    return v
# devuelve un histograma que no depende de dónde están los puntos
# por eso casi no cambia al rotar o desplazar la imagen
# es un filtrado rápido: si dos imágenes tienen histogramas muy distintos, no son copias.


# se verifica el filtrado rápido con RANSAC
# ej: si B es una copia de A girada 30° y corrida a la derecha entonces 
# todos los puntos de A tienen que aparecer en B movidos de la misma manera
# si las imágenes son distintas, algunos puntos pueden parecerse por casualidad
# pero cada uno “se movería” de forma distinta, sin ningún patrón común
# RANSAC busca un patrón común y descarta los puntos que no lo cumplen
def verificar(a, b, bf, ratio=0.75):
    if a is None or b is None: # a y b son lo que devolvió orb_desc para cada imagen
        return 0
    pa, da = a # pa, pb son las coordenadas de los puntos
    pb, db = b # da, db son los descriptores
    pares = bf.knnMatch(da, db, k=2)
    # para cada punto de A busca los 2 puntos de B más parecidos
    buenos = [p[0] for p in pares if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    if len(buenos) < 8:
        return 0
    src = pa[[m.queryIdx for m in buenos]]
    dst = pb[[m.trainIdx for m in buenos]]
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                         ransacReprojThreshold=4.0, maxIters=2000)
    if M is None or inl is None:
        return 0
    escala = float(np.hypot(M[0, 0], M[1, 0]))
    if not 0.5 <= escala <= 2.0:  # zoom absurdo: no es una aumentación
        return 0
    return int(inl.sum())
# devuelve pares de imagenes que son iguales 
# ej: 3 y 7 son la misma, 12 y 7 son la misma

# union-find convierte esos pares sueltos en grupos
# agrupa 3, 7 y 12 en un mismo grupo
class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def raiz(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def unir(self, i, j):
        ri, rj = self.raiz(i), self.raiz(j)
        if ri != rj:
            self.p[rj] = ri


# función que toma los 8 grupos más grandes y arma una figura con hasta 6 imágenes de cada uno
# una fila por grupo, con su número arriba
# sirve para comprobar a ojo que cada fila sea realmente la misma ecografía.
def figura_grupos(rutas, grupos, n_grupos=8, n_por_grupo=6, lado=140):
    tam = Counter(grupos)
    mayores = [g for g, c in tam.most_common(n_grupos) if c > 1]
    filas = []
    for g in mayores:
        idx = [i for i in range(len(rutas)) if grupos[i] == g][:n_por_grupo]
        celdas = []
        for i in idx:
            im = cv2.resize(preparar(rutas[i]), (lado, lado), interpolation=cv2.INTER_AREA)
            celdas.append(cv2.cvtColor(im, cv2.COLOR_GRAY2BGR))
        while len(celdas) < n_por_grupo:
            celdas.append(np.zeros((lado, lado, 3), np.uint8))
        fila = np.hstack(celdas)
        cv2.putText(fila, f"grupo {g} ({tam[g]} img)", (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 255), 1, cv2.LINE_AA)
        filas.append(fila)
    return np.vstack(filas) if filas else None


# main
# splits.csv ya agrupaba las copias casi idénticas (pHash), pero no las rotadas, desplazadas o espejadas
# esta función devuelve splits2.csv con únicamente las imagenes originales y un resumen de los grupos encontrados
def cmd_duplicados(argv=None): # argumentos y atos de entrada
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="splits.csv", help="partición actual (se usa su lista de imágenes y sus grupos pHash)")
    ap.add_argument("--salida", default="splits2.csv")
    ap.add_argument("--vecinos", type=int, default=40, help="candidatos por imagen a verificar")
    ap.add_argument("--inliers-min", type=int, default=25)
    ap.add_argument("--features", type=int, default=800)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    with open(ROOT / args.splits, encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    rutas = [ROOT / r["path"] for r in filas]
    y = np.array([int(r["label"]) for r in filas])
    n = len(rutas)
    faltan = [r for r in rutas if not r.exists()]
    if faltan:
        raise SystemExit(f"Faltan {len(faltan)} imágenes (ej. {faltan[0]}). ¿Corriste limpieza.py limpiar?")
    # si alguna imagen del CSV no está en el disco corta con un aviso
    # descriptores
    # busca el histograma, los puntos ORB y los puntos ORB de su versión espejada de cada imagen
    t0 = time.time()
    orb = cv2.ORB_create(nfeatures=args.features, scaleFactor=1.2, nlevels=8, fastThreshold=10)
    # nfeatures=800: hasta 800 puntos por imagen
    # scaleFactor=1.2, nlevels=8: busca puntos en 8 tamaños de la imagen, cada uno 1,2 veces más chico
    # así encuentra los mismos puntos aunque la copia tenga zoom
    # fastThreshold=10: qué tan “llamativo” tiene que ser un punto
    # un valor bajo encuentra más puntos, útil cuando el contraste es bajo
    hist, desc, desc_esp = [], [], []
    for i, r in enumerate(rutas):
        g = preparar(r) # la pasa a gris, la recorta al área útil y la achica a 320 px de lado mayor
        # despues: histograma, puntos ORB y puntos ORB de la imagen espejada
        hist.append(histograma(g))
        desc.append(orb_desc(orb, g))
        desc_esp.append(orb_desc(orb, cv2.flip(g, 1)))
        if (i + 1) % 200 == 0:
            print(f"  descriptores {i + 1}/{n}  ({time.time() - t0:.0f} s)")
    H = np.stack(hist) # Apila los histogramas en una matriz

    # preseleccion de candidatos
    # para cada imagen elige:
    # los 40 vecinos más cercanos por histograma (distancia L1)
    # los 40 más parecidos por bolsa de palabras ponderada con TF-IDF
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    centros = vocabulario([d[1] for d in desc if d is not None])
    B = np.stack([bolsa([None if a is None else a[1], None if b is None else b[1]], centros, bf)
                  for a, b in zip(desc, desc_esp)])
    # usa TF-IDF para convertir esos conteos en proporciones
    # le baja el peso a los tipos de punto que aparecen en todas las imágenes
    idf = np.log(n / (1 + (B > 0).sum(0))) # IDF
    B = B / np.maximum(B.sum(1, keepdims=True), 1) * idf # TF-IDF
    B /= np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9) # normalización
    sim_bolsa = B @ B.T # calcula qué tan parecida es cada imagen a cada otra
    k = min(args.vecinos, n - 1)
    cand = set()
    for i in range(n):
        d_h = np.abs(H - H[i]).sum(1)
        d_b = -sim_bolsa[i].copy()
        for d in (d_h, d_b):
            d[i] = np.inf
            for j in np.argpartition(d, k)[:k]: # para cada imagen elige las 40 más parecidas como candidatas
                cand.add((min(i, int(j)), max(i, int(j))))
    print(f"Pares candidatos a verificar: {len(cand)}  ({time.time() - t0:.0f} s)")

    # verificación geométrica (normal y espejada)
    # verifica con RANSAC dos veces: i normal contra j, y i espejada contra j
    # se queda con los inliers
    # los inliers son las parejas de puntos que coinciden con el movimiento ganador que encontró RANSAC
    # los outliers son las que no coinciden: emparejamientos equivocados
    uf = UnionFind(n)
    vinculos = []
    for c, (i, j) in enumerate(sorted(cand)):
        inl = max(verificar(desc[i], desc[j], bf), verificar(desc_esp[i], desc[j], bf))
        if inl >= args.inliers_min:
            uf.unir(i, j)
            vinculos.append((i, j, inl))
        if (c + 1) % 5000 == 0:
            print(f"  verificados {c + 1}/{len(cand)}  ({time.time() - t0:.0f} s)")

    # unir con los grupos pHash previos
    ph = [r["grupo"] for r in filas]
    primero = {}
    for i, g in enumerate(ph):
        if g in primero:
            uf.unir(primero[g], i)
        else:
            primero[g] = i
    raices = [uf.raiz(i) for i in range(n)]
    ids = {r: k for k, r in enumerate(sorted(set(raices)))}
    grupos = np.array([ids[r] for r in raices])

    # genera la nueva partición
    # exactamente igual que en cmd_splits, pero con los grupos nuevos
    n_ho = max(2, round(1 / args.holdout))
    idx_cv, idx_ho = next(StratifiedGroupKFold(n_splits=n_ho, shuffle=True, random_state=args.seed)
                          .split(np.zeros(n), y, grupos))
    fold = np.full(n, -1, dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed) # 20% de hold-out
    for kf, (_, val) in enumerate(sgkf.split(np.zeros(len(idx_cv)), y[idx_cv], grupos[idx_cv])):
        fold[idx_cv[val]] = kf
    split = np.where(fold >= 0, "cv", "holdout")

    por_grupo = defaultdict(set) # 80% restante en 5 folds.
    for g, f in zip(grupos, fold):
        por_grupo[g].add(f)
    assert not [g for g, fs in por_grupo.items() if len(fs) > 1], "un grupo cruza folds"
    # Verifica que ningún grupo esté en más de un fold

    with open(ROOT / args.salida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "fold", "split", "clase", "grupo", "origen_kaggle", "grupo_phash"])
        for i, r in enumerate(filas):
            w.writerow([r["path"], int(y[i]), int(fold[i]), split[i], r["clase"], int(grupos[i]),
                        r.get("origen_kaggle", ""), r["grupo"]])
            # Mismas columnas que splits.csv, con los grupos y folds nuevos
            # más una columna grupo_phash con el grupo viejo
    REPORTES.mkdir(exist_ok=True)
    with open(REPORTES / "duplicados_pares.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path_a", "path_b", "inliers"])
        for i, j, inl in sorted(vinculos, key=lambda v: -v[2]):
            w.writerow([filas[i]["path"], filas[j]["path"], inl])
    fig = figura_grupos(rutas, grupos)
    if fig is not None:
        cv2.imwrite(str(REPORTES / "duplicados_ejemplos.png"), fig)
    # grafica los grupos más grandes, una fila por grupo
    # sirve para revisar a ojo que de verdad sean la misma ecografía

    # cuántos grupos pHash viejos quedaron repartidos entre folds en la partición anterior
    fold_viejo = np.array([int(r["fold"]) for r in filas])
    fuga_vieja = sum(1 for g in set(grupos) if len(set(fold_viejo[grupos == g])) > 1)
    # Para cada grupo nuevo, mira en qué folds estaban sus imágenes en splits.csv
    # Si estaban en más de uno, ese grupo tenía fugac (opias rotadas que pHash no había juntado y que quedaron separadas)
    # justifica para qué sirvió todo esto

    tam = Counter(grupos) # Igual que en splits
    # tamaño de cada grupo, grupos con las dos etiquetas y el grupo más grande.
    mixtos = [g for g in tam if len(set(y[grupos == g])) > 1]
    mayor = max(tam.values())
    L = [f"Imágenes: {n} | grupos pHash (antes): {len(set(ph))} | grupos ahora: {len(tam)}",
         f"Vínculos nuevos por ORB+RANSAC: {len(vinculos)} (umbral {args.inliers_min} inliers)",
         f"Grupos con >1 imagen: {sum(c > 1 for c in tam.values())} | grupo más grande: {mayor}",
         f"Imágenes 'únicas' reales (= grupos): {len(tam)} -> {n - len(tam)} eran copias",
         f"Grupos nuevos que en {args.splits} estaban repartidos entre folds/hold-out: {fuga_vieja}"
         "  <- fuga en la partición anterior",
         f"Grupos con clases mezcladas: {len(mixtos)}"
         + ("  <- REVISAR: la misma imagen con las dos etiquetas, o un vínculo falso" if mixtos else ""),
         ""]
    if mayor > max(30, 0.03 * n):
        L.append(f"ATENCIÓN: un grupo de {mayor} imágenes puede ser encadenamiento de vínculos falsos. "
                 f"Revisar duplicados_ejemplos.png y probar --inliers-min más alto (35, 50).")
    L += [f"{'split':8s}{'fold':>5s}{'n':>7s}{'grupos':>8s}{'PCOS':>7s}{'sanas':>7s}"]
    for s, fo in [("cv", kk) for kk in range(args.folds)] + [("holdout", -1)]:
        m = fold == fo
        L.append(f"{s:8s}{fo:>5d}{m.sum():>7d}{len(set(grupos[m])):>8d}{int(y[m].sum()):>7d}{int((1 - y[m]).sum()):>7d}")
    L += ["", f"Tiempo: {time.time() - t0:.0f} s",
          f"Salida: {args.salida} (no pisa {args.splits}). Revisar reports/duplicados_ejemplos.png:",
          "cada fila debería ser UNA misma ecografía con distintas rotaciones/desplazamientos."]
    texto = "\n".join(L)
    print(texto)
    (REPORTES / "resumen_duplicados.txt").write_text(texto + "\n", encoding="utf-8")


COMANDOS = {
    "unir": cmd_unir,
    "limpiar": cmd_limpiar,
    "auditar": cmd_auditar,
    "splits": cmd_splits,
    "duplicados": cmd_duplicados,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMANDOS:
        print("Uso: python procesamiento/limpieza.py <comando> [opciones]\n\nComandos:")
        for k, f in COMANDOS.items():
            print(f"  {k:18s}{(f.__doc__ or '').strip().splitlines()[0] if f.__doc__ else ''}")
        print("\nPara ver las opciones de uno: python procesamiento/limpieza.py <comando> -h")
        raise SystemExit(1)
    COMANDOS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()