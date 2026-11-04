"""Preprocesamiento de ecografías ováricas y elección de filtros (módulo de B).

Pipeline de `preprocess(entrada)`:
  1. gris + quitar marcas (flechas/anotaciones de color, texto y cruces blancos) por inpainting
  2. recortar el área útil (saca bordes negros)
  3. llevar a un tamaño común (INTER_AREA al achicar)  -> reduce pistas de resolución
  4. filtro de speckle (mediana, gaussiano, bilateral, nlm, lee, kuan, frost)
  5. contraste (global, clahe, estirar)

El tamaño común se aplica ANTES de filtrar: el speckle de una imagen de 984x848 y el de una
de 300x300 no son comparables hasta que tienen la misma resolución.

Como módulo:
    from preprocesamiento import preprocess
    img = preprocess("data/clean/infected/x.jpg", filtro="wiener3", contraste="ninguno")

Como script (selección de filtros):
    python procesamiento/preprocesamiento.py comparar         # figuras + SSI/beta por filtro
    python procesamiento/preprocesamiento.py anotar-rois      # marcar bloques homogéneos a mano
    python procesamiento/preprocesamiento.py elegir-filtro    # ranking por ENL con esos bloques
    python procesamiento/preprocesamiento.py procesar         # aplica el filtro elegido a data/clean
"""
from collections import defaultdict
import argparse
import csv
import random
import shutil
import sys
import time
import cv2
import numpy as np
from PIL import Image
from scipy import signal, stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if ROOT.name == "procesamiento": 
    ROOT = ROOT.parent
CLEAN = ROOT / "data" / "clean"
REPORTES = ROOT / "reports"
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
FONT = cv2.FONT_HERSHEY_SIMPLEX

# PIPELINE DE PREPROCESAMIENTO (marcas, recorte, tamaño, speckle, contraste, métricas)

def leer(ruta):
    with Image.open(ruta) as im: #abre la imagen que está en la ruta indicada
        return np.asarray(im.convert("RGB")) #la convierte al espacio de colores --> sirve para estandarizar


def a_gris(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) #hace la transformación al espacio de grises


# Eliminación de las marcas
def _componentes_chicas(binaria, area_max):
    #Construye una máscara con solo las componentes conexas de área <= area_max."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binaria.astype(np.uint8), connectivity=8)
    ids = np.where(stats[:, cv2.CC_STAT_AREA] <= area_max)[0] #se guardan los ids de las eqiquetas con área menor o igual al valor max 
    ids = ids[ids > 0] #elimina id=0 que es el fondo de la imagen para ignorarlo
    return np.isin(lab, ids) #devuelve una matriz del tamaño original con valores 1 en los píxeles 
    #que pertenecen a los objetos pequeños detectados, en el resto devuelve 0


def mascara_marcas(rgb, sat_min=0.55, val_min=0.45, max_frac_color=0.03,
                   quitar_texto=True, max_frac_texto=0.002, max_frac_total=0.15, dilatar=3):
    #Máscara de anotaciones: color muy saturado (flechas, calibres) y
    #componentes blancas chicas (texto, cruces). Las zonas grandes de color (Doppler) y los
    #tintes suaves de toda la imagen NO se enmascaran. Si la máscara cubre más de
    #`max_frac_total` de la imagen se descarta.

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV) #convierte la imagen al espacio HSV
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0
    n = rgb.shape[0] * rgb.shape[1] #calcula el total de los pixeles de la imagen
    color = (s >= sat_min) & (v >= val_min) #umbrales para detectar pixeles de alta saturación y alto brillo
    m = _componentes_chicas(color, int(max_frac_color * n)) #conserva solo lás áreas de color pequeñas --> descarta regiones grandes coloreadas
    if quitar_texto:
        blanco = a_gris(rgb) >= 245 #identifica los pixeles que se ven como blancos 
        m |= _componentes_chicas(blanco, max(30, int(max_frac_texto * n)))  #conserva solos las regiones blancas pequeñas 
    if m.mean() > max_frac_total:
        return np.zeros(m.shape, np.uint8) #si la región de color es muy general de la imagen --> se considera que es un tinte que tiene la imagen y no un área específica
    m = m.astype(np.uint8) * 255
    if dilatar > 0 and m.any(): #si se detecta una marca
        m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=dilatar) #dilata el objeto 
    return m

def quitar_marcas(rgb, **kw):
    """Devuelve (gris sin marcas, máscara usada)."""
    gris = a_gris(rgb) #convierte la imagen a escala de grises
    m = mascara_marcas(rgb, **kw) #guarda en m las marcas detectadas en la imagen rgb
    if m.any(): #si se detecta alguna marca
        gris = cv2.inpaint(gris, m, 3, cv2.INPAINT_TELEA) # se usa inpainting para rellenar las áreas en donde se eliminaron las marcas
    return gris, m #devuelve la imagen limpia y la máscara binaria resultante


def recortar_util(gris, umbral=12, margen=2, min_frac=0.25):
    #Recorta al rectángulo que contiene la componente más grande no negra --> extrae el objeto detectado
    m = (cv2.GaussianBlur(gris, (5, 5), 0) > umbral).astype(np.uint8) #crea una máscara con los píxeles que superan el valor de brillo 12
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))#se hace una apertura para eliminar pequeños puntos brillantes
    n, _, stats, _ = cv2.connectedComponentsWithStats(m) #detecta todas las regiones continuas en la máscara
    if n <= 1:
        return gris #si no se reconocen varios objetos --> devuelve la imagen original
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) #encuentra la etiqueta de la región con la mayor área de píxeles, omitiendo el índice 0 que corresponde al fondo negro
    x, y, w, h = stats[i, :4] 
    if w * h < min_frac * gris.size:  # si la bounding box es muy pequeña asume que no hay un objeto principal
        return gris #devuelve la imagen original
    return gris[max(y - margen, 0):y + h + margen, max(x - margen, 0):x + w + margen] #hace el recorte agregando piceles a cada borde


def a_tamano(gris, tam=256):
   #dimensiona la imagen a una de 256x256
    achica = gris.shape[0] * gris.shape[1] > tam * tam
    interp = cv2.INTER_AREA if achica else cv2.INTER_CUBIC #si la imagen original es más grande hace un submuestreo, agrupa y pormedia áreas de píxeles
    #si la imagen original es más pequeña la agranda interpolando de forma bicúbica
    return cv2.resize(gris, (tam, tam), interpolation=interp) #devuelve la imagen redimensionada

# Para el ruido speckle
def _locales(img, w):
    f = img.astype(np.float32) 
    m = cv2.boxFilter(f, -1, (w, w), borderType=cv2.BORDER_REFLECT) #calcula la media local
    m2 = cv2.boxFilter(f * f, -1, (w, w), borderType=cv2.BORDER_REFLECT) #eleva cada pixel al cuadrado y saca la media local
    v = np.maximum(m2 - m * m, 0) #calcula la varianza
    return f, m, v

def frost(img, w=7, K=2.0):
    #Filtro adaptativo de Frost
    #Calcula el coeficiente de variación local $ci^2 = \frac{v}{m^2}$. 
    # Si la zona es homogénea (baja varianza), aplica un suavizado fuerte; 
    # si está cerca de un borde o estructura importante (alta varianza), 
    # reduce el suavizado para no borrar la información.Pondera los píxeles 
    # vecinos con una función exponencial decreciente en función de la distancia 
    # espacial y de la varianza local
    f, m, v = _locales(img, w)
    ci2 = v / (m * m + 1e-6)
    r = w // 2
    pad = cv2.copyMakeBorder(f, r, r, r, r, cv2.BORDER_REFLECT)
    H, W = f.shape
    num = np.zeros_like(f)
    den = np.zeros_like(f)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            wgt = np.exp(-K * ci2 * np.hypot(dy, dx))
            num += wgt * pad[r + dy:r + dy + H, r + dx:r + dx + W]
            den += wgt
    return np.clip(num / np.maximum(den, 1e-9), 0, 255).astype(np.uint8)


def wiener(img, w=5):
    #Filtro adaptativo de Wiener
    #Se adapta al nivel de ruido estimado localmente.
    # En regiones con alta varianza (bordes), realiza poco filtrado; 
    # en regiones de baja varianza, realiza una mayor atenuación.
    with np.errstate(divide="ignore", invalid="ignore"):
        out = signal.wiener(img.astype(np.float64), (w, w))
    return np.clip(np.nan_to_num(out, nan=0.0), 0, 255).astype(np.uint8)


def homomorfico(img, sigma=1.5): 
    #Como el ruido speckle es multiplicativo, se aplica primero logaritmo para luego 
    # pasar un filtro Gaussiano para sacar el ruido. Al final, se vuelve al dominio 
    # original aplicando la exponencial. 
    u = np.log1p(img.astype(np.float32))
    u = cv2.GaussianBlur(u, (0, 0), sigma)
    return np.clip(np.expm1(u), 0, 255).astype(np.uint8)


def homomorfico_wiener(img, w=3):
    #En vez de aplicar un Gaussiano después del logaritmo,
    #usa el filtro Wiener adaptativo para remover el ruido. 
    u = np.log1p(img.astype(np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
       u = np.nan_to_num(signal.wiener(u, (w, w)), nan=0.0)
    return np.clip(np.expm1(u), 0, 255).astype(np.uint8)


#Se definen distintas variantes y combinaciones de parámetros para los filtros
FILTROS = {"ninguno": lambda im: im}
for w in (3, 5, 7):
    FILTROS[f"promedio{w}"] = lambda im, w=w: cv2.blur(im, (w, w))
    FILTROS[f"mediana{w}"] = lambda im, w=w: cv2.medianBlur(im, w)
    FILTROS[f"frost{w}"] = lambda im, w=w: frost(im, w=w)
for s_ in (1.0, 1.5, 2.0):
    FILTROS[f"gauss{s_}"] = lambda im, s_=s_: cv2.GaussianBlur(im, (0, 0), s_)
FILTROS["wiener3"] = lambda im: wiener(im, 3)
FILTROS["wiener5"] = lambda im: wiener(im, 5)
FILTROS["homomorfico"] = homomorfico
FILTROS["homomorfico_wiener3"] = lambda im: homomorfico_wiener(im, 3)


def filtrar(img, metodo="wiener3"):
    for paso in metodo.split("+"): #permite encadenar filtros separándolos con el signo +
        img = FILTROS[paso](img)
    return img #devuelve la imagen procesada


#Para mejorar el contraste
CONTRASTES = {
    "ninguno": lambda im: im,
    "global": cv2.equalizeHist
}
def contrastar(img, metodo="ninguno"): #pone por default no aplicar un mejoramiento del contraste
    return CONTRASTES[metodo](img)


#Flujo de preprocesamiento --> integra todos los módulos
def base(entrada, tam=256, marcas=True, recortar=True, texto=True):
    rgb = leer(entrada) if isinstance(entrada, (str, Path)) else entrada 
    #detecta si la entrada es una ruta al archivo o una matriz de pixeles ya en memoria
    if marcas:
        gris, _ = quitar_marcas(rgb, quitar_texto=texto) #si tiene marcas, se eliminan
    else:
        gris = a_gris(rgb) #si no tiene, solo se convierte la imagen al espacio de grises
    if recortar:
        gris = recortar_util(gris) #si encuentra algo que recortar, elimina el espacio negro aislando la estructura
    return a_tamano(gris, tam) #redimensiona la imagen recortada


def preprocess(entrada, filtro="wiener3", contraste="ninguno", tam=256,
               marcas=True, recortar=True, texto=True):
    g = base(entrada, tam, marcas, recortar, texto) #se aplica lectura, inpainting, recorte y escalado a la imagen
    return contrastar(filtrar(g, filtro), contraste)


# Definición de métricas
def ssi(orig, filt):
    """Índice de supresión de speckle (σ/μ filtrada ÷ σ/μ original). < 1 = suaviza."""
    o, f = orig.astype(np.float32), filt.astype(np.float32)
    return float((f.std() / (f.mean() + 1e-6)) / (o.std() / (o.mean() + 1e-6) + 1e-9))

def beta(orig, filt):
    """Preservación de bordes: correlación entre laplacianos (1 = bordes intactos)."""
    lo = cv2.Laplacian(orig.astype(np.float32), cv2.CV_32F)
    lf = cv2.Laplacian(filt.astype(np.float32), cv2.CV_32F)
    lo, lf = lo - lo.mean(), lf - lf.mean()
    return float((lo * lf).sum() / (np.sqrt((lo ** 2).sum() * (lf ** 2).sum()) + 1e-9))

def entropia(img):
    h = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())


#Para comparar filtros y contraste sobre una muestra
def hoja(filas, titulos, lado=160, cab=26):
    """Arma una grilla (BGR) con una imagen por celda."""
    out = np.zeros((cab + len(filas) * lado, len(titulos) * lado, 3), np.uint8)
    for j, t in enumerate(titulos):
        cv2.putText(out, t, (j * lado + 4, 18), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for i, fila in enumerate(filas):
        for j, im in enumerate(fila):
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR if im.ndim == 2 else cv2.COLOR_RGB2BGR)
            im = cv2.resize(im, (lado, lado), interpolation=cv2.INTER_AREA)
            out[cab + i * lado:cab + (i + 1) * lado, j * lado:(j + 1) * lado] = im
    return out


def cmd_comparar(argv=None):
    """Compara filtros de speckle y métodos de contraste sobre una muestra de data/clean.

Salidas en reports/:
  preproc_marcas.png      original | máscara de marcas | sin marcas, recortada, tamaño común
  preproc_filtros.png     la imagen base con cada filtro de speckle
  preproc_contraste.png   la imagen filtrada con cada método de contraste
  preproc_metricas.csv    SSI y β por filtro; entropía y desvío por método de contraste

Métricas (sin referencia limpia, como es habitual en ecografía):
  SSI  = índice de supresión de speckle. < 1 suaviza; más bajo = más suavizado.
  beta = correlación de laplacianos con la original. 1 = bordes intactos.
  Un buen filtro baja el SSI sin hundir beta. Complementar siempre mirando las figuras.
"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-fig", type=int, default=8, help="imágenes en las figuras (mitad por clase)")
    ap.add_argument("--n-metricas", type=int, default=20, help="imágenes por clase para las métricas")
    ap.add_argument("--tam", type=int, default=256)
    ap.add_argument("--filtro-ref", default="wiener3", help="filtro previo a la figura de contraste")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    por_clase = defaultdict(list)
    for p in sorted(CLEAN.rglob("*")):
        if p.is_file() and p.suffix.lower() in EXT:
            por_clase[p.parent.name].append(p)
    if not por_clase:
        raise SystemExit(f"No hay imágenes en {CLEAN}.")
    rng = random.Random(args.seed)
    REPORTES.mkdir(exist_ok=True)

    # ---------------- figuras
    k = max(1, args.n_fig // max(len(por_clase), 1))
    muestra = [p for cl in sorted(por_clase) for p in rng.sample(por_clase[cl], min(k, len(por_clase[cl])))]
    f_marcas, f_filtros, f_contraste = [], [], []
    nombres_f = list(FILTROS)
    nombres_c = list(CONTRASTES)
    for p in muestra:
        rgb = leer(p)
        _, mask = quitar_marcas(rgb)
        b = base(rgb, args.tam)
        f_marcas.append([a_gris(rgb), mask if mask.any() else np.zeros_like(a_gris(rgb)), b])
        f_filtros.append([filtrar(b, m) for m in nombres_f])
        ref = filtrar(b, args.filtro_ref)
        f_contraste.append([contrastar(ref, m) for m in nombres_c])
    cv2.imwrite(str(REPORTES / "preproc_marcas.png"),
                hoja(f_marcas, ["original (gris)", "mascara de marcas", "base: sin marcas+recorte"]))
    cv2.imwrite(str(REPORTES / "preproc_filtros.png"), hoja(f_filtros, nombres_f))
    cv2.imwrite(str(REPORTES / "preproc_contraste.png"),
                hoja(f_contraste, [f"{args.filtro_ref}+{m}" for m in nombres_c]))

    # ---------------- métricas
    acum_f = defaultdict(lambda: defaultdict(list))   # (filtro, clase) -> métrica -> valores
    tiempos = defaultdict(list)
    acum_c = defaultdict(lambda: defaultdict(list))
    for cl, rutas in sorted(por_clase.items()):
        for p in rng.sample(rutas, min(args.n_metricas, len(rutas))):
            b = base(p, args.tam)
            for m in nombres_f:
                if m == "ninguno":
                    continue
                t0 = time.perf_counter()
                g = filtrar(b, m)
                tiempos[m].append(time.perf_counter() - t0)
                acum_f[(m, cl)]["ssi"].append(ssi(b, g))
                acum_f[(m, cl)]["beta"].append(beta(b, g))
            ref = filtrar(b, args.filtro_ref)
            for m in nombres_c:
                c = contrastar(ref, m)
                acum_c[(m, cl)]["entropia"].append(entropia(c))
                acum_c[(m, cl)]["desvio"].append(float(c.std()))

    filas = []
    L = [f"Muestra de métricas: hasta {args.n_metricas} imágenes por clase | tamaño {args.tam}", "",
         "FILTROS DE SPECKLE (SSI más bajo = suaviza más; beta más alto = conserva bordes)",
         f"{'filtro':10s}{'clase':14s}{'SSI':>8s}{'beta':>8s}{'ms/img':>9s}"]
    for (m, cl), d in sorted(acum_f.items()):
        s, bt = float(np.mean(d["ssi"])), float(np.mean(d["beta"]))
        ms = 1000 * float(np.mean(tiempos[m]))
        filas.append(["filtro", m, cl, round(s, 4), round(bt, 4), "", "", len(d["ssi"])])
        L.append(f"{m:10s}{cl:14s}{s:8.3f}{bt:8.3f}{ms:9.1f}")
    L += ["", f"CONTRASTE (sobre filtro '{args.filtro_ref}')", f"{'metodo':10s}{'clase':14s}{'entropia':>10s}{'desvio':>9s}"]
    for (m, cl), d in sorted(acum_c.items()):
        e, ds = float(np.mean(d["entropia"])), float(np.mean(d["desvio"]))
        filas.append(["contraste", m, cl, "", "", round(e, 3), round(ds, 2), len(d["entropia"])])
        L.append(f"{m:10s}{cl:14s}{e:10.3f}{ds:9.2f}")
    with open(REPORTES / "preproc_metricas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tipo", "metodo", "clase", "ssi", "beta", "entropia", "desvio", "n"])
        w.writerows(filas)
    L += ["", "Figuras: reports/preproc_marcas.png, preproc_filtros.png, preproc_contraste.png",
          "Elegir mirando las figuras: ¿se ven los folículos nítidos y sin halos? "
          "Anotar la elección y su justificación en el informe."]
    texto = "\n".join(L)
    print(texto)
    (REPORTES / "resumen_comparacion_preproc.txt").write_text(texto + "\n", encoding="utf-8")


# ============================================================================
# COMANDO anotar-rois: bloques homogéneos marcados a mano
# ============================================================================
ZOOM = 3
ROIS = ROOT / "rois.csv"
CAMPOS = ["path", "clase", "x", "y", "lado", "tam"]


def leer_splits(archivo="splits.csv"):
    f = ROOT / archivo
    if not f.exists():
        return None
    with open(f, encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh)]


def armar_muestra(n_por_clase, rng, splits="splits.csv", fuente=None, solo_cv=True):
    """Devuelve [(ruta, clase)], balanceada por clase, independiente de near-duplicados.
    Con splits.csv usa solo split == "cv" (el hold-out no se toca) y una imagen por grupo.
    fuente: si se da (ej. "dv"), solo imágenes cuyo nombre empieza con ese prefijo.
    solo_cv=False: incluye también el hold-out (solo para fuentes que no se usan para ajustar)."""
    filas = leer_splits(splits)
    por_clase = defaultdict(list)
    if filas is not None:
        por_grupo = {}
        for r in filas:
            if solo_cv and r["split"] != "cv":
                continue
            if fuente and not Path(r["path"]).name.startswith(f"{fuente}_"):
                continue
            por_grupo.setdefault((r["clase"], r["grupo"]), []).append(ROOT / r["path"])
        for (cl, _), rutas in por_grupo.items():
            por_clase[cl].append(rng.choice(rutas))
        origen = (f"{splits} ({'solo split=cv' if solo_cv else 'cv + hold-out'}, una imagen por grupo"
                  + (f", fuente {fuente})" if fuente else ")"))
    else:
        for p in sorted(CLEAN.rglob("*")):
            if p.is_file() and p.suffix.lower() in EXT:
                por_clase[p.parent.name].append(p)
        origen = "data/clean al azar (no hay splits.csv)"
    muestra = []
    for cl, rutas in sorted(por_clase.items()):
        muestra += [(p, cl) for p in rng.sample(rutas, min(n_por_clase, len(rutas)))]
    return muestra, origen


class Bloques:
    """Estado de los bloques de una imagen (separado de la ventana para poder probarlo)."""

    def __init__(self, g, lado, k):
        self.g, self.lado, self.k = g, lado, k
        self.lista = []      # [(y, x)]
        self.aviso = ""

    def agregar(self, cx, cy):
        """cx, cy: clic en coordenadas de la imagen base. Devuelve True si se agregó."""
        t = self.g.shape[0]
        x = min(max(cx - self.lado // 2, 0), t - self.lado)
        y = min(max(cy - self.lado // 2, 0), t - self.lado)
        if len(self.lista) >= self.k:
            self.aviso = f"máximo {self.k} bloques (u para deshacer)"
            return False
        if (self.g[y:y + self.lado, x:x + self.lado] <= 10).any():
            self.aviso = "ese bloque toca fondo negro: no se agregó"
            return False
        self.lista.append((y, x))
        self.aviso = ""
        return True

    def deshacer(self):
        if self.lista:
            self.lista.pop()
        self.aviso = ""


def ya_anotadas():
    if not ROIS.exists():
        return set(), None
    with open(ROIS, encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
    conf = {(int(r["lado"]), int(r["tam"])) for r in filas}
    return {ROOT / r["path"] for r in filas}, (conf.pop() if len(conf) == 1 else "mezcla")


def guardar(p, clase, bloques, lado, tam):
    nuevo = not ROIS.exists()
    with open(ROIS, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if nuevo:
            w.writerow(CAMPOS)
        for y, x in bloques:
            w.writerow([p.relative_to(ROOT).as_posix(), clase, x, y, lado, tam])


def dibujar(estado, texto):
    im = cv2.resize(estado.g, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for y, x in estado.lista:
        cv2.rectangle(im, (x * ZOOM, y * ZOOM),
                      ((x + estado.lado) * ZOOM, (y + estado.lado) * ZOOM), (0, 255, 0), 2)
    cv2.putText(im, texto, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    if estado.aviso:
        cv2.putText(im, estado.aviso, (6, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
    return im


def cmd_anotar_rois(argv=None):
    """Marcá a mano bloques de tejido homogéneo en una muestra de imágenes (para el ENL).

Controles (ventana de OpenCV, imagen ampliada 3x):
  clic izquierdo    agrega un bloque centrado en ese punto
  u                 deshace el último bloque
  ENTER / espacio   guarda esta imagen y pasa a la siguiente
  s                 saltea la imagen (no se guarda nada)
  q / ESC           guarda lo hecho y sale

Criterio para elegir un bloque (el mismo en todas las imágenes, y SIN mirar ningún filtro):
  tejido liso (estroma): sin bordes, sin folículos, sin sombras acústicas, sin marcas
  ni cruces, y lejos del borde del ovario. Marcá 3 bloques por imagen (mínimo 1, máximo --k).

Se puede retomar: las imágenes que ya están en rois.csv se saltean.
Los bloques se guardan en coordenadas de la imagen base de --tam x --tam (la misma que usa
preprocesamiento.py elegir-filtro), así que no cambies --tam ni --lado entre sesiones.

Uso:
    python preprocesamiento.py anotar-rois --n-por-clase 20          # 40 imágenes en total
    python preprocesamiento.py elegir-filtro --rois rois.csv
"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-por-clase", type=int, default=20)
    ap.add_argument("--tam", type=int, default=256)
    ap.add_argument("--lado", type=int, default=16)
    ap.add_argument("--k", type=int, default=5, help="máximo de bloques por imagen")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    hechas, conf = ya_anotadas()
    if conf not in (None, (args.lado, args.tam)):
        raise SystemExit(f"rois.csv se anotó con (lado, tam) = {conf}; usá los mismos valores.")
    rng = random.Random(args.seed)
    muestra, origen = armar_muestra(args.n_por_clase, rng)
    rng.shuffle(muestra)   # mezcla las clases para no anotar todas las infected seguidas
    pendientes = [(p, cl) for p, cl in muestra if p not in hechas]
    print(f"Muestra: {origen}. Total {len(muestra)} | ya anotadas {len(muestra) - len(pendientes)} "
          f"| pendientes {len(pendientes)}")

    vent = "anotar ROI"
    cv2.namedWindow(vent, cv2.WINDOW_AUTOSIZE)
    for i, (p, cl) in enumerate(pendientes, 1):
        est = Bloques(base(p, args.tam), args.lado, args.k)
        cv2.setMouseCallback(vent, lambda ev, mx, my, fl, par, e=est:
                             e.agregar(mx // ZOOM, my // ZOOM) if ev == cv2.EVENT_LBUTTONDOWN else None)
        while True:
            cv2.imshow(vent, dibujar(est, f"{i}/{len(pendientes)}  bloques: {len(est.lista)}/{args.k}  "
                                          f"| clic=agregar  u=deshacer  ENTER=guardar  s=saltear  q=salir"))
            tecla = cv2.waitKey(30) & 0xFF
            if tecla == ord("u"):
                est.deshacer()
            elif tecla in (13, 32):
                if est.lista:
                    guardar(p, cl, est.lista, args.lado, args.tam)
                    break
                est.aviso = "marcá al menos un bloque (o s para saltear)"
            elif tecla == ord("s"):
                break
            elif tecla in (ord("q"), 27):
                cv2.destroyAllWindows()
                print(f"Listo. Marcas en {ROIS}")
                return
    cv2.destroyAllWindows()
    print(f"Terminaste la muestra. Marcas en {ROIS}")


# ============================================================================
# COMANDO elegir-filtro: ranking de filtros por ENL
# ============================================================================
# ---------------------------------------------------------------- ENL y bloques marcados
def enl(img, bloques, lado=16, eps=1e-3):
    """Mediana del ENL = media^2 / varianza sobre los bloques dados."""
    f = img.astype(np.float64)
    vals = []
    for y, x in bloques:
        b = f[y:y + lado, x:x + lado]
        vals.append(b.mean() ** 2 / (b.var() + eps))
    return float(np.median(vals))


def leer_rois(archivo):
    """rois.csv -> ({ruta: (clase, [(y, x), ...])}, lado, tam)."""
    por_img, lados, tams = {}, set(), set()
    with open(archivo, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ruta = ROOT / r["path"]
            por_img.setdefault(ruta, (r["clase"], []))[1].append((int(r["y"]), int(r["x"])))
            lados.add(int(r["lado"]))
            tams.add(int(r["tam"]))
    if not por_img:
        raise SystemExit(f"{archivo} está vacío.")
    if len(lados) != 1 or len(tams) != 1:
        raise SystemExit("rois.csv mezcla distintos lado/tam: anotá todo con la misma configuración.")
    return por_img, lados.pop(), tams.pop()


def figura_rois(items, lado_bloque, destino, cols=4, n=12):
    tiles = []
    for g, bloques in items[:n]:
        im = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        for y, x in bloques:
            cv2.rectangle(im, (x, y), (x + lado_bloque, y + lado_bloque), (0, 255, 0), 1)
        tiles.append(im)
    if not tiles:
        return
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    filas = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(destino), np.vstack(filas))


# ---------------------------------------------------------------- main
def cmd_elegir_filtro(argv=None):
    """Elige el filtro de speckle usando ENL como único criterio, sobre los bloques que marcaste a mano.
 
ENL = media^2 / varianza, medido en bloques de tejido homogéneo. Más ENL = más suavizado del speckle.
 
Entrada: rois.csv, generado por preprocesamiento.py anotar-rois (imagen, clase, posición y tamaño de cada bloque).
Los bloques son los MISMOS píxeles para la imagen original y para todos los filtros, así que la
comparación es pareada. Se marcan antes de ver ningún filtro.
 
Salidas en reports/:
  enl_resultados_manual.csv   una fila por imagen y filtro (ENL, cociente vs. original, beta)
  enl_resumen_manual.txt      ranking por ENL + tests estadísticos pareados
  enl_rois_manual.png         bloques sobre algunas imágenes (para verificar)
 
Uso:
    python preprocesamiento.py elegir-filtro                 # lee rois.csv de la raíz del repo
    python preprocesamiento.py elegir-filtro --rois otro.csv
"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--rois", default="rois.csv", help="CSV generado por anotar_rois.py")
    args = ap.parse_args(argv)

    manual, lado, tam = leer_rois(ROOT / args.rois)
    muestra = [(p, cl) for p, (cl, _) in manual.items()]
    origen = f"{args.rois} (bloques marcados a mano)"
    filtros = {k: v for k, v in FILTROS.items() if k != "ninguno"}
    CADENAS = ["wiener3+frost3"]
    filtros.update({c: (lambda im, c=c: filtrar(im, c)) for c in CADENAS})
    nombres = list(filtros)
    REPORTES.mkdir(exist_ok=True)

    filas, enl_o, enl_f, betas, tiempos = [], [], defaultdict(list), defaultdict(list), defaultdict(list)
    para_fig = []
    for p, cl in muestra:
        g = base(p, tam)
        bloques = manual[p][1]
        para_fig.append((g, bloques))
        e0 = enl(g, bloques, lado)
        enl_o.append(e0)
        for m in nombres:
            t0 = time.perf_counter()
            gf = filtros[m](g)
            tiempos[m].append(time.perf_counter() - t0)
            e, b = enl(gf, bloques, lado), beta(g, gf)
            enl_f[m].append(e)
            betas[m].append(b)
            filas.append([p.name, cl, m, round(e0, 3), round(e, 3), round(e / e0, 3), round(b, 4)])

    n = len(enl_o)
    if n < 10:
        raise SystemExit(f"Solo {n} imágenes marcadas: marcá más con anotar_rois.py (mínimo 10).")
    ratio = {m: np.array(enl_f[m]) / np.array(enl_o) for m in nombres}

    with open(REPORTES / "enl_resultados_manual.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["imagen", "clase", "filtro", "enl_original", "enl_filtrada", "cociente", "beta"])
        w.writerows(filas)
    figura_rois(para_fig, lado, REPORTES / "enl_rois_manual.png")

    orden = sorted(nombres, key=lambda m: -np.median(ratio[m]))
    L = [f"Muestra: {origen}",
         f"Imágenes marcadas: {n} | bloque {lado}x{lado} en {tam}x{tam}",
         f"ENL mediano de la imagen original en esos bloques: {np.median(enl_o):.2f}", "",
         "RANKING POR ENL (cociente = ENL filtrada / ENL original; mayor = más supresión de speckle)",
         f"{'filtro':15s}{'cociente med.':>14s}{'IQR':>16s}{'ENL med.':>10s}{'beta*':>8s}{'ms/img':>9s}"]
    for m in orden:
        q1, q3 = np.percentile(ratio[m], [25, 75])
        L.append(f"{m:15s}{np.median(ratio[m]):14.2f}{f'[{q1:.2f}, {q3:.2f}]':>16s}"
                 f"{np.median(enl_f[m]):10.2f}{np.mean(betas[m]):8.3f}{1000 * np.mean(tiempos[m]):9.1f}")
    L += ["", "* beta (preservación de bordes) se reporta pero NO entra en el ranking."]

    # tests pareados (mismas imágenes y mismos bloques para todos)
    chi, p_fr = stats.friedmanchisquare(*[ratio[m] for m in nombres])
    L += ["", f"Friedman entre los {len(nombres)} filtros: chi2 = {chi:.1f}, p = {p_fr:.2g}"]
    top = orden[0]
    L.append(f"Wilcoxon pareado de '{top}' contra cada uno (p con Bonferroni, {len(nombres) - 1} comparaciones):")
    for m in orden[1:]:
        try:
            p = stats.wilcoxon(ratio[top], ratio[m]).pvalue
        except ValueError:
            p = 1.0
        L.append(f"  {top} vs {m:15s} p = {min(1.0, p * (len(nombres) - 1)):.3g}")
    # filtros no dominados: ningún otro tiene a la vez más ENL y más beta
    r = {m: float(np.median(ratio[m])) for m in nombres}
    b = {m: float(np.mean(betas[m])) for m in nombres}
    dominados = {m: [o for o in orden if o != m and r[o] >= r[m] and b[o] >= b[m]
                     and (r[o] > r[m] or b[o] > b[m])] for m in nombres}
    L += ["", "FILTROS NO DOMINADOS (ningún otro suaviza más Y conserva más bordes):",
          "  " + ", ".join(m for m in orden if not dominados[m])]
    L += ["",
          "ATENCIÓN: el ENL sube con cualquier suavizado, así que solo con ENL gana el filtro que más",
          "borronea. Mirá beta y las imágenes filtradas (comparar_preprocesamiento.py) antes de decidir,",
          "y validá el finalista con máscaras manuales (Dice) en unas 15-20 imágenes.",
          "Revisá reports/enl_rois_manual.png: los cuadrados verdes tienen que caer en tejido liso, no en bordes."]
    texto = "\n".join(L)
    print(texto)
    (REPORTES / "enl_resumen_manual.txt").write_text(texto + "\n", encoding="utf-8")


# ============================================================================
# COMANDO procesar: data/clean -> data/pre + control de atajos
# ============================================================================
PRE = ROOT / "data" / "pre"


def estadisticos(g):
    x = g.astype(np.float32) / 255.0
    m, s = float(x.mean()), float(x.std())
    z = (x - m) / (s + 1e-6)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1)
    return {
        "media": m, "desvio": s,
        "asimetria": float((z ** 3).mean()), "curtosis": float((z ** 4).mean()),
        "entropia": entropia(g),
        "grad_medio": float(np.hypot(gx, gy).mean()),
        "lap_var": float(cv2.Laplacian(x, cv2.CV_32F).var()),
        "frac_oscura": float((x < 0.1).mean()), "frac_clara": float((x > 0.9).mean()),
    }


def grupos_de_splits(claves, archivo="splits.csv"):
    """(clase, nombre sin extensión) -> grupo, desde el CSV de splits. None si falta algo."""
    f = ROOT / archivo
    if not f.exists():
        return None
    mapa = {}
    with open(f, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            mapa[(r["clase"], Path(r["path"]).stem)] = int(r["grupo"])
    if not all(k in mapa for k in claves):
        return None
    return np.array([mapa[k] for k in claves])


def control_atajos(filas, claves, seed, archivo="splits.csv"):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict

    clases = sorted({k[0] for k in claves})
    if len(clases) != 2 or len(filas) < 50:
        return None
    nombres = list(filas[0])
    X = np.array([[f[n] for n in nombres] for f in filas])
    y = np.array([k[0] == clases[1] for k in claves], dtype=int)
    grupos = grupos_de_splits(claves, archivo)
    rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    if grupos is not None:
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        p = cross_val_predict(rf, X, y, cv=cv, groups=grupos, method="predict_proba")[:, 1]
        cv_txt = f"CV agrupada por near-duplicados ({archivo})"
    else:
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        p = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]
        cv_txt = "CV sin agrupar (no se encontró splits.csv completo)"
    solas = {}
    for k, n in enumerate(nombres):
        a = roc_auc_score(y, X[:, k])
        solas[n] = max(a, 1 - a)
    return roc_auc_score(y, p), solas, cv_txt


def cmd_procesar(argv=None):
    """Aplica preprocess() a todo data/clean -> data/pre (PNG sin pérdida) y corre un
control de atajos sobre el resultado.

El control entrena un Random Forest que NO ve la imagen sino estadísticos de bajo nivel
(histograma, gradiente, ruido). Si todavía separa las clases, el preprocesamiento no alcanzó
a borrar las pistas de la fuente. Usa los grupos de splits.csv si existe (CV sin fuga).

Uso (desde cualquier carpeta):
    python preprocesamiento.py procesar --filtro mediana --contraste clahe --tam 256
"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--filtro", default="wiener3")
    ap.add_argument("--contraste", default="ninguno")
    ap.add_argument("--tam", type=int, default=256)
    ap.add_argument("--sin-marcas", action="store_true")
    ap.add_argument("--sin-recorte", action="store_true")
    ap.add_argument("--sin-texto", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", default="splits.csv",
                    help="CSV con los grupos para la CV del control (en la raíz del repo)")
    args = ap.parse_args(argv)

    rutas = sorted(p for p in CLEAN.rglob("*") if p.is_file() and p.suffix.lower() in EXT)
    if not rutas:
        raise SystemExit(f"No hay imágenes en {CLEAN}.")
    if PRE.exists():
        shutil.rmtree(PRE)

    filas, claves, fallos = [], [], []
    for i, p in enumerate(rutas, 1):
        try:
            g = preprocess(p, args.filtro, args.contraste, args.tam,
                           marcas=not args.sin_marcas, recortar=not args.sin_recorte,
                           texto=not args.sin_texto)
        except Exception as e:
            fallos.append((p.name, str(e)))
            continue
        dest = PRE / p.parent.name
        dest.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest / f"{p.stem}.png"), g)
        filas.append(estadisticos(g))
        claves.append((p.parent.name, p.stem))
        if i % 200 == 0:
            print(f"  {i}/{len(rutas)}")

    L = [f"Config: filtro={args.filtro} contraste={args.contraste} tam={args.tam} "
         f"marcas={not args.sin_marcas} recorte={not args.sin_recorte} texto={not args.sin_texto}",
         f"Procesadas: {len(filas)} de {len(rutas)} -> {PRE}"]
    if fallos:
        L.append(f"Fallaron {len(fallos)}: " + ", ".join(n for n, _ in fallos[:5]))
    res = control_atajos(filas, claves, args.seed, args.splits)
    if res is not None:
        auc, solas, cv_txt = res
        if auc >= 0.9:
            v = "las clases siguen siendo separables por estadísticos de bajo nivel: hay pistas de la fuente"
        elif auc >= 0.65:
            v = "señal moderada: puede ser patología o fuente, no se puede distinguir con este control"
        else:
            v = "sin señal fuerte"
        L += ["", f"Control de atajos tras preprocesar (solo estadísticos, sin ver el contenido): "
                  f"AUC = {auc:.3f} -> {v}", f"  ({cv_txt})",
              "  AUC por estadístico: " + ", ".join(f"{k}={v:.2f}" for k, v in
                                                    sorted(solas.items(), key=lambda kv: -kv[1]))]
    texto = "\n".join(L)
    print(texto)
    REPORTES.mkdir(exist_ok=True)
    (REPORTES / "resumen_procesamiento.txt").write_text(texto + "\n", encoding="utf-8")


COMANDOS = {
    "comparar": cmd_comparar,
    "anotar-rois": cmd_anotar_rois,
    "elegir-filtro": cmd_elegir_filtro,
    "procesar": cmd_procesar,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMANDOS:
        print("Uso: python procesamiento/preprocesamiento.py <comando> [opciones]\n\nComandos:")
        for k, f in COMANDOS.items():
            print(f"  {k:18s}{(f.__doc__ or '').strip().splitlines()[0] if f.__doc__ else ''}")
        print("\nPara ver las opciones de uno: python procesamiento/preprocesamiento.py <comando> -h")
        raise SystemExit(1)
    COMANDOS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
