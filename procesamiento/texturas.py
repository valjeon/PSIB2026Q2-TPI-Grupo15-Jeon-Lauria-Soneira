"""Calcula, a partir de la segmentación (segmentacion.py), 36 características para cada imagen
agrupadas en: cantidad y tamaños de folículos, forma, distribución espacial, relación folículo-estroma,
textura del estroma (GLCM) e interfaz (borde del folículo).

Flujo:
  1. elegir_muestra: hasta --n-por-grupo imágenes de cada grupo de splits2.csv
  2. segment_detalle (segmentacion.py) -> caracteristicas(): una fila por imagen
  3. resumir_grupos: mediana e IQR de cada característica dentro de cada grupo
  4. comparar: PCOS vs sanas con las medianas de grupo (AUC + permutación), por fuente de dónde se extrajo la imagen
  5. CSV, figuras y caracteristicas_resumen.txt en reports/

Definiciones:
  campo (ROI)  = porción de la ecografía que devuelve la segmentación.
  estroma      = campo - folículos - un anillo del borde de cada folículo.

Datos (extraídos de dos fuentes distintas):
  Kaggle            1.699 imágenes que son aumentaciones (zoom, rotación, recorte)
                         de solo 8 ecografías: 4 PCOS y 4 sanas.
  Dataverse         52 imágenes de 50 ecografías.
Cada ecografía original es un `grupo` en splits2.csv. Las variantes de un grupo no son
muestras independientes, así que PCOS vs sanas se compara con una mediana por grupo, y
dentro de cada fuente, ya que puede haber diferencias de tamaño y encuadre entre ellas.

Salidas (reports/):
caracteristicas_imagenes.csv        una fila por imagen
caracteristicas_grupos.csv          mediana e IQR por grupo
caracteristicas_comparacion.csv     PCOS vs sanas por fuente
caracteristicas_resumen.txt         igual que caracteristicas_comparacion.csv pero en formato de texto,
                                    con la explicación de cada columna
caracteristicas_montaje.png         una ecografía representativa por grupo (folículos en rojo,
                                    estroma en verde); si se quiere omitir se escribe --sin-montaje
figuras_caracteristicas/            carpeta (solo con --imagenes N): un PNG por imagen, hasta N figuras 
                                    por grupo. Cada PNG tiene tres paneles: contornos numerados | 
                                    estroma y folículos | mediciones.
                                    Se vacía al empezar cada corrida.

Uso (por defecto: multiotsu + wiener3 + sin contraste, hasta 40 imágenes por grupo)
    python procesamiento/texturas.py --n-por-grupo 40 --imagenes 3
    python procesamiento/texturas.py --n-por-grupo 0        # todas las imágenes
    python procesamiento/texturas.py --metodo chanvese --filtro frost7 --contraste ninguno #otra segmentación,
                                                                                            se puede usar para evaluar
                                                                                            sensibilidad.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import csv
import itertools
import math
import random
import shutil
import time
import cv2
import numpy as np
from skimage import measure
from skimage.feature import graycomatrix, graycoprops
from segmentacion import metricas_foliculos, segment_detalle

ROOT = Path(__file__).resolve().parent # raíz del repo
if ROOT.name == "procesamiento":
    ROOT = ROOT.parent
REPORTES = ROOT / "reports" # carpeta donde se guardan las salidas
NAN = float("nan")  # NAN se usa para las medidas que no se pueden calcular ej: la forma media de los folículos en una imagen que no tiene ninguno
FONT = cv2.FONT_HERSHEY_SIMPLEX  # fuente de los textos en las figuras


# Agranda una máscara r píxeles hacia todos lados
def _dilatar(mask, r):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)
# dirve para definir el estroma
# se agrandan los folículos 3 píxeles y se excluye ese anillo para segmentar bien los folículos del estroma

# función que mide que tan heterogeneo es el estroma (que tan variados son los grises)
def _entropia(vals, bins=64):
    h, _ = np.histogram(vals, bins=bins, range=(0, 256)) # histograma mide cuántos píxeles hay de cada gris
    p = h[h > 0] / h.sum() # pasa a proporciones
    return float(-(p * np.log2(p)).sum()) # pasa a escala logarítmica
# 0: todos los píxeles tienen el mismo gris (tejido homogéneo)
# 6 (= log₂ 64): los grises están repartidos parejo entre los 64 niveles (tejido muy heterogéneo)


# Compara el área de la forma con su perímetro
# 1: círculo perfecto
# menor a 1: más irregular
def _circularidad(r):
    return 4 * np.pi * r.area / r.perimeter ** 2 if r.perimeter > 0 else NAN


# GLCM del estroma: contraste, disimilitud, homogeneidad, energía y correlación
# cuenta qué grises aparecen uno al lado del otro
# contrast: los vecinos suelen tener grises muy distintos (textura áspera)
# dissimilarity: parecido a contraste, pero crece de forma más gradual
# homogeneity: los vecinos suelen tener grises parecidos (textura suave)
# energy: los vecinos suelen tener grises parecidos (textura suave)
# correlation: el gris de un píxel predice bien el de su vecino (estructura)
def glcm_estroma(img, estroma, niveles=16, distancias=(1, 2),
                 angulos=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4), min_pix=400):
    props = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")
    if estroma.sum() < min_pix:
        return {p: NAN for p in props} # Con menos de 400 píxeles de estroma la textura no es confiable (NaN)
    # reduce 256 grises a 16 niveles sino la matriz queda demasiado grande
    # los píxeles de estroma quedan con valores 1 a 16 y todo lo demás queda en 0
    q = np.zeros(img.shape, np.uint8)
    q[estroma] = 1 + np.minimum(img[estroma].astype(np.int32) * niveles // 256, niveles - 1)
    # matriz de conteos
    # [1:, 1:] borra la fila y la columna del nivel 0
    # desaparece todo par donde algún píxel no es estroma: quedan solo pares estroma-estroma.
    g = graycomatrix(q, list(distancias), list(angulos), levels=niveles + 1,
                     symmetric=True, normed=False)[1:, 1:].astype(np.float64) # queda simetrica
    # calcula la tabla para 2 distancias (vecino a 1 y a 2 píxeles) y 4 direcciones (0°, 45°, 90°, 135°)
    # normalizar para que queden valores de probabilidad (suma = 1).
    tot = g.sum(axis=(0, 1), keepdims=True)     # total de pares por distancia y ángulo
    if (tot == 0).all():
        return {p: NAN for p in props}
    g = g / np.maximum(tot, 1)
    # silenciar warnings de NaN y división por cero (si algún par de distancia-ángulo no tiene pares estroma-estroma)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {p: float(np.nanmean(graycoprops(g, p))) for p in props}
    # Calcula las 5 propiedades en las 8 tablas y las promedia


def caracteristicas(d, margen_estroma=3):
    """Características de una imagen a partir de segment_detalle(). Devuelve un dict;
    lo que no está definido queda en NaN.

    Conteo      n_foliculos, area_roi_px, densidad_x1e4px, frac_area_rechazos
    Tamaño      area_total/media/mediana/max/min_px, area_cv, dominancia_max,
                diam_eq_rel_medio/max (relativos a sqrt(área del campo))
    Forma       circularidad, excentricidad, solidez y ratio_ejes (medias)
    Distrib.    dist_vecino_rel, clark_evans_R
    Fol-estr.   frac_area_foliculos, ratio_area_fol_estroma, brillo_foliculos,
                ratio_brillo_fol_estroma, contraste_michelson, cnr_fol_estroma
    Estroma     brillo_estroma, estroma_std, estroma_entropia, glcm_* (5 propiedades)
    Interfaz    contraste, CNR y gradiente en el borde (medias de metricas_foliculos)
    """
    # d = salida de segment_detalle (segmentacion.py).
    #   base = sin filtrar ; pre = filtrada ; rech = regiones oscuras descartadas por la segmentación.
    roi, mask, lab, rech = d["roi"], d["mask"], d["labels"], d["rechazos"]
    # roi: máscara del campo: la parte útil de la ecografía
    # mask: máscara de todos los folículos juntos
    # rechazos: zonas oscuras que la segmentación descartó como folículos
    # base: la imagen sin filtrar
    # pre: la imagen filtrada (sin ruido)
    base, pre = d["base"], d["pre"]
    a_roi = float(roi.sum())
    estroma = roi & ~_dilatar(mask, margen_estroma) 
    a_fol, a_est = float(mask.sum()), float(estroma.sum())
    escala = np.sqrt(a_roi) # longitud típica de la ROI

    n = int(lab.max()) # cantidad de folículos
    out = dict(n_foliculos=n, area_roi_px=a_roi,
               densidad_x1e4px=1e4 * n / a_roi, # folículos cada 10.000 px de ROI
               frac_area_foliculos=a_fol / a_roi, # qué parte del campo son folículos
               ratio_area_fol_estroma=a_fol / a_est if a_est > 0 else NAN,
               frac_area_rechazos=float(rech.sum()) / a_roi) # cuánto descartó la segmentación

    # brillo medio, desvío y entropía
    # se dividen por 255 para que queden entre 0 y 1
    if a_est > 0:
        pe = pre[estroma].astype(np.float32) 
        # pre[estroma] es una lista con los grises de los píxeles de estroma
        out.update(brillo_estroma=float(base[estroma].mean()) / 255, 
                   # el brillo se mide en base (sin filtrar) para no alterar el valor real
                   estroma_std=float(pe.std()) / 255, 
                   # el desvío se mide en pre (filtrada) para no incluir ruido
                   estroma_entropia=_entropia(pre[estroma]))
    else:
        out.update(brillo_estroma=NAN, estroma_std=NAN, estroma_entropia=NAN)
    out.update({f"glcm_{k}": v for k, v in glcm_estroma(pre, estroma).items()})

    props = measure.regionprops(lab) 
    # regionprops mide propiedades de cada folículo: área, perímetro, centro, ejes
    if not props: # sin folículos: lo que no está definido queda en NaN
        out.update(area_total_px=0.0, area_media_px=NAN, area_mediana_px=NAN, area_max_px=NAN,
                   area_min_px=NAN, area_cv=NAN, dominancia_max=NAN, diam_eq_rel_medio=NAN,
                   diam_eq_rel_max=NAN, circularidad_media=NAN, excentricidad_media=NAN,
                   solidez_media=NAN, ratio_ejes_medio=NAN, dist_vecino_rel=NAN,
                   clark_evans_R=NAN, brillo_foliculos=NAN, ratio_brillo_fol_estroma=NAN,
                   contraste_michelson=NAN, cnr_fol_estroma=NAN,
                   contraste_interfaz_medio=NAN, cnr_interfaz_medio=NAN,
                   gradiente_borde_medio=NAN)
        return out

    # tamaño y forma
    a = np.array([r.area for r in props], float)
    deq = np.sqrt(4 * a / np.pi) # diámetro del círculo equivalente (de igual área)
    # Sirve para hablar de diámetro aunque el folículo no sea redondo
    circ = [c for c in map(_circularidad, props) if np.isfinite(c)]
    out.update(
        area_total_px=float(a.sum()), area_media_px=float(a.mean()), 
        area_mediana_px=float(np.median(a)), area_max_px=float(a.max()), area_min_px=float(a.min()),
        # tamaños de los folículos
        area_cv=float(a.std() / a.mean()), # coef. de variación
        # 0 = todos el mismo tamaño
        # alto = tamaños muy desparejos
        dominancia_max=float(a.max() / a.sum()), # área del más grande / área total
        # cerca de 1 = un folículo domina (típico de un folículo)
        # bajo = muchos parecidos (más típico de PCOS)
        diam_eq_rel_medio=float(deq.mean() / escala), diam_eq_rel_max=float(deq.max() / escala), # diámetro relativo al tamaño del campo
        circularidad_media=float(np.mean(circ)) if circ else NAN, # que tan redondos son
        excentricidad_media=float(np.mean([r.eccentricity for r in props])), # 0 = círculo, 1 = muy alargado
        solidez_media=float(np.mean([r.solidity for r in props])), # 1 = sin entrantes, bajo = forma con concavidades
        ratio_ejes_medio=float(np.mean([r.axis_minor_length / max(r.axis_major_length, 1e-6)
                                        for r in props]))) # eje menor/eje mayor: 1 = redondo

    # distribución espacial (se necesitan por lo menos 2 folículos)
    if len(props) >= 2:
        c = np.array([r.centroid for r in props]) # c: el centro de cada folículo.
        dd = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1)) # distancias entre todos los centroides
        np.fill_diagonal(dd, np.inf) 
        # la distancia de un folículo a sí mismo (0) se reemplaza por infinito
        # para que no cuente como su vecino más cercano
        dmin = float(dd.min(1).mean()) # distancia media al vecino más cercano
        # Clark-Evans: observada / esperada si estuviera al azar en la misma ROI
        # R cociente entre dos distancias 
        # R aprox 1: distribución al azar
        # R < 1: folículos amontonados
        # R > 1: folículos espaciados de forma regular
        out.update(dist_vecino_rel=dmin / escala,
                   clark_evans_R=dmin / (0.5 * np.sqrt(a_roi / len(props))))
    else:
        out.update(dist_vecino_rel=NAN, clark_evans_R=NAN)

    # brillo medio de los folículos
    bf = float(base[mask].mean()) / 255
    out["brillo_foliculos"] = bf # los folículos son líquido: se ven oscuros
    if a_est > 0:
        be = out["brillo_estroma"]
        out["ratio_brillo_fol_estroma"] = bf / be if be > 0 else NAN # cerca de 0 = folículos mucho más oscuros que el estroma
        # Michelson:(Ie - If)/(Ie + If)
        out["contraste_michelson"] = (be - bf) / (be + bf + 1e-6) # contraste entre 0 y 1
        # CNR: diferencia de medias en unidades de desvío
        # CNR: contrast-to-noise ratio
        # una diferencia de brillo de 50 grises no significa lo mismo en una imagen limpia que en una muy ruidosa
        # CNR divide la diferencia por el ruido
        pf = pre[mask].astype(np.float32)
        out["cnr_fol_estroma"] = float((pe.mean() - pf.mean()) / np.sqrt(pe.var() + pf.var() + 1e-6))
    else:
        out.update(ratio_brillo_fol_estroma=NAN, contraste_michelson=NAN, cnr_fol_estroma=NAN)

    # interfaz
    fol = metricas_foliculos(d) # funcion de segmentacion.py
    # mide cada folículo contra un anillo alrededor (NO con todo el estroma)
    # contraste local, CNR local y gradiente
    # se promedia entre folículos
    for nombre, clave in (("contraste_interfaz_medio", "contraste_interfaz"),
                          ("cnr_interfaz_medio", "cnr_interfaz"),
                          ("gradiente_borde_medio", "gradiente_borde")):
        out[nombre] = float(np.mean([f[clave] for f in fol])) if fol else NAN
    return out


# imágenes de resultado
def _f(v, nd=2):
    return "-" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"
# pasa los número a texto con nd decimales
# pasa los espacios a NaN

# amplía una imagen z veces, solo para figuras
# suaviza con interpolación bicúbica para que no se vean los píxeles cuadrados
def _zoom_img(img, z):
    return cv2.resize(img, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)


# aplía una máscara z veces
def _zoom_mask(m, z):
    return np.repeat(np.repeat(m, z, axis=0), z, axis=1)


# imagen base ampliada z veces, con el contorno de la ROI (celeste) y los folículos numerados (rojo)
def panel_foliculos(d, z=2):
    # se pasa la imagen a RGB para dibujar en color
    img = _zoom_img(cv2.cvtColor(d["base"], cv2.COLOR_GRAY2RGB), z)
    roi = _zoom_mask(d["roi"], z).astype(np.uint8)
    cr, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cr, -1, (0, 160, 255), 1)        # ROI
    labz = _zoom_mask(d["labels"], z)
    for r in measure.regionprops(labz):                     # un folículo por loop
        cs, _ = cv2.findContours((labz == r.label).astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cs, -1, (255, 40, 40), 1)     # folículo
        # número del folículo centrado en el centroide
        txt = str(r.label)
        (w, h), _ = cv2.getTextSize(txt, FONT, 0.4, 1)
        cy, cx = r.centroid
        cv2.putText(img, txt, (int(cx - w / 2), int(cy + h / 2)), FONT, 0.4, (255, 255, 0), 1,
                    cv2.LINE_AA)
    return img

# imagen base con el estroma (verde), folículos (rojo) y zonas oscuras que la segmentación descartó (amarillo)
def panel_estroma(d, margen_estroma=3, z=2):
    img = _zoom_img(cv2.cvtColor(d["base"], cv2.COLOR_GRAY2RGB), z).astype(np.float32)
    roi, mask = _zoom_mask(d["roi"], z), _zoom_mask(d["mask"], z)
    # se usa el mismo estroma que en caracteristicas()
    estroma = roi & ~_dilatar(mask, margen_estroma * z)
    img[estroma] = 0.65 * img[estroma] + 0.35 * np.array([0, 200, 0]) # el color se mezcla
    # 0,65 × imagen + 0,35 × verde
    # así se tiñe sin tapar la ecografía
    img[mask] = 0.45 * img[mask] + 0.55 * np.array([255, 0, 0])
    out = img.astype(np.uint8)
    rech = _zoom_mask(d["rechazos"], z)
    if rech.any():
        cr, _ = cv2.findContours(rech.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cr, -1, (255, 220, 0), 1) # descartados
    return out


# panel oscuro con las mediciones de la imagen y la tabla de folículos
# muestra medidas principales y una tabla por folículo que tenga número, área, circularidad
def panel_texto(d, c, titulo, lado):
    p = np.full((lado, lado, 3), 30, np.uint8)
    L = [titulo, "",
         f"foliculos: {c['n_foliculos']}    campo (ROI): {_f(c['area_roi_px'], 0)} px",
         f"area fol. (px): total {_f(c['area_total_px'], 0)} | media {_f(c['area_media_px'], 0)}"
         f" | max {_f(c['area_max_px'], 0)}",
         f"fraccion area fol/campo:    {_f(c['frac_area_foliculos'], 3)}",
         f"relacion area fol/estroma:  {_f(c['ratio_area_fol_estroma'], 3)}",
         f"brillo fol/estroma: {_f(c['ratio_brillo_fol_estroma'])}"
         f" (Michelson {_f(c['contraste_michelson'])})",
         f"CNR fol-estroma: {_f(c['cnr_fol_estroma'])}",
         f"circularidad {_f(c['circularidad_media'])} | excentr. {_f(c['excentricidad_media'])}",
         f"Clark-Evans R: {_f(c['clark_evans_R'])}  (<1 apinados)",
         f"estroma: brillo {_f(c['brillo_estroma'])} | GLCM contr. {_f(c['glcm_contrast'])}"
         f" | homog. {_f(c['glcm_homogeneity'])}",
         "", "id   area(px)  circ.   (rojo en la imagen)"]
    props = measure.regionprops(d["labels"])
    ancho_fila, y = 17, 22
    libres = max((lado - y - ancho_fila * len(L)) // ancho_fila, 0)
    for r in props[:libres]:
        circ = _circularidad(r)
        L.append(f"{r.label:<4d} {r.area:8.0f}  {_f(circ)}")
    if len(props) > libres:
        L.append(f"... (+{len(props) - libres} mas)")
    for s in L:
        cv2.putText(p, s, (10, y), FONT, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
        y += ancho_fila
    return p


# une los tres paneles uno al lado del otro
# contornos numerados, estroma (verde) y folículos (rojo), mediciones
def figura_imagen(d, c, titulo, margen_estroma=3, z=2):
    a = panel_foliculos(d, z)
    return np.hstack([a, panel_estroma(d, margen_estroma, z), panel_texto(d, c, titulo, a.shape[0])])


# arma una grilla de 'cols' columnas con los paneles 
# es una grilla de paneles con texto abajo
def montaje_grupos(items, cols=4, pie=40):
    lado = items[0][0].shape[0]
    filas = -(-len(items) // cols)
    hoja = np.full((filas * (lado + pie), cols * lado, 3), 20, np.uint8)
    for k, (panel, lineas) in enumerate(items):
        r, col = divmod(k, cols)
        y0, x0 = r * (lado + pie), col * lado
        hoja[y0:y0 + lado, x0:x0 + lado] = panel
        for j, s in enumerate(lineas):
            cv2.putText(hoja, s, (x0 + 6, y0 + lado + 16 + 16 * j), FONT, 0.42, (235, 235, 235), 1,
                        cv2.LINE_AA)
    return hoja


# toma una imagen representativa por grupo
# la imagen representativa de cada grupo es la que tiene la frac_area_foliculos más cercana a la mediana del grupo
def hacer_montaje(filas, grupos, args):
    items = []
    # orden: PCOS primero, después por número de grupo
    for g in sorted(grupos.values(), key=lambda g: (-int(g["label"]), int(g["grupo"]))):
        fs = [x for x in filas if x["grupo"] == g["grupo"]]
        rep = min(fs, key=lambda x: abs(x["frac_area_foliculos"] - g["frac_area_foliculos_med"]))
        # se vuelve a segmentar porque no se guardan los `d` de cada imagen para ahorrar memoria
        d = segment_detalle(resolver_imagen(rep["path"], rep["clase"]), filtro=args.filtro,
                            contraste=args.contraste, metodo=args.metodo)
        # se vuelve a segmentar porque no se guardaron los resultados de la segmentación de cada imagen
        # para ahorrar memoria
        items.append((panel_estroma(d, args.margen_estroma, 1),         # z=1 --> no se amplía
                      [f"g{g['grupo']} {rep['clase']}  n={rep['n_foliculos']}",
                       f"fol/campo {_f(rep['frac_area_foliculos'], 3)}  "
                       f"fol/estroma {_f(rep['ratio_area_fol_estroma'], 3)}"]))
    cv2.imwrite(str(REPORTES / "caracteristicas_montaje.png"),
                cv2.cvtColor(montaje_grupos(items), cv2.COLOR_RGB2BGR))  # guarda en BGR


# resumen por grupo

# para cada grupo (cada ecografía original) y cada medida calcula 
# la mediana (el valor representativo)
# el rango intercuartílico (IQR: percentil 75 − percentil 25) (cuánto varía esa medida entre copias de la misma ecografía)
def resumir_grupos(filas, claves):
    por_grupo = {}
    for f in filas:
        por_grupo.setdefault(f["grupo"], []).append(f)
    res = {}
    for g, fs in sorted(por_grupo.items(), key=lambda kv: int(kv[0])):
        fila = dict(grupo=g, clase=fs[0]["clase"], label=fs[0]["label"], fuente=fs[0]["fuente"],
                    n_imagenes=len(fs))
        for k in claves:
            v = np.array([x[k] for x in fs], float)
            v = v[np.isfinite(v)]
            if len(v) == 0:
                fila[f"{k}_med"], fila[f"{k}_iqr"] = NAN, NAN
            else:
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                fila[f"{k}_med"], fila[f"{k}_iqr"] = float(med), float(q3 - q1)
        res[g] = fila
    return res
# toma los valores de la medida k en todas las copias descarta los NaN y calcula los percentiles
# si una copia rotada da un valor muy distinto al de otra copia de la misma ecografía, el IQR es grande (no debería suceder)

# de dónde salió la imagen, según el prefijo del archivo: train_/test_ = Kaggle original;
# cualquier otro prefijo = otra fuente
# sin prefijo: "otra".
def fuente_de(path):
    pref = Path(path).name.partition("_")[0].lower()
    if pref in ("train", "test"):
        return "kaggle"
    return pref if "_" in Path(path).name else "otra"


# fuentes con las dos clases (PCOS, sanas)
def _estratos(et, fu):
    out = []
    for s_ in sorted(set(fu)):
        m = fu == s_
        if (et[m] == 1).any() and (et[m] == 0).any():
            out.append((np.where(m & (et == 1))[0], np.where(m & (et == 0))[0]))
    return out


# comparacion estadistica
# evalua si las ecografías con PCOS tienen valores distintos de las sanas
# y si es una diferencia real o puede ser casualidad
def _estadistico(v, estratos, pos_por_estrato): # estrato es una fuente
    # separar por fuente evita que la diferencia entre equipos se asuma como diferencia entre PCOS y sanas
    num = den = 0.0
    for (ip, ineg), pos in zip(estratos, pos_por_estrato):
        idx = np.concatenate([ip, ineg])
        resto = np.setdiff1d(idx, pos)
        w = len(pos) * len(resto) / len(idx)
        num += w * (v[pos].mean() - v[resto].mean())
        den += w # calcula la diferencia de medias PCOS − sana dentro de cada fuente y la promedia entre fuentes con un peso
        # las fuentes más grandes y más balanceadas pesan más (w = peso)
    return num / den
# devuelve para cada fuente que tiene las dos clases las posiciones de sus grupos PCOS y de sus grupos sanos
# dentro de una fuente se compara PCOS contra sanas de esa misma fuente
# si una fuente tuviera solo PCOS no habría sanas con qué compararlas: se descarta una fuenta con una sola clase


# compara cada grupo PCOS con cada grupo sano de la misma fuente y cuenta en qué proporción de pares el PCOS tiene el valor más alto
# AUC 1: en todos los pares el PCOS supera a la sana
# AUC 0: en los todos los pares la sana supera al PCOS
# AUC 0,5: no hay diferencia
# no se comparan PCOS de una fuente con sanas de otra: calcula el AUC dentro de cada una y los promedia
def auc_grupos(v, et, fu=None):
    fu = np.zeros(len(v), int) if fu is None else fu
    aucs, pesos = [], []
    for ip, ineg in _estratos(et, fu):
        aucs.append(np.mean([(p > n) + 0.5 * (p == n) for p in v[ip] for n in v[ineg]]))
        pesos.append(len(ip) * len(ineg))
    return float(np.average(aucs, weights=pesos))


# estadistico p de permutacion: probabilidad de una diferencia así por azar
def permutacion(v, et, fu=None, max_exacta=200000, n_mc=20000, seed=0):
    fu = np.zeros(len(v), int) if fu is None else fu
    est = _estratos(et, fu)
    obs = abs(_estadistico(v, est, [e[0] for e in est]))
    total = 1
    for ip, ineg in est:
        total *= math.comb(len(ip) + len(ineg), len(ip))
    if total <= max_exacta:
        opciones = [list(itertools.combinations(np.concatenate([ip, ineg]), len(ip))) for ip, ineg in est]
        cuenta = sum(abs(_estadistico(v, est, [np.array(c) for c in comb])) >= obs - 1e-12
                     for comb in itertools.product(*opciones))
        return cuenta / total
    rng = np.random.default_rng(seed)
    num = np.zeros(n_mc)
    den = 0.0
    for ip, ineg in est: # vectorizado: n_mc reparticiones de las etiquetas dentro de la fuente
        idx = np.concatenate([ip, ineg])
        x, k, n = v[idx], len(ip), len(idx)
        elegidos = rng.random((n_mc, n)).argsort(1)[:, :k] # Monte Carlo: en lugar de probar todos, prueba 20 000 repartos al azar.
        suma = x[elegidos].sum(1)
        w = k * (n - k) / n
        num += w * (suma / k - (x.sum() - suma) / (n - k))
        den += w
    cuenta = 1 + int((np.abs(num / den) >= obs - 1e-12).sum())
    return cuenta / (n_mc + 1)


# PCOS vs sanas con las medianas de grupo
# para cada una de las 36 medidas calcula si difiere entre las clases, cuánto y qué tan confiable es esa diferencia
def comparar(grupos, claves, fuentes=None, seed=0): # usa semillas para Monte Carlo
    # fuente= None porque usa todas (comparando siempre dentro de cada fuente)
    gs = [g for g in grupos.values() if fuentes is None or g["fuente"] in fuentes] # gs: los grupos de las fuentes pedidas
    et = np.array([int(g["label"]) for g in gs]) # et: la etiqueta de cada uno (1 o 0)
    fu = np.array([g["fuente"] for g in gs]) # fu: la fuente de cada uno
    if not _estratos(et, fu):
        return []
    # si ninguna fuente tiene las dos clases, no hay nada que comparar y devuelve una tabla vacía
    filas = []
    for k in claves:
        v = np.array([g[f"{k}_med"] for g in gs], float) # v es la mediana de esa medida en cada grupo
        if not np.isfinite(v).all(): # si algún grupo tiene NaN en esa medida se saltea esa medida
            continue
        iqr = np.array([g[f"{k}_iqr"] for g in gs], float) # iqr: cuánto varía la medida dentro de cada grupo
        rango = v.max() - v.min() # rango: cuánto varía entre grupos
        # signo de la diferencia en cada fuente: se fija si coincide entre fuentes
        signos = {np.sign(np.median(v[ip]) - np.median(v[ineg])) for ip, ineg in _estratos(et, fu)}
        filas.append(dict(
            caracteristica=k, med_pcos=float(np.median(v[et == 1])), med_sana=float(np.median(v[et == 0])),
            auc_grupos=auc_grupos(v, et, fu), p_perm=permutacion(v, et, fu, seed=seed),
            inestabilidad=float(np.mean(iqr) / rango) if rango > 0 and iqr.any() else NAN,
            mismo_sentido=("si" if len(signos) == 1 and 0 not in signos else "no")))
        # usa todas las funciones: _estratos, auc_grupos, permutacion
    return sorted(filas, key=lambda f: (f["p_perm"], -abs(f["auc_grupos"] - 0.5)))


# entrada y salida

# busca la imagen en data/clean/.... 
# si no está, prueba en data/raw/<train|test>/<clase>/<nombre original> sacando el prefijo
# si tampoco está devuelve None
# sirve para que el script funcione aunque alguien no haya corrido limpiar.
def resolver_imagen(path_rel, clase):
    p = ROOT / path_rel
    if p.exists():
        return p
    pref, _, resto = Path(path_rel).name.partition("_")
    raw = ROOT / "data" / "raw" / pref / clase / resto
    return raw if raw.exists() else None


# escribe una lista de diccionarios como CSV 
def _guardar_csv(ruta, filas):
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        if filas:
            w = csv.DictWriter(f, fieldnames=list(filas[0]))
            w.writeheader() # vacio si no hay filas
            w.writerows(filas)


# lee splits2.csv, junta las imágenes por grupo y toma hasta 40 al azar de cada grupo 
# procesarlas todas tarda mucho y no aporta porque al final cada grupo se resume en una mediana
# 0 = todas
def elegir_muestra(ruta_splits, n_por_grupo, seed):
    with open(ruta_splits, encoding="utf-8") as f:
        filas_csv = list(csv.DictReader(f))
    por_grupo = {}
    for r in filas_csv:
        por_grupo.setdefault(r["grupo"], []).append(r)
    rng = random.Random(seed)
    muestra = []
    for g, rs in sorted(por_grupo.items(), key=lambda kv: int(kv[0])):
        rs = sorted(rs, key=lambda r: r["path"])
        # ordena antes de sortear para que con la misma semilla salgan siempre las mismas imágenes
        muestra += rs if n_por_grupo <= 0 else rng.sample(rs, min(n_por_grupo, len(rs)))
    return muestra


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--splits", default="splits2.csv", help="CSV con path, label, clase, grupo")
    ap.add_argument("--n-por-grupo", type=int, default=40, help="0 = todas las imágenes")
     # qué segmentación usar
     # estan estas por default pero puede cambiarse dependiendo la fuente
    ap.add_argument("--metodo", default="multiotsu")
    ap.add_argument("--filtro", default="wiener3")
    ap.add_argument("--contraste", default="ninguno")
    ap.add_argument("--margen-estroma", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--imagenes", type=int, default=0,
                    help="guardar una figura con las mediciones de N imágenes por grupo")
    ap.add_argument("--sin-montaje", action="store_true", help="no generar el montaje por grupo")
    args = ap.parse_args()

    muestra = elegir_muestra(ROOT / args.splits, args.n_por_grupo, args.seed)

    filas, claves, t0, saltadas = [], None, time.time(), 0
    guardadas = defaultdict(int)
    carpeta_fig = REPORTES / "figuras_caracteristicas"
    if args.imagenes > 0 and carpeta_fig.exists():
        shutil.rmtree(carpeta_fig)
    for i, r in enumerate(muestra, 1):
        ruta = resolver_imagen(r["path"], r["clase"])
        if ruta is None:
            saltadas += 1
            continue
        d = segment_detalle(ruta, filtro=args.filtro, contraste=args.contraste, metodo=args.metodo)
        f = caracteristicas(d, args.margen_estroma)
        claves = claves or list(f)
        filas.append(dict(path=r["path"], clase=r["clase"], label=r["label"], grupo=r["grupo"],
                          fuente=fuente_de(r["path"]), **f))
        # para cada imagen: 
        # la busca (si no existe, la cuenta como saltada)
        # la segmenta
        # calcula las 36 medidas
        # guarda una fila con los datos de la imagen más sus medidas.
        if guardadas[r["grupo"]] < args.imagenes:
            guardadas[r["grupo"]] += 1
            carpeta_fig.mkdir(parents=True, exist_ok=True)
            fig = figura_imagen(d, f, f"{r['clase']} | grupo {r['grupo']} | {ruta.name[:36]}",
                                args.margen_estroma)
            cv2.imwrite(str(carpeta_fig / f"g{r['grupo']}__{r['clase']}__{ruta.stem}.png"),
                        cv2.cvtColor(fig, cv2.COLOR_RGB2BGR))
        if i % 50 == 0:
            print(f"  {i}/{len(muestra)}  ({time.time() - t0:.0f} s)", flush=True)
    if not filas:
        raise SystemExit("No encontré ninguna imagen (ni en data/clean ni en data/raw).")

    REPORTES.mkdir(exist_ok=True)
    _guardar_csv(REPORTES / "caracteristicas_imagenes.csv", filas)
    grupos = resumir_grupos(filas, claves)
    _guardar_csv(REPORTES / "caracteristicas_grupos.csv", list(grupos.values()))

    if not args.sin_montaje:
        hacer_montaje(filas, grupos, args)

    fuentes = sorted({g["fuente"] for g in grupos.values()})
    L = [f"Imágenes procesadas: {len(filas)} de {len(muestra)} ({saltadas} sin archivo) | "
         f"grupos: {len(grupos)} | segmentación: {args.metodo}+{args.filtro}+{args.contraste}",
         "Grupos (= ecografías originales; las de Kaggle son aumentaciones de 8, las demás fuentes",
         "  suelen traer una imagen por ecografía):"]
    for fte in fuentes:
        gs = [g for g in grupos.values() if g["fuente"] == fte]
        npos = sum(int(g["label"]) for g in gs)
        L.append(f"  fuente {fte}: {len(gs)} grupos ({npos} PCOS, {len(gs) - npos} sanas), "
                 f"{sum(g['n_imagenes'] for g in gs)} imágenes")
    tablas = [(fte, comparar(grupos, claves, [fte], args.seed)) for fte in fuentes]
    if len(fuentes) > 1:
        tablas.append(("TODAS (permutando etiquetas dentro de cada fuente)", comparar(grupos, claves, None, args.seed)))
    comp_csv = []
    for nombre, comp in tablas:
        gs = [g for g in grupos.values() if nombre in fuentes and g["fuente"] == nombre] or list(grupos.values())
        npos = sum(int(g["label"]) for g in gs)
        L += ["", f"PCOS vs sanas, fuente {nombre}: {npos} vs {len(gs) - npos} grupos",
              f"{'característica':28s}{'PCOS':>10s}{'sana':>10s}{'AUC':>7s}{'p perm.':>9s}{'inestab.':>10s}"
              + (f"{'mismo sentido':>15s}" if nombre not in fuentes else "")]
        for c in comp:
            L.append(f"{c['caracteristica']:28s}{c['med_pcos']:10.3f}{c['med_sana']:10.3f}"
                     f"{c['auc_grupos']:7.2f}{c['p_perm']:9.4f}{_f(c['inestabilidad']):>10s}"
                     + (f"{c['mismo_sentido']:>15s}" if nombre not in fuentes else ""))
            comp_csv.append(dict(fuente=nombre.split(" ")[0], **c))
    _guardar_csv(REPORTES / "caracteristicas_comparacion.csv", comp_csv)
    # hace una tabla de comparación por cada fuente y, si hay más de una, otra combinando todas. L
    # la columna mismo_sentido solo aparece en la tabla combinada porque con una sola fuente no tiene sentido
    L += ["", "AUC: sobre medianas de grupo, calculado dentro de cada fuente (1 = todas las PCOS por encima",
          "  de todas las sanas, 0,5 = nada). Con varias fuentes es el promedio de los AUC de cada una.",
          "p perm.: permutación de etiquetas dentro de cada fuente (exacta si hay pocas asignaciones;",
          "  con 4 vs 4 el mínimo es 0,029; si no, Monte Carlo con 20.000 reparticiones, mínimo 0,00005).",
          "  SIN corregir por comparaciones múltiples: con ~35 características, varias dan p < 0,05 por azar.",
          "mismo sentido: la diferencia PCOS - sana tiene el mismo signo en todas las fuentes. Una",
          "  característica que cambia de signo entre fuentes depende de la fuente, no de la patología.",
          "inestab.: ('-' si hay una imagen por grupo) rango intercuartílico entre variantes / rango entre grupos (< 1 = la diferencia entre",
          "  ecografías supera lo que cambia una misma ecografía al rotarla, hacerle zoom o recortarla).",
          "En Kaggle el efectivo es 4 vs 4 y clase y encuadre van juntos; las otras fuentes son la prueba",
          "  más limpia. Se describe qué mide cada característica; no se estima capacidad diagnóstica."]
    texto = "\n".join(L)
    print(texto)
    (REPORTES / "caracteristicas_resumen.txt").write_text(texto + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()