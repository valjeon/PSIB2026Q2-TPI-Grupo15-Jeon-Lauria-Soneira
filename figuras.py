"""Figuras de resultados para el informe y la presentación (a partir de los CSV de reports/).

No recalcula nada: lee lo que dejaron los otros módulos y dibuja. Cada figura se saltea con un
aviso si falta su CSV, así que se puede correr en cualquier momento.

  fig1_dataset.png          de las imágenes a las ecografías independientes (por fuente) y qué se descartó
  fig2_tamanos.png          tamaño de imagen por clase y fuente (el atajo de adquisición)
  fig3_filtros.png          filtros de speckle: supresión (ENL) vs. bordes (beta), con los no dominados
  fig4_segmentacion.png     Dice de las combinaciones de segmentación contra las máscaras manuales
  fig5_caracteristicas.png  características clave por ecografía, por fuente y clase
  fig6_auc_fuentes.png      AUC de cada característica en cada fuente (¿se sostiene fuera de Kaggle?)
  fig7_preproc_ejemplo.png  una ecografía antes y después del preprocesamiento (necesita data/clean/)
  fig8_seg_ejemplo.png      folículos manuales vs. automáticos, con Dice (necesita mascaras.csv y data/clean/)

Entradas (reports/): excluidas.csv, audit.csv, enl_resultados_manual.csv, seg_dice.csv,
seg_finalistas.csv, caracteristicas_grupos.csv, caracteristicas_comparacion.csv y splits2.csv
(raíz del repo); fig7 y fig8 además usan data/clean/, mascaras.csv y mascaras/. Salida: reports/figuras/*.png

Uso:
    python procesamiento/figuras.py
    python procesamiento/figuras.py --filtro wiener3 --dpi 300
    python procesamiento/figuras.py --sin-titulo        # para pegarlas en diapositivas con título propio
"""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
if ROOT.name == "procesamiento":  # funciona dentro de procesamiento/ o en la raíz del repo
    ROOT = ROOT.parent
REPORTES = ROOT / "reports"

# Paleta categórica validada (modo claro): PCOS = azul, sana = naranja; fuentes = violeta / verde
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
C_PCOS, C_SANA = "#2a78d6", "#eb6834"
C_KAGGLE, C_OTRA = "#6250d6", "#008300"
NOMBRE_CLASE = {"infected": "PCOS", "notinfected": "Sana"}

NOMBRES = {
    "n_foliculos": "Número de folículos",
    "densidad_x1e4px": "Densidad (folículos por 10⁴ px)",
    "dominancia_max": "Dominancia del folículo mayor",
    "frac_area_foliculos": "Fracción de área de folículos",
    "ratio_area_fol_estroma": "Cociente de áreas folículo/estroma",
    "area_total_px": "Área total de folículos (px)",
    "dist_vecino_rel": "Distancia al vecino (relativa)",
    "brillo_foliculos": "Brillo de folículos",
    "ratio_brillo_fol_estroma": "Cociente de brillos fol./estroma",
    "contraste_michelson": "Contraste de Michelson",
    "area_cv": "Variación del área (CV)",
    "area_roi_px": "Área del campo (px)",
}


def etiqueta(k):
    t = NOMBRES.get(k, k.replace("_", " "))
    return t[:1].upper() + t[1:]


# ---------------------------------------------------------------- utilidades
def estilo():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 13, "legend.frameon": False, "legend.fontsize": 12,
        "font.family": "DejaVu Sans",
    })


def leer_csv(ruta):
    ruta = Path(ruta)
    if not ruta.exists():
        return None
    with open(ruta, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fuente_de(nombre):
    pref = Path(nombre).name.partition("_")[0].lower()
    if pref in ("train", "test"):
        return "Kaggle"
    return pref.upper() if "_" in Path(nombre).name else "otra"


def nombre_fuente(f):
    return "Kaggle" if f.lower() == "kaggle" else f.upper()


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def guardar(fig, destino, args):
    fig.savefig(destino, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  {destino.relative_to(ROOT)}")


def titulo(ax, texto, args):
    if not args.sin_titulo:
        ax.set_title(texto, pad=12)


# ---------------------------------------------------------------- fig1: dataset
def fig1_dataset(args, out):
    excl = leer_csv(REPORTES / "excluidas.csv")
    sp = leer_csv(ROOT / args.splits)
    if sp is None:
        return print(f"  fig1: falta {args.splits}")
    imgs, grupos = Counter(), defaultdict(set)
    for r in sp:
        f = fuente_de(r["path"])
        imgs[f] += 1
        grupos[f].add(r["grupo"])
    fuentes = sorted(imgs, key=lambda f: (f != "Kaggle", f))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(fuentes))
    h = 0.36
    n_img = [imgs[f] for f in fuentes]
    n_gr = [len(grupos[f]) for f in fuentes]
    a1.barh(y - h / 2 - 0.01, n_img, h, color=MUTED, label="Imágenes conservadas")
    a1.barh(y + h / 2 + 0.01, n_gr, h, color=C_PCOS, label="Ecografías independientes (grupos)")
    top = max(n_img)
    for yi, a, b in zip(y, n_img, n_gr):
        a1.text(a + top * 0.01, yi - h / 2, f"{a:,}".replace(",", "."), va="center", fontsize=13, color=INK)
        a1.text(b + top * 0.01, yi + h / 2, f"{b}", va="center", fontsize=13, color=INK, fontweight="bold")
    a1.set_yticks(y)
    a1.set_yticklabels(fuentes)
    a1.invert_yaxis()
    a1.set_xlim(0, top * 1.15)
    a1.set_xlabel("Cantidad")
    a1.grid(axis="y", visible=False)
    a1.legend(loc="lower right")
    titulo(a1, "Muchas imágenes, pocas ecografías", args)
    if excl:
        c = Counter(r["motivo"] for r in excl)
        orden = c.most_common()
        etiquetas = [m.capitalize() for m, _ in orden]
        vals = [v for _, v in orden]
        a2.barh(np.arange(len(vals)), vals, 0.55, color=MUTED)
        for i, v in enumerate(vals):
            a2.text(v + max(vals) * 0.01, i, f"{v:,}".replace(",", "."), va="center", fontsize=13)
        a2.set_yticks(np.arange(len(vals)))
        a2.set_yticklabels(etiquetas)
        a2.invert_yaxis()
        a2.set_xlim(0, max(vals) * 1.15)
        a2.set_xlabel("Imágenes excluidas")
        a2.grid(axis="y", visible=False)
        titulo(a2, "Qué descartó la limpieza", args)
    else:
        a2.axis("off")
    guardar(fig, out / "fig1_dataset.png", args)


# ---------------------------------------------------------------- fig2: tamaños
def fig2_tamanos(args, out):
    au = leer_csv(REPORTES / "audit.csv")
    if au is None:
        return print("  fig2: falta reports/audit.csv")
    por_tam = defaultdict(Counter)
    for r in au:
        por_tam[(r["ancho"] + "×" + r["alto"], fuente_de(Path(r["path"]).name))][r["clase"]] += 1
    filas = sorted(por_tam.items(), key=lambda kv: -sum(kv[1].values()))[:14]
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(filas) + 1.8))
    y = np.arange(len(filas))
    pos = [c.get("infected", 0) for _, c in filas]
    sana = [c.get("notinfected", 0) for _, c in filas]
    ax.barh(y, pos, 0.62, color=C_PCOS, label="PCOS", edgecolor=SURFACE, linewidth=1.5)
    ax.barh(y, sana, 0.62, left=pos, color=C_SANA, label="Sana", edgecolor=SURFACE, linewidth=1.5)
    for yi, p, s in zip(y, pos, sana):
        ax.text(p + s + max(np.add(pos, sana)) * 0.01, yi, f"{p + s}", va="center", fontsize=12, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{t}  ({f})" for (t, f), _ in filas])
    ax.invert_yaxis()
    ax.set_xlabel("Imágenes por tamaño (ancho × alto en píxeles)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    titulo(ax, "En Kaggle el tamaño de la imagen delata la clase", args)
    guardar(fig, out / "fig2_tamanos.png", args)


# ---------------------------------------------------------------- fig3: filtros
def fig3_filtros(args, out):
    enl = leer_csv(REPORTES / "enl_resultados_manual.csv")
    if enl is None:
        return print("  fig3: falta reports/enl_resultados_manual.csv")
    coc, beta = defaultdict(list), defaultdict(list)
    for r in enl:
        coc[r["filtro"]].append(num(r["cociente"]))
        beta[r["filtro"]].append(num(r["beta"]))
    nombres = list(coc)
    x = {m: float(np.median(coc[m])) for m in nombres}
    yb = {m: float(np.mean(beta[m])) for m in nombres}
    dom = {m: any(o != m and x[o] >= x[m] and yb[o] >= yb[m] and (x[o] > x[m] or yb[o] > yb[m])
                  for o in nombres) for m in nombres}
    fig, ax = plt.subplots(figsize=(11.5, 7))
    for m in nombres:
        ax.scatter(x[m], yb[m], s=90, color=MUTED if dom[m] else C_PCOS, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
    if args.filtro in x:
        ax.scatter(x[args.filtro], yb[args.filtro], s=240, facecolor="none", edgecolor=INK,
                   linewidth=2.4, zorder=4)
    ax.set_xlim(min(x.values()) - 0.25, max(x.values()) + 0.6)
    ax.set_ylim(min(yb.values()) - 0.06, max(yb.values()) + 0.1)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ocupado = [ax.transData.transform((x[m], yb[m])) for m in nombres]
    cajas = [(px - 9, py - 9, px + 9, py + 9) for px, py in ocupado]
    candidatos = [(9, -4, "left"), (-9, -4, "right"), (0, 11, "center"), (0, -20, "center"),
                  (9, 10, "left"), (9, -18, "left"), (-9, 10, "right"), (-9, -18, "right"),
                  (14, 24, "left"), (14, -32, "left"), (-14, 24, "right"), (-14, -32, "right")]
    # primero los que importan (no dominados y el elegido), después el resto
    orden = sorted(nombres, key=lambda k: (dom[k] and k != args.filtro, -yb[k]))
    for m in orden:
        resaltado = (not dom[m]) or m == args.filtro
        mejor = None
        for dx, dy, ha in candidatos:
            t = ax.annotate(m, (x[m], yb[m]), xytext=(dx, dy), textcoords="offset points", ha=ha,
                            fontsize=12 if resaltado else 10.5, color=INK if resaltado else MUTED,
                            fontweight="bold" if m == args.filtro else "normal")
            bb = t.get_window_extent(rend)
            caja = (bb.x0, bb.y0, bb.x1, bb.y1)
            choca = any(not (caja[2] < o[0] or caja[0] > o[2] or caja[3] < o[1] or caja[1] > o[3])
                        for o in cajas)
            dentro = (ax.bbox.x0 <= bb.x0 and bb.x1 <= ax.bbox.x1 and ax.bbox.y0 <= bb.y0 and bb.y1 <= ax.bbox.y1)
            if not choca and dentro:
                mejor = caja
                break
            t.remove()
        if mejor is None:  # sin lugar libre: el último candidato, aunque pise
            ax.annotate(m, (x[m], yb[m]), xytext=candidatos[0][:2], textcoords="offset points",
                        fontsize=10.5, color=MUTED)
        else:
            cajas.append(mejor)
    ax.set_xlabel("Supresión de speckle: cociente de ENL filtrada / original (mayor = más suavizado)")
    ax.set_ylabel("Preservación de bordes: β (mayor = bordes intactos)")
    ax.scatter([], [], color=C_PCOS, s=90, label="No dominado (ningún otro suaviza más y conserva más bordes)")
    ax.scatter([], [], color=MUTED, s=90, label="Dominado")
    ax.scatter([], [], s=200, facecolor="none", edgecolor=INK, linewidth=2.4, label=f"Usado en el pipeline: {args.filtro}")
    ax.legend(loc="upper right")
    titulo(ax, "Elección del filtro de speckle: suavizar sin borrar bordes", args)
    guardar(fig, out / "fig3_filtros.png", args)


# ---------------------------------------------------------------- fig4: segmentación
def fig4_segmentacion(args, out):
    fin = leer_csv(REPORTES / "seg_finalistas.csv")
    if fin:
        filas = [(r["combinacion"], num(r["dice"]), r.get("finalista") == "1") for r in fin]
    else:
        d = leer_csv(REPORTES / "seg_dice.csv")
        if d is None:
            return print("  fig4: faltan reports/seg_finalistas.csv y seg_dice.csv")
        acum = defaultdict(list)
        for r in d:
            acum["+".join((r["metodo"], r["filtro"], r["contraste"]))].append(num(r["dice"]))
        filas = [(k, float(np.mean(v)), False) for k, v in acum.items()]
    filas = sorted(filas, key=lambda t: -t[1])[:args.top_seg]
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(filas) + 1.8))
    y = np.arange(len(filas))
    ax.barh(y, [d for _, d, _ in filas], 0.6, color=[C_PCOS if f else MUTED for _, _, f in filas])
    for yi, (_, d, _) in zip(y, filas):
        ax.text(d + 0.008, yi, f"{d:.3f}".replace(".", ","), va="center", fontsize=12)
    ax.set_yticks(y)
    ax.set_yticklabels([k for k, _, _ in filas])
    ax.invert_yaxis()
    ax.set_xlim(0, max(d for _, d, _ in filas) * 1.12)
    ax.set_xlabel("Dice contra las máscaras manuales (conjunto de ajuste)")
    ax.grid(axis="y", visible=False)
    if any(f for _, _, f in filas):
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=C_PCOS, label="Finalista"), Patch(color=MUTED, label="No finalista")],
                  loc="lower right")
    titulo(ax, "Segmentación: método + filtro + contraste", args)
    guardar(fig, out / "fig4_segmentacion.png", args)


# ---------------------------------------------------------------- fig5: características por grupo
def fig5_caracteristicas(args, out):
    g = leer_csv(REPORTES / "caracteristicas_grupos.csv")
    if g is None:
        return print("  fig5: falta reports/caracteristicas_grupos.csv")
    claves = [k for k in args.claves if f"{k}_med" in g[0]]
    if not claves:
        return print("  fig5: ninguna de las características pedidas está en el CSV")
    fuentes = sorted({r.get("fuente", "Kaggle") for r in g}, key=lambda f: (f != "kaggle", f))
    cols = [(f, c) for f in fuentes for c in ("notinfected", "infected")]
    n = len(claves)
    ncol = 2 if n > 2 else n
    nfil = -(-n // ncol)
    fig, axs = plt.subplots(nfil, ncol, figsize=(6.2 * ncol, 4.6 * nfil), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, k in zip(axs.ravel(), claves):
        for i, (f, c) in enumerate(cols):
            v = np.array([num(r[f"{k}_med"]) for r in g if r.get("fuente", "Kaggle") == f and r["clase"] == c])
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            color = C_PCOS if c == "infected" else C_SANA
            ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=46, color=color, alpha=0.8,
                       edgecolor=SURFACE, linewidth=0.8, zorder=3, marker="o" if c == "infected" else "s")
            ax.hlines(np.median(v), i - 0.3, i + 0.3, color=INK, linewidth=2.4, zorder=4)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([f"{nombre_fuente(f)}\n{NOMBRE_CLASE.get(c, c)}  (n={sum(1 for r in g if r.get('fuente', 'Kaggle') == f and r['clase'] == c)})"
                            for f, c in cols], fontsize=11)
        ax.set_ylabel(etiqueta(k))
        ax.grid(axis="x", visible=False)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    axs.ravel()[0].scatter([], [], color=C_PCOS, marker="o", label="PCOS")
    axs.ravel()[0].scatter([], [], color=C_SANA, marker="s", label="Sana")
    axs.ravel()[0].plot([], [], color=INK, linewidth=2.4, label="Mediana")
    axs.ravel()[0].legend(loc="upper left", fontsize=11)
    if not args.sin_titulo:
        fig.suptitle("Una ecografía = un punto (mediana de sus variantes)", x=0.01, ha="left",
                     fontsize=15, fontweight="bold")
    fig.tight_layout()
    guardar(fig, out / "fig5_caracteristicas.png", args)


# ---------------------------------------------------------------- fig6: AUC por fuente
def fig6_auc(args, out):
    c = leer_csv(REPORTES / "caracteristicas_comparacion.csv")
    if c is None:
        return print("  fig6: falta reports/caracteristicas_comparacion.csv")
    por = defaultdict(dict)
    sentido = {}
    for r in c:
        f = r["fuente"]
        if f.upper().startswith("TODAS"):
            sentido[r["caracteristica"]] = r["mismo_sentido"] == "si"
        else:
            por[f][r["caracteristica"]] = num(r["auc_grupos"])
    fuentes = sorted(por, key=lambda f: (f != "kaggle", f))
    if len(fuentes) < 2:
        return print("  fig6: hace falta más de una fuente en caracteristicas_comparacion.csv")
    ref = args.fuente_ref if args.fuente_ref in por else fuentes[-1]
    otras = [f for f in fuentes if f != ref]
    claves = [k for k in por[ref] if all(k in por[f] for f in otras)]
    claves.sort(key=lambda k: -abs(por[ref][k] - 0.5))
    claves = claves[:args.top_auc][::-1]
    colores = {f: (C_KAGGLE if f.lower() == "kaggle" else C_OTRA) for f in fuentes}
    marcas = {f: ("o" if f.lower() == "kaggle" else "D") for f in fuentes}
    fig, ax = plt.subplots(figsize=(11.5, 0.42 * len(claves) + 2.4))
    y = np.arange(len(claves))
    ax.axvline(0.5, color=INK2, linewidth=1.2, zorder=1)
    for yi, k in zip(y, claves):
        v = [por[f][k] for f in fuentes]
        ax.hlines(yi, min(v), max(v), color=GRID, linewidth=3, zorder=2)
    for f in fuentes:
        ax.scatter([por[f][k] for k in claves], y, s=110, color=colores[f], marker=marcas[f],
                   edgecolor=SURFACE, linewidth=1.5, zorder=3,
                   label=f"{nombre_fuente(f)} ({'referencia: ordena las filas' if f == ref else 'comparación'})")
    ax.set_yticks(y)
    etiquetas = [etiqueta(k) + (" ✓" if sentido.get(k) else "") for k in claves]
    ax.set_yticklabels(etiquetas)
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUC entre ecografías (0,5 = no separa; > 0,5 = mayor en PCOS; < 0,5 = menor en PCOS)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(fuentes), borderaxespad=0.4)
    ax.annotate("✓ = la diferencia PCOS − sana va en el mismo sentido en todas las fuentes",
                xy=(0, 0), xycoords="axes fraction", xytext=(0, -52), textcoords="offset points",
                fontsize=11, color=INK2)
    if not args.sin_titulo:
        ax.set_title("¿Qué se sostiene fuera de Kaggle?", pad=40)
    guardar(fig, out / "fig6_auc_fuentes.png", args)


# ---------------------------------------------------------------- fig7 y fig8: ejemplos con imágenes
def _ejemplos(n_por_clase, conjunto="reporte"):
    """Imágenes con máscara manual del conjunto de reporte (no el de ajuste). PCOS: las que tienen
    MÁS folículos marcados a mano (para que haya algo que comparar); sanas: las más típicas, cerca
    de la mediana de folículos. Decirlo en el pie de figura."""
    filas = leer_csv(ROOT / "mascaras.csv")
    if filas is None:
        return None
    out = []
    for cl in ("infected", "notinfected"):
        fs = [r for r in filas if r["clase"] == cl and r["conjunto"] == conjunto
              and (ROOT / r["imagen"]).exists() and int(r["n_foliculos"]) > 0]
        if not fs:
            continue
        if cl == "infected":
            fs.sort(key=lambda r: (-int(r["n_foliculos"]), r["imagen"]))
        else:
            med = np.median([int(r["n_foliculos"]) for r in fs])
            fs.sort(key=lambda r: (abs(int(r["n_foliculos"]) - med), r["imagen"]))
        out += fs[:n_por_clase]
    return out


def fig7_preproc_ejemplo(args, out):
    from preprocesamiento import a_gris, base, filtrar, leer
    ej = _ejemplos(1)
    if not ej:
        return print("  fig7: faltan mascaras.csv o data/clean/")
    filtros = [args.filtro] + [f for f in ("wiener5", "gauss1.0") if f != args.filtro]
    titulos = ["Original", "Sin marcas, recortada,\n256 × 256"] + [f"Filtro {f}" for f in filtros]
    fig, axs = plt.subplots(len(ej), len(titulos), figsize=(3.3 * len(titulos), 3.3 * len(ej) + 0.4), squeeze=False)
    for i, r in enumerate(ej):
        ruta = ROOT / r["imagen"]
        b = base(ruta, 256)
        imgs = [a_gris(leer(ruta)), b] + [filtrar(b, f) for f in filtros]
        for j, im in enumerate(imgs):
            ax = axs[i, j]
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            if i == 0:
                ax.set_title(titulos[j], fontsize=12, fontweight="normal", loc="center")
        axs[i, 0].text(-0.04, 0.5, NOMBRE_CLASE.get(r["clase"], r["clase"]), transform=axs[i, 0].transAxes,
                       rotation=90, va="center", ha="right", fontsize=13, color=C_PCOS if r["clase"] == "infected" else C_SANA,
                       fontweight="bold")
    guardar(fig, out / "fig7_preproc_ejemplo.png", args)


def fig8_seg_ejemplo(args, out):
    from segmentacion import metricas_mascara, recortar_a_zona, segment_detalle
    import cv2
    ej = _ejemplos(2)
    if not ej:
        return print("  fig8: faltan mascaras.csv o data/clean/")
    fig, axs = plt.subplots(len(ej), 3, figsize=(10.2, 3.4 * len(ej) + 0.4), squeeze=False)
    for i, r in enumerate(ej):
        d = segment_detalle(ROOT / r["imagen"], filtro=args.seg_filtro, contraste=args.seg_contraste,
                            metodo=args.seg_metodo)
        gt = cv2.imread(str(ROOT / r["mascara"]), cv2.IMREAD_UNCHANGED)
        zlado = int(r["zlado"])
        zona = np.ones(gt.shape, bool)
        if zlado > 0:
            zona[:] = False
            zx, zy = int(r["zx"]), int(r["zy"])
            zona[zy:zy + zlado, zx:zx + zlado] = True
        gt_z = (gt > 0) & zona
        pred_z = recortar_a_zona(d["labels"], zona) > 0
        dice = metricas_mascara(pred_z, gt_z)["dice"]
        base = d["base"]
        paneles = [("Imagen", None), ("Manual", (gt > 0) & zona), (f"Automática   Dice {dice:.2f}".replace(".", ","), pred_z)]
        for j, (tit, m) in enumerate(paneles):
            ax = axs[i, j]
            ax.imshow(base, cmap="gray", vmin=0, vmax=255)
            if m is not None:
                color = "#00c853" if j == 1 else "#e53935"
                ax.imshow(np.ma.masked_where(~m, m), cmap=matplotlib.colors.ListedColormap([color]), alpha=0.45)
                ax.contour(m.astype(float), levels=[0.5], colors=[color], linewidths=1.6)
            if zlado > 0 and j > 0:
                ax.add_patch(plt.Rectangle((int(r["zx"]), int(r["zy"])), zlado, zlado, fill=False,
                                           edgecolor="#fdd835", linewidth=1.4, linestyle="--"))
            ax.axis("off")
            if i == 0 or j == 2:
                ax.set_title(tit if i == 0 else tit.split("   ")[-1], fontsize=12, fontweight="normal", loc="center")
        axs[i, 0].text(-0.04, 0.5, NOMBRE_CLASE.get(r["clase"], r["clase"]), transform=axs[i, 0].transAxes,
                       rotation=90, va="center", ha="right", fontsize=13, fontweight="bold",
                       color=C_PCOS if r["clase"] == "infected" else C_SANA)
    guardar(fig, out / "fig8_seg_ejemplo.png", args)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Figuras de resultados a partir de los CSV de reports/")
    ap.add_argument("--splits", default="splits2.csv", help="CSV de grupos (raíz del repo)")
    ap.add_argument("--filtro", default="wiener3", help="filtro elegido (se resalta en fig3 y va primero en fig7)")
    ap.add_argument("--claves", nargs="+",
                    default=["n_foliculos", "densidad_x1e4px", "dominancia_max", "frac_area_foliculos"],
                    help="características de fig5")
    ap.add_argument("--fuente-ref", default="dv", help="fuente que ordena las filas de fig6")
    ap.add_argument("--top-auc", type=int, default=14, help="filas en fig6")
    ap.add_argument("--top-seg", type=int, default=10, help="combinaciones en fig4")
    ap.add_argument("--seg-metodo", default="multiotsu", help="método de segmentación para fig8")
    ap.add_argument("--seg-filtro", default="wiener3", help="filtro de la segmentación para fig8")
    ap.add_argument("--seg-contraste", default="ninguno", help="contraste de la segmentación para fig8")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--sin-titulo", action="store_true", help="sin títulos dentro de las figuras")
    ap.add_argument("--salida", default="reports/figuras")
    args = ap.parse_args()

    estilo()
    out = ROOT / args.salida
    out.mkdir(parents=True, exist_ok=True)
    print("Figuras:")
    for f in (fig1_dataset, fig2_tamanos, fig3_filtros, fig4_segmentacion, fig5_caracteristicas, fig6_auc,
              fig7_preproc_ejemplo, fig8_seg_ejemplo):
        try:
            f(args, out)
        except Exception as e:  # una figura que falla no tira abajo a las demás
            print(f"  {f.__name__}: error ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
