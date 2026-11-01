from pathlib import Path
import shutil

CARPETA = Path("data/raw_dataverse")      # donde están las fotos mezcladas
FUENTE = "dv"                      # prefijo corto de la fuente (sin guiones bajos)
DESTINO = Path("data/all")

n = {"infected": 0, "notinfected": 0, "sin_clase": []}
for f in sorted(CARPETA.rglob("*")):
    if not f.is_file() or f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
        continue
    nombre = f.name.lower()
    if nombre.startswith("pco"):
        clase = "infected"
    elif nombre.startswith("normal"):
        clase = "notinfected"
    else:
        n["sin_clase"].append(f.name)
        continue
    (DESTINO / clase).mkdir(parents=True, exist_ok=True)
    shutil.copy2(f, DESTINO / clase / f"{FUENTE}_{f.name}")
    n[clase] += 1

print(n["infected"], "infected |", n["notinfected"], "notinfected |",
      len(n["sin_clase"]), "sin clase", n["sin_clase"][:5])