# main.py
import importlib
import sys
from gui import MeteorologicalImageProcessor

# Vérification des dépendances
required_libraries = {
    'cv2': 'opencv-python',
    'numpy': 'numpy',
    'PIL': 'pillow',
    'matplotlib': 'matplotlib',
    'fpdf': 'fpdf',
    'requests': 'requests'
}

missing_libs = []
for lib, pkg in required_libraries.items():
    try:
        importlib.import_module(lib)
    except ImportError:
        missing_libs.append(pkg)

if missing_libs:
    raise ImportError(f"Bibliothèques manquantes: {', '.join(missing_libs)}. Installez-les avec `pip install {' '.join(missing_libs)}`")

if __name__ == "__main__":
    app = MeteorologicalImageProcessor()
    app.run()