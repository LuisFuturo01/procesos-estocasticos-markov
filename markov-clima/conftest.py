# Permite ejecutar pytest desde la raíz del repositorio: `python -m pytest markov-clima/tests`.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
