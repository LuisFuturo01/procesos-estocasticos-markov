# markov-clima

Cadena de Markov (3 estados) sobre precipitación diaria de Open-Meteo. DAT262 Cap. 3.
Estados: S1 seco, S2 lluvia leve, S3 lluvia intensa (umbrales ajustables).

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests          # pruebas del motor y datos estáticos
streamlit run app/ui/app.py     # http://localhost:8501
```

- **Datos estáticos:** `data.json` con 1,096 registros diarios (2023 - 2026) de La Paz para ejecución 100% offline o despliegues sin dependencia de la API externa.
- **Caché:** `app/database/cache.sqlite3` (esquema en `app/database/schema.sql`).
- **Despliegue:** subir a GitHub y conectar en Streamlit Community Cloud (archivo principal `app/ui/app.py`).
