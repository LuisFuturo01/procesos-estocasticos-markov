# markov-cripto

Cadena de Markov (3 estados) sobre retornos diarios R_t de BTC, ETH o tipos de cambio.
Estados: S1 bajista (R < -1.5%), S2 lateral, S3 alcista (R > +1.5%), umbrales ajustables.

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests
streamlit run app/ui/app.py     # http://localhost:8501
```

Caché: `app/database/cache.sqlite3` (esquema en `app/database/schema.sql`).
