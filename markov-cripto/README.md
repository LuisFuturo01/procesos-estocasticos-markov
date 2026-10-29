# markov-cripto

Cadena de Markov (3 estados) sobre retornos diarios R_t = (P_t − P_{t−1}) / P_{t−1} de BTC, ETH o tipos de cambio.
Estados: S1 bajista, S2 lateral, S3 alcista.

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests
streamlit run app/ui/app.py     # http://localhost:8501
```

- **Umbrales:** *Fijo* (±1.5 %), *Volatilidad* (media ± z·σ) o *Terciles*. Las divisas casi nunca se mueven
  ±1.5 % en un día (USD/EUR: 2 bajistas y 1 alcista en 3 años), así que para ellas se usa por defecto
  el modo por volatilidad; con umbrales fijos las filas S1/S3 de P quedarían sin datos.
- **Datos locales:** `data.json` (BTC/ETH 365 días, divisas 3 años) se usa como fuente offline o como
  respaldo si la API limita las consultas. Para regenerarlo:
  `python -c "from app.services.data_fetcher import build_static_snapshot; build_static_snapshot()"`
- **CoinGecko** gratuito solo entrega los últimos 365 días; el selector de fechas ya lo respeta.
- **Caché:** `app/database/cache.sqlite3` (esquema en `app/database/schema.sql`).
- Herramienta educativa: no constituye recomendación de inversión.
