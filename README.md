# Procesos Estocásticos - Cadenas de Markov

Este repositorio contiene dos proyectos interactivos basados en Cadenas de Markov en tiempo discreto (3 estados), desarrollados con **Python**, **Streamlit**, **Plotly**, **NumPy**, **Pandas** y **SciPy**:

1. **`markov-clima`**: Analizador estocástico de precipitación diaria (Seco, Lluvia leve, Lluvia intensa) usando la API abierta de Open-Meteo.
2. **`markov-cripto`**: Analizador estocástico de mercados financieros y criptomonedas (Bajista, Lateral, Alcista) usando CoinGecko y Frankfurter.

---

## 📁 Estructura del Repositorio

```text
procesos-estocasticos-markov/
├── requirements.txt              # Dependencias globales del repositorio
├── .gitignore
├── README.md
├── markov-clima/
│   ├── requirements.txt
│   ├── data.json                 # Datos locales estáticos de precipitación (1,096 días)
│   ├── app/
│   │   ├── core/                 # Motor de Markov y discretización
│   │   ├── database/             # Esquema y caché SQLite
│   │   ├── services/             # Carga estática y cliente API Open-Meteo
│   │   └── ui/                   # app.py y componentes visuales Plotly
│   └── tests/                    # Pruebas unitarias (pytest)
└── markov-cripto/
    ├── requirements.txt
    ├── app/
    │   ├── core/                 # Motor de Markov y discretización
    │   ├── database/             # Esquema y caché SQLite
    │   ├── services/             # Cliente APIs CoinGecko y Frankfurter
    │   └── ui/                   # app.py y componentes visuales Plotly
    └── tests/                    # Pruebas unitarias (pytest)
```

---

## 🚀 Ejecución Local

### 1. Clonar el repositorio y crear entorno virtual

#### En Windows (PowerShell):
```powershell
# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
.venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt
```

#### En Linux / macOS / Git Bash:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Ejecutar Pruebas Unitarias
```bash
python -m pytest markov-clima/tests
python -m pytest markov-cripto/tests
```

### 3. Ejecutar las Aplicaciones

- **Para lanzar Markov Clima:**
  ```bash
  streamlit run markov-clima/app/ui/app.py
  ```
  *(Disponible en: http://localhost:8501)*

- **Para lanzar Markov Cripto:**
  ```bash
  streamlit run markov-cripto/app/ui/app.py --server.port 8502
  ```
  *(Disponible en: http://localhost:8502)*

---

## 🌐 Despliegue en la Nube

> **⚠️ Nota importante sobre Vercel:**  
> Streamlit requiere un servidor Python en ejecución continua con WebSockets persistentes para manejar la interactividad y el estado de la sesión. **Vercel es una plataforma serverless**, por lo que no soporta la arquitectura de WebSockets persistentes de Streamlit.  
> La plataforma oficial, estándar y 100% gratuita para desplegar estas aplicaciones es **Streamlit Community Cloud** (o servicios web en **Render** / **Railway**).

### Despliegue en Streamlit Community Cloud (Recomendado - Gratis)

1. Sube tu código a GitHub:
   ```bash
   git add .
   git commit -m "Preparar proyectos y requirements para despliegue"
   git push origin master
   ```
2. Entra en [share.streamlit.io](https://share.streamlit.io/) e inicia sesión con tu cuenta de GitHub.
3. Haz clic en **"Create app"**:
   - **Repository:** `LuisFuturo01/procesos-estocasticos-markov`
   - **Branch:** `master`
   - **Main file path:** `markov-clima/app/ui/app.py`
   - Haz clic en **Deploy**.
4. Para la app de criptomonedas, repite el proceso seleccionando:
   - **Main file path:** `markov-cripto/app/ui/app.py`
