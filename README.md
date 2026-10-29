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
    ├── data.json                 # Copia local de BTC, ETH (365 d) y USD/EUR, GBP, JPY (3 años)
    ├── app/
    │   ├── core/                 # Motor de Markov y discretización
    │   ├── database/             # Esquema y caché SQLite
    │   ├── services/             # Cliente APIs CoinGecko y Frankfurter (+ respaldo local)
    │   └── ui/                   # app.py y componentes visuales Plotly
    └── tests/                    # Pruebas unitarias (pytest)
```

---

## 📐 Teoría DAT262 (Cap. 3) → código → interfaz

Ambos proyectos comparten el mismo motor (`app/core/markov_engine.py`) y la misma interfaz
(`app/ui/components.py`); solo cambian los datos y la discretización. La app está organizada
como un recorrido en pestañas que sigue el orden del capítulo:

| Pestaña | Concepto (DAT262) | Fórmula | Función del motor |
|---|---|---|---|
| ① Datos y estados | Def. 1.1 proceso estocástico, espacio de estados | $X:T\times\Omega\to S$ | `discretize.py` |
| ② Matriz P | Def. 3.1 matriz estocástica (+ IC 95 %) | $\hat p_{ij}=N_{ij}/N_{i\cdot}$ | `transition_counts`, `transition_matrix`, `transition_ci` |
| ③ Propiedad de Markov | Ausencia de memoria (contraste χ² de Anderson–Goodman) | $P(j\mid i,h)=p_{ij}$ | `markov_property_test` |
| ④ Predicción Pⁿ | Ej. 3.1, Chapman–Kolmogorov | $\pi_n=\pi_0P^n$ | `n_step`, `n_step_matrix`, `chapman_kolmogorov_terms` |
| ⑤ Estacionaria π | Ej. 3.2 y Aplic. 7.2 (3 métodos: sistema lineal, autovector, potencia) | $\pi P=\pi,\ \sum\pi_i=1$ | `stationary`, `stationary_eig`, `stationary_power` |
| ⑤ Tiempos medios | Teorema ergódico, MTTF | $\mu_i=1/\pi_i,\ 1/(1-p_{ii})$ | `mean_return_times`, `mean_sojourn_times`, `empirical_*` |
| ⑥ Convergencia | Propiedad de convergencia, Fig. 3.1, clasificación de estados | $P^n\to\mathbf 1\pi$, $d(i)=\mathrm{mcd}\{n\}$ | `convergence`, `power_distance`, `mixing_time`, `classify_states`, `period` |
| ⑥ Monte Carlo | Ejercicio 3.2 | $\frac1T\sum\mathbf 1\{X_t=j\}\to\pi_j$ | `simulate` |

Las pruebas (`tests/test_engine.py`) validan el motor contra los ejemplos resueltos del texto
(Ej. 3.1, 3.2 y Aplicación 7.2).

> **Fe de erratas del texto DAT262 detectada por las pruebas:** en la Aplicación 7.2 el texto
> reporta π ≈ (0.813, 0.136, 0.051), pero ese vector no cumple πP = π. La solución exacta es
> π = (5/6, 1/8, 1/24) ≈ (0.8333, 0.1250, 0.0417), es decir, disponibilidad del 83.3 %.
> En el Ejemplo 3.2 el valor exacto es π = (170, 37, 22)/229 ≈ (0.7424, 0.1616, 0.0961).

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
