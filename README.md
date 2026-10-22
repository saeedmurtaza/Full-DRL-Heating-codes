# Full-DRL-Heating-codes

Deep Reinforcement Learning (DQN) supervisory control for the **heating mode of two VRF outdoor units (ODU#1, ODU#2)**. The agent chooses, every 30 minutes, the ODU **target high pressure** and **on/off state** so that the capacity‑weighted indoor air temperature tracks the capacity‑weighted set temperature while minimising ODU power.

The RL environment is **model‑based**: two LightGBM surrogate models (indoor‑air‑temperature forecaster and energy forecaster) stand in for the real building during training. In deployment the same agent + surrogates run against live 1‑minute ODU/IDU data in SQLite plus a KMA (Korea Meteorological Administration) outdoor forecast, and write a 2‑hour‑ahead control policy back to the database.

> Snapshot name: `Updated DRL Heating training and deployment codes_20230818` (training data 2022‑12‑15 → 2023‑01‑11, live DB data 2023‑02‑06 → 2023‑03‑10).

---

## Table of contents

1. [Pipeline overview](#1-pipeline-overview)
2. [Repository layout](#2-repository-layout)
3. [Installation](#3-installation)
4. [Data and feature definitions](#4-data-and-feature-definitions)
5. [Stage 1 — Surrogate models (IAT & Energy)](#5-stage-1--surrogate-models-iat--energy)
6. [Stage 2 — DRL training](#6-stage-2--drl-training)
7. [Stage 3 — Deployment](#7-stage-3--deployment)
8. [ODU#1 vs ODU#2](#8-odu1-vs-odu2)
9. [Known issues and required changes before use](#9-known-issues-and-required-changes-before-use)
10. [Reproducing from scratch](#10-reproducing-from-scratch)

---

## 1. Pipeline overview

```
 1-min BMS log (IDU on/off, set temp, room temp, capacity; ODU out temp, target/high pressure, power, mode)
        │
        ▼  capacity-weighted ST / IAT, 30-min mean, time-of-day & year sin/cos, wet-bulb (OTW)
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ Stage 1  1_Energy and IAT models prediction/                                   │
 │   LightGBM regressors (Optuna-tuned), 2 lagged timesteps as input              │
 │   → IAT.pkl   (predicts next IAT)                                              │
 │   → Energy.pkl (predicts next ODU power)                                       │
 └───────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ Stage 2  2_DRL  Training/                                                      │
 │   gym.Env (ODUx_env.py) steps through the 30-min training CSV; each step the   │
 │   agent's action is fed to IAT.pkl → Energy.pkl → reward                       │
 │   DQN (stable-baselines3) → trained_DQN_agent.zip                              │
 │   control.py replays the agent on the test CSV → RL_results.csv               │
 └───────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ Stage 3  3_Deployment/                                                         │
 │   KMA_requests.py  : KMA ultra-short-term forecast → outdoor_data table        │
 │   control.py       : last 3 × 30-min rows from oduN_data → 4 agent steps      │
 │                      (t+30 … t+120 min) → oduN_drl_policy table               │
 └───────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Repository layout

```
Updated DRL Heating training and deployment codes_20230818/
├── requirements.txt                      # pinned Windows/Python 3.9 environment (see §3)
│
├── 1_Energy and IAT models prediction/   # STAGE 1 – surrogate models
│   ├── ODU#1/
│   │   ├── Energy/ ODU1_Energy_model.ipynb, 07_ODU1_5min_for_energy_prediction_model.xlsx
│   │   └── IAT/    ODU1_IAT_Model.ipynb,    07_ODU1_5min_for_IAT_prediction_model.xlsx
│   └── ODU#2/      (same structure)
│
├── 2_DRL  Training/                      # STAGE 2 – offline RL   (note: two spaces in the folder name)
│   ├── ODU#1/
│   │   ├── ODU1_env.py              # gym.Env: state, 6 discrete actions, surrogate roll-out, reward
│   │   ├── main.py                  # trains DQN with fixed hyper-parameters
│   │   ├── tune.py                  # Optuna search over DQN hyper-parameters
│   │   ├── custom_callback.py       # checkpoint / best-model callback
│   │   ├── utils.py                 # directory helpers
│   │   ├── control.py               # replays trained agent on the test CSV → RL_results.csv
│   │   ├── ALL_data_30min_train.csv # 278 rows, 2022-12-21 08:30 → 2023-01-11 19:00
│   │   ├── ALL_data_30min_test.csv  # 234 rows, 2022-12-15 16:00 → 2023-01-11 11:00
│   │   ├── IAT.pkl, Energy.pkl      # Stage-1 LightGBM models
│   │   └── trained_DQN_agent.zip    # SB3 DQN checkpoint (120 000 steps)
│   └── ODU#2/                       # same files; ODU2_env.py; 317 train / 244 test rows
│
└── 3_Deployment/                         # STAGE 3 – online control
    ├── 1_KMA/KMA_requests.py        # weather forecast fetcher → SQLite outdoor_data
    ├── 2_ODU#1/
    │   ├── control.py               # live policy generator (ODU#1)
    │   ├── utils.py                 # SQLite readers, feature engineering, (de)normalisation, wet-bulb
    │   ├── ODU1_env.py              # env copy used only to build observation/action spaces
    │   ├── IAT.pkl, Energy.pkl, trained_DQN_agent.zip
    │   ├── NED_SQLite.db            # sample database (schema in §7.1)
    │   ├── ODU1_1min_data.xlsx      # sample of the raw 1-min log (121 rows)
    │   ├── test.py                  # EMPTY placeholder
    │   ├── energy_weights_biases.json, iat_weights_biases.json   # EMPTY placeholders
    │   └── __pycache__/             # committed by mistake
    └── 3_ODU#2/                     # same as 2_ODU#1 (no DB copy, no json/test placeholders)
```

---

## 3. Installation

The code targets **Python 3.9 on Windows** (the saved agents report `Windows-10`, `Python 3.9.16`, `PyTorch 1.13.1+cpu`, `Stable-Baselines3 1.7.0`, `Gym 0.21.0`).

`requirements.txt` is a full `pip freeze` of the original machine (121 packages, including unrelated ones such as TensorFlow, Selenium, PyQt5, Pyrogram and `pywin32`). The packages the code actually imports are:

| Package | Pinned in requirements.txt | Used by |
|---|---|---|
| `gym` | 0.21.0 | env (old `gym` API: `reset()` returns obs only, `step()` returns 4‑tuple) |
| `stable-baselines3` | 1.8.0 (agents were saved with 1.7.0) | DQN training / loading |
| `torch` | 2.0.1 (agents saved with 1.13.1) | DQN backend |
| `lightgbm` | 3.3.2 | IAT / Energy surrogates (`.pkl` are `lightgbm.sklearn.LGBMRegressor`) |
| `optuna` | 3.0.3 | hyper‑parameter search (notebooks + `tune.py`) |
| `pandas` / `numpy` | 1.5.1 / 1.23.4 | everywhere |
| `scikit-learn` | 1.1.3 | metrics, `train_test_split` |
| `openpyxl` | 3.0.7 | reading the `.xlsx` datasets |
| `lxml` | 4.9.1 | XML parser for BeautifulSoup |
| `requests` | **not listed** | `KMA_requests.py` |
| `beautifulsoup4` | **not listed** | `KMA_requests.py` |
| `sqlite3` | stdlib | deployment |

Minimal environment (recommended instead of installing the full freeze):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   Linux/macOS: source .venv/bin/activate
pip install "gym==0.21.0" "stable-baselines3==1.8.0" "torch==2.0.1" \
            "lightgbm==3.3.2" "optuna==3.0.3" "pandas==1.5.1" "numpy==1.23.4" \
            "scikit-learn==1.1.3" "openpyxl==3.0.7" "lxml==4.9.1" \
            requests beautifulsoup4
```

> `gym==0.21.0` needs `setuptools<66` and `wheel<0.40` to build on recent pip; if installation fails run `pip install "setuptools<66" "wheel<0.40"` first. Loading the `.pkl` surrogates requires a LightGBM version that can unpickle 3.3.x models; 3.3.2 is the safe choice.

---

## 4. Data and feature definitions

### 4.1 Raw 1‑minute BMS columns (`oduN_data` table / `ODUN_1min_data.xlsx`)

| Column | Meaning |
|---|---|
| `timestamp` | 1‑min sample time |
| `IDUk_onoff` | indoor unit k state, text `운전 On` / `운전 Off` (k = 0..3 for ODU#1, 0..4 for ODU#2) |
| `IDUk_settemp` | indoor unit k set temperature (°C) |
| `IDUk_intemp` | indoor unit k return‑air temperature (°C) |
| `IDUk_capa` | indoor unit k rated capacity (used as weight) |
| `ODU_outtemp` | outdoor air temperature at the ODU (°C) |
| `ODU_targetpress` | ODU target high pressure |
| `ODU_highpress` | measured ODU high pressure |
| `ODU_power` | ODU electrical power (W) |
| `ODU_onoff` | ODU operating mode text; only `일반 운전` (normal run) and `정지` (stop) are used, all other modes (defrost, oil recovery, safety start, …) are filtered out |

### 4.2 Engineered 30‑minute features (used by both surrogates and the RL state)

| Feature | Definition (see `3_Deployment/*/utils.py::get_oduN_data`) |
|---|---|
| `ST` | capacity‑weighted set temperature over **running** IDUs: Σ(on·settemp·capa) / Σ(on·capa). If all IDUs are off every IDU is treated as on so the denominator is non‑zero |
| `IAT` | capacity‑weighted indoor air temperature, same weighting |
| `OAT` | `ODU_outtemp` |
| `High_Pressure` | `ODU_highpress` |
| `Energy` | `ODU_power`; rows with `ODU_onoff == 0` **and** `ODU_power > 100 W` are dropped |
| `Op_Mode_1` / `Op_Mode_0` | one‑hot ODU on / off (`일반 운전`→1, `정지`→0; 30‑min mean ≥ 0.5 → 1) |
| `Day_sin`, `Day_cos` | sin/cos of time‑of‑day (period 86 400 s) |
| `Year_sin`, `Year_cos` | sin/cos of time‑of‑year (period 365.2425 days) |
| `OTW` | outdoor **wet‑bulb** temperature from OAT and relative humidity (Stull 2011 formula, `compute_WBT`) |
| `Delta_IAT_OAT`, `Delta_ST_IAT`, `Delta_ST_OAT` | pairwise differences, recomputed after normalisation |

All raw samples are averaged with `pd.Grouper(freq='30min')`.

### 4.3 Min–max normalisation (identical in every file)

| Variable | min | max |
|---|---|---|
| `ST` | 18 | 31 |
| `IAT` | 15 | 35 |
| `OAT` | −20 | 20 |
| `OTW` | −15 | 10 |
| `High_Pressure` | 5 | 35 |
| `Energy` | 0 | **15 000** (ODU#1) / **18 000** (ODU#2) |

The deltas are computed **after** normalisation, so they live in [−1, 1]. `inverse_normalize` rounds `High_Pressure` to an integer when writing the policy.

### 4.4 Training CSVs (`ALL_data_30min_{train,test}.csv`)

Columns: `timestamp, hour, minutes, Op_Mode_1, ST, IAT, OAT, Op_Mode_0, Energy, High_Pressure, Day_sin, Day_cos, Year_sin, Year_cos, OTW, Delta_IAT_OAT, Delta_ST_IAT, Delta_ST_OAT` — **un‑normalised** values; the env normalises on load.

---

## 5. Stage 1 — Surrogate models (IAT & Energy)

Folder: `1_Energy and IAT models prediction/ODU#N/{IAT,Energy}/`

Each notebook:

1. Loads the 5‑minute dataset `07_ODUN_5min_for_*_prediction_model.xlsx` (ODU#1 ≈ 1 540 rows, ODU#2 ≈ 1 735 rows — note the ODU#2 sheets have trailing empty columns; drop them).
2. Applies the normalisation of §4.3.
3. Builds a supervised set with `series_to_supervised(values, n_in=2, n_out=1)` — i.e. **two lagged timesteps + the current exogenous inputs** predict the current target.
4. Splits 80/20 (`train_test_split`, `shuffle` as in the notebook), runs an **Optuna** search (`n_trials=1000`, maximise R²) over `num_leaves, n_estimators, learning_rate, max_depth, reg_alpha, reg_lambda`.
5. Fits `LGBMRegressor` with the best parameters and pickles it.

### 5.1 Input layout

| Model | Column order of each timestep | Current‑step columns dropped | Feature count |
|---|---|---|---|
| **IAT** | `ST, OAT, Delta_IAT_OAT, Delta_ST_IAT, Delta_ST_OAT, High_Pressure, Day_sin, Day_cos, Op_Mode_0, Op_Mode_1, Year_sin, Year_cos, OTW, IAT` (14) | `var3(t)`=Delta_IAT_OAT, `var4(t)`=Delta_ST_IAT (unknown before IAT is known) and the target `var14(t)`=IAT | 14·2 + 11 = **39** |
| **Energy** | `ST, IAT, OAT, Delta_IAT_OAT, Delta_ST_IAT, Delta_ST_OAT, High_Pressure, Day_sin, Day_cos, Op_Mode_0, Op_Mode_1, Year_sin, Year_cos, OTW, Energy` (15) | only the target `var15(t)`=Energy | 15·2 + 14 = **44** |

The env/deployment code rebuilds exactly these arrays (`IAT_forecast_array` 3×14, `energy_forecast_array` 3×15), so **the column order above must not change** without retraining.

### 5.2 Final hyper‑parameters and reported test metrics (values in the notebooks, de‑normalised units)

| Model | num_leaves | n_estimators | lr | max_depth | reg_alpha | reg_lambda | Test R² | Test RMSE | Test MAE |
|---|---|---|---|---|---|---|---|---|---|
| ODU#1 Energy | 2720 | 500 | 0.1 | 11 | 0.1075 | 0.7253 | 0.922 | 611 W | 321 W |
| ODU#1 IAT | 1920 | 500 | 0.3 | 3 | 0.0098 | 0.5538 | 0.859 | 0.64 °C | 0.34 °C |
| ODU#2 Energy | 200 | 500 | 0.1 | 5 | 0.0170 | 0.6996 | 0.885 | 943 W | 520 W |
| ODU#2 IAT | 840 | 500 | 0.1 | 20 | 0.0446 | 0.6630 | 0.918 | 0.59 °C | 0.34 °C |

Outputs: `IAT.pkl`, `Energy.pkl` (copied into `2_DRL  Training/ODU#N/` and `3_Deployment/*/`), plus `training_results_past_2.xlsx` / `testing_results_past_2.xlsx`.

---

## 6. Stage 2 — DRL training

Folder: `2_DRL  Training/ODU#N/` (run every script **from inside this folder** — all paths are relative).

### 6.1 Environment (`ODUN_env.py::ODUEnvironment`)

**Observation** — `Box(14,)` in normalised units, index order used throughout:

| idx | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | Day_sin | Day_cos | ST | IAT | OAT | Δ IAT‑OAT | Δ ST‑IAT | Δ ST‑OAT | High_Pressure | Op_Mode_1 | Year_sin | Year_cos | OTW | Energy |

**Action** — `Discrete(6)`; `actions_represented[a] = (normalised target high pressure, ODU on)`:

| a | normalised HP | HP (de‑normalised, 5–35 scale) | ODU |
|---|---|---|---|
| 0 | 0.7000 | 26 | **off** |
| 1 | 0.7000 | 26 | on |
| 2 | 0.7333 | 27 | on |
| 3 | 0.7667 | 28 | on |
| 4 | 0.8000 | 29 | on |
| 5 | 0.8333 | 30 | on |

**Episode** — one pass over `ALL_data_30min_train.csv`. `reset()` sets the cursor to row 2 (rows 0–1 seed the two‑lag history). `step()`:

1. Assemble the IAT forecast window (rows t‑2, t‑1 from history, row t from the current state with the agent's HP / on‑off), predict `IAT(t)`, recompute the IAT‑dependent deltas.
2. Assemble the energy window with the forecasted IAT, predict `Energy(t)`.
3. Compute the reward (below) from de‑normalised `ST − IAT` and **normalised** forecasted energy `E ∈ [0,1]`.
4. Advance the cursor; the next state takes exogenous values (OAT, OTW, time encodings, ST) from the CSV row and overwrites `IAT`, the deltas, `High_Pressure`, `Op_Mode_1`, `Energy` with the forecasts / action.
5. `done = True` when the cursor reaches the last row.

**Reward** (`ODUN_env.py`, identical for both ODUs):

| `ST − IAT` (°C) | reward |
|---|---|
| > 2 | −50 − 2·E |
| (1.5, 2] | −30 − 3·E |
| (1, 1.5] | −10 − 4·E |
| (0.5, 1] | −5 − 5·E |
| (0, 0.5] | −5·E |
| **(−0.5, 0]** | **+10 + 5 / E** |
| (−1, −0.5] | −30 − 15·E |
| ≤ −1 | −50 − 20·E |

i.e. the only positive band is IAT slightly **above** set‑point (0–0.5 °C); under‑heating and over‑heating are penalised asymmetrically, and energy is penalised more strongly when the room is already warm.

### 6.2 Training (`main.py`)

```bash
cd "2_DRL  Training/ODU#1"
python main.py
```

`main.py` runs `check_env`, seeds everything (`seed=1`), empties `./logs/` (TensorBoard) and `./saved_agents_1/`, then trains `DQN('MlpPolicy')`:

| hyper‑parameter | ODU#1 | ODU#2 |
|---|---|---|
| learning_rate | 0.01 | 0.01 |
| buffer_size | 10 000 | 100 000 |
| tau | 0.2 | 0.2 |
| gamma | 0.99 | 0.95 |
| train_freq / gradient_steps | 4 / 1 | 4 / 1 |
| target_update_interval | 5 | 10 |
| exploration_fraction | 0.3 | 0.1 |
| total_timesteps in `main.py` | 80 000 | 1 000 000 |

`TrainLoggingCallback` (`custom_callback.py`) saves `dqn_checkpoint_<n>` every 10 000 steps, saves `dqn_checkpoint_best_model` whenever a one‑step evaluation on a fresh `eval_env` beats the previous best (evaluation starts after `start_saving_timesteps=50 000`; `eval_freq` is accepted but not used — evaluation happens every step after that point), and saves a final checkpoint on `_on_training_end`.

The bundled `trained_DQN_agent.zip` files are **120 000‑step checkpoints** of runs configured for 1 000 000 steps (SB3 1.7.0, PyTorch 1.13.1 CPU, `exploration_final_eps=0.05`); the ODU#1 hyper‑parameters inside match `main.py`, the ODU#2 ones match its `main.py` ("used 41 trial" of `tune.py`).

### 6.3 Hyper‑parameter search (`tune.py`)

Optuna study (`n_trials=1000`, maximise the callback's best one‑step reward, 100 000 steps per trial) over
`learning_rate ∈ {1e‑4,1e‑3,1e‑2,1e‑1}`, `buffer_size ∈ {1e4,1e5,1e6}`, `tau ∈ {0,0.2,0.5,1}`, `gamma ∈ {0.99,0.95,0.8,0.7}`, `train_freq ∈ {4,10}`, `gradient_steps ∈ {4,1}`, `target_update_interval ∈ {1,5,10}`, `exploration_fraction ∈ {0.1…0.5}`, `seed ∈ {1,3,13}`. Results are printed by Optuna; `OUTPUT_FILE` is declared but never written.

### 6.4 Offline evaluation (`control.py`)

```bash
python control.py          # → RL_results.csv
```

Loads `trained_DQN_agent.zip`, `IAT.pkl`, `Energy.pkl`, then for **each calendar day** in `ALL_data_30min_test.csv` (days with ≤ 2 rows are skipped) replays the agent step by step exactly as in `step()` (without reward) and writes the de‑normalised trajectory — `timestamp, ST, IAT, OAT, deltas, High_Pressure, Day_sin, Day_cos, Op_mode_0.0, Op_mode_1.0, Year_sin, Year_cos, OTW, Energy` — to `RL_results.csv`.

---

## 7. Stage 3 — Deployment

Folder: `3_Deployment/`. Designed to be run by a scheduler every 30 min on the site PC. Both scripts talk to one SQLite file (`NED_SQLite.db`).

### 7.1 Database schema (from the bundled `NED_SQLite.db`)

| Table | Written by | Columns |
|---|---|---|
| `odu1_data` | BMS logger (external) | raw 1‑min columns of §4.1, 4 IDUs (41 597 rows) |
| `odu2_data` | BMS logger (external) | raw 1‑min columns, 5 IDUs (40 606 rows) |
| `outdoor_data` | `KMA_requests.py` (replaced each run) | `timestamp, OAT (°C int), ORH (% int)` — 6 hourly forecast rows |
| `oud1_policy_temp` / `oud2_policy_temp` | `control.py` (temporary) | `timestamp, ODU_targetpress, ODU_onoff` |
| `odu1_drl_policy` / `odu2_drl_policy` | `control.py` (merged history) | `timestamp, ODU_targetpress, ODU_onoff` |

### 7.2 `1_KMA/KMA_requests.py` — outdoor forecast

* Calls the public‑data‑portal endpoint `VilageFcstInfoService_2.0/getUltraSrtFcst` (초단기예보, ultra‑short‑term forecast) for grid cell `nx=61, ny=125`, `base_date` = today, `base_time` = current hour `HH00`.
* Parses the XML with BeautifulSoup/lxml, keeps category `T1H` (air temperature) and `REH` (relative humidity) items, builds `timestamp / OAT / ORH` and **replaces** `outdoor_data`.
* Retries up to 40 times (30 s apart) on empty responses and 20 times (60 s apart) on exceptions.

The service key is **hard‑coded in the script** — see §9.

### 7.3 `2_ODU#1/control.py` (and `3_ODU#2/control.py`) — live policy

1. `get_odu1_data()` — `SELECT … FROM odu1_data WHERE ODU_onoff IN ('일반 운전','정지') ORDER BY timestamp DESC LIMIT 100`, builds the §4.2 features, 30‑min mean, keeps the **last 3 rows** (two lags + current).
2. `normalize()` — also computes `OTW` from the *current* outdoor T/RH (`outdoor_data` row 0).
3. Loads `trained_DQN_agent.zip`, `IAT.pkl`, `Energy.pkl` and builds an `ODUEnvironment` (only to initialise spaces).
4. Loops **4 steps** (`start` 2→5): agent action → OAT/RH for the step taken from `outdoor_data` row 1 (next forecast hour, same row for all 4 steps) → IAT forecast → energy forecast → next state built with time encodings at `now + 30, 60, 90, 120 min`.
5. De‑normalises, rounds HP, and writes rows `(timestamp, ODU_targetpress int, ODU_onoff)` to `oud1_policy_temp`, then merges into `odu1_drl_policy`: existing timestamps are **updated**, new ones **appended**, table rewritten.

ODU#1 writes `ODU_onoff` as text (`일반 운전` / `정지`); ODU#2 leaves it numeric (1 / 0). Downstream consumers must handle both or the ODU#2 script should add the same `replace` mapping.

### 7.4 Running

```bash
# 1. fetch weather (replaces outdoor_data)
python "3_Deployment/1_KMA/KMA_requests.py"

# 2. generate policy for each ODU (run from the ODU folder)
cd "3_Deployment/2_ODU#1" && python control.py
cd "3_Deployment/3_ODU#2" && python control.py
```

Before this works you **must** fix the absolute paths listed in §9.

---

## 8. ODU#1 vs ODU#2

| | ODU#1 | ODU#2 |
|---|---|---|
| Indoor units | 4 (`IDU0`–`IDU3`) | 5 (`IDU0`–`IDU4`) |
| Energy normalisation max | 15 000 W | 18 000 W |
| DQN buffer / γ / target_update / explore_frac | 10 000 / 0.99 / 5 / 0.3 | 100 000 / 0.95 / 10 / 0.1 |
| `main.py` total_timesteps | 80 000 | 1 000 000 |
| Surrogate R² (test) IAT / Energy | 0.859 / 0.922 | 0.918 / 0.885 |
| Policy table `ODU_onoff` type | text (Korean) | numeric |

Everything else (state, actions, reward, callbacks, `series_to_supervised`, wet‑bulb formula, time encodings) is byte‑for‑byte the same apart from comments.

---

## 9. Known issues and required changes before use

**Security / portability**

1. **A KMA public‑data API service key is committed in `3_Deployment/1_KMA/KMA_requests.py`.** The repository is public — rotate the key on data.go.kr and load it from an environment variable (e.g. `os.environ["KMA_SERVICE_KEY"]`) or a `.env` file (`python-dotenv` is already pinned).
2. Hard‑coded Windows paths must be replaced (relative paths or a config/env variable):
   * `sqlite3.connect(r"C:\Users\user\Desktop\NED_SQLite.db")` — `KMA_requests.py`, `2_ODU#1/control.py`, `2_ODU#1/utils.py`
   * `C:\Users\user\Desktop\Synology_Friday_backup\…\2_ODU#1\{trained_DQN_agent.zip, Energy.pkl, IAT.pkl}` — `2_ODU#1/control.py`, `2_ODU#1/ODU1_env.py`
   * `C:/Users/HUBEL/Desktop/Backup_Patrick/…/3_Deployment/{2_ODU#1/NED_SQLite.db, 3_ODU#2/…}` — `3_ODU#2/control.py`, `3_ODU#2/utils.py`, `3_ODU#2/ODU2_env.py`
3. `3_Deployment/2_ODU#1/utils.py` opens the DB connection at import time; `KMA_requests.py` closes its connection inside the loop.

**Code‑level**

4. Reward band `(−0.5, 0]` returns `10 + 5/E`; if the forecasted normalised energy `E` is 0 (ODU off) this raises `ZeroDivisionError` / produces `inf`. Guard with a small epsilon or a cap.
5. `TrainLoggingCallback` ignores `eval_freq`; after 50 000 steps it evaluates **every** step with a single `step()` from `reset()`, so "best model" means best first‑step reward, not best episode return.
6. ODU#1 `main.py` trains for 80 000 steps but saving/evaluation only begins at 50 000; the bundled agents come from 1 000 000‑step runs stopped at 120 000.
7. The surrogates are trained on **5‑minute** data but driven with **30‑minute** states in the env and in deployment; retrain on 30‑min aggregates (or resample) if you change the control interval.
8. Deployment uses the same `outdoor_data` row (index 1) for all four 30‑min horizons instead of the matching hourly forecast.
9. `requirements.txt` omits `requests` and `beautifulsoup4`; it also pins `stable-baselines3==1.8.0` / `torch==2.0.1` while the agents were saved with 1.7.0 / 1.13.1 (they load fine, but retraining will differ).
10. `test.py`, `energy_weights_biases.json`, `iat_weights_biases.json` are empty placeholders; `__pycache__/` should be git‑ignored.
11. The folder `2_DRL  Training` contains two spaces — quote it in shell commands.

---

## 10. Reproducing from scratch

```bash
# 0. environment (see §3)

# 1. surrogates — run the four notebooks, each writes IAT.pkl or Energy.pkl next to itself
jupyter nbconvert --to notebook --execute "1_Energy and IAT models prediction/ODU#1/IAT/ODU1_IAT_Model.ipynb"
#  … repeat for ODU#1/Energy, ODU#2/IAT, ODU#2/Energy, then copy the .pkl files into
#  "2_DRL  Training/ODU#N/" and "3_Deployment/{2_ODU#1,3_ODU#2}/"

# 2. (optional) hyper-parameter search, then training
cd "2_DRL  Training/ODU#1"
python tune.py            # long: 1000 trials × 100k steps
python main.py            # → saved_agents_1/dqn_checkpoint_*.zip, logs/ (tensorboard --logdir logs)
cp saved_agents_1/dqn_checkpoint_best_model.zip trained_DQN_agent.zip

# 3. offline evaluation
python control.py         # → RL_results.csv

# 4. deployment (after fixing paths/keys per §9)
python "../../3_Deployment/1_KMA/KMA_requests.py"
cd "../../3_Deployment/2_ODU#1" && python control.py
```

A suggested `.gitignore` for the repo:

```
__pycache__/
*.pyc
logs/
saved_agents*/
RL_results.csv
*results_past_2.xlsx
.env
```
