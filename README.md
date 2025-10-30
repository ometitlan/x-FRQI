# xFRQI

Correlated Multi‑Channel Feature‑Map FRQI for Quantum Classifiers.

Each color qubit carries a feature map (image or other object), and inter‑channel correlations are treated as first‑class signals for optimization and downstream classification.

## Features
- Multi‑channel FRQI: `n_color_qubits ∈ {1,2,3}` → `2^C` feature channels.
- Two core validations:
  - Theoretical (formula) vs Circuit (statevector)
  - Circuit (statevector) vs Sampling (√p)
- Ready‑to‑use utilities for notebooks: generation, plots, and information metrics.
- Clean layout: core modules in `src/`, historical variants in `src/legacy/` (git‑ignored).

## Requirements
- Python 3.9+
- numpy, matplotlib, pennylane (optional: pennylane‑lightning for fast statevector)

Quick setup (Windows PowerShell):
```
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install numpy matplotlib pennylane pennylane-lightning
```

Quick setup (Linux/macOS):
```
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib pennylane pennylane-lightning
```

## Repository Structure
- `src/README.md` — Technical guide and recommended flow
- `src/frqi_module.py` — FRQI/FRQI2/FRQI3 + main API
- `src/measurements.py` — Z‑basis samplers and classical MI from samples
- `src/info_measures.py` — Entropies and MI on statevectors
- `src/utils.py` — `generate_image`, `ensure_uint8`, `select_frqi`
- `src/legacy/` — Legacy modules (excluded by `.gitignore`)
- `notebooks/Information_encode.ipynb` — End‑to‑end comparisons and examples

## Key Concepts
**Wires and indexing**
- Big‑endian: wire 0 is MSB.
- Position wires: `0..n_pos-1` (MSB→LSB). Color wires: `n_pos..n_pos+n_color-1` (channel 0 = MSB of color block).
- Basis index: `idx = (pos_idx << n_color) | color_bits`.

**Angles and amplitudes**
- For channel k and pixel i: `θ_k(i) = (π/2) * (I_k(i)/255)`; uniform superposition scale `1/√P`.
- Amplitude for `|pos⟩⊗|color_bits⟩`:
  - `α(pos, color_bits) = (1/√P) * ∏_k [cos(θ_k) or sin(θ_k)]` per color bit.
- With `θ ∈ [0, π/2]`, coefficients are real and non‑negative.

## Recommended Workflow
1) Model Validation — Theoretical (formula) vs Circuit (statevector)
   - `theoretical_state(...)` vs `encode(...)`
   - Device: `lightning.qubit` for precision/speed
2) Measurement Validation — Circuit (statevector) vs Sampling (√p)
   - Compare `|α|` vs `√p_j` estimated by counts with `shots > 0`
   - Device: `default.qubit` for sampling
3) (Optional) Reconstruction & Metrics
   - `recover(state, shots=None|>0)`, `analyze_state(state)`, `mi_samples(...)`

## Minimal Usage (Notebook)
Imports and config:
```
from src import FRQI, FRQI2, FRQI3, generate_image, select_frqi
Model = select_frqi(n_color_qubits)
model = Model(image_size, device="lightning.qubit")
img = generate_image(image_size=image_size, use_pattern=True)
images = [img] * n_color_qubits  # or different per channel [MSB→LSB]
```

Two‑panel comparison (what to check):
- A) Theoretical (formula) vs Circuit (statevector)
- B) Circuit (statevector) vs Sampling (√p, `shots=8192`)

## Main APIs
`src/frqi_module.py`
- `encode(*images)` → circuit statevector
- `theoretical_state(*images)` → analytical FRQI state
- `recover(state, shots=None|>0)` → exact or sampling‑based reconstruction
- `analyze_state(state)` → entropies and correlations
- `mi_samples(*images, shots=None|>0, n_blocks=None)` → exact/sampled MI
- `stem_plot_amplitudes(images, shots=None|>0, device=..., threshold=...)` → stem plots

`src/utils.py`
- `generate_image(image_size, use_pattern=True, seed=None, pattern=None)`
- `select_frqi(n_color_qubits)`

## Theoretical vs Circuit, and Sampling
**Teórico vs Circuito**
- Teórico (fórmula): `theoretical_state(...)` calcula |ψ⟩ analíticamente con la fórmula FRQI.
- Circuito (statevector): `encode(...)` ejecuta el circuito y devuelve |ψ⟩ del simulador.
- Validación de modelo: |Teórico| ≈ |Circuito| con diferencias numéricas típicas ~1e−12…1e−8.

**Medición y Muestreo**
- Statevector (“exacto”): lee |ψ⟩ y obtén |α_j| directamente; requiere backend de statevector (p. ej., `lightning.qubit`).
- Muestreo (“hardware‑like”): mide en base computacional con `shots > 0`, estima `p_j ≈ N_j/shots` y compara `|α_j| ≈ √p_j`.
- Error esperado por muestreo: O(1/√shots); aumentar `shots` reduce la discrepancia.

## Correlated Color Channels
Correlations between color qubits capture interactions among feature channels. Use them as signals for optimization and classification.
- `analyze_state(state)`, `mi_samples(...)` for information metrics
- `color_Z_stats(...)` for ⟨Z⟩, ⟨ZZ⟩ on color qubits

## License
MIT (suggested) or Apache‑2.0 — choose according to your needs.

