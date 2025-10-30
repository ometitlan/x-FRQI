FRQI Project — Source Layout

Recommended modules (stable)
- frqi_module.py
  - Classes: FRQI, FRQI2, FRQI3 (via QuantumImageBase)
  - API: encode (circuit → statevector), theoretical_state (formula), recover (exact/sampled),
    analyze_state (entropies/MI), mi_samples (MI exact/sampled), stem_plot_amplitudes, color_Z_stats
- measurements.py
  - Reusable PennyLane samplers/builders for Z-basis measurements and classical MI from samples
- info_measures.py
  - Entropy, conditional entropy, mutual information on statevectors
- utils.py
  - Notebook helpers: generate_image, ensure_uint8, select_frqi
- __init__.py
  - Exposes: FRQI, FRQI2, FRQI3, generate_image, select_frqi

Legacy modules (kept for reference)
- legacy/quantum_image.py
- legacy/quantum_image_sf.py
- legacy/info_measures_sf.py
- legacy/measurements_sf.py

Notes
- Devices
  - Use lightning.qubit for statevector (exact) analysis
  - Use default.qubit with shots>0 for sampling (hardware-like)
- Color channels
  - With n_color_qubits=C, pass C images in [MSB → LSB] order
- Import pattern (from notebooks)
  - Ensure project root on sys.path, then: from src import FRQI, FRQI2, FRQI3, generate_image, select_frqi
  - Optional: from src.measurements import build_expval_Z_family

