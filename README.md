<p align="center">
  <img src="assets/logo_ing.jpg" alt="Facultad de Ingeniería UNAM" width="180"/>
</p>

<h1 align="center">xFRQI — Multi-Channel Correlated Feature Maps</h1>
<p align="center">Codificación FRQI extendida para representar canales correlacionados y alimentar clasificadores/cuánticos híbridos.</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"/></a>
  <a href="https://pennylane.ai/"><img src="https://img.shields.io/badge/PennyLane-quantum%20ML-22c55e" alt="PennyLane"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"/></a>
</p>

---

Proyecto desarrollado con el apoyo del Consejo Nacional de Ciencia y Tecnología (CONAHCYT) mediante la beca de Estancias Posdoctorales por México 2022 (modalidad Académica - Inicial), CVU 469604.

- **Institución:** Facultad de Ingeniería, UNAM  
- **Director de Proyecto:** Dr. Boris Escalante Ramírez  
- **Período:** Diciembre 2022 - Noviembre 2024

Autores:
- Dr. Mario Alberto Mercado Sánchez — ometitlan@gmail.com  

---

## Tabla de contenidos

- [Visión general](#visión-general)
- [Cuadernos destacados](#cuadernos-destacados)
- [Inicio rápido](#inicio-rápido)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Componentes principales](#componentes-principales)
- [Conceptos clave de xFRQI](#conceptos-clave-de-xfrqi)
- [Flujo de trabajo recomendado](#flujo-de-trabajo-recomendado)
- [APIs principales](#apis-principales)
- [Canales de color correlacionados](#canales-de-color-correlacionados)
- [Recuperación de imagen](#recuperación-de-imagen)
- [Agradecimientos y créditos](#agradecimientos-y-créditos)
- [Licencia](#licencia)

---

## Visión general

Muchas de las propiedades relevantes de un sistema cuántico multipartito dependen directamente de cómo decidimos codificar la información clásica en el estado cuántico. En el caso de imágenes cuánticas, esta codificación puede aprovechar la estructura ondulatoria de la función de estado, utilizando tanto los estados base para representar índices discretos (como las posiciones de píxeles) como las amplitudes y fases relativas para representar magnitudes continuas (como intensidades o atributos derivados).

La representación FRQI (Flexible Representation of Quantum Images) establece un mapeo en el que un único qubit de color codifica la intensidad del píxel mediante una amplitud rotada, mientras que los qubits restantes describen la posición del píxel en los estados base del registro.

La extensión xFRQI generaliza este esquema incorporando múltiples qubits de “color”, de forma que cada qubit adicional puede codificar un canal o atributo independiente (por ejemplo, componentes RGB, filtros convolucionales, kernels, mapas de características o máscaras). Esto no solo permite representar información multicanal, sino que además las correlaciones cuánticas entre los qubits de color y de posición (y entre los propios canales) se vuelven parte del espacio de información disponible para optimización, análisis y clasificación.

El proyecto se basa en esta idea para construir un codificador cuántico multicanal que permita extraer medidas informacionales (entropía, información mutua, correlaciones físicas) directamente de la estructura del estado cuántico resultante.

- Implementaciones FRQI mono/multi-canal (`FRQI`, `FRQI2`, `FRQI3`) con soporte para imágenes cuadradas y patrones sintéticos.
- Herramientas para validar el modelo comparando estado teórico vs circuito cuántico vs muestreo “hardware-like”.
- Métricas de información (entropías, información mutua) que cuantifican correlaciones entre canales de color.
- Cuadernos Jupyter listos para experimentar con codificación, reconstrucción de imágenes y visualizaciones de amplitudes.

---

## Cuadernos destacados

- **`notebooks/Information_encode.ipynb`**  
  Comparaciones entre la fórmula teórica FRQI y los statevectors producidos por circuitos, análisis de errores, validación mediante muestreo y gráficos de amplitudes.

- **`notebooks/pruebas.ipynb`**  
  Banco de experimentos y versiones de funciones auxiliares (generación de imágenes sintéticas, análisis de correlaciones, etc.).

---

## Inicio rápido

1. **Crea y activa un entorno virtual**
   - **Windows (venv `goq`, ignorado por Git)**  
     - CMD (evita policies de PowerShell):
       ```cmd
       cd /d D:\Documents\GitHub\x-FRQI
       C:\Users\TECNOCOSMOS\AppData\Local\Programs\Python\Python311\python.exe -m venv goq
       goq\Scripts\activate
       ```
     - PowerShell (en la sesión actual si hay restricciones):
       ```powershell
       Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
       .\goq\Scripts\Activate.ps1
       ```
     - Alternativa: crea el entorno fuera del repo (p. ej. `D:\envs\goq`) y actívalo con `D:\envs\goq\Scripts\activate`.
   - **Linux/macOS**
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```
2. **Instala dependencias básicas**
   ```bash
   pip install numpy matplotlib pennylane pennylane-lightning
   ```
3. **Abre el notebook principal**
   - `notebooks/Information_encode.ipynb`
   - Ejecuta las celdas de validación (estado teórico vs circuito vs muestreo).
4. **Explora módulos desde scripts**
   ```python
   from src import FRQI, FRQI2, FRQI3, generate_image, select_frqi
   Model = select_frqi(n_color_qubits=3)
   model = Model(image_size=8, device="lightning.qubit")
   images = [generate_image(8, use_pattern=True)] * 3
   state = model.encode(*images)
   ```

---

## Estructura del repositorio

| Ruta | Contenido |
| --- | --- |
| `src/frqi_module.py` | Implementación FRQI/FRQI2/FRQI3, funciones `encode`, `theoretical_state`, `recover`, `analyze_state`, etc. |
| `src/measurements.py` | Rutinas de muestreo en base Z e información mutua clásica. |
| `src/info_measures.py` | Cálculo de entropías e información mutua directamente en statevectors. |
| `src/utils.py` | Utilidades (`generate_image`, `ensure_uint8`, `select_frqi`, kernels Sobel/Laplaciano/Gabor dimensionables, `build_kernel_patch`, `apply_kernel_bank`). |
| `src/legacy/` | Versiones anteriores excluidas por `.gitignore`. |
| `notebooks/` | Cuadernos `Information_encode.ipynb`, `pruebas.ipynb` y ejercicios de mapas de información con kernels clásicos. |
| `examples/` | Material complementario (datasets de prueba, scripts auxiliares). |
| `assets/` | Recursos gráficos (logo institucional). |

---

## Componentes principales

- **Motor FRQI (`src/frqi_module.py`)**  
  - `FRQI`, `FRQI2`, `FRQI3`: manejan 1, 2 o 3 qubits de color.  
  - `encode(*images)`: ejecuta el circuito y devuelve el statevector.  
  - `theoretical_state(*images)`: calcula el estado analíticamente.  
  - `recover(...)`, `analyze_state(...)`, `mi_samples(...)`.

- **Métricas e información (`src/info_measures.py`, `src/measurements.py`)**  
  Cálculo de entropías, información mutua y estadísticas de medición (`color_Z_stats`, `mi_samples`, etc.).

- **Utilidades (`src/utils.py`)**  
  Generación de imágenes sintéticas, selección automática del modelo según el número de canales, aseguramiento de tipos `uint8`, kernels dimensionables (Sobel/Laplaciano/Gabor), parches de kernel (`build_kernel_patch`) y bancos de kernels (`apply_kernel_bank`).

- **Mapas locales de información**  
  Puedes codificar ventanas locales (p. ej. 4×4 u 8×8) con `FRQI2/FRQI3`, usando un canal para el parche de imagen y otro para el parche del kernel. Con `analyze_state` obtienes entropías e informaciones mutuas; con `color_Z_stats` obtienes `<Z_k>` y `<ZZ_ij>` (analíticos o por muestreo con `shots>0`, aptos para hardware real). Ejemplo:
  ```python
  from src import FRQI2
  from src.utils import ensure_uint8, build_kernel_patch
  win = 8
  img_u8 = ensure_uint8(img * 255)  # img 28x28 en [0,1]
  kpatch = build_kernel_patch("sobel_x", win)
  model = FRQI2(image_size=win, device="default.qubit")
  info = model.analyze_state(model.encode(img_u8[:win,:win], kpatch))
  zstats = model.color_Z_stats(img_u8[:win,:win], kpatch, shots=1024)
  ```
  Claves típicas de `analyze_state`: `H_total`, `H_position`, `H_color0/1`, `I(color*:position)`, `H_color*|position`, `I(color0:color1)`, `I3(position:color0:color1)`.

---

## Conceptos clave de xFRQI

- **Convención de wires (big-endian):**  
  - Posición: `0 .. n_pos-1` (MSB→LSB).  
  - Color: `n_pos .. n_pos + n_color - 1` (el canal 0 es el MSB del bloque de color).  
  - Índice base: `idx = (pos_idx << n_color) | color_bits`.

- **Codificación de ángulos:**  
  - `θ_k(i) = (π/2) × (I_k(i) / 255)` para cada píxel `i` y canal `k`.  
  - Superposición uniforme con factor `1/√P`.  
  - Amplitudes reales y no negativas: `α(pos, color_bits) = (1/√P) × ∏ [cos(θ_k) o sin(θ_k)]`.

---

## Flujo de trabajo recomendado

1. **Validación del modelo**  
   Compara estado teórico (`theoretical_state`) vs circuito (`encode`) usando `lightning.qubit` para máxima precisión.
2. **Validación de medición**  
   Ejecuta muestreos con `shots > 0`, compara `|α|` vs `√p_j` y cuantifica el error `O(1/√shots)`.
3. **Reconstrucción y análisis**  
   - `recover(state, shots=None|>0)` para obtener imágenes exactas o muestreadas.  
   - `analyze_state(state)` y `mi_samples(...)` para entropías, correlaciones y métricas de información.  
   - `stem_plot_amplitudes(...)` para visualizar amplitudes canal a canal.

---

## APIs principales

```python
from src import FRQI, FRQI2, FRQI3, generate_image, select_frqi

Model = select_frqi(n_color_qubits=2)
model = Model(image_size=16, device="lightning.qubit")

images = [generate_image(16, use_pattern=True, seed=42) for _ in range(2)]

state_theoretical = model.theoretical_state(*images)
state_circuit = model.encode(*images)
rec_image = model.recover(state_circuit)
info = model.analyze_state(state_circuit)
```

Funciones clave:

- `encode(*images)`
- `theoretical_state(*images)`
- `recover(state, shots=None|>0)`
- `analyze_state(state)`
- `mi_samples(*images, shots=None|>0, n_blocks=None)`
- `stem_plot_amplitudes(images, shots=None|>0, device=..., threshold=...)`

---

## Canales de color correlacionados

La principal aportación de xFRQI es tratar las correlaciones entre qubits de color como *features* útiles:

- Capturan interacciones entre canales (p. ej. R/G/B, mapas de gradiente, filtros CNN).
- Sirven para optimizar circuitos, mejorar clasificadores y enriquecer descriptores cuánticos.
- Herramientas: `analyze_state(state)`, `mi_samples(...)`, `color_Z_stats(...)` para obtener ⟨Z⟩ y ⟨ZZ⟩.

---

## Recuperación de imagen

- **Exacta:** usa el statevector `|ψ⟩` para reconstruir probabilidades conjuntas posición-color sin ruido.
- **Por muestreo:** emplea conteos de mediciones; la fidelidad aumenta con más `shots`.
- **Orden de canales:** se respeta MSB→LSB al reconstruir (coherente con `select_frqi`).

---

## Licencia

Este proyecto se distribuye bajo licencia [MIT](LICENSE). Si empleas xFRQI en investigaciones o demostraciones, menciona a los autores y referencia este repositorio.
