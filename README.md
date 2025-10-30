# xFRQI

Mapa de Características Correlacionadas Multi-Canal FRQI para Clasificadores Cuánticos.

En el modelo extendido X-FRQI cada qubit "de color" representa un canal que puede representar un mapa de características de una imagen u otro objeto, y las correlaciones entre canales se tratan como nuevas características útiles para ejercicios de optimización y clasificación.

## Usando Information_encode.ipynb

### Comparación Teórico vs Circuito

- **Teórico (fórmula)**: `theoretical_state(...)` calcula |ψ⟩ analíticamente usando la fórmula FRQI
- **Circuito (statevector)**: `encode(...)` ejecuta el circuito cuántico y devuelve |ψ⟩ del simulador
- **Validación esperada**: |Teórico| ≈ |Circuito| con diferencias típicas de ~1e⁻¹² a 1e⁻⁸

### Medición y Muestreo

Dos formas de obtener información del estado cuántico:

**Statevector ("exacto")**:
- Lee directamente el vector de estado |ψ⟩ y obtiene todas las amplitudes |α_j|
- Requiere un backend de statevector como `lightning.qubit`
- No disponible en hardware cuántico real

**Muestreo ("hardware-like")**:
- Mide en la base computacional con `shots > 0` (simula hardware real)
- Estima probabilidades: `p_j ≈ N_j/shots`
- Compara amplitudes: `|α_j| ≈ √p_j`
- Error esperado: O(1/√shots) — más shots = menor error

- Puedes usar un modelo FRQI multi-canal: `n_color_qubits ∈ {1,2,3}` → hasta tres canales de características.
- Dos validaciones principales:
  - El valor Teórico de la función de estado dado por la fórmula del modelo vs La función de estado producida por el Circuito (statevector)
  - Circuito (statevector) vs Muestreo (√p) usando un simulador 
- Utilidades listas para usar en notebooks: generación de gráficos y métricas de información.

## Requisitos

Para ejecutar este proyecto necesitas:
- Python 3.9 o superior
- Las bibliotecas: numpy, matplotlib, pennylane
- Opcional pero recomendado: pennylane-lightning para cálculos más rápidos

### Configuración rápida (Windows PowerShell):
```
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install numpy matplotlib pennylane pennylane-lightning
```

### Configuración rápida (Linux/macOS):
```
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib pennylane pennylane-lightning
```

## Estructura del Repositorio

El proyecto está organizado de la siguiente manera:

- `src/README.md` — Guía técnica detallada y flujo de trabajo recomendado
- `src/frqi_module.py` — Implementaciones FRQI/FRQI2/FRQI3 y API principal
- `src/measurements.py` — Funciones para muestreo en base Z e información mutua clásica
- `src/info_measures.py` — Cálculo de entropías e información mutua en statevectors
- `src/utils.py` — Utilidades: `generate_image`, `ensure_uint8`, `select_frqi`
- `src/legacy/` — Módulos legacy (excluidos por `.gitignore`)
- `notebooks/Information_encode.ipynb` — Comparaciones completas y ejemplos de uso
- `notebooks/pruebas.ipynb` — Experimentos y versiones de funciones

## Conceptos Clave

### Organización de Qubits (Wires)

Usamos la convención **big-endian**: el wire 0 es el bit más significativo (MSB).

- **Wires de posición**: `0..n_pos-1` (ordenados de MSB a LSB)
- **Wires de color**: `n_pos..n_pos+n_color-1` (el canal 0 es el MSB del bloque de color)
- **Índice de base**: `idx = (pos_idx << n_color) | color_bits`

### Codificación de Ángulos y Amplitudes

Para cada canal k y píxel i, el ángulo de codificación es:
- `θ_k(i) = (π/2) × (I_k(i)/255)`, donde I_k(i) es la intensidad del píxel (0-255)
- La superposición uniforme usa una escala de `1/√P`

La amplitud para un estado `|pos⟩⊗|color_bits⟩` se calcula como:
- `α(pos, color_bits) = (1/√P) × ∏_k [cos(θ_k) o sin(θ_k)]` según cada bit de color

Como `θ ∈ [0, π/2]`, todos los coeficientes son reales y no negativos.

## Flujo de Trabajo Recomendado

Sigue estos pasos para validar y usar el modelo:

### 1) Validación del Modelo 
Compara el estado teórico (calculado con la fórmula del modelo FRQI) vs el estado del circuito:
- Usa `theoretical_state(...)` para el cálculo analítico
- Usa `encode(...)` para ejecutar el circuito
- Dispositivo recomendado: `lightning.qubit` para mejor precisión y velocidad

### 2) Validación de Medición
Compara el statevector del circuito vs las probabilidades obtenidas por muestreo:
- Compara `|α|` (amplitudes exactas) vs `√p_j` (estimado con conteos)
- Usa `shots > 0` para simular mediciones reales
- Dispositivo recomendado: `default.qubit` para muestreo

### 3) Reconstrucción y Análisis
Recupera la imagen y analiza las métricas de información:
- `recover(state, shots=None|>0)` — reconstruye la imagen
- `analyze_state(state)` — calcula entropías y correlaciones
- `mi_samples(...)` — calcula información mutua

## Uso Básico

### Importaciones y configuración inicial:
```python
from src import FRQI, FRQI2, FRQI3, generate_image, select_frqi

# Selecciona el modelo según el número de canales
Model = select_frqi(n_color_qubits)
model = Model(image_size, device="lightning.qubit")

# Genera una imagen de prueba
img = generate_image(image_size=image_size, use_pattern=True)

# Crea las imágenes para cada canal (puedes usar la misma o diferentes)
images = [img] * n_color_qubits  # [MSB→LSB]
```
## APIs Principales

### Módulo `src/frqi_module.py`

Funciones principales para codificación y análisis:

- `encode(*images)` — Ejecuta el circuito y devuelve el statevector
- `theoretical_state(*images)` — Calcula el estado FRQI analíticamente
- `recover(state, shots=None|>0)` — Reconstruye la imagen (exacta o con muestreo)
- `analyze_state(state)` — Calcula entropías y correlaciones cuánticas
- `mi_samples(*images, shots=None|>0, n_blocks=None)` — Información mutua (exacta/muestreada)
- `stem_plot_amplitudes(images, shots=None|>0, device=..., threshold=...)` — Genera gráficos de amplitudes

### Módulo `src/utils.py`

Utilidades auxiliares:

- `generate_image(image_size, use_pattern=True, seed=None, pattern=None)` — Genera imágenes de prueba
- `select_frqi(n_color_qubits)` — Selecciona automáticamente FRQI, FRQI2 o FRQI3

## Validación: Teórico vs Circuito vs Muestreo


## Canales de Color Correlacionados

Una de las características principales de X-FRQI es el uso de correlaciones entre canales:

Las correlaciones entre qubits de color capturan interacciones entre diferentes canales de características. Estas correlaciones se pueden usar como señales adicionales para:
- Optimización de circuitos cuánticos
- Mejora en tareas de clasificación
- Extracción de características avanzadas

### Herramientas disponibles:

- `analyze_state(state)` y `mi_samples(...)` — Calcula métricas de información cuántica
- `color_Z_stats(...)` — Calcula valores esperados ⟨Z⟩ y ⟨ZZ⟩ en qubits de color

## Recuperación de Imagen

El modelo permite recuperar la imagen codificada de dos formas:

- **Exacta**: Desde el statevector |ψ⟩, usando probabilidades conjuntas posición-color
- **Por muestreo**: Desde los conteos de medición, con mayor ruido estadístico; la fidelidad aumenta con más shots
- **Orden de canales**: Al reconstruir, se respeta el orden MSB→LSB

## Licencia

Este proyecto está disponible bajo licencia MIT (sugerida) o Apache-2.0 — elige según tus necesidades.
