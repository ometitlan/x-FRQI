<p align="center">
  <img src="assets/logo_ing.jpg" alt="Facultad de Ingeniería UNAM" width="180"/>
</p>

<h1 align="center">xFRQI — Extended Flexible Representation of Quantum Images</h1>
<p align="center">Codificación cuántica multicanal para representar imágenes correlacionadas y alimentar clasificadores híbridos cuántico-clásicos.</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"/></a>
  <a href="https://pennylane.ai/"><img src="https://img.shields.io/badge/PennyLane-quantum%20ML-22c55e" alt="PennyLane"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"/></a>
</p>

---

## Información del Proyecto

Proyecto desarrollado con el apoyo del Consejo Nacional de Ciencia y Tecnología (CONAHCYT) mediante la beca de Estancias Posdoctorales por México 2022 (modalidad Académica - Inicial), CVU 469604.

| | |
|---|---|
| **Institución** | Facultad de Ingeniería, UNAM |
| **Director de Proyecto** | Dr. Boris Escalante Ramírez |
| **Período** | Diciembre 2022 - Noviembre 2024 |
| **Autor** | Dr. Mario Alberto Mercado Sánchez — ometitlan@gmail.com |

---

## Tabla de Contenidos

1. [Visión General](#visión-general)
2. [Fundamentos Matemáticos](#fundamentos-matemáticos)
   - [Modelo FRQI Original](#modelo-frqi-original)
   - [Extensión xFRQI Multicanal](#extensión-xfrqi-multicanal)
   - [Codificación de Ángulos](#codificación-de-ángulos)
3. [Medidas de Información Cuántica](#medidas-de-información-cuántica)
   - [Entropía de von Neumann](#entropía-de-von-neumann)
   - [Información Mutua Cuántica](#información-mutua-cuántica)
   - [Entropía Condicional](#entropía-condicional)
   - [Discord Cuántico](#discord-cuántico)
   - [Correlación Tripartita](#correlación-tripartita)
4. [Estructura del Repositorio](#estructura-del-repositorio)
5. [Instalación y Configuración](#instalación-y-configuración)
6. [APIs Principales](#apis-principales)
7. [Cuadernos de Trabajo](#cuadernos-de-trabajo)
8. [Flujo de Trabajo Recomendado](#flujo-de-trabajo-recomendado)
9. [Aplicaciones y Casos de Uso](#aplicaciones-y-casos-de-uso)
10. [Referencias](#referencias)
11. [Licencia](#licencia)

---

## Visión General

Las propiedades relevantes de un sistema cuántico multipartito dependen directamente de cómo decidimos codificar la información clásica en el estado cuántico. En el caso de imágenes cuánticas, esta codificación puede aprovechar la estructura ondulatoria de la función de estado, utilizando tanto los estados base para representar índices discretos (como las posiciones de píxeles) como las amplitudes y fases relativas para representar magnitudes continuas (como intensidades o atributos derivados).

El proyecto **xFRQI** (Extended Flexible Representation of Quantum Images) extiende el modelo FRQI original para incorporar múltiples qubits de "color", de forma que cada qubit adicional puede codificar un canal o atributo independiente. Esto permite:

- **Representación multicanal**: Componentes RGB, filtros convolucionales, kernels, mapas de características o máscaras.
- **Correlaciones cuánticas**: Las correlaciones entre los qubits de color y de posición se vuelven parte del espacio de información disponible para optimización, análisis y clasificación.
- **Medidas informacionales**: Extracción directa de entropía, información mutua y correlaciones físicas del estado cuántico resultante.

---

## Fundamentos Matemáticos

### Modelo FRQI Original

El modelo FRQI (Flexible Representation of Quantum Images) fue propuesto para representar imágenes mediante un estado cuántico normalizado que es el producto directo del color y la posición para cada píxel:

$$|\psi\rangle = \frac{1}{\sqrt{2^n}} \sum_{i=0}^{2^{2n}-1} |c_i\rangle \otimes |i\rangle = \frac{1}{\sqrt{2^n}} \sum_{i=0}^{2^{2n}-1} (\cos\theta_i|0\rangle + \sin\theta_i|1\rangle) \otimes |i\rangle$$

donde:
- $|c_i\rangle = \cos\theta_i|0\rangle + \sin\theta_i|1\rangle$ codifica el **color** (intensidad) del píxel $i$
- $|i\rangle = |0\rangle, |1\rangle, |2\rangle, \ldots, |2^{2n}-1\rangle$ representa la **posición** en una secuencia de estados base
- $\theta_i \in [0, \pi/2]$ es el ángulo de rotación que codifica la intensidad

La función de onda está normalizada:

$$||\psi|| = \frac{1}{\sqrt{2^n}} \sqrt{\sum_{i=0}^{2^{2n}-1} (\cos^2\theta_i + \sin^2\theta_i)} = 1$$

#### Ejemplo: Imagen de 2×2 píxeles

Para una imagen de 4 píxeles con configuración de color $\{51, 204, 204, 51\}$:

**Vector de ángulos:**
$$\vec{\theta} = \begin{pmatrix} 0.314 \\ 1.256 \\ 1.256 \\ 0.314 \end{pmatrix}$$

**Vector de estados de color:**
$$q_{color} = \begin{pmatrix} 0.951|0\rangle + 0.309|1\rangle \\ 0.309|0\rangle + 0.951|1\rangle \\ 0.309|0\rangle + 0.951|1\rangle \\ 0.951|0\rangle + 0.309|1\rangle \end{pmatrix}$$

**Función de onda completa:**
$$|\psi\rangle = 0.475|000\rangle + 0.154|001\rangle + 0.154|010\rangle + 0.475|011\rangle + 0.154|100\rangle + 0.475|101\rangle + 0.475|110\rangle + 0.154|111\rangle$$

### Extensión xFRQI Multicanal

La extensión xFRQI generaliza el esquema FRQI incorporando múltiples qubits de color. Con $C$ qubits de color, el estado se construye como:

$$|\psi\rangle = \frac{1}{\sqrt{P}} \sum_{i=0}^{P-1} \bigotimes_{k=0}^{C-1} (\cos\theta_k(i)|0\rangle + \sin\theta_k(i)|1\rangle) \otimes |i\rangle$$

donde:
- $P = 2^{n_{pos}}$ es el número de píxeles (posiciones)
- $C$ es el número de qubits de color (canales)
- $\theta_k(i)$ es el ángulo para el canal $k$ en la posición $i$

#### Convención de Wires (Big-Endian)

| Subsistema | Wires | Descripción |
|------------|-------|-------------|
| Posición | $0, \ldots, n_{pos}-1$ | MSB → LSB |
| Color | $n_{pos}, \ldots, n_{pos}+n_{color}-1$ | Canal 0 es MSB del bloque de color |

**Índice base compuesto:**
$$\text{idx} = (\text{pos\_idx} \ll n_{color}) \,|\, \text{color\_bits}$$

### Codificación de Ángulos

La transformación de intensidades de píxel (rango $[0, 255]$) a ángulos de rotación:

$$\theta_k(i) = \frac{\pi}{2} \times \frac{I_k(i)}{255}$$

donde $I_k(i)$ es la intensidad del píxel $i$ en el canal $k$.

**Amplitudes del estado:**
$$\alpha(\text{pos}, \text{color\_bits}) = \frac{1}{\sqrt{P}} \prod_{k} \begin{cases} \cos(\theta_k) & \text{si bit } k = 0 \\ \sin(\theta_k) & \text{si bit } k = 1 \end{cases}$$

Las amplitudes son **reales y no negativas** por construcción.

---

## Medidas de Información Cuántica

Las correlaciones en sistemas cuánticos multipartitos no se limitan a dependencias estadísticas: incluyen efectos puramente cuánticos como la coherencia global y la no separabilidad entre subsistemas. Para capturar estas relaciones utilizamos medidas basadas en la entropía de von Neumann.

### Entropía de von Neumann

La entropía de von Neumann de un estado $\rho$ cuantifica el grado de mezcla del sistema:

$$S(\rho) = -\text{Tr}(\rho \log_2 \rho) = -\sum_i \lambda_i \log_2 \lambda_i$$

donde $\lambda_i$ son los eigenvalores de $\rho$.

**Propiedades clave:**
- Para un estado puro: $S(\rho) = 0$
- Para un sistema bipartito puro $\rho_{AB}$: $S(\rho_A) = S(\rho_B)$
- Máximo: $S_{max} = \log_2 d$ para un sistema de dimensión $d$

#### Cálculo de Matrices de Densidad Reducidas

Para obtener la entropía de subsistemas, se realiza la traza parcial:

$$\rho_A = \text{Tr}_B(\rho_{AB})$$

**Ejemplo:** Para el sistema de 2×2 píxeles con 1 qubit de color:

$$\rho_{1p2p} = \text{Tr}_c(\rho_{color,position}) = \frac{1}{4} \begin{pmatrix} 0.250 & 0.146 & 0.146 & 0.250 \\ 0.146 & 0.250 & 0.250 & 0.146 \\ 0.146 & 0.250 & 0.250 & 0.146 \\ 0.250 & 0.146 & 0.146 & 0.250 \end{pmatrix}$$

### Información Mutua Cuántica

La información mutua cuántica mide las correlaciones totales (clásicas + cuánticas) entre subsistemas:

$$I(A;B) = S(A) + S(B) - S(A,B) = S(\rho_{AB} \| \rho_A \otimes \rho_B)$$

**Propiedades:**
- $I(A;B) \geq 0$ (no negativa)
- $I(A;B) = 0$ solo si $\rho_{AB} = \rho_A \otimes \rho_B$ (estados producto)
- Para un estado bipartito puro: $I(A;B) = 2S(A) = 2S(B)$

Esta última propiedad implica que **la información mutua cuántica puede ser el doble de la entropía de un subsistema individual**, lo cual es una diferencia fundamental respecto al caso clásico.

### Entropía Condicional

#### Entropía Condicional Cuántica

La generalización cuántica de la entropía condicional:

$$S(A|B) = S(A,B) - S(B)$$

**Nota importante:** A diferencia del caso clásico donde $H(A|B) \geq 0$, la entropía condicional cuántica **puede ser negativa** para estados entrelazados. La negatividad indica correlaciones cuánticas fuertes.

#### Entropía Condicional Inducida por Medición

Cuando se realiza una medición proyectiva de von Neumann en el subsistema $A$ con proyectores $\{\Pi_i^A\}$:

$$S_{\{\Pi_i^A\}}(\rho_{B|A}) = \sum_i p_i^A S(\rho_{B|i})$$

donde:
- $p_i^A = \text{Tr}[(\Pi_i^A \otimes I)\rho_{AB}]$ es la probabilidad del resultado $i$
- $\rho_{B|i} = \text{Tr}_A[(\Pi_i^A \otimes I)\rho_{AB}(\Pi_i^A \otimes I)] / p_i^A$ es el estado post-medición

### Discord Cuántico

El discord cuántico captura las correlaciones puramente cuánticas (no clásicas):

$$D_A(A,B) = I(A,B) - J_A(A,B)$$

donde la **información mutua clásica** es:

$$J_A(A,B) = S(B) - S_{\{\Pi_i^A\}}(\rho_{B|A})$$

**Interpretación:**
- $D = 0$: Solo correlaciones clásicas
- $D > 0$: Presencia de correlaciones cuánticas no clásicas
- Para imágenes binarias (blanco/negro): $D = 0$
- Para imágenes en escala de grises: $D$ puede variar significativamente

### Correlación Tripartita

Para sistemas con tres partes (posición + 2 canales de color), se definen:

**Información de interacción:**
$$I_0(A;B;C) = I(A;B) - I(A;B|C)$$

donde $I(A;B|C) = S(A|C) + S(B|C) - S(A,B|C)$

**Correlación total:**
$$I_T(A;B;C) = I(A;B) + I(AB;C) = S(A) + S(B) + S(C) - S(A,B,C)$$

**Correlación dual total:**
$$I_D(A;B;C) = I(A;BC) + I(B;C|A) = S(A,B) + S(A,C) + S(B,C) - 2S(A,B,C)$$

**Propiedades para estados puros:**
- $I_0 = 0$ (la información de interacción es cero)
- $I_T = I_D$ (correlación total igual a correlación dual)
- $I_T$ puede alcanzar el **doble** de la entropía conjunta clásica

---

## Estructura del Repositorio

```
xFRQI/
├── src/
│   ├── __init__.py
│   ├── frqi_module.py      # Implementación FRQI/FRQI2/FRQI3
│   ├── measurements.py     # Rutinas de muestreo y MI clásica
│   ├── info_measures.py    # Entropías e información mutua cuántica
│   └── utils.py            # Utilidades (generate_image, kernels, etc.)
├── notebooks/
│   ├── Information_encode.ipynb   # Validación y métricas
│   └── feature_maps.ipynb         # Mapas de información locales
├── examples/
├── assets/
├── requirements.txt
├── LICENSE
└── README.md
```

### Descripción de Módulos

| Módulo | Descripción |
|--------|-------------|
| `frqi_module.py` | Clases `FRQI`, `FRQI2`, `FRQI3` con métodos `encode`, `theoretical_state`, `recover`, `analyze_state` |
| `measurements.py` | Rutinas de muestreo en base Z e información mutua clásica (`color_Z_stats`, `mi_samples`) |
| `info_measures.py` | Cálculo de entropías e información mutua directamente en statevectors |
| `utils.py` | `generate_image`, `ensure_uint8`, `select_frqi`, kernels Sobel/Laplaciano/Gabor, `build_kernel_patch`, `apply_kernel_bank` |

---

## Instalación y Configuración

### 1. Crear Entorno Virtual

**Conda:**
```bash
conda create -n x_FRQI python=3.11 -y
conda activate x_FRQI
```

**Venv (Linux/macOS):**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Venv (Windows PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### Dependencias Principales

- `pennylane` / `pennylane-lightning` — Simulación cuántica
- `numpy`, `scipy` — Cálculo numérico
- `matplotlib` — Visualización
- `tensorflow` / `keras` (opcional) — Datasets MNIST

---

## APIs Principales

### Selección de Modelo

```python
from src import FRQI, FRQI2, FRQI3, generate_image, select_frqi

# Selección automática según número de canales
Model = select_frqi(n_color_qubits=2)
model = Model(image_size=16, device="lightning.qubit")
```

### Codificación y Validación

```python
# Generar imágenes de prueba
images = [generate_image(16, use_pattern=True, seed=42) for _ in range(2)]

# Estado teórico (fórmula analítica)
state_theoretical = model.theoretical_state(*images)

# Estado del circuito (statevector)
state_circuit = model.encode(*images)

# Reconstrucción
rec_exact = model.recover(state_circuit, shots=None)     # Exacta
rec_sampled = model.recover(state_circuit, shots=8192)  # Por muestreo
```

### Análisis de Información

```python
# Entropías y correlaciones
stats = model.analyze_state(state_circuit)

# Métricas disponibles:
# - H_total: Entropía total del estado
# - H_position: Entropía del subsistema posicional
# - H_color0, H_color1, ...: Entropía de cada qubit de color
# - I(color0:position), I(color1:position): Información mutua color-posición
# - H_color0|position: Entropía condicional
# - I(color0:color1): Información mutua entre canales
# - I3(position:color0:color1): Correlación tripartita

# Información mutua clásica (por muestreo)
mi_exact = model.mi_samples(*images, shots=None)
mi_sampled = model.mi_samples(*images, shots=8192)

# Expectativas Z y ZZ
zstats = model.color_Z_stats(*images, shots=1024)
# Incluye: <Z_0>, <Z_1>, <ZZ_01>
```

### Mapas de Información Locales

```python
from src import FRQI2
from src.utils import ensure_uint8, build_kernel_patch

# Ventana local
win = 8
img_u8 = ensure_uint8(img * 255)  # img 28x28 en [0,1]
kpatch = build_kernel_patch("sobel_x", win)

model = FRQI2(image_size=win, device="default.qubit")

# Análisis de una ventana
info = model.analyze_state(model.encode(img_u8[:win,:win], kpatch))
zstats = model.color_Z_stats(img_u8[:win,:win], kpatch, shots=1024)
```

---

## Cuadernos de Trabajo

### `Information_encode.ipynb`

**Propósito:** Validar la implementación del modelo xFRQI y calcular métricas de información.

**Contenido:**
1. Comparación: Estado teórico vs Circuito vs Muestreo
2. Reconstrucción de imágenes (exacta y por muestreo)
3. Cálculo de entropías e información mutua
4. Análisis de correlaciones cuánticas

**Validaciones:**
- Teórico ≈ Circuito → Valida implementación del `encode()`
- Circuito ≈ Muestreo → Valida consistencia estadística (error ∝ 1/√shots)

### `feature_maps.ipynb`

**Propósito:** Generar mapas de información locales para análisis de imágenes.

**Contenido:**
1. Carga de dataset MNIST
2. Procesamiento por ventanas deslizantes
3. Generación de mapas:
   - `I(color0:position)`, `I(color1:position)`
   - `I(color0:color1)`, `I3(position:color0:color1)`
   - `H_color0|position`, `H_color1|position`
   - `<Z_0>`, `<Z_1>`, `<ZZ_01>`

---

## Flujo de Trabajo Recomendado

### 1. Validación del Modelo

```python
# Comparar estado teórico vs circuito
state_th = model.theoretical_state(*images)
state_ct = model.encode(*images)
error = np.max(np.abs(state_th - state_ct))
assert error < 1e-10, f"Error de implementación: {error}"
```

### 2. Validación de Medición

```python
# Comparar amplitudes vs √p (muestreo)
# Error esperado: O(1/√shots)
rec_exact = model.recover(state, shots=None)
rec_samp = model.recover(state, shots=8192)
mae = np.mean(np.abs(rec_exact - rec_samp))
print(f"MAE exacta vs muestreo: {mae:.3f}")
```

### 3. Análisis de Correlaciones

```python
stats = model.analyze_state(state)

# Verificar estado puro
assert stats["H_total"] < 1e-6, "Estado no puro"

# Analizar correlaciones
print(f"I(color:position) = {stats['I(color0:position)']:.4f} bits")
print(f"I(color0:color1) = {stats['I(color0:color1)']:.4f} bits")
```

### Dispositivos Recomendados

| Dispositivo | Uso | Características |
|-------------|-----|-----------------|
| `lightning.qubit` | Estado exacto | Rápido, alta precisión |
| `default.qubit` | Muestreo | Simula hardware real |

---

## Aplicaciones y Casos de Uso

### 1. Comparación de Imágenes

Las medidas de información mutua permiten comparar imágenes de forma cuántica:
- **Máxima MI**: Imágenes idénticas o complementarias
- **Mínima MI**: Imágenes sin correlación

### 2. Registro de Imágenes

Basado en el método de información mutua para alineación:
- La MI alcanza su máximo cuando las imágenes están correctamente registradas
- Aplicaciones en imagenología médica (PET/MRI, CT/MRI)

### 3. Mapas de Características

Codificación de imagen + kernel en canales separados:
- Análisis de correlaciones locales
- Detección de bordes y texturas
- Descriptores cuánticos para clasificación

### 4. Sensibilidad a Transformaciones

El modelo cuántico es sensible a:
- **Cambios de color**: La entropía cuántica varía (a diferencia de la clásica que permanece invariante)
- **Traslaciones**: Los extremos de correlación indican el registro óptimo
- **Distribución de intensidades**: El discord captura información no clásica

---

## Referencias

1. **Mercado Sánchez M. A., Sun G. H., Dong S. H.**  
   *Correlation Property of Multipartite Quantum Image.*  
   International Journal of Theoretical Physics, vol. 58, pp. 3773–3796 (2019).  
   https://doi.org/10.1007/s10773-019-04247-9

2. **Le P. Q., Dong F., Hirota K.**  
   *A flexible representation of quantum images for polynomial preparation, image compression, and processing operations.*  
   Quantum Information Processing, vol. 10, pp. 63–84 (2011).

3. **Yan F., Iliyasu A. M., Venegas-Andraca S. E.**  
   *A survey of quantum image representations.*  
   Quantum Information Processing, vol. 15, pp. 1–35 (2016).  
   https://doi.org/10.1007/s11128-015-1195-6

4. **Modi K., Brodutch A., Cable H., Paterek T., Vedral V.**  
   *The classical-quantum boundary for correlations: discord and related measures.*  
   Reviews of Modern Physics, vol. 84, pp. 1655–1707 (2012).

5. **Phoenix S. J. D.**  
   *Quantum information as a measure of multipartite correlation.*  
   Quantum Information Processing, vol. 14, pp. 3723–3738 (2015).

6. **Documentación de PennyLane**  
   Sección sobre mediciones y operadores de observación.  
   https://docs.pennylane.ai/en/stable/introduction/measurements.html

---

## Licencia

Este proyecto se distribuye bajo licencia [MIT](LICENSE).

Si empleas xFRQI en investigaciones o demostraciones, por favor menciona a los autores y referencia este repositorio junto con la publicación:

```bibtex
@article{mercado2019correlation,
  title={Correlation Property of Multipartite Quantum Image},
  author={Mercado Sánchez, M. A. and Sun, Guo-Hua and Dong, Shi-Hai},
  journal={International Journal of Theoretical Physics},
  volume={58},
  pages={3773--3796},
  year={2019},
  publisher={Springer}
}
```

---

<p align="center">
  <em>xFRQI: Explorando las correlaciones cuánticas en representaciones de imágenes</em>
</p>