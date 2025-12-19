# Guía de Features Cuánticos para Clasificación de Imágenes

## Descripción del Proyecto

Este proyecto implementa un pipeline para extraer **features cuánticos** de imágenes usando el modelo **xFRQI** (Extended Flexible Representation of Quantum Images) y utilizarlos para clasificación de dígitos MNIST.

---

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                        FASE 1: EXTRACCIÓN                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Imagen MNIST (28×28)                                           │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────┐                                            │
│  │ Ventanas 4×4    │ ──► 49 ventanas por imagen                 │
│  └─────────────────┘                                            │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────┐     ┌─────────────────┐                    │
│  │ Canal 0: Imagen │  +  │ Canal 1: Kernel │                    │
│  └─────────────────┘     └─────────────────┘                    │
│         │                       │                               │
│         └───────────┬───────────┘                               │
│                     ▼                                           │
│            ┌─────────────────┐                                  │
│            │  FRQI2.encode() │ ──► Estado cuántico |ψ⟩          │
│            └─────────────────┘                                  │
│                     │                                           │
│         ┌───────────┼───────────┬───────────┐                   │
│         ▼           ▼           ▼           ▼                   │
│    ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐              │
│    │Teóricos │ │Hardware │ │Hardware │ │  Probs  │              │
│    │  (13)   │ │Analít(3)│ │Sampl(3) │ │   (4)   │              │
│    └─────────┘ └─────────┘ └─────────┘ └─────────┘              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FASE 2: CLASIFICACIÓN                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Features por imagen: (8 kernels × 49 ventanas × 20 features)   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────┐                                            │
│  │   Agregación    │ ──► mean, std, max, min por kernel         │
│  └─────────────────┘                                            │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────┐                                            │
│  │  Clasificador   │ ──► SVM, MLP, Random Forest, XGBoost       │
│  └─────────────────┘                                            │
│         │                                                       │
│         ▼                                                       │
│    Predicción (0-9)                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tipos de Features Extraídos

### 1. Features Teóricos (13 features)

Requieren acceso al **statevector completo** — solo disponibles en simulación.

| Feature | Descripción | Interpretación |
|---------|-------------|----------------|
| `H_total` | Entropía total del estado | ≈0 para estado puro |
| `H_position` | Entropía del subsistema posición | Mezcla espacial |
| `H_color0` | Entropía qubit imagen | Variabilidad de intensidades |
| `H_color1` | Entropía qubit kernel | Variabilidad del kernel |
| `I(color0:position)` | Información mutua imagen-posición | Estructura espacial |
| `I(color1:position)` | Información mutua kernel-posición | Respuesta del filtro |
| `H_color0\|position` | Entropía condicional | Puede ser negativa (cuántico) |
| `H_color1\|position` | Entropía condicional | Puede ser negativa |
| `H_colors_joint` | Entropía conjunta de canales | |
| `I(color0:color1)` | Info mutua entre canales | **Correlación imagen-kernel** |
| `H_color0\|color1` | Condicional | |
| `H_color1\|color0` | Condicional | |
| `I3` | Información tripartita | Correlación de 3 cuerpos |

### 2. Features Hardware (3 features)

Accesibles en **hardware cuántico real** mediante mediciones.

| Feature | Descripción | Rango |
|---------|-------------|-------|
| `⟨Z_0⟩` | Expectativa Pauli Z en canal imagen | [-1, +1] |
| `⟨Z_1⟩` | Expectativa Pauli Z en canal kernel | [-1, +1] |
| `⟨Z_0 Z_1⟩` | Correlación entre canales | [-1, +1] |

**Interpretación física:**
- `⟨Z⟩ = +1`: Todos los píxeles en intensidad baja (θ ≈ 0)
- `⟨Z⟩ = -1`: Todos los píxeles en intensidad alta (θ ≈ π/2)
- `⟨ZZ⟩`: Correlación cuántica entre imagen y kernel

### 3. Probabilidades de Medición (4 features)

Distribución de resultados al medir ambos qubits de color.

| Feature | Descripción |
|---------|-------------|
| `p_00` | Prob(imagen=bajo, kernel=bajo) |
| `p_01` | Prob(imagen=bajo, kernel=alto) |
| `p_10` | Prob(imagen=alto, kernel=bajo) |
| `p_11` | Prob(imagen=alto, kernel=alto) |

---

## Kernels Utilizados

| Kernel | Propósito | Qué detecta |
|--------|-----------|-------------|
| `identity` | Baseline | Sin filtrado |
| `sobel_x` | Gradiente horizontal | Bordes verticales |
| `sobel_y` | Gradiente vertical | Bordes horizontales |
| `laplacian` | Segunda derivada | Bordes isotrópicos |
| `gaussian` | Suavizado | Promedio local |
| `gabor_0` | Textura 0° | Patrones horizontales |
| `gabor_45` | Textura 45° | Patrones diagonales |
| `gabor_90` | Textura 90° | Patrones verticales |

---

## Archivos del Proyecto

```
project/
├── quantum_features_utils.py     # Utilidades para extracción
├── quantum_classification_mnist.ipynb  # Notebook principal
├── QUANTUM_FEATURES_GUIDE.md     # Este documento
└── results/
    └── features/
        ├── quantum_features_n{N}.npz    # Features guardados
        └── classification_results.json  # Resultados
```

---

## Uso Básico

### Extracción de Features

```python
from quantum_features_utils import (
    FeatureExtractionConfig,
    extract_features_dataset,
    save_features
)
from src import FRQI2

# Configurar
config = FeatureExtractionConfig(
    window_size=4,
    n_windows_x=7,
    n_windows_y=7,
    shots_hardware=1024
)

# Extraer
features = extract_features_dataset(
    images=images,  # (N, 28, 28) uint8
    labels=labels,  # (N,)
    config=config,
    model_class=FRQI2
)

# Guardar
save_features(features, "quantum_features.npz")
```

### Clasificación

```python
from quantum_features_utils import (
    load_features,
    prepare_classification_data
)
from sklearn.svm import SVC

# Cargar
features = load_features("quantum_features.npz")

# Preparar
X, y, names = prepare_classification_data(
    features,
    feature_type='all',      # 'theoretical', 'hardware_*', 'probs', 'all'
    aggregation='stats'      # 'stats' o 'flatten'
)

# Clasificar
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
```

---

## Sugerencias para Continuar

### 1. Escalar el Dataset

```python
# Incrementar gradualmente
N_SAMPLES = 100   # ~1 hora
N_SAMPLES = 1000  # ~10 horas
N_SAMPLES = 5000  # ~2 días (usar GPU/cluster)
```

**Recomendación:** Guardar features incrementalmente para no perder trabajo.

### 2. Clasificación Binaria

Empezar con pares de dígitos difíciles:

```python
# Pares desafiantes
pairs = [
    (3, 8),  # Formas similares
    (4, 9),  # Confusión común
    (1, 7),  # Trazo vertical
    (5, 6),  # Curvas similares
]

# Filtrar dataset
mask = np.isin(labels, [3, 8])
X_binary = X[mask]
y_binary = (labels[mask] == 8).astype(int)
```

### 3. Análisis de Features Específicos

Investigar qué features son más discriminativos:

```python
# ¿I(color0:color1) discrimina mejor que ⟨ZZ⟩?
# ¿Qué kernels son más informativos?
# ¿Los features teóricos aportan más que los hardware?

from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(X, y)
ranking = sorted(zip(feature_names, mi_scores), 
                 key=lambda x: x[1], reverse=True)
```

### 4. Comparación con Baseline Clásico

Implementar features clásicos equivalentes:

```python
from scipy.signal import convolve2d

def classical_features(image, kernel):
    """Respuesta clásica del filtro."""
    response = convolve2d(image, kernel, mode='same')
    return {
        'mean': response.mean(),
        'std': response.std(),
        'max': response.max(),
        'energy': (response**2).sum(),
    }
```

### 5. Diferentes Agregaciones

```python
# Por regiones espaciales
def aggregate_by_region(features, n_regions=3):
    """Agregar features por región (centro, bordes, esquinas)."""
    # Dividir las 7×7 ventanas en regiones
    pass

# Histogramas
def aggregate_histogram(features, n_bins=10):
    """Histograma de cada feature."""
    pass
```

### 6. Modelos Más Sofisticados

```python
# XGBoost para mejor ranking de features
import xgboost as xgb
model = xgb.XGBClassifier(n_estimators=100, max_depth=5)

# CNN sobre los mapas de features
# Cada imagen tiene (K, Wx, Wy, F) → tratar como imagen multicanal
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    # ...
])
```

### 7. Análisis de Sensibilidad

```python
# ¿Cómo varían los features con ruido?
def add_noise(image, sigma=10):
    noise = np.random.normal(0, sigma, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)

# ¿Cómo varían con transformaciones?
# - Rotación
# - Traslación
# - Escala
```

### 8. Visualizaciones Avanzadas

```python
# Mapas promedio por clase
for digit in range(10):
    mask = labels == digit
    avg_map = features['features_theoretical'][mask].mean(axis=0)
    # Visualizar

# Diferencias entre clases
diff_3_8 = avg_maps[3] - avg_maps[8]
# ¿Dónde difieren más?
```

---

## Consideraciones sobre Hardware Cuántico

### ¿Qué es realista en hardware real?

| Aspecto | Simulación | Hardware Real |
|---------|------------|---------------|
| Statevector | ✅ Accesible | ❌ No accesible |
| Features teóricos | ✅ Calculables | ❌ Requiere tomografía |
| ⟨Z⟩, ⟨ZZ⟩ | ✅ Exactos | ✅ Por muestreo |
| Probabilidades | ✅ Exactas | ✅ Por muestreo |
| Shots necesarios | N/A | ~1000-10000 |
| Ruido | ❌ Sin ruido | ⚠️ Significativo |

### Recomendación para Hardware

Si el objetivo es demostrar viabilidad en hardware:

1. **Enfocarse en features ⟨Z⟩ y ⟨ZZ⟩**
2. **Estudiar el efecto del número de shots**
3. **Agregar modelos de ruido en simulación**
4. **Comparar analítico vs muestreado**

```python
# Comparar efecto de shots
for shots in [100, 500, 1000, 5000, 10000]:
    features_sampled = extract_with_shots(images, shots)
    acc = classify(features_sampled)
    print(f"shots={shots}: accuracy={acc:.4f}")
```

---

## Preguntas de Investigación

1. **¿Los features cuánticos capturan información diferente a los clásicos?**
   - Comparar accuracy, pero también: ¿hay imágenes que un método clasifica bien y el otro no?

2. **¿Qué rol juega la correlación ⟨ZZ⟩?**
   - ¿Es solo el producto ⟨Z₀⟩⟨Z₁⟩ o hay información adicional?

3. **¿Qué kernels son más informativos para cada dígito?**
   - El dígito "1" probablemente responde más a Sobel vertical

4. **¿La información mutua I(color0:color1) es útil para clasificación?**
   - Esta métrica captura cómo la imagen "responde" al kernel

5. **¿Hay ventaja al usar entropías condicionales negativas?**
   - H(color|position) < 0 indica correlaciones cuánticas fuertes

---

## Referencias

1. Mercado Sánchez, Sun, Dong (2019). "Correlation Property of Multipartite Quantum Image"
2. Le, Dong, Hirota (2011). "A flexible representation of quantum images"
3. Yan, Iliyasu, Venegas-Andraca (2016). "A survey of quantum image representations"

---

## Contacto

Dr. Mario Alberto Mercado Sánchez  
ometitlan@gmail.com

Proyecto xFRQI - Facultad de Ingeniería, UNAM
