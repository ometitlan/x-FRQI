"""
quantum_features_utils.py
Utilidades para extracción de features cuánticos de imágenes usando xFRQI.

Autor: Dr. Mario Alberto Mercado Sánchez
Proyecto: xFRQI - Quantum Feature Maps for Image Classification
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm
import warnings

# ============================================================
# Configuración de Features
# ============================================================

FEATURE_NAMES_THEORETICAL = [
    'H_total',
    'H_position', 
    'H_color0',
    'H_color1',
    'I_color0_position',
    'I_color1_position',
    'H_color0_given_position',
    'H_color1_given_position',
    'H_colors_joint',
    'I_color0_color1',
    'H_color0_given_color1',
    'H_color1_given_color0',
    'I3_position_color0_color1',
]

FEATURE_NAMES_HARDWARE = [
    'Z_0',
    'Z_1', 
    'ZZ_01',
]

FEATURE_NAMES_PROBS = [
    'p_00',
    'p_01',
    'p_10',
    'p_11',
]

# Mapeo de claves de analyze_state a índices
THEORETICAL_KEY_MAP = {
    'H_total': 0,
    'H_position': 1,
    'H_color0': 2,
    'H_color1': 3,
    'I(color0:position)': 4,
    'I(color1:position)': 5,
    'H_color0|position': 6,
    'H_color1|position': 7,
    'H_colors_joint': 8,
    'I(color0:color1)': 9,
    'H_color0|color1': 10,
    'H_color1|color0': 11,
    'I3(position:color0:color1)': 12,
}

# ============================================================
# Generación de Kernels
# ============================================================

def get_kernel(name: str, size: int = 4) -> np.ndarray:
    """
    Genera un kernel normalizado al rango [0, 255] para codificación FRQI.
    
    Args:
        name: Nombre del kernel ('identity', 'sobel_x', 'sobel_y', 'laplacian', 
              'gaussian', 'gabor_0', 'gabor_45', 'gabor_90')
        size: Tamaño del kernel (size x size)
    
    Returns:
        Kernel normalizado como array uint8 de forma (size, size)
    """
    if name == 'identity':
        # Kernel identidad: valor medio uniforme
        kernel = np.ones((size, size), dtype=np.float32) * 0.5
        
    elif name == 'sobel_x':
        # Gradiente horizontal (detecta bordes verticales)
        base = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        kernel = _resize_kernel(base, size)
        
    elif name == 'sobel_y':
        # Gradiente vertical (detecta bordes horizontales)
        base = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        kernel = _resize_kernel(base, size)
        
    elif name == 'laplacian':
        # Laplaciano (bordes isotrópicos)
        base = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        kernel = _resize_kernel(base, size)
        
    elif name == 'gaussian':
        # Suavizado gaussiano
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        sigma = 0.5
        kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        
    elif name.startswith('gabor_'):
        # Filtros Gabor para diferentes orientaciones
        angle = int(name.split('_')[1])
        kernel = _gabor_kernel(size, theta=np.deg2rad(angle))
        
    else:
        raise ValueError(f"Kernel desconocido: {name}")
    
    # Normalizar al rango [0, 255]
    kernel = kernel.astype(np.float32)
    k_min, k_max = kernel.min(), kernel.max()
    if k_max - k_min > 1e-8:
        kernel = (kernel - k_min) / (k_max - k_min)
    else:
        kernel = np.ones_like(kernel) * 0.5
    
    return (kernel * 255).astype(np.uint8)


def _resize_kernel(base: np.ndarray, size: int) -> np.ndarray:
    """Redimensiona un kernel base al tamaño deseado usando interpolación."""
    from scipy.ndimage import zoom
    if base.shape[0] == size:
        return base
    factor = size / base.shape[0]
    return zoom(base, factor, order=1)


def _gabor_kernel(size: int, theta: float = 0, sigma: float = 1.0, 
                  lambd: float = 2.0, gamma: float = 0.5) -> np.ndarray:
    """Genera un kernel Gabor."""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Rotación
    X_theta = X * np.cos(theta) + Y * np.sin(theta)
    Y_theta = -X * np.sin(theta) + Y * np.cos(theta)
    
    # Gabor
    gb = np.exp(-0.5 * (X_theta**2 + gamma**2 * Y_theta**2) / sigma**2)
    gb *= np.cos(2 * np.pi * X_theta / lambd)
    
    return gb


def get_default_kernels() -> List[str]:
    """Retorna la lista de kernels por defecto."""
    return ['identity', 'sobel_x', 'sobel_y', 'laplacian', 
            'gaussian', 'gabor_0', 'gabor_45', 'gabor_90']


# ============================================================
# Extracción de Features
# ============================================================

@dataclass
class FeatureExtractionConfig:
    """Configuración para extracción de features."""
    window_size: int = 4
    n_windows_x: int = 7
    n_windows_y: int = 7
    kernels: List[str] = None
    shots_hardware: int = 1024
    device: str = "default.qubit"
    
    def __post_init__(self):
        if self.kernels is None:
            self.kernels = get_default_kernels()


def extract_window(image: np.ndarray, wx: int, wy: int, 
                   window_size: int) -> np.ndarray:
    """
    Extrae una ventana de la imagen.
    
    Args:
        image: Imagen de entrada (H, W)
        wx, wy: Índices de la ventana
        window_size: Tamaño de la ventana
    
    Returns:
        Parche de la imagen (window_size, window_size)
    """
    x_start = wx * window_size
    y_start = wy * window_size
    return image[y_start:y_start+window_size, x_start:x_start+window_size]


def extract_features_single_window(
    model,
    image_patch: np.ndarray,
    kernel_patch: np.ndarray,
    shots_hardware: int = 1024
) -> Dict[str, np.ndarray]:
    """
    Extrae features cuánticos de una ventana imagen-kernel.
    
    Args:
        model: Instancia de FRQI2
        image_patch: Parche de imagen (uint8)
        kernel_patch: Parche de kernel (uint8)
        shots_hardware: Número de shots para features hardware
    
    Returns:
        Diccionario con 'theoretical', 'hardware_analytic', 
        'hardware_sampled', 'probs'
    """
    # Codificar estado
    state = model.encode(image_patch, kernel_patch)
    
    # Features teóricos (del statevector)
    stats = model.analyze_state(state)
    theoretical = np.zeros(len(FEATURE_NAMES_THEORETICAL), dtype=np.float32)
    for key, idx in THEORETICAL_KEY_MAP.items():
        theoretical[idx] = stats.get(key, 0.0)
    
    # Features hardware - analíticos (shots=None)
    z_stats_analytic = model.color_Z_stats(image_patch, kernel_patch, shots=None)
    hardware_analytic = np.array([
        z_stats_analytic.get('<Z_0>', 0.0),
        z_stats_analytic.get('<Z_1>', 0.0),
        z_stats_analytic.get('<ZZ_01>', 0.0),
    ], dtype=np.float32)
    
    # Features hardware - muestreados
    z_stats_sampled = model.color_Z_stats(image_patch, kernel_patch, shots=shots_hardware)
    hardware_sampled = np.array([
        z_stats_sampled.get('<Z_0>', 0.0),
        z_stats_sampled.get('<Z_1>', 0.0),
        z_stats_sampled.get('<ZZ_01>', 0.0),
    ], dtype=np.float32)
    
    # Probabilidades de medición (del statevector)
    probs = _compute_color_probs(state, model.n_position_qubits, model.n_color_qubits)
    
    return {
        'theoretical': theoretical,
        'hardware_analytic': hardware_analytic,
        'hardware_sampled': hardware_sampled,
        'probs': probs,
    }


def _compute_color_probs(state: np.ndarray, n_pos: int, n_color: int) -> np.ndarray:
    """
    Calcula probabilidades marginales de los qubits de color.
    
    Returns:
        Array [p_00, p_01, p_10, p_11] para 2 qubits de color
    """
    n_total = n_pos + n_color
    probs = np.abs(state)**2
    
    # Marginalizar sobre posiciones
    n_positions = 2**n_pos
    n_colors = 2**n_color
    
    # Reshape: (n_positions, n_colors) considerando orden de bits
    probs_reshaped = probs.reshape(n_positions, n_colors)
    
    # Sumar sobre posiciones
    color_probs = probs_reshaped.sum(axis=0)
    
    return color_probs.astype(np.float32)


def extract_features_single_image(
    model,
    image: np.ndarray,
    config: FeatureExtractionConfig,
    kernels_cache: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Extrae todos los features de una imagen.
    
    Args:
        model: Instancia de FRQI2
        image: Imagen (H, W) en rango [0, 255] uint8
        config: Configuración de extracción
        kernels_cache: Diccionario con kernels pre-generados
    
    Returns:
        Diccionario con arrays de features
    """
    n_kernels = len(config.kernels)
    n_wx = config.n_windows_x
    n_wy = config.n_windows_y
    
    # Inicializar arrays
    theoretical = np.zeros((n_kernels, n_wx, n_wy, len(FEATURE_NAMES_THEORETICAL)), dtype=np.float32)
    hardware_analytic = np.zeros((n_kernels, n_wx, n_wy, len(FEATURE_NAMES_HARDWARE)), dtype=np.float32)
    hardware_sampled = np.zeros((n_kernels, n_wx, n_wy, len(FEATURE_NAMES_HARDWARE)), dtype=np.float32)
    probs = np.zeros((n_kernels, n_wx, n_wy, len(FEATURE_NAMES_PROBS)), dtype=np.float32)
    
    for k_idx, kernel_name in enumerate(config.kernels):
        kernel_patch = kernels_cache[kernel_name]
        
        for wx in range(n_wx):
            for wy in range(n_wy):
                # Extraer parche de imagen
                image_patch = extract_window(image, wx, wy, config.window_size)
                
                # Verificar tamaño
                if image_patch.shape != (config.window_size, config.window_size):
                    continue
                
                # Extraer features
                feats = extract_features_single_window(
                    model, image_patch, kernel_patch, config.shots_hardware
                )
                
                theoretical[k_idx, wx, wy] = feats['theoretical']
                hardware_analytic[k_idx, wx, wy] = feats['hardware_analytic']
                hardware_sampled[k_idx, wx, wy] = feats['hardware_sampled']
                probs[k_idx, wx, wy] = feats['probs']
    
    return {
        'theoretical': theoretical,
        'hardware_analytic': hardware_analytic,
        'hardware_sampled': hardware_sampled,
        'probs': probs,
    }


def extract_features_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    config: FeatureExtractionConfig,
    model_class,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Extrae features de un dataset completo.
    
    Args:
        images: Array de imágenes (N, H, W) en [0, 255] uint8
        labels: Array de labels (N,)
        config: Configuración de extracción
        model_class: Clase FRQI2 (o similar)
        verbose: Mostrar barra de progreso
    
    Returns:
        Diccionario con todos los features y metadata
    """
    n_images = len(images)
    n_kernels = len(config.kernels)
    n_wx = config.n_windows_x
    n_wy = config.n_windows_y
    
    # Crear modelo
    model = model_class(image_size=config.window_size, device=config.device)
    
    # Pre-generar kernels
    kernels_cache = {name: get_kernel(name, config.window_size) 
                     for name in config.kernels}
    
    # Inicializar arrays de salida
    all_theoretical = np.zeros(
        (n_images, n_kernels, n_wx, n_wy, len(FEATURE_NAMES_THEORETICAL)), 
        dtype=np.float32
    )
    all_hardware_analytic = np.zeros(
        (n_images, n_kernels, n_wx, n_wy, len(FEATURE_NAMES_HARDWARE)),
        dtype=np.float32
    )
    all_hardware_sampled = np.zeros(
        (n_images, n_kernels, n_wx, n_wy, len(FEATURE_NAMES_HARDWARE)),
        dtype=np.float32
    )
    all_probs = np.zeros(
        (n_images, n_kernels, n_wx, n_wy, len(FEATURE_NAMES_PROBS)),
        dtype=np.float32
    )
    
    # Iterar sobre imágenes
    iterator = tqdm(range(n_images), desc="Extrayendo features") if verbose else range(n_images)
    
    for i in iterator:
        image = images[i]
        
        # Asegurar uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        feats = extract_features_single_image(model, image, config, kernels_cache)
        
        all_theoretical[i] = feats['theoretical']
        all_hardware_analytic[i] = feats['hardware_analytic']
        all_hardware_sampled[i] = feats['hardware_sampled']
        all_probs[i] = feats['probs']
    
    return {
        # Metadata
        'n_images': n_images,
        'n_kernels': n_kernels,
        'n_windows_x': n_wx,
        'n_windows_y': n_wy,
        'window_size': config.window_size,
        'kernel_names': config.kernels,
        'feature_names_theoretical': FEATURE_NAMES_THEORETICAL,
        'feature_names_hardware': FEATURE_NAMES_HARDWARE,
        'feature_names_probs': FEATURE_NAMES_PROBS,
        'shots_hardware': config.shots_hardware,
        
        # Labels
        'labels': labels,
        
        # Features
        'features_theoretical': all_theoretical,
        'features_hardware_analytic': all_hardware_analytic,
        'features_hardware_sampled': all_hardware_sampled,
        'features_probs': all_probs,
    }


# ============================================================
# Agregación de Features
# ============================================================

def aggregate_features_stats(features: np.ndarray) -> np.ndarray:
    """
    Agrega features por imagen usando estadísticas (mean, std, max, min).
    
    Args:
        features: Array (N, K, Wx, Wy, F)
    
    Returns:
        Array (N, K * F * 4) con mean, std, max, min por kernel y feature
    """
    N, K, Wx, Wy, F = features.shape
    
    # Calcular estadísticas sobre ventanas
    feat_mean = features.mean(axis=(2, 3))  # (N, K, F)
    feat_std = features.std(axis=(2, 3))
    feat_max = features.max(axis=(2, 3))
    feat_min = features.min(axis=(2, 3))
    
    # Concatenar y aplanar
    aggregated = np.concatenate([feat_mean, feat_std, feat_max, feat_min], axis=-1)  # (N, K, F*4)
    
    return aggregated.reshape(N, -1)


def aggregate_features_flatten(features: np.ndarray) -> np.ndarray:
    """
    Aplana todos los features.
    
    Args:
        features: Array (N, K, Wx, Wy, F)
    
    Returns:
        Array (N, K * Wx * Wy * F)
    """
    N = features.shape[0]
    return features.reshape(N, -1)


def prepare_classification_data(
    data: Dict[str, Any],
    feature_type: str = 'theoretical',
    aggregation: str = 'stats'
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepara datos para clasificación.
    
    Args:
        data: Diccionario de features (output de extract_features_dataset)
        feature_type: 'theoretical', 'hardware_analytic', 'hardware_sampled', 
                      'probs', o 'all'
        aggregation: 'stats' o 'flatten'
    
    Returns:
        X: Features (N, F_total)
        y: Labels (N,)
        feature_names: Lista de nombres de features
    """
    y = data['labels']
    kernel_names = data['kernel_names']
    
    feature_arrays = []
    feature_names = []
    
    def add_features(feat_array, base_names, prefix):
        if aggregation == 'stats':
            agg = aggregate_features_stats(feat_array)
            for k_name in kernel_names:
                for f_name in base_names:
                    for stat in ['mean', 'std', 'max', 'min']:
                        feature_names.append(f"{prefix}_{k_name}_{f_name}_{stat}")
        else:
            agg = aggregate_features_flatten(feat_array)
            for k_idx, k_name in enumerate(kernel_names):
                for wx in range(data['n_windows_x']):
                    for wy in range(data['n_windows_y']):
                        for f_name in base_names:
                            feature_names.append(f"{prefix}_{k_name}_w{wx}{wy}_{f_name}")
        feature_arrays.append(agg)
    
    if feature_type in ['theoretical', 'all']:
        add_features(data['features_theoretical'], 
                    data['feature_names_theoretical'], 'th')
    
    if feature_type in ['hardware_analytic', 'all']:
        add_features(data['features_hardware_analytic'],
                    data['feature_names_hardware'], 'hw_an')
    
    if feature_type in ['hardware_sampled', 'all']:
        add_features(data['features_hardware_sampled'],
                    data['feature_names_hardware'], 'hw_sp')
    
    if feature_type in ['probs', 'all']:
        add_features(data['features_probs'],
                    data['feature_names_probs'], 'prob')
    
    X = np.concatenate(feature_arrays, axis=1)
    
    return X, y, feature_names


# ============================================================
# Visualización
# ============================================================

def plot_feature_maps(
    features: np.ndarray,
    feature_names: List[str],
    kernel_names: List[str],
    title: str = "Feature Maps",
    figsize: Tuple[int, int] = None
):
    """
    Visualiza mapas de features para una imagen.
    
    Args:
        features: Array (K, Wx, Wy, F) para una imagen
        feature_names: Lista de nombres de features
        kernel_names: Lista de nombres de kernels
        title: Título de la figura
        figsize: Tamaño de figura (opcional)
    """
    import matplotlib.pyplot as plt
    
    K, Wx, Wy, F = features.shape
    
    if figsize is None:
        figsize = (3 * K, 3 * min(F, 6))
    
    n_features_to_show = min(F, 6)
    
    fig, axes = plt.subplots(n_features_to_show, K, figsize=figsize)
    
    if K == 1:
        axes = axes.reshape(-1, 1)
    if n_features_to_show == 1:
        axes = axes.reshape(1, -1)
    
    for f_idx in range(n_features_to_show):
        for k_idx in range(K):
            ax = axes[f_idx, k_idx]
            im = ax.imshow(features[k_idx, :, :, f_idx], cmap='viridis')
            
            if f_idx == 0:
                ax.set_title(kernel_names[k_idx], fontsize=9)
            if k_idx == 0:
                ax.set_ylabel(feature_names[f_idx], fontsize=8)
            
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    return fig


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
    title: str = "Feature Importance",
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Visualiza importancia de features.
    """
    import matplotlib.pyplot as plt
    
    # Ordenar por importancia
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(top_n)
    ax.barh(y_pos, importances[indices][::-1])
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]], fontsize=8)
    ax.set_xlabel('Importancia')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    title: str = "Matriz de Confusión",
    figsize: Tuple[int, int] = (8, 6),
    normalize: bool = True
):
    """
    Visualiza matriz de confusión.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='Etiqueta Real',
           xlabel='Etiqueta Predicha')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Añadir valores en celdas
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_tsne_embedding(
    X: np.ndarray,
    y: np.ndarray,
    title: str = "t-SNE Embedding",
    figsize: Tuple[int, int] = (10, 8),
    perplexity: int = 30
):
    """
    Visualiza embedding t-SNE de los features.
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    # Reducir dimensionalidad si es muy alta
    if X.shape[1] > 50:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=50)
        X_reduced = pca.fit_transform(X)
    else:
        X_reduced = X
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(X)-1), 
                random_state=42, n_iter=1000)
    X_embedded = tsne.fit_transform(X_reduced)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    classes = np.unique(y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    
    for i, cls in enumerate(classes):
        mask = y == cls
        ax.scatter(X_embedded[mask, 0], X_embedded[mask, 1], 
                  c=[colors[i]], label=str(cls), alpha=0.7, s=50)
    
    ax.legend(title="Clase", loc='best')
    ax.set_title(title)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    plt.tight_layout()
    return fig


def plot_pca_variance(
    X: np.ndarray,
    n_components: int = 20,
    title: str = "Varianza Explicada por PCA",
    figsize: Tuple[int, int] = (10, 5)
):
    """
    Visualiza varianza explicada por componentes PCA.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    n_components = min(n_components, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_components)
    pca.fit(X)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Varianza individual
    ax1.bar(range(1, n_components+1), pca.explained_variance_ratio_)
    ax1.set_xlabel('Componente')
    ax1.set_ylabel('Varianza Explicada')
    ax1.set_title('Varianza por Componente')
    
    # Varianza acumulada
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    ax2.plot(range(1, n_components+1), cumsum, 'bo-')
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95%')
    ax2.axhline(y=0.99, color='g', linestyle='--', label='99%')
    ax2.set_xlabel('Número de Componentes')
    ax2.set_ylabel('Varianza Acumulada')
    ax2.set_title('Varianza Acumulada')
    ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


# ============================================================
# I/O
# ============================================================

def save_features(data: Dict[str, Any], filepath: str):
    """Guarda features en formato .npz"""
    # Convertir listas a arrays para guardar
    save_dict = {}
    for key, value in data.items():
        if isinstance(value, list):
            save_dict[key] = np.array(value, dtype=object)
        else:
            save_dict[key] = value
    
    np.savez_compressed(filepath, **save_dict)
    print(f"Features guardados en: {filepath}")


def load_features(filepath: str) -> Dict[str, Any]:
    """Carga features desde archivo .npz"""
    loaded = np.load(filepath, allow_pickle=True)
    
    data = {}
    for key in loaded.files:
        value = loaded[key]
        # Convertir arrays de objetos a listas
        if value.dtype == object:
            data[key] = value.tolist()
        elif value.ndim == 0:
            data[key] = value.item()
        else:
            data[key] = value
    
    return data
