"""
Utilidades para notebooks y ejemplos FRQI.

- Generación de imágenes de prueba
- Selección de clase FRQI según número de qubits de color
"""

from __future__ import annotations

from typing import Optional, Type

import numpy as np

# Import relativo dentro del paquete `src`
from .frqi_module import FRQI, FRQI2, FRQI3, QuantumImageBase

__all__ = [
    "generate_image",
    "ensure_uint8",
    "select_frqi",
    "get_kernel",
    "build_kernel_bank",
    "apply_kernel_bank",
    "sobel_kernel_size",
    "laplacian_kernel_size",
    "kernel_patch_from_base",
    "build_kernel_patch",
]


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    """Asegura que la imagen esté en rango [0,255] y dtype uint8.

    - Si el dtype es float se asume rango [0,1] o [0,255] y se reescala si procede.
    - Si el dtype es entero se recorta al rango válido.
    """
    if img.dtype.kind == "f":
        m = float(img.max()) if img.size else 1.0
        # Reescala si el valor máximo aparenta ser <= 1.0
        if m <= 1.0 + 1e-8:
            img = (img * 255.0)
        img = np.clip(img, 0.0, 255.0).round().astype(np.uint8)
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def generate_image(
    image_size: int = 4,
    use_pattern: bool = True,
    seed: Optional[int] = None,
    pattern: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Genera una imagen de tamaño ``image_size x image_size``.

    - ``use_pattern=True``: usa un patrón fijo o el provisto en ``pattern``.
    - ``use_pattern=False``: genera valores aleatorios uniformes en [0,255].
    - ``seed`` controla la reproducibilidad cuando ``use_pattern=False``.
    """
    if use_pattern:
        if pattern is not None:
            if pattern.shape != (image_size, image_size):
                raise ValueError("pattern debe tener shape (image_size, image_size)")
            return ensure_uint8(pattern)
        base = np.array(
            [
                [0, 150, 0, 0],
                [200, 255, 200, 0],
                [0, 200, 255, 200],
                [0, 0, 150, 0],
            ],
            dtype=np.uint8,
        )
        if image_size == 4:
            return base
        # Si se pide otro tamaño, repite el mosaico para mantener el motivo
        reps = image_size // 4 + int(image_size % 4 != 0)
        tiled = np.tile(base, (reps, reps))[:image_size, :image_size]
        return tiled.astype(np.uint8)
    # Aleatoria reproducible
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(image_size, image_size), dtype=np.uint8)


def select_frqi(n_color_qubits: int) -> Type[QuantumImageBase]:
    """Devuelve la clase FRQI apropiada según ``n_color_qubits`` (1–3)."""
    if n_color_qubits == 1:
        return FRQI
    if n_color_qubits == 2:
        return FRQI2
    if n_color_qubits == 3:
        return FRQI3
    raise ValueError("n_color_qubits debe estar en {1,2,3}")


# --- Kernels y convoluciones basicas para ejercicios con xFRQI ---

def _identity_kernel() -> np.ndarray:
    return np.array([[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]], dtype=np.float32)


def _sobel_kernel(axis: str = "x") -> np.ndarray:
    if axis.lower() == "x":
        return np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]], dtype=np.float32)
    if axis.lower() == "y":
        return np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]], dtype=np.float32)
    raise ValueError("axis debe ser 'x' o 'y'")


def _laplacian_kernel(kind: str = "4") -> np.ndarray:
    if kind == "4":
        return np.array([[0, 1, 0],
                         [1, -4, 1],
                         [0, 1, 0]], dtype=np.float32)
    if kind == "8":
        return np.array([[1, 1, 1],
                         [1, -8, 1],
                         [1, 1, 1]], dtype=np.float32)
    raise ValueError("kind debe ser '4' o '8'")


def _gabor_kernel(theta: float = 0.0, sigma: float = 1.0, freq: float = 0.25, size: int = 7) -> np.ndarray:
    if size % 2 == 0:
        raise ValueError("size debe ser impar")
    half = size // 2
    y, x = np.meshgrid(np.arange(-half, half + 1), np.arange(-half, half + 1))
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    gaussian = np.exp(-(xr**2 + yr**2) / (2 * sigma**2))
    carrier = np.cos(2 * np.pi * freq * xr)
    kern = gaussian * carrier
    kern = kern - kern.mean()
    norm = np.linalg.norm(kern.ravel(), ord=2)
    return kern / (norm + 1e-8)


def get_kernel(name: str, **kwargs) -> np.ndarray:
    """Devuelve un kernel 2D por nombre."""
    key = name.lower()
    if key == "identity":
        return _identity_kernel()
    if key in ("sobel_x", "sobel-x"):
        return _sobel_kernel("x")
    if key in ("sobel_y", "sobel-y"):
        return _sobel_kernel("y")
    if key in ("laplacian4", "laplace4", "laplacian_4"):
        return _laplacian_kernel("4")
    if key in ("laplacian8", "laplace8", "laplacian_8"):
        return _laplacian_kernel("8")
    if key == "gabor":
        return _gabor_kernel(**kwargs)
    raise ValueError(f"kernel desconocido: {name}")


def build_kernel_bank(specs):
    """Construye una lista de kernels a partir de especificaciones."""
    bank = []
    for spec in specs:
        if isinstance(spec, str):
            bank.append(get_kernel(spec))
        elif isinstance(spec, (tuple, list)) and len(spec) == 2:
            name, params = spec
            bank.append(get_kernel(name, **(params or {})))
        elif isinstance(spec, dict) and "name" in spec:
            bank.append(get_kernel(spec["name"], **spec.get("params", {})))
        else:
            raise ValueError(f"Especificacion no reconocida: {spec}")
    return bank


def _conv2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolucion 2D simple con padding reflejado."""
    img = image.astype(np.float32)
    k = kernel.astype(np.float32)
    kh, kw = k.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("El kernel debe tener dimensiones impares")
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            window = padded[i:i+kh, j:j+kw]
            out[i, j] = np.sum(window * k)
    return out


def apply_kernel_bank(image: np.ndarray, kernels, normalize: bool = True):
    """Aplica un banco de kernels a una imagen 2D (grises)."""
    outputs = []
    for k in kernels:
        res = _conv2d(image, k)
        if normalize:
            res = res - res.min()
            m = res.max()
            if m > 1e-8:
                res = (res / m) * 255.0
            res = ensure_uint8(res)
        outputs.append(res)
    return outputs


# --- Kernels dimensionables y parches auxiliares ---

def sobel_kernel_size(size: int = 3, axis: str = "x") -> np.ndarray:
    """Sobel 3x3 o 5x5 (eje x o y)."""
    axis = axis.lower()
    if size == 3:
        if axis == "x":
            return np.array([[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]], dtype=np.float32)
        if axis == "y":
            return np.array([[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]], dtype=np.float32)
    elif size == 5:
        if axis == "x":
            return np.array([[2, 1, 0, -1, -2],
                             [3, 2, 0, -2, -3],
                             [4, 3, 0, -3, -4],
                             [3, 2, 0, -2, -3],
                             [2, 1, 0, -1, -2]], dtype=np.float32)
        if axis == "y":
            return np.array([[2, 3, 4, 3, 2],
                             [1, 2, 3, 2, 1],
                             [0, 0, 0, 0, 0],
                             [-1, -2, -3, -2, -1],
                             [-2, -3, -4, -3, -2]], dtype=np.float32)
    raise ValueError("size debe ser 3 o 5 y axis debe ser 'x' o 'y'")


def laplacian_kernel_size(size: int = 3) -> np.ndarray:
    """Laplaciano 3x3 (4 vecinos) o 5x5."""
    if size == 3:
        return np.array([[0, 1, 0],
                         [1, -4, 1],
                         [0, 1, 0]], dtype=np.float32)
    if size == 5:
        return np.array([[0, 0, -1, 0, 0],
                         [0, -1, -2, -1, 0],
                         [-1, -2, 16, -2, -1],
                         [0, -1, -2, -1, 0],
                         [0, 0, -1, 0, 0]], dtype=np.float32)
    raise ValueError("size debe ser 3 o 5")


def kernel_patch_from_base(kbase: np.ndarray, win: int) -> np.ndarray:
    """Inserta un kernel base en el centro de un parche win x win y normaliza a [0,255]."""
    kpatch = np.zeros((win, win), dtype=np.float32)
    kh, kw = kbase.shape
    r0, c0 = (win - kh) // 2, (win - kw) // 2
    kpatch[r0:r0+kh, c0:c0+kw] = kbase
    kpatch = kpatch - kpatch.min()
    kpatch = kpatch / (kpatch.max() + 1e-8) * 255.0
    return ensure_uint8(kpatch)


def build_kernel_patch(name: str, win: int, **kwargs) -> np.ndarray:
    """Construye un parche de kernel del mismo tamaño que la ventana FRQI.

    - Sobel/Laplaciano: usa tamaño base 3 si win<=4, si win>=8 usa 5.
    - Gabor: usa params pasados o defaults coherentes con win.
    """
    key = name.lower()
    if key.startswith("sobel"):
        size = 3 if win <= 4 else 5
        axis = "x" if "x" in key else "y"
        base = sobel_kernel_size(size=size, axis=axis)
    elif key.startswith("laplacian"):
        size = 3 if win <= 4 else 5
        base = laplacian_kernel_size(size=size)
    elif key == "gabor":
        params = kwargs.copy()
        params.setdefault("theta", 0.0)
        params.setdefault("sigma", win / 4)
        params.setdefault("freq", 0.25)
        size = params.get("size")
        if size is None:
            size = win if win % 2 else win - 1
            if size < 3:
                size = 3
            params["size"] = size
        base = get_kernel("gabor", **params)
    else:
        base = get_kernel(name, **kwargs)
    return kernel_patch_from_base(base, win)
