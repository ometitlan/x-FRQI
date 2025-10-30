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
