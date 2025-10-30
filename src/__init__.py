# src/__init__.py

"""
FRQI Project - Quantum Image Representation
"""

# Importa las clases/funciones principales de cada módulo
from .frqi_module import FRQI, FRQI2, FRQI3
from .utils import generate_image, select_frqi
# Exportaciones principales y minimalistas para evitar dependencias cruzadas
__all__ = [
    'FRQI',
    'FRQI2',
    'FRQI3',
    'generate_image',
    'select_frqi',
]

# Nota: módulos completos pueden importarse explícitamente si se necesitan
#   from src.quantum_image import QuantumImageBase
#   from src.measurements import build_expval_Z_family, build_sampler_Z, mi_color_vs_position

__version__ = '0.1.0'
