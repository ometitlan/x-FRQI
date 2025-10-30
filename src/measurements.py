"""
Funciones de medición y estadística a partir de circuitos FRQI.

El objetivo es reutilizar el *mismo* circuito de preparación tanto
en simulador (shots=None → estado exacto) como en hardware (shots>0).
Todas las mediciones están pensadas para base Z física.
"""

from itertools import combinations
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import pennylane as qml


# -------------------------------------------------------------------
# 1) Expval ⟨Z⟩ individuales y ⟨ZZ⟩ de todos los pares de qubits color
# -------------------------------------------------------------------
def build_expval_Z_family(
    circuit_fn: Callable,
    device: qml.device,
    pos_wires: List[int],
    color_wires: List[int],
):
    """
    Devuelve un QNode que prepara el circuito_fn(*angles)
    y retorna:
      (<Z_color0>, <Z_color1>, ...,
       <Z_color0 Z_color1>, <Z_color0 Z_color2>, ...)
    en UN solo disparo de simulador o en una misma tanda de shots.

    Parameters
    ----------
    circuit_fn : Callable
        Función que implementa el circuito de preparación y acepta un número variable de ángulos como argumentos: circuit_fn(*angles).
    device : qml.Device
        Dispositivo de PennyLane.
    pos_wires : List[int]
        Lista de índices de qubits de posición.
    color_wires : List[int]
        Lista de índices de qubits de color.
    """

    obs = [qml.PauliZ(w) for w in color_wires]  # 1-qubit
    obs += [
        qml.PauliZ(w1) @ qml.PauliZ(w2)
        for w1, w2 in combinations(color_wires, 2)
    ]

    @qml.qnode(device)
    def qnode(*angles: float) -> list:
        """
        Prepares the circuit with the given angles and returns expectation values
        for single and pairwise Z observables on color qubits.

        Args:
            *angles (float): Rotation angles for the circuit preparation.

        Returns:
            list: Expectation values for Z observables.
        """
        circuit_fn(*angles)
        return [qml.expval(o) for o in obs]

    return qnode


# -------------------------------------------------------------------
# 2) Muestreo completo en base Z
# -------------------------------------------------------------------
def build_sampler_Z(
    circuit_fn,
    device: qml.device,
    n_qubits: int,
):
    """
    Devuelve un QNode que prepara circuit_fn(*angles)
    y devuelve `qml.sample(wires=range(n_qubits))`
    (matriz de shape (shots, n_qubits) con bits 0/1).

    Parameters
    @qml.qnode(device)
    def qnode(*angles: float) -> np.ndarray:
        circuit_fn(*angles)
        return qml.sample(wires=range(n_qubits))
        Dispositivo de PennyLane.
    n_qubits : int
        Número total de qubits en el circuito.
    """

    @qml.qnode(device)
    def qnode(*angles):
        circuit_fn(*angles)
        return qml.sample(wires=list(range(n_qubits)))

    return qnode


# -------------------------------------------------------------------
# 3) Información mutua clásica I(B:X) a partir de samples
# -------------------------------------------------------------------
def mi_color_vs_position(
    samples: np.ndarray,
    pos_wires: List[int],
    color_wires: List[int],
    n_blocks: Optional[int] = None,
) -> float:
    """
    Calcula la información mutua clásica entre
      B = bits de color (interpretados como entero)
      X = índice de píxel (o bloque si n_blocks está definido)
    a partir de la matriz de muestras (shots × n_qubits).

    n_blocks : si es potencia de 2, se usan los k-MSBs de posición
               para agrupar la imagen en 2^k bloques.
               Ej.: n_blocks=16 → bloques 4×4 en imagen 8×8.
    """
    # entero de color
    B = (samples[:, color_wires] @ (1 << np.arange(len(color_wires))[::-1])).astype(
        int
    )

    # entero de posición (big-endian)
    Xbits = samples[:, pos_wires]
    X_int = (Xbits @ (1 << np.arange(len(pos_wires))[::-1])).astype(int)

    # coarse graining si se pide
    if n_blocks is not None:
        if n_blocks <= 0 or (n_blocks & (n_blocks - 1)) != 0:
            raise ValueError("n_blocks must be a positive power of 2.")
        k = int(np.log2(n_blocks))
        X = X_int >> (len(pos_wires) - k)
    else:
        X = X_int

    # histograma conjunto
    joint = np.histogram2d(
        B, X, bins=[2 ** len(color_wires), X.max() + 1]
    )[0].astype(float)
    joint /= joint.sum()

    pB = joint.sum(axis=1, keepdims=True)
    pX = joint.sum(axis=0, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(joint * (np.log(joint) - np.log(pB) - np.log(pX))) / np.log(2)

    return float(mi)
