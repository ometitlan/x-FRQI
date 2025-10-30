# src/quantum_image.py

import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from typing import List, Tuple
from itertools import combinations

from .info_measures import entropy, conditional_entropy, mutual_information
from . import measurements

__all__ = ["FRQI", "FRQI2", "FRQI3", "QuantumImageBase"]

class QuantumImageBase:
    """
    FRQI genérico con n_color_qubits (1-3).
    Wires 0..n_pos-1   : posición   (MSB primero)
    Wires n_pos..end   : colores    (canal 0 = wire n_pos)
    """

    def __init__(self, image_size, n_color_qubits=1, device="default.qubit"):
        self.image_size = image_size
        self.n_color_qubits = n_color_qubits
        self.n_position_qubits = int(np.log2(image_size ** 2))
        self.n_qubits = self.n_position_qubits + n_color_qubits
        self.dev = qml.device(device, wires=self.n_qubits)
        self.device_str = device

    def _apply_hadamards(self):
        for w in range(self.n_position_qubits):
            qml.Hadamard(wires=w)

    def _angle_encoding(self, data, target_wire):
        for idx, angle in enumerate(data):
            binary = f"{idx:0{self.n_position_qubits}b}"
            for j, bit in enumerate(binary):
                if bit == "0": qml.PauliX(wires=j)
            qml.ctrl(qml.RY, control=list(range(self.n_position_qubits)))(2 * angle, wires=target_wire)
            for j, bit in enumerate(binary):
                if bit == "0": qml.PauliX(wires=j)

    def _flatten_angles(self, image):
        return (np.pi / 2) * (image.flatten() / 255.0)

    def _build_circuit(self, *angles):
        self._apply_hadamards()
        for i, ang in enumerate(angles):
            self._angle_encoding(ang, target_wire=self.n_position_qubits + i)

    def encode(self, *images):
        angles = [self._flatten_angles(img) for img in images]
        @qml.qnode(self.dev)
        def circuit(*args):
            self._build_circuit(*args)
            return qml.state()
        return circuit(*angles)

    def theoretical_state(self, *images):
        n_pixels = self.image_size ** 2
        dim = 2 ** self.n_qubits
        state = np.zeros(dim, dtype=complex)
        angles = [self._flatten_angles(img) for img in images]
        amp = 1 / np.sqrt(n_pixels)
        for pos_idx in range(n_pixels):
            for color_bits in range(2 ** self.n_color_qubits):
                rev = 0
                for k in range(self.n_color_qubits):
                    if (color_bits >> k) & 1:
                        rev |= 1 << (self.n_color_qubits - 1 - k)
                prod = 1
                for k in range(self.n_color_qubits):
                    theta = angles[k][pos_idx]
                    prod *= np.sin(theta) if (color_bits>>k)&1 else np.cos(theta)
                idx = (pos_idx << self.n_color_qubits) | rev
                state[idx] = amp * prod
        return state

    def recover(self, state, shots=None):
        """
        Recover images: exact (shots=None) or statistical (shots>0).
        """
        # get probabilities (compatibilidad con versiones de PennyLane)
        def _prepare_state():
            try:
                qml.StatePrep(state, wires=range(self.n_qubits))
            except Exception:
                qml.QubitStateVector(state, wires=range(self.n_qubits))

        if shots is None:
            @qml.qnode(self.dev)
            def meas():
                _prepare_state()
                return qml.probs(wires=range(self.n_qubits))
            probs = meas()
        else:
            dev = qml.device(self.device_str, wires=self.n_qubits, shots=shots)
            @qml.qnode(dev)
            def samp():
                _prepare_state()
                return qml.sample(wires=range(self.n_qubits))
            counts = {}
            for s in samp(): counts[tuple(s)] = counts.get(tuple(s),0)+1
            probs = np.zeros(2**self.n_qubits)
            for bits,c in counts.items():
                idx=0
                for b in bits: idx=2*idx+int(b)
                probs[idx]=c/shots
        # reconstruct each channel
        rec=[]
        n_pixels=self.image_size**2
        for ch in range(self.n_color_qubits):
            img=np.zeros((self.image_size,self.image_size))
            for pos_idx in range(n_pixels):
                tot=p1=0
                for cb in range(2**self.n_color_qubits):
                    rev=0
                    for k in range(self.n_color_qubits):
                        if (cb>>k)&1: rev|=1<<(self.n_color_qubits-1-k)
                    idx=(pos_idx<<self.n_color_qubits)|rev
                    p=probs[idx]
                    tot+=p
                    if (rev>>ch)&1: p1+=p
                theta=np.arcsin(np.sqrt(p1/tot)) if tot>0 else 0
                img.flat[pos_idx]=(2/np.pi)*theta*255
            rec.append(img.astype(np.uint8))
        if self.n_color_qubits>1: rec=rec[::-1]
        return rec

    def analyze_state(self,state):
        pos=list(range(self.n_position_qubits))
        stats={"H_total":entropy(state,wires=range(self.n_qubits)),"H_position":entropy(state,wires=pos)}
        for k in range(self.n_color_qubits):
            cw=[self.n_position_qubits+k]
            stats[f"H_color{k}"]=entropy(state,wires=cw)
            stats[f"I(color{k}:position)"]=mutual_information(state,wires_a=cw,wires_b=pos)
            stats[f"H_color{k}|position"]=conditional_entropy(state,wires_a=cw,wires_b=pos)
        if self.n_color_qubits>=2:
            c0, c1=[self.n_position_qubits],[self.n_position_qubits+1]
            stats["H_colors_joint"]=entropy(state,wires=c0+c1)
            stats["I(color0:color1)"]=mutual_information(state,wires_a=c0,wires_b=c1)
            stats["H_color0|color1"]=conditional_entropy(state,wires_a=c0,wires_b=c1)
            stats["H_color1|color0"]=conditional_entropy(state,wires_a=c1,wires_b=c0)
            HA,HB,HC=stats["H_position"],stats["H_color0"],stats["H_color1"]
            HAB, HAC, HBC = entropy(state,wires=pos+c0),entropy(state,wires=pos+c1),entropy(state,wires=c0+c1)
            HABC=stats["H_total"]
            stats["I3(position:color0:color1)"]=HA+HB+HC-HAB-HAC-HBC+HABC
        return stats

    def color_Z_stats(self,*images,shots=None,backend=None):
        backend=backend or self.device_str
        angles=[self._flatten_angles(img) for img in images]
        dev=qml.device(backend,wires=self.n_qubits,shots=shots)
        qnode=measurements.build_expval_Z_family(self._build_circuit,dev,
            pos_wires=list(range(self.n_position_qubits)),
            color_wires=list(range(self.n_position_qubits,self.n_qubits)))
        vals=qnode(*angles)
        labels=[f"<Z_{k}>" for k in range(self.n_color_qubits)]
        labels+=[f"<ZZ_{i}{j}>" for i,j in combinations(range(self.n_color_qubits),2)]
        return dict(zip(labels,vals))

    def mi_samples(self, *images, shots=None, n_blocks=None, backend=None):
        """
        Compute classical mutual information between color qubits and position.
        If shots=None: analytic calculation from exact state probabilities.
        If shots>0: empirical estimation via sampling in Z basis.
        """
        from .info_measures import entropy  # ensure package-relative import
        # Analytic MI (exact probabilities)
        if shots is None:
            # get state vector and joint probabilities
            state = self.encode(*images)
            probs = np.abs(state)**2
            N_pos = 2 ** self.n_position_qubits
            N_col = 2 ** self.n_color_qubits
            # reshape to matrix [pos, color]
            probs_matrix = probs.reshape((N_pos, N_col))
            # marginal distributions
            p_pos = probs_matrix.sum(axis=1)
            p_col = probs_matrix.sum(axis=0)
            # compute MI = sum p_ij log2(p_ij/(p_i p_j))
            mi = 0.0
            for i in range(N_pos):
                for j in range(N_col):
                    p_ij = probs_matrix[i, j]
                    if p_ij > 0 and p_pos[i] > 0 and p_col[j] > 0:
                        mi += p_ij * np.log2(p_ij / (p_pos[i] * p_col[j]))
            return mi
        # Sampling-based MI
        backend = backend or self.device_str
        angles = [self._flatten_angles(img) for img in images]
        dev = qml.device(backend, wires=self.n_qubits, shots=shots)
        sampler = measurements.build_sampler_Z(self._build_circuit, dev, self.n_qubits)
        samples = sampler(*angles)
        return measurements.mi_color_vs_position(
            samples,
            pos_wires=list(range(self.n_position_qubits)),
            color_wires=list(range(self.n_position_qubits, self.n_qubits)),
            n_blocks=n_blocks,
        )

    def stem_plot_amplitudes(self, images: List[np.ndarray], shots: int = None, device: str = None, threshold: float = 1e-3, figsize: Tuple[int, int] = (8, 4)):
        dev_str = device or self.device_str
        if shots is None:
            state = self.encode(*images)
            amps = np.abs(state)
            nz = np.nonzero(amps > threshold)[0]
            vals = amps[nz]
            title = f"Stem Plot Exacto ({self.n_color_qubits} color qubits)"
        else:
            dev = qml.device(dev_str, wires=self.n_qubits, shots=shots)
            @qml.qnode(dev)
            def sampler_q(*angles):
                self._build_circuit(*angles)
                return qml.sample(wires=range(self.n_qubits))
            angles = [self._flatten_angles(img) for img in images]
            bits = np.array(sampler_q(*angles))
            idxs = bits.dot(1 << np.arange(self.n_qubits)[::-1])
            counts = np.bincount(idxs, minlength=2 ** self.n_qubits)
            probs = counts / shots
            nz = np.nonzero(probs > threshold)[0]
            vals = np.sqrt(probs[nz])
            title = f"Sampling Stem-Plot (~{shots} shots)"
        labels = [f"|{idx >> self.n_color_qubits}⟩⊗|{format(idx & ((1 << self.n_color_qubits) - 1), f'0{self.n_color_qubits}b')}⟩" for idx in nz]
        fig, ax = plt.subplots(figsize=figsize)
        ml, sl, bl = ax.stem(np.arange(len(nz)), vals)
        plt.setp(bl, visible=False)
        ax.set_xticks(np.arange(len(nz)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel("Proxy amplitude")
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
        return nz, vals

class FRQI(QuantumImageBase):
    def __init__(self,image_size,device="default.qubit"): super().__init__(image_size,n_color_qubits=1,device=device)
class FRQI2(QuantumImageBase):
    def __init__(self,image_size,device="default.qubit"): super().__init__(image_size,n_color_qubits=2,device=device)
class FRQI3(QuantumImageBase):
    def __init__(self,image_size,device="default.qubit"): super().__init__(image_size,n_color_qubits=3,device=device)
