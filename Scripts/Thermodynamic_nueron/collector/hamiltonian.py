from qutip import basis, tensor
from ..config.parametres import e0, e1, ez, g
from ..collector.operators import n0, n1, nz

H0 = e0 * n0 + e1 * n1 + ez * nz

ket101 = tensor(basis(2, 1), basis(2, 0), basis(2, 1))
ket010 = tensor(basis(2, 0), basis(2, 1), basis(2, 0))

H_int = g * (ket101 * ket010.dag() + ket010 * ket101.dag())

H = H0 + H_int

