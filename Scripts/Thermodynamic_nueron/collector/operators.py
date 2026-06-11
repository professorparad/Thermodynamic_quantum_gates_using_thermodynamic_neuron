from qutip import qeye, destroy, tensor

I = qeye(2)
sm = destroy(2)
sp = sm.dag()
n = sp * sm

n0 = tensor(n, I, I)
n1 = tensor(I, n, I)
nz = tensor(I, I, n)

sm0 = tensor(sm, I, I)
sm1 = tensor(I, sm, I)
smz = tensor(I, I, sm)
sp0 = sm0.dag()
sp1 = sm1.dag()
spz = smz.dag()
