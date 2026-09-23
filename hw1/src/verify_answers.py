import numpy as np
from itertools import combinations

def net(w, x):
    return w[0] + w[1]*x[0] + w[2]*x[1]

def step01(n):
    return 1 if n >= 0 else 0

w = np.array([0.5, 1.0, -0.3])
pts = {'a': (1.0, 0.2), 'b': (0.0, 0.0), 'c': (-0.1, -0.5), 'd': (1.7, 0.1), 'e': (1.8, -1.4)}

print("=== Part 1.3 ===")
for k, p in pts.items():
    n = net(w, p)
    print(f"{k} {p}: net={n:+.4f} -> class {step01(n)}")

print("\n=== Part 1.7 sigmoid ===")
for k, p in pts.items():
    n = net(w, p)
    s = 1/(1+np.exp(-n))
    print(f"{k} {p}: net={n:+.4f} sigmoid={s:.4f} -> thresh0.5 class {1 if s>=0.5 else 0}")

print("\n=== Part 1.4 check: (0.9,-0.7) is midpoint of b and e ===")
b, e = np.array(pts['b']), np.array(pts['e'])
print("midpoint b,e =", (b+e)/2)
print("net at midpoint =", net(w, (b+e)/2), " (affine: avg of", net(w,b), net(w,e), "=", (net(w,b)+net(w,e))/2, ")")

# brute force: is (0.9,-0.7) labeled 0 with the 5 pts labeled 1 linearly separable?
from scipy.optimize import linprog
P = np.array([[1, *p] for p in pts.values()])          # want w.P >= eps
Q = np.array([[1, 0.9, -0.7]])                          # want w.Q <= -eps
A_ub = np.vstack([-P, Q])
b_ub = -np.ones(len(P)+len(Q)) * 1e-6 * 0 - np.concatenate([np.ones(len(P))*1e-3, -(-np.ones(len(Q))*1e-3)])
# simpler: minimize 0 s.t. -P w <= -1 and Q w <= -1
A = np.vstack([-P, Q])
bb = np.concatenate([-np.ones(len(P)), -np.ones(len(Q))])
res = linprog(c=np.zeros(3), A_ub=A, b_ub=bb, bounds=[(None, None)]*3)
print("LP separable?", res.status == 0, res.message)

print("\n=== Part 2 ===")
data = [((-0.1, -1.0), 0), ((0.2, -0.9), 0), ((0.6, 0.8), 1), ((0.0, 0.0), 1)]
names = ['a', 'b', 'c', 'd']
w = np.array([0.5, 1.0, -0.3])
print("w0 =", w)
for p in range(1, 5):
    print(f"\n--- Pass {p} ---")
    changed = False
    for nm, (x, t) in zip(names, data):
        n = net(w, x)
        a = step01(n)
        pv = np.array([1.0, x[0], x[1]])
        if t - a != 0:
            wnew = w + (t - a)*pv
            print(f" {nm} {x} t={t}: net={n:+.4f} a={a}  (t-a)={t-a:+d}  w: {w} -> {wnew}")
            w = wnew
            changed = True
        else:
            print(f" {nm} {x} t={t}: net={n:+.4f} a={a}  correct, no change  w={w}")
    print(f" end of pass {p}: w = {w}")
    print("  status:", [(nm, f"{net(w,x):+.4f}", step01(net(w,x)), 'OK' if step01(net(w,x))==t else 'WRONG')
                        for nm,(x,t) in zip(names,data)])
    if not changed:
        print("  CONVERGED (no updates this pass)")
        break
