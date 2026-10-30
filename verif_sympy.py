# Contrôle en SymPy des calculs exacts du notebook SageMath (Sage absent de cet environnement)
import sympy as sp, numpy as np, networkx as nx, itertools
from networkx.algorithms.isomorphism import GraphMatcher
# 1. deux cliques K5 reliées par une arête
G = nx.disjoint_union(nx.complete_graph(5), nx.complete_graph(5)); G.add_edge(0, 5)
L = sp.Matrix(nx.laplacian_matrix(G, nodelist=range(10)).toarray())
lam = sp.symbols('lambda')
print("charpoly :", sp.factor(L.charpoly(lam).as_expr()))
print("valeurs propres :", L.eigenvals())
auts = list(GraphMatcher(G, G).isomorphisms_iter()); print("ordre Aut(G) :", len(auts))
# orbites
orb = {}
for v in G: orb.setdefault(frozenset(a[v] for a in auts), None)
print("orbites :", [sorted(o) for o in orb])
# 2. formule de l'atrophie sur le chemin P3 (symbolique)
b, t, s = sp.symbols('beta t s', positive=True)
L3 = sp.Matrix([[1,-1,0],[-1,2,-1],[0,-1,1]])
x0 = sp.Matrix([1,0,0])
E = (-b*s*L3).exp()
a_int = (E*x0).applyfunc(lambda f: sp.integrate(f, (s, 0, t)))
P, Dg = L3.diagonalize(normalize=True)
mus = [Dg[k,k] for k in range(3)]
a_spec = sp.zeros(3,1)
for k in range(3):
    u = P[:,k]; c = (u.T*x0)[0]
    a_spec += (t if mus[k]==0 else (1-sp.exp(-b*mus[k]*t))/(b*mus[k]))*c*u
print("écart a(t) intégrale - formule spectrale :", sp.simplify(a_int - a_spec).T)
# 3. coefficients de Taylor de Fisher-KPP sur le chemin pondéré 0-1-2-3, graine en 0
al, eps = sp.symbols('alpha epsilon', positive=True)
w1, w2, w3 = sp.symbols('w1 w2 w3', positive=True)
W = sp.Matrix([[0,w1,0,0],[w1,0,w2,0],[0,w2,0,w3],[0,0,w3,0]])
Lw = sp.diag(*[sum(W.row(i)) for i in range(4)]) - W
X = sp.Matrix(sp.symbols('x0:4'))
F = -b*Lw*X + al*X.applyfunc(lambda z: z*(1-z))
def lie(g):  # dérivée de g le long du champ F
    return sum(sp.diff(g, X[i])*F[i] for i in range(4))
init = {X[0]: eps, X[1]: 0, X[2]: 0, X[3]: 0}
for j in range(4):
    g = X[j]; ders = []
    for k in range(j+1):
        ders.append(sp.factor(sp.expand(g.subs(init)))); g = sp.expand(lie(g))
    print(f"x_{j} : dérivées 0..{j} en t=0 :", ders)
# 4. équivariance et non-identifiabilité sous observation bilatérale (numérique)
Ln = np.array(L, dtype=float); from scipy.linalg import expm
sig = auts[[i for i,a in enumerate(auts) if a[0]==5 and all(a[k]==k+5 for k in range(5))][0]]
Pm = np.zeros((10,10))
for i,j in sig.items(): Pm[j,i] = 1
def a_num(x0, t, beta=1.0):
    mu,U = np.linalg.eigh(Ln); w = np.where(mu>1e-10, -np.expm1(-beta*mu*t)/(beta*np.maximum(mu,1e-10)), t)
    return U@(w*(U.T@x0))
e1 = np.eye(10)[1]; e6 = np.eye(10)[6]
print("||a(e6) - P a(e1)|| =", np.abs(a_num(e6,2.0) - Pm@a_num(e1,2.0)).max())
H = np.hstack([np.eye(5), np.eye(5)])/2    # observation moyennée sur les régions homologues
print("||H a(e1) - H a(e6)|| =", np.abs(H@a_num(e1,2.0) - H@a_num(e6,2.0)).max())
print("||a(e1) - a(e6)|| =", np.abs(a_num(e1,2.0) - a_num(e6,2.0)).max())
# 5. limite aux grands temps : a(t) - t*mean(x0) 1 -> L^+ x0 / beta
Lp = np.linalg.pinv(Ln); print("écart à la limite t=60 :", np.abs(a_num(e1,60.)-60*e1.mean()-Lp@e1).max())
