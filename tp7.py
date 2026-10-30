# tp7.py -- code de depart du cours (TP 7), reproduit sans modification
import numpy as np
from scipy.linalg import expm

def connectome_modulaire(n=80, k=4, p_in=0.3, p_out=0.005, seed=5):
    """Graphe pondéré à k modules (substitut d'un connectome de référence)."""
    rng = np.random.default_rng(seed)
    mod = np.repeat(np.arange(k), n//k)
    P = np.where(mod[:, None] == mod[None, :], p_in, p_out)
    W = np.triu((rng.random((n, n)) < P)*rng.uniform(0.5, 1.5, (n, n)), 1)
    return W + W.T, mod

def laplacien(W):
    return np.diag(W.sum(1)) - W

def diffusion(L, x0, t, beta=1.0):
    return expm(-beta*t*L) @ x0

def atrophie(L, x0, t, beta=1.0, spectre=None):
    """a(t) = ∫₀ᵗ x(s) ds, forme spectrale exacte (exercice 7.3)."""
    mu, U = np.linalg.eigh(L) if spectre is None else spectre
    poids = np.where(mu > 1e-10, -np.expm1(-beta*mu*t)/(beta*np.maximum(mu, 1e-10)), t)
    return U @ (poids*(U.T @ x0))

def epicentre(L, a_obs, t_grille):
    """Pour chaque graine et chaque t, corrélation entre carte prédite et carte observée."""
    n = len(L); score = np.full(n, -np.inf); spectre = np.linalg.eigh(L)
    for s in range(n):
        e = np.zeros(n); e[s] = 1
        for t in t_grille:
            score[s] = max(score[s], np.corrcoef(atrophie(L, e, t, spectre=spectre), a_obs)[0, 1])
    return np.argmax(score), score

if __name__ == "__main__":
    W, mod = connectome_modulaire()
    L = laplacien(W); mu, U = np.linalg.eigh(L)
    print("4 plus petites valeurs propres :", mu[:4].round(3), "(module = goulots d'étranglement)")
    x0 = np.zeros(80); x0[3] = 1
    a = atrophie(L, x0, t=2.0) + np.random.default_rng(6).normal(0, 0.02, 80)
    s, _ = epicentre(L, a, t_grille=(1.0, 2.0, 4.0))
    print("épicentre vrai 3, estimé", s, "(même module :", mod[s] == mod[3], ")")
