# -*- coding: utf-8 -*-
"""
sujet7.py -- Sujet 7 : retrouver l'épicentre d'une maladie neurodégénérative.
Simulations NumPy. Dans le notebook SageMath :  load('tp7.py'); load('sujet7.py')
En Python pur : python3 sujet7.py  (reproduit toutes les figures et tableaux).
"""
import numpy as np
import time
from scipy.sparse.csgraph import shortest_path

try:                      # sous SageMath, load('tp7.py') a déjà défini ces fonctions
    connectome_modulaire
except NameError:         # en Python pur
    from tp7 import connectome_modulaire, laplacien, diffusion, atrophie, epicentre

GRAINE = 2026             # graine globale de reproductibilité
T_GRILLE = np.geomspace(0.25, 16.0, 25)   # grille des temps pour le problème inverse

# ---------------------------------------------------------------------------
# 1. Outils de base
# ---------------------------------------------------------------------------
def spectre(L):
    """Valeurs propres croissantes et base orthonormée de vecteurs propres de L."""
    return np.linalg.eigh(L)

def cartes_diffusion(spec, t, beta=1.0):
    """Matrice n x n dont la colonne s est a(t) pour la graine e_s (formule de l'ex. 7.3).
    a(t) = U diag(w(t)) U^T e_s,  w_k = (1 - exp(-beta mu_k t)) / (beta mu_k),  w_1 = t."""
    mu, U = spec
    w = np.where(mu > 1e-10, -np.expm1(-beta*mu*t)/(beta*np.maximum(mu, 1e-10)), t)
    return (U*w) @ U.T

def fkpp_toutes_graines(L, alpha, beta, eps, t_points, dt=0.005, X0=None):
    """Fisher-KPP sur réseau x' = -beta L x + alpha x(1-x), intégré par Runge-Kutta 4
    pour TOUTES les graines à la fois (colonne s : x(0) = eps e_s).
    Renvoie (X, A) : états et atrophies cumulées a(t) = ∫ x, aux instants t_points."""
    n = len(L)
    X = eps*np.eye(n) if X0 is None else X0.copy()
    f = lambda Y: -beta*(L @ Y) + alpha*Y*(1 - Y)
    t_points = np.sort(np.asarray(t_points, float))
    Xs, As = [], []
    t, A = 0.0, np.zeros_like(X)
    for tp in t_points:
        while t < tp - 1e-12:
            h = min(dt, tp - t)
            k1 = f(X); k2 = f(X + h/2*k1); k3 = f(X + h/2*k2); k4 = f(X + h*k3)
            Xn = X + h/6*(k1 + 2*k2 + 2*k3 + k4)
            A += h/2*(X + Xn)          # règle des trapèzes pour a(t)
            X, t = Xn, t + h
        Xs.append(X.copy()); As.append(A.copy())
    return np.array(Xs), np.array(As)

def correlations(C, b):
    """Corrélation de Pearson entre chaque colonne de C et le vecteur b."""
    Cc = C - C.mean(0); bc = b - b.mean()
    return (Cc.T @ bc)/(np.linalg.norm(Cc, axis=0)*np.linalg.norm(bc) + 1e-300)

def epicentre_diffusion(spec, a_obs, t_grille=T_GRILLE, beta=1.0):
    """Version vectorisée de epicentre() (tp7.py) : score(s) = max_t corr(a_s(t), a_obs)."""
    S = np.max([correlations(cartes_diffusion(spec, t, beta), a_obs) for t in t_grille], axis=0)
    return int(np.argmax(S)), S

T_FKPP = np.linspace(1, 8, 29)            # grille des temps pour Fisher-KPP (pas 0,25)

def bibliotheque_fkpp(L, alpha=2.0, beta=1.0, eps=0.01, t_grille=T_FKPP, dt=0.02):
    """Cartes d'atrophie FKPP pré-calculées pour toutes les graines et tous les t de la grille
    (RK4, pas dt = 0,02 : stable car beta*mu_max*dt < 2,78)."""
    return fkpp_toutes_graines(L, alpha, beta, eps, t_grille, dt=dt)[1]

def epicentre_bibliotheque(biblio, a_obs):
    S = np.max([correlations(A, a_obs) for A in biblio], axis=0)
    return int(np.argmax(S)), S

def distances(W):
    """Distance de graphe (nombre minimal d'arêtes)."""
    return shortest_path((W > 0).astype(float), unweighted=True, directed=False)

def perturber(W, p, rng, mode="poids"):
    """Connectome imparfait.
    mode='poids'  : W_ij -> W_ij (1 + p u_ij), u ~ U(-1, 1)  (erreur de tractographie sur les poids)
    mode='aretes' : une fraction p des arêtes est supprimée et autant d'arêtes fausses ajoutées."""
    n = len(W); iu = np.triu_indices(n, 1); w = W[iu].copy()
    if mode == "poids":
        w = w*(1 + p*rng.uniform(-1, 1, w.size))
    else:
        pos = np.flatnonzero(w > 0); neg = np.flatnonzero(w == 0)
        m = int(round(p*pos.size))
        if m > 0:
            out = rng.choice(pos, m, replace=False); inn = rng.choice(neg, m, replace=False)
            vals = w[out].copy(); w[out] = 0; w[inn] = vals
    Wp = np.zeros_like(W); Wp[iu] = w
    return Wp + Wp.T

def bruiter(a, sigma_rel, rng):
    """Bruit gaussien i.i.d. d'écart-type sigma_rel * (écart-type spatial de la carte)."""
    return a + sigma_rel*a.std()*rng.standard_normal(a.shape)

# ---------------------------------------------------------------------------
# 2. Étude de simulation principale
# ---------------------------------------------------------------------------
def experience(sigmas=(0, 0.1, 0.25, 0.5, 1.0, 1.5), ps=(0, 0.1, 0.2, 0.3),
               generateurs=("diffusion", "fkpp"), modeles=("diffusion", "fkpp"),
               mode="poids", R=150, seed=GRAINE, alpha=2.0, eps=0.01):
    """Taux de retrait de l'épicentre selon le bruit, la perturbation du connectome,
    le modèle générateur et le modèle inverse. Renvoie un dictionnaire de résultats."""
    rng = np.random.default_rng(seed)
    W, mod = connectome_modulaire(); L = laplacien(W); n = len(W)
    Dg = distances(W)
    # bibliothèques de cartes "vraies" (connectome exact) pour la génération
    spec_vrai = spectre(L)
    tg_fkpp = T_FKPP
    lib_vrai_fkpp = bibliotheque_fkpp(L, alpha, 1.0, eps, tg_fkpp, dt=0.005)
    res = {}
    for p in ps:
        for r in range(R):
            Wp = perturber(W, p, rng, mode) if p > 0 else W
            Lp = laplacien(Wp); spec_p = spectre(Lp)
            lib_p = bibliotheque_fkpp(Lp, alpha, 1.0, eps, tg_fkpp) if "fkpp" in modeles else None
            s = int(rng.integers(n))
            for gen in generateurs:
                if gen == "diffusion":
                    t = rng.uniform(1, 8); a = cartes_diffusion(spec_vrai, t)[:, s]
                else:
                    k = rng.integers(6, 21); a = lib_vrai_fkpp[k][:, s]   # t entre 2,5 et 6
                for sig in sigmas:
                    a_obs = bruiter(a, sig, rng)
                    for mdl in modeles:
                        if mdl == "diffusion":
                            sh, _ = epicentre_diffusion(spec_p, a_obs)
                        else:
                            sh, _ = epicentre_bibliotheque(lib_p, a_obs)
                        cle = (gen, mdl, p, sig)
                        res.setdefault(cle, []).append((sh == s, mod[sh] == mod[s], Dg[s, sh]))
    return {k: np.array(v, float) for k, v in res.items()}

def tableau(res, gen, mdl, ps, sigmas, col=0):
    """Matrice (p x sigma) des taux (col=0 : région exacte ; 1 : bon module ; 2 : distance)."""
    return np.array([[res[(gen, mdl, p, s)][:, col].mean() for s in sigmas] for p in ps])

# ---------------------------------------------------------------------------
# 3. Extension : loi a posteriori de l'épicentre
# ---------------------------------------------------------------------------
def posterior_epicentre(spec, a_obs, sigma_bruit, t_grille=T_GRILLE, beta=1.0):
    """Modèle a_obs = c * a_s(t) + b + bruit N(0, sigma^2 I), loi a priori uniforme sur
    s et sur t (grille) ; (b, c) profilés par moindres carrés.
    p(s | a_obs) ∝ Σ_t exp(-RSS(s,t) / (2 sigma^2))."""
    n = len(a_obs); logv = []
    for t in t_grille:
        A = cartes_diffusion(spec, t, beta)
        Ac = A - A.mean(0); yc = a_obs - a_obs.mean()
        c = (Ac.T @ yc)/np.sum(Ac**2, 0)
        rss = np.sum((yc[:, None] - Ac*c)**2, 0)
        logv.append(-rss/(2*sigma_bruit**2))
    logv = np.array(logv)
    m = logv.max()
    post = np.exp(logv - m).sum(0); return post/post.sum()

def ensemble_credible(post, niveau=0.9):
    o = np.argsort(post)[::-1]; k = np.searchsorted(np.cumsum(post[o]), niveau) + 1
    return o[:k]

def experience_bayes(sigmas=(0.1, 0.25, 0.5, 1.0), R=300, seed=GRAINE + 1):
    rng = np.random.default_rng(seed)
    W, mod = connectome_modulaire(); L = laplacien(W); spec = spectre(L)
    out = {}
    for sig in sigmas:
        cov, taille, pmax, mod_ok = [], [], [], []
        for r in range(R):
            s = int(rng.integers(80)); t = rng.uniform(1, 8)
            a = cartes_diffusion(spec, t)[:, s]; sd = sig*a.std()
            a_obs = a + sd*rng.standard_normal(80)
            post = posterior_epicentre(spec, a_obs, sd)
            E = ensemble_credible(post, 0.9)
            cov.append(s in E); taille.append(len(E)); pmax.append(post.max())
            mod_ok.append(post[mod == mod[s]].sum())
        out[sig] = dict(couverture=np.mean(cov), taille=np.median(taille),
                        pmax=np.median(pmax), proba_module=np.mean(mod_ok))
    return out

# ---------------------------------------------------------------------------
# 4. Figures (toutes enregistrées dans figures/)
# ---------------------------------------------------------------------------
def figures_toutes(dossier="figures", R=150):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(dossier, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
    chrono = {}
    W, mod = connectome_modulaire(); L = laplacien(W); spec = spectre(L); mu, U = spec
    n = len(W)

    # Figure 1 : connectome et spectre
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    im = ax[0].imshow(W, cmap="viridis"); ax[0].set_title("Matrice de poids $W$ (80 régions)")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    ax[1].plot(np.arange(1, n+1), mu, ".", ms=4)
    ax[1].plot(np.arange(1, 5), mu[:4], "o", mfc="none", c="C3", label=r"$\mu_1,\dots,\mu_4$")
    ax[1].set_xlabel("rang $k$"); ax[1].set_ylabel(r"$\mu_k$"); ax[1].legend()
    ax[1].set_title("Spectre du laplacien")
    fig.tight_layout(); fig.savefig(f"{dossier}/fig1_connectome.pdf"); plt.close(fig)

    # Figure 2 : diffusion depuis la région 3 et modes propres
    x0 = np.zeros(n); x0[3] = 1
    ts = np.geomspace(0.01, 30, 200)
    Xt = np.array([U @ (np.exp(-mu*t)*(U.T @ x0)) for t in ts])
    Xc = Xt - Xt.mean(1, keepdims=True)
    proj = Xc @ U[:, :6]
    part = proj**2/np.sum(Xc**2, 1, keepdims=True)
    masse = Xt.sum(1)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for j, (lab, idx) in enumerate([("module de la graine", mod == 0), ("autres modules", mod != 0)]):
        ax[0].plot(ts, Xt[:, idx].mean(1), label=lab)
    ax[0].plot(ts, Xt[:, 3], "k--", label="région 3 (graine)")
    ax[0].axhline(1/n, c="gray", lw=0.6); ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_ylim(1e-5, 1.5); ax[0].set_xlabel("$t$"); ax[0].set_ylabel("concentration moyenne")
    ax[0].legend(fontsize=7); ax[0].set_title("Diffusion $x'=-Lx$, graine en 3")
    ax[1].plot(ts, part[:, 1:4].sum(1), label=r"part de $u_2,u_3,u_4$")
    ax[1].plot(ts, part[:, 4:].sum(1), label="part des modes 5–6")
    ax[1].plot(ts, 1 - (part[:, 1:].sum(1)), label="modes $k\\geq 7$")
    ax[1].set_xscale("log"); ax[1].set_xlabel("$t$"); ax[1].set_ylabel("fraction d'énergie (centrée)")
    ax[1].legend(fontsize=7); ax[1].set_title("Dominance des premiers modes")
    fig.tight_layout(); fig.savefig(f"{dossier}/fig2_modes.pdf"); plt.close(fig)
    chrono["masse_conservee_ecart_max"] = float(np.abs(masse - 1).max())
    chrono["min_x"] = float(Xt.min())

    # Figure 3 : Fisher-KPP, ordre d'arrivée et vérification t^d(s,j)
    Dg = distances(W); s = 3; alpha, beta, eps = 2.0, 1.0, 0.01
    tt = np.linspace(0, 10, 801)[1:]
    X, _ = fkpp_toutes_graines(L, alpha, beta, eps, tt, dt=0.005, X0=eps*np.eye(n)[:, [s]])
    X = X[:, :, 0]
    t_demi = np.array([tt[np.argmax(X[:, j] >= 0.5)] if X[-1, j] >= 0.5 else np.nan for j in range(n)])
    # coefficients théoriques N_j (somme des produits de poids sur les plus courts chemins)
    d = Dg[s].astype(int); N = np.zeros(n); N[s] = 1.0
    for k in range(1, d.max()+1):
        for j in np.flatnonzero(d == k):
            N[j] = sum(W[j, i]*N[i] for i in np.flatnonzero((d == k-1) & (W[j] > 0)))
    from math import factorial
    tpetit = np.geomspace(1e-4, 1e-1, 40)
    Xp, _ = fkpp_toutes_graines(L, alpha, beta, eps, tpetit, dt=1e-5, X0=eps*np.eye(n)[:, [s]])
    Xp = Xp[:, :, 0]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for k, c in zip(range(0, d.max()+1), ["k", "C0", "C1", "C2", "C3", "C4"]):
        j = np.flatnonzero(d == k)[0]
        ax[0].loglog(tpetit, Xp[:, j], c=c, label=f"$d={k}$ (région {j})")
        th = eps*(beta*tpetit)**k*N[j]/factorial(k)
        ax[0].loglog(tpetit, th, c=c, ls=":")
    ax[0].set_xlabel("$t$"); ax[0].set_ylabel("$x_j(t)$"); ax[0].set_ylim(1e-30, 1)
    ax[0].legend(fontsize=6.5); ax[0].set_title(r"Temps courts : $x_j\sim\varepsilon N_j(\beta t)^d/d!$")
    jit = np.random.default_rng(0).uniform(-0.12, 0.12, n)
    ax[1].scatter(d + jit, t_demi, s=10, c=mod, cmap="tab10", vmax=9)
    ax[1].set_xlabel("distance $d(s,j)$ à la graine"); ax[1].set_ylabel(r"temps où $x_j=1/2$")
    ax[1].set_title("Fisher–KPP : ordre d'arrivée")
    fig.tight_layout(); fig.savefig(f"{dossier}/fig3_fkpp.pdf"); plt.close(fig)
    ratio = Xp[0, :]/(eps*(beta*tpetit[0])**d*N/np.array([factorial(int(k)) for k in d]))
    chrono["ratio_temps_courts"] = (float(ratio.min()), float(ratio.max()))
    chrono["corr_tdemi_distance"] = float(np.corrcoef(d[~np.isnan(t_demi)], t_demi[~np.isnan(t_demi)])[0, 1])

    # Figure 4 : exemple de problème inverse (score par région)
    rng = np.random.default_rng(GRAINE)
    a = cartes_diffusion(spec, 3.0)[:, 3]; a_obs = bruiter(a, 0.25, rng)
    sh, S = epicentre_diffusion(spec, a_obs)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ax[0].bar(np.arange(n), a_obs, color=[f"C{m}" for m in mod]); ax[0].axvline(3, c="k", ls="--", lw=0.8)
    ax[0].set_title(r"Carte observée ($t=3$, bruit 25 %)"); ax[0].set_xlabel("région")
    ax[1].plot(np.arange(n), S, ".-", lw=0.6, ms=3); ax[1].plot(sh, S[sh], "r*", ms=10, label=f"estimé : {sh}")
    ax[1].axvline(3, c="k", ls="--", lw=0.8, label="vrai : 3"); ax[1].legend(fontsize=7)
    ax[1].set_title("Score $\\max_t \\rho(a_s(t), a^{obs})$"); ax[1].set_xlabel("graine candidate $s$")
    fig.tight_layout(); fig.savefig(f"{dossier}/fig4_inverse.pdf"); plt.close(fig)

    # Étude principale
    sigmas = (0, 0.1, 0.25, 0.5, 1.0, 1.5); ps = (0, 0.1, 0.2, 0.3)
    t0 = time.time(); res = experience(sigmas, ps, R=R); chrono["etude_poids_s"] = time.time() - t0
    t0 = time.time(); res_ar = experience(sigmas, ps, R=R, mode="aretes", seed=GRAINE + 7)
    chrono["etude_aretes_s"] = time.time() - t0

    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=True)
    for gi, (gen, mdl, tit) in enumerate([("diffusion", "diffusion", "généré : diffusion\ninversé : diffusion"),
                                           ("fkpp", "diffusion", "généré : Fisher–KPP\ninversé : diffusion"),
                                           ("fkpp", "fkpp", "généré : Fisher–KPP\ninversé : Fisher–KPP")]):
        T = tableau(res, gen, mdl, ps, sigmas, 0); Tm = tableau(res, gen, mdl, ps, sigmas, 1)
        for i, p in enumerate(ps):
            ax[gi].plot(sigmas, T[i], "o-", ms=3, c=f"C{i}", label=f"$p={int(100*p)}$ %")
            ax[gi].plot(sigmas, Tm[i], "s:", ms=2.5, c=f"C{i}")
        ax[gi].set_title(tit, fontsize=8); ax[gi].set_xlabel(r"bruit relatif $\sigma$")
    ax[0].set_ylabel("taux de succès"); ax[0].legend(fontsize=7)
    fig.text(0.5, -0.02, "trait plein : région exacte ;   pointillé : bon module", ha="center", fontsize=7.5)
    fig.tight_layout(); fig.savefig(f"{dossier}/fig5_succes.pdf", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for gi, gen in enumerate(("diffusion", "fkpp")):
        for k, (R_, lab) in enumerate([(res, "poids perturbés"), (res_ar, "arêtes déplacées")]):
            T = tableau(R_, gen, gen, ps, sigmas, 0)
            ax[gi].plot([100*p for p in ps], T[:, 2], "o-" if k == 0 else "s--", label=f"{lab}, exact")
            T = tableau(R_, gen, gen, ps, sigmas, 1)
            ax[gi].plot([100*p for p in ps], T[:, 2], "o:" if k == 0 else "s:", label=f"{lab}, module")
        ax[gi].set_xlabel("perturbation du connectome $p$ (%)")
        ax[gi].set_title(f"générateur = modèle inverse = {gen}\n(bruit $\\sigma$ = 0,25)", fontsize=8)
    ax[0].set_ylabel("taux de succès"); ax[0].legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(f"{dossier}/fig6_connectome.pdf"); plt.close(fig)

    # Extension bayésienne
    t0 = time.time(); bay = experience_bayes(); chrono["bayes_s"] = time.time() - t0
    a_obs = bruiter(cartes_diffusion(spec, 3.0)[:, 3], 0.5, np.random.default_rng(GRAINE + 3))
    post = posterior_epicentre(spec, a_obs, 0.5*cartes_diffusion(spec, 3.0)[:, 3].std())
    fig, ax = plt.subplots(figsize=(7.2, 2.3))
    ax.bar(np.arange(n), post, color=[f"C{m}" for m in mod]); ax.axvline(3, c="k", ls="--", lw=0.8)
    E = ensemble_credible(post); ax.set_xlabel("région $s$"); ax.set_ylabel(r"$p(s\mid a^{obs})$")
    ax.set_title(f"Loi a posteriori de l'épicentre (bruit 50 %) ; ensemble crédible à 90 % : {len(E)} régions")
    fig.tight_layout(); fig.savefig(f"{dossier}/fig7_bayes.pdf"); plt.close(fig)
    return dict(res=res, res_ar=res_ar, bayes=bay, chrono=chrono, sigmas=sigmas, ps=ps)

if __name__ == "__main__" and not globals().get("_SOUS_SAGE", False):
    import platform, scipy
    print("Python", platform.python_version(), "| NumPy", np.__version__, "| SciPy", scipy.__version__)
    # Validation : la version vectorisée donne le même score que epicentre() du cours
    W, mod = connectome_modulaire(); L = laplacien(W)
    x0 = np.zeros(80); x0[3] = 1
    a = atrophie(L, x0, t=2.0) + np.random.default_rng(6).normal(0, 0.02, 80)
    s1, S1 = epicentre(L, a, (1.0, 2.0, 4.0)); s2, S2 = epicentre_diffusion(spectre(L), a, (1.0, 2.0, 4.0))
    print("validation epicentre :", s1, s2, "écart max des scores", np.abs(S1 - S2).max())
    t0 = time.time(); out = figures_toutes(); print("temps total", round(time.time() - t0, 1), "s")
    print(out["chrono"])
    for gen, mdl in [("diffusion", "diffusion"), ("fkpp", "diffusion"), ("fkpp", "fkpp")]:
        for col, nom in [(0, "exact"), (1, "module"), (2, "distance")]:
            print(gen, mdl, nom); print(np.round(tableau(out["res"], gen, mdl, out["ps"], out["sigmas"], col), 3))
    for gen in ("diffusion", "fkpp"):
        print("ARETES", gen); print(np.round(tableau(out["res_ar"], gen, gen, out["ps"], out["sigmas"], 0), 3))
        print(np.round(tableau(out["res_ar"], gen, gen, out["ps"], out["sigmas"], 1), 3))
    print(out["bayes"])
    np.save("resultats.npy", out, allow_pickle=True)

def reference_naive(sigmas=(0, 0.1, 0.25, 0.5, 1.0, 1.5), R=600, seed=GRAINE + 11, alpha=2.0, eps=0.01):
    """Estimateur naïf : épicentre = région la plus atrophiée (argmax de la carte observée)."""
    rng = np.random.default_rng(seed)
    W, mod = connectome_modulaire(); L = laplacien(W); spec = spectre(L)
    lib = bibliotheque_fkpp(L, alpha, 1.0, eps, T_FKPP, dt=0.005)
    out = {"diffusion": [], "fkpp": []}
    for sig in sigmas:
        ok = {"diffusion": 0, "fkpp": 0}
        for r in range(R):
            s = int(rng.integers(80))
            for gen in ok:
                a = cartes_diffusion(spec, rng.uniform(1, 8))[:, s] if gen == "diffusion" \
                    else lib[rng.integers(6, 21)][:, s]
                ok[gen] += int(np.argmax(bruiter(a, sig, rng)) == s)
        for gen in ok: out[gen].append(ok[gen]/R)
    return out
