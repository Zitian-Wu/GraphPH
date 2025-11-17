from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Dict, Tuple, List, Literal, Optional, Any

import time
import json

import numpy as np
import jax
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS

ClassID = Literal["1G", "2G_11", "2G_37"]
SimMode = Literal["anchored", "iid"]
Reorder = Literal["labels", "spectral", "rowsum", "none"]

@dataclass
class SimConfig:
    n: int = 150
    S: int = 10
    sigma: float = 0.25
    delta: float = 2.0
    trans_sd: float = 0.2
    scale_lo: float = 0.9
    scale_hi: float = 1.1
    jitter_sd: float = 0.07
    seed: int = 2025


@dataclass
class FitConfig:
    m: int = 3
    kappa: float = 6.0
    alpha: float = 0.2
    num_warmup: int = 600
    num_samples: int = 800
    num_chains: int = 1
    target_accept: float = 0.9
    seed: int = 2025
    dense_mass: bool = True


@dataclass
class SaveConfig:
    out_dir: str = "runs/run_001"
    thin_every: int = 10
    save_full_lambda: bool = False


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

# ---------------------------
# Simulation (component setups)
# ---------------------------

def _class_mixture_setup(cid: ClassID, delta: float):
    if cid == "1G":
        weights = [1.0]
        mus = [np.array([0.0, 0.0])]
    elif cid == "2G_11":
        weights = [0.5, 0.5]
        mus = [np.array([-delta/2, 0.0]), np.array([+delta/2, 0.0])]
    elif cid == "2G_37":
        weights = [0.3, 0.7]
        mus = [np.array([-delta/2, 0.0]), np.array([+delta/2, 0.0])]
    else:
        raise ValueError("Unknown class_id")
    return weights, mus

# -------- IID subjects (no shared node identities) --------

def sample_subject_iid(cid: ClassID, cfg: SimConfig, seed: int) -> np.ndarray:
    """
    One subject as a fresh mixture sample, plus random rigid+scale transforms.
    """
    rng = _rng(seed)
    n, sigma, delta = cfg.n, cfg.sigma, cfg.delta
    weights, mus = _class_mixture_setup(cid, delta)

    comp = rng.choice(len(weights), size=n, p=weights)
    X = np.zeros((n,2), dtype=float)
    for k, mu in enumerate(mus):
        idx = np.where(comp == k)[0]
        X[idx] = mu + rng.normal(0.0, sigma, size=(len(idx), 2))

    # rotation + scale + translation (distances change if scale != 1)
    theta = rng.uniform(0.0, 2*np.pi)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    s = rng.uniform(cfg.scale_lo, cfg.scale_hi)
    t = rng.normal(0.0, cfg.trans_sd, size=2)
    return s * (X @ R.T) + t

# -------- Anchored subjects (shared node identities) --------

def make_class_template(cid: ClassID, cfg: SimConfig, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns template point set X0 (n×2) and per-node component labels comp0 (n,)
    """
    rng = _rng(seed)
    n, sigma, delta = cfg.n, cfg.sigma, cfg.delta
    weights, mus = _class_mixture_setup(cid, delta)
    comp0 = rng.choice(len(weights), size=n, p=weights)
    X0 = np.zeros((n,2), dtype=float)
    for k, mu in enumerate(mus):
        idx = np.where(comp0 == k)[0]
        X0[idx] = mu + rng.normal(0.0, sigma, size=(len(idx), 2))
    return X0, comp0

def sample_subject_from_template(X0: np.ndarray, cfg: SimConfig, seed: int) -> np.ndarray:
    """
    Subject is template + small per-node jitter. Keep rigid transforms off to stabilize Δt.
    """
    rng = _rng(seed)
    return X0 + rng.normal(0.0, cfg.jitter_sd, size=X0.shape)

def make_three_oracles_with_subjects(simcfg: SimConfig) -> Dict[str, Any]:
    """
    Three related *population oracles* μ^(g) in R^2, then S noisy subjects per group.
    Oracle construction (using 0-based indices; matches the paper text when n=150):
      • Group 1 ("1G"):      μ^(1)_i ~ N(0, σ^2 I2)               for i=0..n-1
      • Group 2 ("2G_11"):   μ^(2)_i = μ^(1)_i                    for i=0..k12-1
                             μ^(2)_i ~ N(Δ·1, σ^2 I2)             for i=k12..n-1
      • Group 3 ("2G_37"):   μ^(3)_i = μ^(1)_i                    for i=0..k13-1
                             μ^(3)_i ~ N(Δ·1, σ^2 I2)             for i=k13..k12-1
                             μ^(3)_i = μ^(2)_i                    for i=k12..n-1
    with k12 = round(0.50*n), k13 = round(0.30*n).

    Subjects: for each group g and subject s=1..S, draw
        X^{(g,s)} = μ^(g) + ε^{(g,s)},   ε^{(g,s)}_i ~ i.i.d. N(0, jitter_sd^2 I2).

    NOTE: Here σ = simcfg.sigma (component sd), jitter_sd = simcfg.jitter_sd (subject noise).
    """
    rng = np.random.default_rng(simcfg.seed)
    n, S, delta = simcfg.n, simcfg.S, simcfg.delta
    comp_sd  = simcfg.sigma           # CHANGED: component (mixture) sd
    noise_sd = simcfg.jitter_sd       # CHANGED: subject jitter sd

    # enforce splits; for n=150 → k12=75, k13=45
    k12 = int(round(0.50 * n))
    k13 = int(round(0.30 * n))
    k13 = min(k13, k12)  # just in case of extreme rounding

    # --- Oracles (variance = σ^2) ---
    mu1 = rng.normal(0.0, comp_sd, size=(n, 2))                   # CHANGED
    mu2 = mu1.copy()
    if k12 < n:
        mu2[k12:n] = rng.normal(delta, comp_sd, size=(n - k12, 2))  # CHANGED
    mu3 = np.empty_like(mu1)
    mu3[0:k13] = mu1[0:k13]
    if k13 < k12:
        mu3[k13:k12] = rng.normal(delta, comp_sd, size=(k12 - k13, 2))  # CHANGED
    if k12 < n:
        mu3[k12:n] = mu2[k12:n]  # μ^(3) copies μ^(2) on the tail

    # --- Labels (bookkeeping; not used by MST) ---
    # 0 = base N(0,σ^2 I), 1 = shifted N(Δ·1, σ^2 I), 10 = copied from G1, 20 = copied from G2
    lab1 = np.zeros(n, dtype=int)
    lab2 = np.empty(n, dtype=int); lab2[:k12] = 10;  lab2[k12:] = 1 if k12 < n else []
    lab3 = np.empty(n, dtype=int); lab3[:k13] = 10
    if k13 < k12: lab3[k13:k12] = 1
    if k12 < n:   lab3[k12:n]   = 20

    def subjects_from(mu: np.ndarray) -> List[np.ndarray]:
        return [mu + rng.normal(0.0, noise_sd, size=mu.shape) for _ in range(S)]  # CHANGED

    subs1 = subjects_from(mu1)
    subs2 = subjects_from(mu2)
    subs3 = subjects_from(mu3)

    # sanity: identical fractions at the oracle (useful for logging)
    same12 = float(np.mean(np.all(mu1 == mu2, axis=1)))
    same13 = float(np.mean(np.all(mu1 == mu3, axis=1)))
    same23 = float(np.mean(np.all(mu2 == mu3, axis=1)))

    return {
        "1G":     {"oracle": mu1, "subjects": subs1, "labels": lab1},
        "2G_11":  {"oracle": mu2, "subjects": subs2, "labels": lab2},
        "2G_37":  {"oracle": mu3, "subjects": subs3, "labels": lab3},
        "meta": {
            "n": n, "S": S, "delta": delta,
            "component_sd": comp_sd,           # CHANGED (renamed for clarity)
            "noise_sd": noise_sd,              # CHANGED: subject jitter sd
            "splits": {"k12": k12, "k13": k13},
            "oracle_identical_fractions": {"1G-2G_11": same12, "1G-2G_37": same13, "2G_11-2G_37": same23}
        }
    }

def simulate_dataset(simcfg: SimConfig,
                     classes: Tuple[ClassID, ...] = ("1G","2G_11","2G_37"),
                     mode: SimMode = "anchored"):
    """
    Returns:
      results: {cid: {s: {"X":..., "events":...}}}
      anchors: {cid: {"X0":..., "labels":...}} or {}

    Notes:
      • anchored: uses pre-generated subjects from make_three_oracles_with_subjects(simcfg)
                  (ε ~ N(0, simcfg.sigma^2 I)) exactly as described in the paper text.
      • iid:      unchanged; uses your existing sample_subject_iid()
    """
    results: Dict[ClassID, Dict[int, Dict]] = {cid: {} for cid in classes}
    anchors: Dict[ClassID, Dict] = {}

    if mode == "anchored":
        # Use the new generator (builds three related oracles + S noisy subjects per group)
        bundle = make_three_oracles_with_subjects(simcfg)

        for cid in classes:
            # Adapt to the old "anchors" shape: expose oracle as "X0"
            oracle = bundle[cid]["oracle"]
            labels = bundle[cid]["labels"]
            anchors[cid] = {"X0": oracle, "labels": labels}

            # Subjects are already jittered with N(0, sigma^2 I)
            subs = bundle[cid]["subjects"]
            # Respect simcfg.S in case subs length differs (shouldn't, but safe)
            for s, Xs in enumerate(subs[:simcfg.S]):
                ev = mst_h0_events(Xs)
                results[cid][s] = {"X": Xs, "events": ev}

    elif mode == "iid":
        # Keep your previous iid pathway unchanged
        for c, cid in enumerate(classes):
            for s in range(simcfg.S):
                Xs = sample_subject_iid(cid, simcfg, seed=simcfg.seed + 1000*c + s)
                ev = mst_h0_events(Xs)
                results[cid][s] = {"X": Xs, "events": ev}
    else:
        raise ValueError("mode must be 'anchored' or 'iid'")

    return results, anchors

class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a: int, b: int) -> bool:
        a, b = self.find(a), self.find(b)
        if a == b: return False
        if self.r[a] < self.r[b]:
            self.p[a] = b
        elif self.r[a] > self.r[b]:
            self.p[b] = a
        else:
            self.p[b] = a
            self.r[a] += 1
        return True
    
def mst_h0_events(X: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Returns sufficient statistics for H0 MST race:
      pairs: (M,2) i<j
      e_idx: (r,) indices of chosen MST edges among pairs
      delta_t: (r,) increments of death time (edge_len/2 increments)
      E_counts: (M,) 1-hot counts for chosen edges
      A_weights: (M,) sum_i delta_t[i] * 1{edge∈A_i}
      w_mst: (r,) MST edge lengths
    """
    n = X.shape[0]
    I, J = np.triu_indices(n, k=1)
    pairs = np.stack([I, J], axis=1).astype(np.int32)
    M = len(I)
    # pairwise distances
    Dx, Dy = X[:,0], X[:,1]
    w = np.sqrt((Dx[I]-Dx[J])**2 + (Dy[I]-Dy[J])**2)

    # Kruskal
    order = np.argsort(w, kind="mergesort")
    Iord, Jord, word = I[order], J[order], w[order]
    tab = -np.ones((n,n), dtype=np.int32)
    tab[I, J] = np.arange(M, dtype=np.int32)

    dsu = DSU(n)
    e_pairs, e_weights, comp_labels_list = [], [], []
    for u, v, ww in zip(Iord, Jord, word):
        fu, fv = dsu.find(u), dsu.find(v)
        if fu != fv:
            comp = np.array([dsu.find(t) for t in range(n)], dtype=np.int32)
            comp_labels_list.append(comp)
            dsu.union(fu, fv)
            e_pairs.append((u, v))
            e_weights.append(ww)
            if len(e_pairs) == n-1:
                break

    e_pairs = np.array(e_pairs, dtype=np.int32)
    e_weights = np.array(e_weights, dtype=float)
    d = e_weights / 2.0
    delta_t = np.empty_like(d)
    delta_t[0] = d[0]
    if len(d) > 1:
        delta_t[1:] = d[1:] - d[:-1]

    # indices of chosen edges
    e_idx = np.array([tab[min(u,v), max(u,v)] for (u,v) in e_pairs], dtype=np.int32)

    # vectors instead of A-matrix
    E_counts = np.zeros(M, dtype=np.float32)
    E_counts[e_idx] = 1.0
    A_weights = np.zeros(M, dtype=np.float32)
    for i, comp in enumerate(comp_labels_list):
        mask = (comp[I] != comp[J])
        if np.any(mask):
            A_weights[mask] += float(delta_t[i])

    return dict(pairs=pairs, e_idx=e_idx, delta_t=delta_t,
                E_counts=E_counts, A_weights=A_weights,
                w_mst=e_weights)


# ---------------------------
# Integrity checks
# ---------------------------

def check_events_integrity(ev):
    n = int(0.5 + 0.5*(1 + np.sqrt(1 + 8*len(ev["pairs"]))))
    assert len(ev["e_idx"]) == n-1, "MST must have n-1 edges"
    assert abs(ev["E_counts"].sum() - (n-1)) < 1e-6, "E_counts sum mismatch"
    # rebuild A_weights naively and compare
    I, J = ev["pairs"][:,0], ev["pairs"][:,1]
    dsu = DSU(n)
    A_check = np.zeros_like(ev["A_weights"])
    for i, e_lin in enumerate(ev["e_idx"]):
        comp = np.array([dsu.find(t) for t in range(n)])
        mask = (comp[I] != comp[J])
        A_check[mask] += float(ev["delta_t"][i])
        u, v = int(I[e_lin]), int(J[e_lin])
        dsu.union(u, v)
    assert np.allclose(A_check, ev["A_weights"], atol=1e-6), "A_weights mismatch"

def check_vector_ll_equals_naive(ev, Lambda_hat):
    # vector form
    I,J = ev["pairs"][:,0], ev["pairs"][:,1]
    lam = Lambda_hat[I,J]
    ll_vec = float(np.sum(ev["E_counts"]*np.log(np.clip(lam,1e-12)) - ev["A_weights"]*lam))
    # naive step-by-step
    n = int(0.5 + 0.5*(1 + np.sqrt(1 + 8*len(ev["pairs"]))))
    dsu = DSU(n); ll_naive = 0.0
    for i, e_lin in enumerate(ev["e_idx"]):
        comp = np.array([dsu.find(t) for t in range(n)])
        mask = (comp[I] != comp[J])
        R = float(np.sum(lam[mask]))
        ll_naive += np.log(max(lam[e_lin], 1e-12)) - ev["delta_t"][i]*R
        u, v = int(I[e_lin]), int(J[e_lin]); dsu.union(u, v)
    assert abs(ll_vec - ll_naive) < 1e-6, (ll_vec, ll_naive)

def build_h0_potential_pooled(n: int, m: int, kappa: float, alpha: float,
                              iu: Tuple[np.ndarray, np.ndarray],
                              E_total: jnp.ndarray, A_total: jnp.ndarray):
    """
    Negative log posterior:
      -[ <E_total, log λ> - <A_total, λ> + logprior(Z) + logJ(softplus) ]
    with Λ = ZZᵀ and Z = softplus(Φ).
    """
    iu0, iu1 = jnp.array(iu[0]), jnp.array(iu[1])
    E_total = E_total.astype(jnp.float32)
    A_total = A_total.astype(jnp.float32)
    eps = 1e-12

    def sum_upper_triangle_log(G: jnp.ndarray) -> jnp.ndarray:
        iu_bar = jnp.triu_indices(G.shape[0], 0)  # include diagonal
        return jnp.sum(jnp.log(jnp.clip(G[iu_bar], eps)))

    def potential_fn(phi_flat: jnp.ndarray) -> jnp.ndarray:
        Phi = phi_flat.reshape((n, m))
        Z = jax.nn.softplus(Phi)
        G = Z @ Z.T
        lam_vec = G[iu0, iu1]
        loglam_vec = jnp.log(jnp.clip(lam_vec, eps))

        loglik = jnp.dot(E_total, loglam_vec) - jnp.dot(A_total, lam_vec)

        quad = -0.5 * kappa * jnp.sum(Z * Z)
        barrier = alpha * sum_upper_triangle_log(G)
        logprior = quad + barrier

        logJ = jnp.sum(jax.nn.log_sigmoid(Phi))
        return -(loglik + logprior + logJ)

    return potential_fn

def _pool_class_vectors(subj_dict: Dict[int, Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sum E and A across subjects within a class (pairs must match shape).
    Returns pairs, E_total, A_total.
    """
    first = next(iter(subj_dict.values()))
    pairs = first["events"]["pairs"]
    M = pairs.shape[0]
    E_tot = np.zeros(M, dtype=np.float32)
    A_tot = np.zeros(M, dtype=np.float32)
    for pack in subj_dict.values():
        ev = pack["events"]
        assert ev["pairs"].shape == pairs.shape, "pairs mismatch across subjects"
        E_tot += ev["E_counts"].astype(np.float32)
        A_tot += ev["A_weights"].astype(np.float32)
    return pairs, E_tot, A_tot

def _extract_phi_samples(samples):
    # numpyro returns dict when using potential_fn; fallback to array if already dense
    if isinstance(samples, dict):
        if "z" in samples:             # common in potential_fn mode
            return samples["z"]
        # else take the first item deterministically
        return samples[next(iter(samples.keys()))]
    return samples  # already an array

def fit_class_lambda_via_nuts(subj_dict: Dict[int, Dict], fitcfg: FitConfig,
                              save_thin_every: int = 10,
                              return_draws: bool = True):
    """
    Fit one Λ per class using pooled H0 vectors via NUTS.
    Returns (Lambda_hat, time_sec, mcmc, draws) where draws may include thinned Lambda and Z.
    """
    pairs, E_tot, A_tot = _pool_class_vectors(subj_dict)
    n = int(0.5 + 0.5*(1 + np.sqrt(1 + 8*len(pairs))))
    iu = np.triu_indices(n, k=1)

    potential_fn = build_h0_potential_pooled(
        n=n, m=fitcfg.m, kappa=fitcfg.kappa, alpha=fitcfg.alpha,
        iu=iu, E_total=jnp.array(E_tot), A_total=jnp.array(A_tot)
    )

    kernel = NUTS(potential_fn=potential_fn,
                  target_accept_prob=fitcfg.target_accept,
                  dense_mass=fitcfg.dense_mass)

    phi0 = jnp.zeros((n, fitcfg.m)).ravel()
    init_params = phi0 if fitcfg.num_chains == 1 else jnp.tile(phi0[None,:], (fitcfg.num_chains,1))

    mcmc = MCMC(kernel,
                num_warmup=fitcfg.num_warmup,
                num_samples=fitcfg.num_samples,
                num_chains=fitcfg.num_chains,
                chain_method=("vectorized" if fitcfg.num_chains>1 else "sequential"),
                progress_bar=True)

    key = jax.random.PRNGKey(fitcfg.seed + 9001)
    t0 = time.time()
    mcmc.run(key, init_params=init_params)
    elapsed = time.time() - t0

    raw = mcmc.get_samples(group_by_chain=False)
    phi_samples = _extract_phi_samples(raw)         # (T, n*m)
    T = phi_samples.shape[0]

    def phi_to_Z(phi_flat):
        Phi = phi_flat.reshape((n, fitcfg.m))
        return jax.nn.softplus(Phi)

    def phi_to_Lam(phi_flat):
        Z = phi_to_Z(phi_flat)
        return (Z @ Z.T)

    # Posterior mean Λ̂
    Lam_draws = jax.vmap(phi_to_Lam)(phi_samples)   # (T, n, n)
    Lambda_hat = np.array(Lam_draws.mean(axis=0))

    draws = {}
    if return_draws:
        thin_idx = np.arange(0, T, max(1, save_thin_every))
        Z_draws_thin   = jax.vmap(phi_to_Z)(phi_samples[thin_idx])   # (T_thin, n, m)
        Lam_draws_thin = Lam_draws[thin_idx]                         # (T_thin, n, n)
        draws = {
            "phi_thin": np.array(phi_samples[thin_idx]),
            "Z_thin":   np.array(Z_draws_thin),
            "Lambda_thin": np.array(Lam_draws_thin),
            "thin_every": int(save_thin_every),
            "nsamples": int(T),
        }

    return Lambda_hat, float(elapsed), mcmc, draws

def train_per_class_Lambda(results: Dict[ClassID, Dict[int, Dict]],
                           fitcfg: FitConfig,
                           save_thin_every: int = 10):
    """
    Fit Λ̂ for each class.
    Returns models: {cid: {"Lambda_hat":..., "time_sec":..., "mcmc":..., "draws": {...}}}
    """
    models = {}
    for cid, subj_dict in results.items():
        Lam_hat, t_sec, mcmc, draws = fit_class_lambda_via_nuts(subj_dict, fitcfg,
                                                                save_thin_every=save_thin_every,
                                                                return_draws=True)
        models[cid] = {"Lambda_hat": Lam_hat, "time_sec": t_sec, "mcmc": mcmc, "draws": draws}
        print(f"[train] class={cid}  time={t_sec:.2f}s")
    return models

# ---------------------------
# Predictive log-likelihood & classification
# ---------------------------

def predictive_loglik_subject_given_Lambda(events: Dict[str, np.ndarray], Lambda_hat: np.ndarray) -> float:
    I, J = events["pairs"][:,0], events["pairs"][:,1]
    lam_vec = Lambda_hat[I, J].astype(float)
    E = events["E_counts"].astype(float)
    A = events["A_weights"].astype(float)
    eps = 1e-12
    return float(np.sum(E * np.log(np.maximum(lam_vec, eps)) - A * lam_vec))

def classify_with_class_Lambda(results: Dict[ClassID, Dict[int, Dict]],
                               models: Dict[ClassID, Dict],
                               loo: bool = False,
                               fitcfg_for_loo: Optional[FitConfig] = None):
    """
    Score each subject under each class model and build confusion matrix.
    Optional LOO refit for the subject's true class (expensive).
    """
    classes = list(results.keys())
    idx = {c:i for i,c in enumerate(classes)}
    C = np.zeros((len(classes), len(classes)), dtype=int)
    for cid_true, subj_dict in results.items():
        for s_idx, pack in subj_dict.items():
            ev = pack["events"]
            scores = []
            for cid_m in classes:
                if loo and cid_m == cid_true:
                    excl = {k:v for k,v in subj_dict.items() if k != s_idx}
                    Lam_hat_c, _, _ = fit_class_lambda_via_nuts(excl, fitcfg_for_loo or pack.get("fitcfg", FitConfig()))
                    ll = predictive_loglik_subject_given_Lambda(ev, Lam_hat_c)
                else:
                    ll = predictive_loglik_subject_given_Lambda(ev, models[cid_m]["Lambda_hat"])
                scores.append(ll)
            C[idx[cid_true], int(np.argmax(scores))] += 1
    acc = C.trace() / C.sum()
    return C, acc

def fit_and_classify(results: Dict[ClassID, Dict[int, Dict]], fitcfg: FitConfig, save_thin_every: int = 10):
    models = train_per_class_Lambda(results, fitcfg, save_thin_every=save_thin_every)
    C, acc = classify_with_class_Lambda(results, models, loo=False)
    return models, C, acc

def reorder_perm(Lambda_hat: np.ndarray,
                 labels: Optional[np.ndarray] = None,
                 how: Reorder = "rowsum") -> np.ndarray:
    """
    Returns a permutation of nodes for prettier heatmaps.
      - 'labels': group by labels (then rowsum within each group)
      - 'rowsum': sort by row sums
      - 'spectral': Fiedler vector order (Laplacian eigenvector #2)
      - 'none': identity
    """
    n = Lambda_hat.shape[0]
    Lam = np.array(Lambda_hat, float)
    Lam = (Lam + Lam.T) / 2.0

    if how == "none":
        return np.arange(n)

    if how == "rowsum":
        return np.argsort(Lam.sum(axis=1))

    if how == "labels":
        if labels is None:
            raise ValueError("labels must be provided for how='labels'")
        labels = np.asarray(labels)
        groups = [np.where(labels == lab)[0] for lab in np.unique(labels)]
        def _sort_block(idx):
            block = Lam[np.ix_(idx, idx)]
            return idx[np.argsort(block.sum(axis=1))]
        return np.concatenate([_sort_block(g) for g in groups])

    if how == "spectral":
        d = Lam.sum(axis=1)
        L = np.diag(d) - Lam
        # tiny jitter for numerical stability
        w, V = np.linalg.eigh(L + 1e-8*np.eye(n))
        fied = V[:, 1] if n > 1 else np.zeros(n)
        return np.argsort(fied)

    raise ValueError(f"Unknown reordering: {how}")

def plot_lambda_heatmap(Lambda_hat: np.ndarray,
                        perm: Optional[np.ndarray] = None,
                        title: str = "Λ̂",
                        out_png: Optional[str] = None,
                        logscale: bool = True):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    Lam = np.array(Lambda_hat, float)
    Lam = (Lam + Lam.T) / 2.0
    if perm is not None:
        Lam = Lam[perm][:, perm]
    vmin = max(1e-8, np.percentile(Lam, 0.5))
    vmax = np.percentile(Lam, 99.5)
    norm = LogNorm(vmin=vmin, vmax=vmax) if logscale else Normalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(Lam, origin="lower", interpolation="nearest", norm=norm)
    ax.set_title(title); ax.set_xlabel("nodes"); ax.set_ylabel("nodes")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("λ" + (" [log]" if logscale else ""))
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=160); print("Saved heatmap →", out_png)
    plt.close(fig)

# ---------------------------
# Saving helpers
# ---------------------------

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _save_numpy_dict(npz_path: Path, **arrays):
    np.savez_compressed(npz_path, **arrays)
    print("Saved:", npz_path)

def save_experiment(savecfg: SaveConfig,
                    simcfg: SimConfig,
                    fitcfg: FitConfig,
                    mode: SimMode,
                    results: Dict[ClassID, Dict[int, Dict]],
                    anchors: Dict[ClassID, Dict],
                    models: Dict[ClassID, Dict],
                    C: np.ndarray,
                    acc: float):
    out = Path(savecfg.out_dir)
    _ensure_dir(out)

    # 1) configs + manifest
    meta = {
        "simcfg": asdict(simcfg),
        "fitcfg": asdict(fitcfg),
        "mode": mode,
        "classes": list(results.keys()),
        "accuracy": float(acc),
        "thin_every": savecfg.thin_every
    }
    (out / "manifest.json").write_text(json.dumps(meta, indent=2))
    print("Saved:", out / "manifest.json")

    # 2) anchors (templates)
    anc_dir = out / "anchors"
    _ensure_dir(anc_dir)
    for cid, pack in anchors.items():
        _save_numpy_dict(anc_dir / f"{cid}_template.npz",
                         X0=pack["X0"].astype(np.float32),
                         labels=pack["labels"].astype(np.int32))

    # 3) subjects: clouds + H0 events (per class/per subject)
    subj_dir = out / "subjects"
    _ensure_dir(subj_dir)
    for cid, subj_dict in results.items():
        class_dir = subj_dir / cid
        _ensure_dir(class_dir)
        for s_idx, pack in subj_dict.items():
            ev = pack["events"]
            _save_numpy_dict(class_dir / f"subj_{s_idx:03d}.npz",
                             X=pack["X"].astype(np.float32),
                             pairs=ev["pairs"],
                             e_idx=ev["e_idx"],
                             delta_t=ev["delta_t"].astype(np.float32),
                             E_counts=ev["E_counts"].astype(np.float32),
                             A_weights=ev["A_weights"].astype(np.float32),
                             w_mst=ev["w_mst"].astype(np.float32))

    # 4) models: Λ̂ + posterior draws (thinned) + basic summary
    model_dir = out / "models"
    _ensure_dir(model_dir)
    for cid, m in models.items():
        draws = m.get("draws", {})
        _save_numpy_dict(model_dir / f"{cid}_Lambda_hat.npz",
                         Lambda_hat=m["Lambda_hat"].astype(np.float32))
        if draws:
            _save_numpy_dict(model_dir / f"{cid}_posterior_thin.npz",
                             phi_thin=draws["phi_thin"].astype(np.float32),
                             Z_thin=draws["Z_thin"].astype(np.float32),
                             Lambda_thin=draws["Lambda_thin"].astype(np.float32))
        # optional: write a tiny JSON summary per class
        summary = {
            "time_sec": m["time_sec"],
            "nsamples": int(draws.get("nsamples", 0)),
            "thin_every": int(draws.get("thin_every", savecfg.thin_every))
        }
        (model_dir / f"{cid}_summary.json").write_text(json.dumps(summary, indent=2))

    # 5) confusion matrix + accuracy
    eval_dir = out / "eval"
    _ensure_dir(eval_dir)
    _save_numpy_dict(eval_dir / "confusion.npz", C=C.astype(np.int32), accuracy=np.array([acc], dtype=np.float32))
    with open(eval_dir / "confusion.txt", "w") as fh:
        fh.write("Confusion (rows=true, cols=pred):\n")
        fh.write(np.array2string(C, max_line_width=200))
        fh.write(f"\nAccuracy: {acc:.3f}\n")
    print("Saved:", eval_dir / "confusion.txt")










