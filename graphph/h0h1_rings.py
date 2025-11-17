
"""
H0+H1 simulation on concentric rings
- Single class of subjects, all jittered from one reference concentric-rings cloud
- Uses H0+H1 likelihood with NumPyro NUTS
- Saves:
    * per-class Lambda_hat
    * phi draws
    * trace & ACF plots for selected phi entries
    * latent 2D coordinates
"""

import os, csv, json, math, time, shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from numpyro.infer import MCMC, NUTS
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------- Matplotlib style ----------
HAS_TEX = shutil.which("latex") is not None
mpl.rcParams.update({
    "text.usetex": HAS_TEX,
    "savefig.dpi": 450,     # ≥400 dpi
    "font.size": 18,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

# ============================================================
# Generic helpers
# ============================================================
def segsum(data: jnp.ndarray, segment_ids: jnp.ndarray, num_segments: int):
    """Segmented sum via scatter-add. JIT/grad friendly. Handles ND `data`."""
    segment_ids = segment_ids.astype(jnp.int32)
    out_shape = (int(num_segments),) + data.shape[1:]
    out = jnp.zeros(out_shape, dtype=data.dtype)
    return out.at[segment_ids].add(data)

def _log1mexp(x: jnp.ndarray) -> jnp.ndarray:
    """Stable log(1 - exp(-x)) for x>0."""
    x = jnp.asarray(x)
    log2 = jnp.log(jnp.array(2.0, dtype=x.dtype))
    return jnp.where(x <= log2, jnp.log(-jnp.expm1(-x)), jnp.log1p(-jnp.exp(-x)))

def acf(series: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """Unbiased ACF up to max_lag."""
    x = np.asarray(series, float)
    x = x - x.mean()
    n = x.size
    var = np.dot(x, x) / n
    if var == 0:
        return np.zeros(max_lag+1)
    corr = np.correlate(x, x, mode='full')[n-1 : n+max_lag] / n
    return corr / var

def plot_trace(series: np.ndarray, png_path: Path, xlabel=r"$\mathrm{draw}$", ylabel=r"$\mathrm{value}$"):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(np.arange(len(series)), series, lw=1.0)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(png_path); plt.close(fig)

def plot_acf(series: np.ndarray, png_path: Path, max_lag: int = 100, xlabel=r"$\mathrm{lag}$", ylabel=r"$\mathrm{ACF}$"):
    r = acf(series, max_lag=max_lag)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.vlines(np.arange(len(r)), 0, r, lw=1.2)
    ax.scatter(np.arange(len(r)), r, s=12)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(png_path); plt.close(fig)

# ============================================================
# Pooled H0+H1 features loader
# ============================================================
def _pairs_to_iu(pairs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (iu0, iu1) = the exact i<j pairs (already provided)."""
    return pairs[:,0], pairs[:,1]

def pool_class_h0h1_from_feat_dir(
    feat_dir: str,
    cid: str,
    exclude_undying: bool = True,
    tol: float = 1e-9,   # numerical cushion when comparing to cap
):
    """
    Pool H0+H1 features across subjects of a class in `feat_dir`.
    Expects:
      - index.csv with columns: sid, cid, file
      - per-subject <file>.replace('.npz','_feat.npz') containing:
          pairs, h0_E_counts, h0_A_weights,
          L_e_idx, L_f_idx, L_b, L_d,
          B1_data, B1_indptr, B2_data, B2_indptr,
          vr_cap (optional), divide_by_two (optional)
    """
    base = Path(feat_dir)
    idx = base / "index.csv"
    rows = [row for row in csv.DictReader(idx.open()) if row["cid"] == cid]
    if not rows:
        raise FileNotFoundError(f"No subjects for class {cid} in {idx}")

    # Optional global fallback cap from meta.json
    fallback_cap = None
    meta_path = base / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            vr = (meta.get("distance_info", {}) or {}).get("vr", {})
            if "cap" in vr:
                fallback_cap = float(vr["cap"])
        except Exception:
            pass

    # Pairs template from first subject
    first_feat = base / rows[0]["file"].replace(".npz", "_feat.npz")
    with np.load(first_feat) as z0:
        pairs = z0["pairs"].astype(np.int32)
        M = pairs.shape[0]

    # H0 pools
    E_tot = np.zeros(M, dtype=np.float32)
    A_tot = np.zeros(M, dtype=np.float32)

    # H1 pools
    e_all = []; f_all = []; b_all = []; d_all = []
    B1_edge_idx = []; B1_loop_ids = []
    B2_edge_idx = []; B2_loop_ids = []

    loops_offset = 0  # pooled loop id base

    for row in rows:
        zpath = base / row["file"].replace(".npz", "_feat.npz")
        with np.load(zpath) as z:
            # H0 accumulate (sanity: shared pairs)
            if z["pairs"].shape != pairs.shape or not np.array_equal(z["pairs"], pairs):
                raise AssertionError(f"'pairs' mismatch in {zpath}")
            E_tot += z["h0_E_counts"].astype(np.float32)
            A_tot += z["h0_A_weights"].astype(np.float32)

            # H1 arrays for this subject
            L_b = z["L_b"]; L = int(L_b.shape[0])
            if L == 0:
                continue

            e = z["L_e_idx"].astype(np.int32)
            f = z["L_f_idx"].astype(np.int32)
            b = z["L_b"].astype(np.float32)
            d = z["L_d"].astype(np.float32)

            # Determine effective cap for this subject
            # Prefer subject-level vr_cap; fallback to meta.json
            subj_cap = float(z["vr_cap"]) if ("vr_cap" in z.files) else fallback_cap
            div2 = bool(z["divide_by_two"].item()) if ("divide_by_two" in z.files) else False
            cap_eff = None
            if exclude_undying and (subj_cap is not None):
                cap_eff = subj_cap / (2.0 if div2 else 1.0)

            # Keep only loops that actually die (< cap_eff)
            if cap_eff is not None:
                keep = d < (cap_eff - tol)
            else:
                keep = np.ones_like(d, dtype=bool)

            if not np.any(keep):
                continue

            # Append kept loops
            e_k = e[keep]; f_k = f[keep]; b_k = b[keep]; d_k = d[keep]
            e_all.append(e_k); f_all.append(f_k); b_all.append(b_k); d_all.append(d_k)

            # Map old per-subject loop indices -> new pooled loop IDs
            new_id = -np.ones(L, dtype=np.int32)
            new_id[keep] = loops_offset + np.arange(np.count_nonzero(keep), dtype=np.int32)

            # Append B1/B2 rows only for kept loops, preserving loop IDs
            def _append_csr_rows(data, indptr, dest_edge, dest_loop):
                if data.size == 0:
                    return
                for i in range(L):
                    nid = new_id[i]
                    if nid < 0:
                        continue
                    start, stop = int(indptr[i]), int(indptr[i+1])
                    if stop > start:
                        seg = data[start:stop]
                        dest_edge.append(seg.astype(np.int32, copy=False))
                        dest_loop.append(np.full(seg.size, nid, dtype=np.int32))

            _append_csr_rows(z["B1_data"], z["B1_indptr"], B1_edge_idx, B1_loop_ids)
            _append_csr_rows(z["B2_data"], z["B2_indptr"], B2_edge_idx, B2_loop_ids)

            # Advance pooled loop ID base by number kept
            loops_offset += int(np.count_nonzero(keep))

    # Concatenate pooled arrays (possibly empty)
    e_idx = np.concatenate(e_all, axis=0) if e_all else np.zeros((0,), np.int32)
    f_idx = np.concatenate(f_all, axis=0) if f_all else np.zeros((0,), np.int32)
    b = np.concatenate(b_all, axis=0) if b_all else np.zeros((0,), np.float32)
    d = np.concatenate(d_all, axis=0) if d_all else np.zeros((0,), np.float32)

    B1_edge_idx = (np.concatenate(B1_edge_idx, axis=0) if B1_edge_idx
                   else np.zeros((0,), np.int32))
    B1_loop_ids = (np.concatenate(B1_loop_ids, axis=0) if B1_loop_ids
                   else np.zeros((0,), np.int32))
    B2_edge_idx = (np.concatenate(B2_edge_idx, axis=0) if B2_edge_idx
                   else np.zeros((0,), np.int32))
    B2_loop_ids = (np.concatenate(B2_loop_ids, axis=0) if B2_loop_ids
                   else np.zeros((0,), np.int32))

    h1 = dict(
        e_idx=e_idx, f_idx=f_idx, b=b, d=d,
        B1_edge_idx=B1_edge_idx, B1_loop_ids=B1_loop_ids,
        B2_edge_idx=B2_edge_idx, B2_loop_ids=B2_loop_ids,
    )
    return pairs, E_tot, A_tot, h1

# ============================================================
# H0+H1 potential (edgewise implementation)
# ============================================================
def compress_active_edges(pairs, E_tot, A_tot, h1):
    used = (E_tot > 0) | (A_tot > 0)
    for k in ["e_idx", "f_idx", "B1_edge_idx", "B2_edge_idx"]:
        if h1[k].size:
            used[h1[k]] = True
    old2new = -np.ones(len(pairs), np.int32)
    act = np.flatnonzero(used)
    old2new[act] = np.arange(act.size, dtype=np.int32)
    pairs_c = pairs[act]
    E_c, A_c = E_tot[act], A_tot[act]
    h1_c = dict(
        e_idx=old2new[h1["e_idx"]] if h1["e_idx"].size else h1["e_idx"],
        f_idx=old2new[h1["f_idx"]] if h1["f_idx"].size else h1["f_idx"],
        b=h1["b"], d=h1["d"],
        B1_edge_idx=old2new[h1["B1_edge_idx"]] if h1["B1_edge_idx"].size else h1["B1_edge_idx"],
        B1_loop_ids=h1["B1_loop_ids"],
        B2_edge_idx=old2new[h1["B2_edge_idx"]] if h1["B2_edge_idx"].size else h1["B2_edge_idx"],
        B2_loop_ids=h1["B2_loop_ids"],
    )
    return pairs_c, E_c, A_c, h1_c

def build_h0h1_potential_edgewise(n, m, kappa, alpha,
                                  pairs_active, E_c, A_c, h1,
                                  use_h1: bool = True,
                                  barrier: str = "diag",   # {"diag","upper_full","upper_active"}
                                  barrier_k: int = 0):     # 0=include diag, 1=exclude diag
    ei = jnp.array(pairs_active[:,0], jnp.int32)
    ej = jnp.array(pairs_active[:,1], jnp.int32)
    E_c = jnp.array(E_c, jnp.float32)
    A_c = jnp.array(A_c, jnp.float32)
    e_idx = jnp.array(h1["e_idx"], jnp.int32)
    f_idx = jnp.array(h1["f_idx"], jnp.int32)
    b = jnp.array(h1["b"], jnp.float32)
    d = jnp.array(h1["d"], jnp.float32)
    B1_e = jnp.array(h1["B1_edge_idx"], jnp.int32)
    B1_l = jnp.array(h1["B1_loop_ids"], jnp.int32)
    B2_e = jnp.array(h1["B2_edge_idx"], jnp.int32)
    B2_l = jnp.array(h1["B2_loop_ids"], jnp.int32)
    L = b.shape[0]
    eps = jnp.array(1e-12, jnp.float32)

    def potential_fn(phi_flat):
        Phi = phi_flat.reshape((n, m))
        Z = jax.nn.softplus(Phi)

        # edgewise λ (avoid forming G for likelihood)
        lam = jnp.sum(Z[ei] * Z[ej], axis=1)  # (M_act,)
        loglam = jnp.log(jnp.maximum(lam, eps))
        loglik_h0 = jnp.dot(E_c, loglam) - jnp.dot(A_c, lam)

        loglik_h1 = jnp.array(0., jnp.float32)
        if use_h1 and (L > 0):
            lam_e = lam[e_idx]; lam_f = lam[f_idx]
            acc = jnp.log(jnp.maximum(lam_e, eps)) + jnp.log(jnp.maximum(lam_f, eps)) \
                  - lam_e * b - lam_f * d
            if B1_e.size > 0:
                lam_B1 = lam[B1_e]
                add = _log1mexp(jnp.maximum(lam_B1 * b[B1_l], eps))
                acc = acc + segsum(add, B1_l, L)
            if B2_e.size > 0:
                lam_B2 = lam[B2_e]
                add = (-lam_B2 * b[B2_l]) + _log1mexp(jnp.maximum(lam_B2 * (d - b)[B2_l], eps))
                acc = acc + segsum(add, B2_l, L)
            loglik_h1 = jnp.sum(acc)

        # ---- prior & barrier options ----
        quad = -0.5 * kappa * jnp.sum(Z * Z)

        if barrier == "diag":
            diagG = jnp.sum(Z * Z, axis=1)
            barrier_term = jnp.sum(jnp.log(jnp.maximum(diagG, eps)))

        elif barrier == "upper_full":
            G = Z @ Z.T                                        # O(n^2 m)
            iu0, iu1 = jnp.triu_indices(n, k=barrier_k)        # k=0 include diag; k=1 exclude
            barrier_term = jnp.sum(jnp.log(jnp.maximum(G[iu0, iu1], eps)))

        elif barrier == "upper_active":
            # fast: use diagonal + only the active off-diagonal pairs in the likelihood
            diagG = jnp.sum(Z * Z, axis=1)
            barrier_term = jnp.sum(jnp.log(jnp.maximum(diagG, eps))) if barrier_k == 0 else 0.0
            barrier_term = barrier_term + jnp.sum(jnp.log(jnp.maximum(lam, eps)))
        else:
            raise ValueError("barrier must be one of {'diag','upper_full','upper_active'}")

        logprior = quad + alpha * barrier_term
        logJ = jnp.sum(jax.nn.log_sigmoid(Phi))
        return -(loglik_h0 + loglik_h1 + logprior + logJ)

    return potential_fn

# ============================================================
# NUTS fit, diagnostics, confusion, latent coords
# ============================================================
def nuts_fit_class_from_feats(feat_dir: str, cid: str, fitcfg, use_h1: bool = True):
    pairs, E_tot, A_tot, h1 = pool_class_h0h1_from_feat_dir(feat_dir, cid)
    pairs_c, E_c, A_c, h1_c = compress_active_edges(pairs, E_tot, A_tot, h1)
    M = pairs.shape[0]
    # infer n from M = n*(n-1)/2
    n = int(0.5 + 0.5 * (1 + math.sqrt(1 + 8 * M)))
    iu = _pairs_to_iu(pairs)

    potential_fn = build_h0h1_potential_edgewise(
        n=n, m=fitcfg.m, kappa=fitcfg.kappa, alpha=fitcfg.alpha,
        pairs_active=pairs_c, E_c=E_c, A_c=A_c, h1=h1_c,
        use_h1=use_h1,
        barrier="upper_full",   # exact upper triangle
        barrier_k=0             # include diagonal in the barrier
    )

    kernel = NUTS(potential_fn=potential_fn,
                  target_accept_prob=fitcfg.target_accept,
                  dense_mass=fitcfg.dense_mass,
                  max_tree_depth=7)

    phi0 = jnp.zeros((n, fitcfg.m)).ravel()
    init_params = phi0 if fitcfg.num_chains == 1 else jnp.tile(phi0[None, :], (fitcfg.num_chains, 1))
    mcmc = MCMC(kernel,
                num_warmup=fitcfg.num_warmup,
                num_samples=fitcfg.num_samples,
                num_chains=fitcfg.num_chains,
                chain_method=("vectorized" if fitcfg.num_chains > 1 else "sequential"),
                progress_bar=True)

    key = jax.random.PRNGKey(fitcfg.seed + 9401)
    t0 = time.time()
    mcmc.run(key, init_params=init_params)
    elapsed = time.time() - t0

    raw = mcmc.get_samples(group_by_chain=False)
    # if NumPyro returns a dict, try a few keys; otherwise treat as flat array
    if isinstance(raw, dict):
        if "phi" in raw:
            phi_samples = raw["phi"]
        elif "z" in raw:
            phi_samples = raw["z"]
        else:
            phi_samples = raw[next(iter(raw.keys()))]
    else:
        phi_samples = raw

    phi_samples = np.array(phi_samples)

    Phi_mean = jnp.mean(phi_samples, axis=0).reshape((n, fitcfg.m))
    Z_bar = jax.nn.softplus(Phi_mean)
    Lambda_hat = np.array(Z_bar @ Z_bar.T)
    return dict(pairs=pairs, n=n, iu=iu,
                mcmc=mcmc, phi_samples=phi_samples,
                Lambda_hat=Lambda_hat, elapsed=elapsed)

def save_diagnostics(out_dir: Path, phi_samples: np.ndarray, n: int, m: int,
                     phi_indices: List[Tuple[int,int]], max_lag: int = 100, thin_every: int = 1):
    """
    phi_samples: (T, n*m)
    phi_indices: list of (node, factor) to plot
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    T = phi_samples.shape[0]
    idx = np.arange(0, T, max(1, thin_every))
    phi_thin = phi_samples[idx]

    for (i, k) in phi_indices:
        col = i * m + k
        series = np.asarray(phi_thin[:, col], float)
        trace_png = out_dir / f"trace_phi_i{i}_k{k}.png"
        acf_png   = out_dir / f"acf_phi_i{i}_k{k}.png"
        plot_trace(series, trace_png, ylabel=rf"$\phi_{{{i},{k}}}$")
        plot_acf(series, acf_png, max_lag=max_lag)

def _subject_loglik_from_feat(zpath: str, Lambda: np.ndarray) -> float:
    with np.load(zpath) as z:
        pairs = z["pairs"]
        lam_vec = Lambda[pairs[:,0], pairs[:,1]].astype(np.float64)
        eps = 1e-12

        # H0
        E = z["h0_E_counts"].astype(np.float64)
        A = z["h0_A_weights"].astype(np.float64)
        loglik_h0 = np.dot(E, np.log(np.maximum(lam_vec, eps))) - np.dot(A, lam_vec)

        # H1
        L = int(z["L_b"].shape[0])
        if L == 0:
            return float(loglik_h0)

        e = z["L_e_idx"].astype(np.int64)
        f = z["L_f_idx"].astype(np.int64)
        b = z["L_b"].astype(np.float64)
        d = z["L_d"].astype(np.float64)

        lam_e = lam_vec[e]; lam_f = lam_vec[f]
        term_edge = np.log(np.maximum(lam_e, eps)) + np.log(np.maximum(lam_f, eps)) - lam_e*b - lam_f*d

        # B1
        B1_data = z["B1_data"].astype(np.int64); B1_indptr = z["B1_indptr"].astype(np.int64)
        b1 = np.zeros(L, dtype=np.float64)
        if B1_data.size > 0:
            lam_B1 = lam_vec[B1_data]
            b_rep = np.repeat(b, B1_indptr[1:] - B1_indptr[:-1])
            x = np.clip(lam_B1 * b_rep, eps, None)
            entries = np.where(x <= np.log(2.0), np.log(-np.expm1(-x)), np.log1p(-np.exp(-x)))
            for i in range(L):
                b1[i] = entries[B1_indptr[i]:B1_indptr[i+1]].sum()

        # B2
        B2_data = z["B2_data"].astype(np.int64); B2_indptr = z["B2_indptr"].astype(np.int64)
        b2 = np.zeros(L, dtype=np.float64)
        if B2_data.size > 0:
            lam_B2 = lam_vec[B2_data]
            counts = (B2_indptr[1:] - B2_indptr[:-1])
            b_rep  = np.repeat(b, counts)
            db_rep = np.repeat(d-b, counts)
            x = np.clip(lam_B2 * db_rep, eps, None)
            entries = (-lam_B2 * b_rep) + np.where(x <= np.log(2.0), np.log(-np.expm1(-x)), np.log1p(-np.exp(-x)))
            for i in range(L):
                b2[i] = entries[B2_indptr[i]:B2_indptr[i+1]].sum()

        loglik_h1 = (term_edge + b1 + b2).sum()
        return float(loglik_h0 + loglik_h1)

def save_confusion_from_feats(feat_dir: str, class_Lambda: Dict[str, np.ndarray], out_png: Path, out_csv: Path):
    base = Path(feat_dir)
    # collect rows
    with (base/"index.csv").open() as f:
        rows = list(csv.DictReader(f))

    classes = sorted(set(r["cid"] for r in rows))
    class_to_idx = {c:i for i,c in enumerate(classes)}
    K = len(classes)

    # confusion counts
    C = np.zeros((K, K), dtype=int)  # rows=true, cols=pred

    # cache Lambda vecs per class to avoid recomputing for each subject
    lam_vec_by_class = {}
    # infer n/M from the first feat file
    sample_feat = base / rows[0]["file"].replace(".npz", "_feat.npz")
    with np.load(sample_feat) as z0:
        pairs = z0["pairs"]

    for c in classes:
        Lam = class_Lambda[c]
        lam_vec_by_class[c] = Lam[pairs[:,0], pairs[:,1]].astype(np.float64)

    for r in rows:
        zpath = base / r["file"].replace(".npz", "_feat.npz")
        logliks = []
        with np.load(zpath) as z:
            E = z["h0_E_counts"].astype(np.float64)
            A = z["h0_A_weights"].astype(np.float64)

            e = z["L_e_idx"].astype(np.int64)
            f = z["L_f_idx"].astype(np.int64)
            b = z["L_b"].astype(np.float64)
            d = z["L_d"].astype(np.float64)

            B1_data = z["B1_data"].astype(np.int64); B1_indptr = z["B1_indptr"].astype(np.int64)
            B2_data = z["B2_data"].astype(np.int64); B2_indptr = z["B2_indptr"].astype(np.int64)

            for c in classes:
                lam_vec = lam_vec_by_class[c]
                eps = 1e-12
                ll_h0 = np.dot(E, np.log(np.maximum(lam_vec, eps))) - np.dot(A, lam_vec)
                if e.size == 0:
                    logliks.append(ll_h0); continue
                lam_e = lam_vec[e]; lam_f = lam_vec[f]
                term_edge = np.log(np.maximum(lam_e, eps)) + np.log(np.maximum(lam_f, eps)) - lam_e*b - lam_f*d
                # B1
                b1 = 0.0
                if B1_data.size > 0:
                    lam_B1 = lam_vec[B1_data]
                    b_rep = np.repeat(b, B1_indptr[1:] - B1_indptr[:-1])
                    x = np.clip(lam_B1 * b_rep, eps, None)
                    entries = np.where(x <= np.log(2.0), np.log(-np.expm1(-x)), np.log1p(-np.exp(-x)))
                    tmp = np.zeros_like(b)
                    for i in range(len(b)):
                        tmp[i] = entries[B1_indptr[i]:B1_indptr[i+1]].sum()
                    b1 = tmp.sum()
                # B2
                b2 = 0.0
                if B2_data.size > 0:
                    lam_B2 = lam_vec[B2_data]
                    counts = (B2_indptr[1:] - B2_indptr[:-1])
                    b_rep  = np.repeat(b, counts)
                    db_rep = np.repeat(d-b, counts)
                    x = np.clip(lam_B2 * db_rep, eps, None)
                    entries = (-lam_B2 * b_rep) + np.where(x <= np.log(2.0), np.log(-np.expm1(-x)), np.log1p(-np.exp(-x)))
                    tmp = np.zeros_like(b)
                    for i in range(len(b)):
                        tmp[i] = entries[B2_indptr[i]:B2_indptr[i+1]].sum()
                    b2 = tmp.sum()
                logliks.append((term_edge.sum() + b1 + b2) + ll_h0)

        true_i = class_to_idx[r["cid"]]
        pred_i = int(np.argmax(logliks))
        C[true_i, pred_i] += 1

    # save CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f); w.writerow([""] + classes)
        for i,c in enumerate(classes):
            w.writerow([c] + list(map(int, C[i])))

    # plot heatmap
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(C, aspect='auto', interpolation='nearest')
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, str(C[i,j]), ha='center', va='center')
    ax.set_xlabel(r"$\mathrm{Predicted}$")
    ax.set_ylabel(r"$\mathrm{True}$")
    fig.tight_layout()
    fig.savefig(out_png); plt.close(fig)

    acc = np.trace(C) / np.sum(C) if np.sum(C)>0 else float('nan')
    return C, classes, acc

def save_latent_coords_plot(Lambda_hat: np.ndarray, png_path: Path, node_names: List[str] = None):
    # eigen-embedding (top-2)
    vals, vecs = np.linalg.eigh(Lambda_hat)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]; vecs = vecs[:, idx]
    k = min(2, np.sum(vals > 1e-10))
    Z2 = vecs[:, :k] * np.sqrt(np.clip(vals[:k], 0, None))
    if Z2.shape[1] == 1:
        Z2 = np.hstack([Z2, np.zeros_like(Z2)])
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(Z2[:,0], Z2[:,1], s=24)
    if node_names is not None:
        for i, name in enumerate(node_names):
            ax.annotate(str(name), (Z2[i,0], Z2[i,1]), fontsize=10, alpha=0.7)
    ax.set_xlabel(r"$z_1$"); ax.set_ylabel(r"$z_2$")
    fig.tight_layout()
    fig.savefig(png_path); plt.close(fig)

def run_and_save_all(
    feat_dir: str,
    out_root: str,
    fitcfg,
    phi_diag_indices: List[Tuple[int,int]] = [(0,0),(10,0)],
    max_lag: int = 100,
    thin_every_for_diag: int = 1,
    use_h1: bool = True,
):
    out_root = Path(out_root); out_root.mkdir(parents=True, exist_ok=True)

    # figure out classes from index
    with (Path(feat_dir)/"index.csv").open() as f:
        classes = sorted({row["cid"] for row in csv.DictReader(f)})

    class_Lambda = {}
    for cid in classes:
        print(f"[fit] {cid} …")
        fit = nuts_fit_class_from_feats(feat_dir, cid, fitcfg, use_h1=use_h1)
        pairs, n = fit["pairs"], fit["n"]
        Lambda_hat = fit["Lambda_hat"]
        mcmc = fit["mcmc"]; phi_samples = fit["phi_samples"]

        # save numeric artifacts
        cdir = out_root / cid
        cdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cdir/"draws_phi.npz", phi=phi_samples)
        np.savez_compressed(cdir/"Lambda_hat.npz", Lambda_hat=Lambda_hat, pairs=pairs)

        # traces & ACF for a few phi entries
        save_diagnostics(cdir, phi_samples, n, fitcfg.m,
                         phi_indices=phi_diag_indices, max_lag=max_lag,
                         thin_every=thin_every_for_diag)

        # latent coordinate plot, with optional node names
        node_names = None
        meta_json = Path(feat_dir)/"meta.json"
        if meta_json.exists():
            try:
                meta = json.loads(meta_json.read_text())
                if "node_names_file" in meta:
                    node_file = Path(feat_dir)/meta["node_names_file"]
                    node_names = json.loads(node_file.read_text())
            except Exception:
                node_names = None
        save_latent_coords_plot(Lambda_hat, cdir/"latent_coords.png", node_names=node_names)

        class_Lambda[cid] = Lambda_hat

    # confusion and accuracy using Λ̂ per class
    C, cls, acc = save_confusion_from_feats(
        feat_dir,
        class_Lambda=class_Lambda,
        out_png=out_root/"confusion.png",
        out_csv=out_root/"confusion.csv"
    )
    with (out_root/"metrics.json").open("w") as f:
        json.dump({"classes": cls, "accuracy": float(acc)}, f, indent=2)

    print(f"[done] saved to {out_root}  |  accuracy={acc:.3f}")
    return class_Lambda, C, acc

# ============================================================
# Batch: build & save H0+H1 features for all saved distance shards
# - Input  dir: base_dir/ (from save_distances_sharded)
#     index.csv, meta.json, <cid>/subj_XXXX.npz with D (and maybe D_filled)
# - Output dir: out_dir/
#     index.csv (mirrors input), <cid>/subj_XXXX_feat.npz
#
# Each *_feat.npz contains:
#   # Shared edge index / times
#   pairs              (M,2) int32
#   t_edge             (M,)  float32
#   divide_by_two      ()    bool (as uint8)
#
#   # H0 (MST race, VR time scale)
#   h0_e_idx           (r,)  int32
#   h0_delta_t         (r,)  float32
#   h0_E_counts        (M,)  float32
#   h0_A_weights       (M,)  float32
#   h0_w_mst_time      (r,)  float32
#   h0_mst_mask        (M,)  uint8
#
#   # H1 loops (ragged; CSR-like encoding)
#   L_e_idx            (L,)  int32
#   L_f_idx            (L,)  int32
#   L_b                (L,)  float32
#   L_d                (L,)  float32
#   L_uv               (L,2,2) int32   # [[(u_e,v_e),(u_f,v_f)], ...]
#
#   L_vertices_data    (sum|V_i|,) int32
#   L_vertices_indptr  (L+1,)     int32
#
#   B1_data            (sum|B1_i|,) int32
#   B1_indptr          (L+1,)       int32
#   B2_data            (sum|B2_i|,) int32
#   B2_indptr          (L+1,)       int32
#   Buniq_data         (sum|B_i|,)  int32
#   Buniq_indptr       (L+1,)       int32
#
#   # (optional) node_names_json in the first subject per class if you want
# ============================================================

import csv, json, numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# ---------- Unified extractor (shared H0 + H1 on distance matrix) ----------
# (identical to the one we discussed; kept here self-contained)

def _pairs_index(n: int):
    I, J = np.triu_indices(n, k=1)
    pairs = np.stack([I, J], axis=1).astype(np.int32)
    tab = -np.ones((n, n), dtype=np.int32)
    tab[I, J] = np.arange(len(I), dtype=np.int32)
    return pairs, tab

def _edge_times_from_D(D: np.ndarray, divide_by_two: bool) -> np.ndarray:
    I, J = np.triu_indices(D.shape[0], k=1)
    t = np.asarray(D, float)[I, J]
    return t/2.0 if divide_by_two else t

class _DSU:
    def __init__(self, n): self.p=list(range(n)); self.r=[0]*n
    def find(self, x):
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]]; x=self.p[x]
        return x
    def union(self, a,b):
        a,b=self.find(a),self.find(b)
        if a==b: return False
        if self.r[a]<self.r[b]: self.p[a]=b
        elif self.r[a]>self.r[b]: self.p[b]=a
        else: self.p[b]=a; self.r[a]+=1
        return True

def _extract_h0_from_index(n, pairs, t_edge):
    M = len(pairs)
    edge_ids = np.arange(M, dtype=np.int32)
    order = edge_ids[np.argsort(np.lexsort((edge_ids, np.where(np.isfinite(t_edge), t_edge, np.inf))))]
    dsu = _DSU(n)
    e_idx, comp_labels_list = [], []
    for eid in order:
        te = t_edge[eid]
        if not np.isfinite(te): continue
        u,v = pairs[eid]
        fu, fv = dsu.find(u), dsu.find(v)
        if fu != fv:
            comp = np.array([dsu.find(k) for k in range(n)], dtype=np.int32)
            comp_labels_list.append(comp)
            dsu.union(fu, fv)
            e_idx.append(eid)
            if len(e_idx)==n-1: break
    e_idx = np.array(e_idx, dtype=np.int32)
    w_mst_time = t_edge[e_idx].astype(np.float32)
    d = w_mst_time
    delta_t = np.empty_like(d)
    if d.size:
        delta_t[0]=d[0]; delta_t[1:]=d[1:]-d[:-1] if d.size>1 else delta_t[1:]
    E_counts = np.zeros(M, dtype=np.float32)
    if e_idx.size: E_counts[e_idx]=1.0
    I, J = pairs[:,0], pairs[:,1]
    A_weights = np.zeros(M, dtype=np.float32)
    for i, comp in enumerate(comp_labels_list):
        mask = (comp[I]!=comp[J]) & np.isfinite(t_edge)
        if np.any(mask): A_weights[mask]+=float(delta_t[i])
    mst_mask = np.zeros(M, dtype=np.uint8)
    if e_idx.size: mst_mask[e_idx]=1
    return dict(
        e_idx=e_idx,
        delta_t=delta_t.astype(np.float32),
        E_counts=E_counts,
        A_weights=A_weights,
        w_mst_time=w_mst_time,
        mst_mask=mst_mask
    )

from collections import deque
def _graph_edges_up_to_time(n, pairs, t_edge, t_thresh, exclude_edge=None):
    adj=[[] for _ in range(n)]
    for eid,(u,v) in enumerate(pairs):
        te = t_edge[eid]
        if not np.isfinite(te): continue
        if exclude_edge is not None and eid==exclude_edge: continue
        if te<=t_thresh:
            adj[u].append(v); adj[v].append(u)
    return adj

def _path_vertices(adj, s, t):
    prev=[-1]*len(adj); q=deque([s]); prev[s]=s
    while q:
        u=q.popleft()
        if u==t: break
        for v in adj[u]:
            if prev[v]==-1:
                prev[v]=u; q.append(v)
    if prev[t]==-1: return None
    path=[t]; cur=t
    while cur!=s:
        cur=prev[cur]; path.append(cur)
    path.reverse(); return path

def _extract_h1_from_index(
    n: int,
    pairs: np.ndarray,
    t_edge: np.ndarray,
    mst_mask: np.ndarray,
    max_eps: float | None = None,
    min_pers: float = 0.0,
):
    """
    Vietoris–Rips H1 via triangle columns + Z2 reduction, aligned with PH ordering.

    - Edges with t_edge > max_eps are treated as +inf (excluded) if max_eps is provided.
    - Row (edge) order is by (t_edge, edge_id) — the same order used in PH.
    - Pivot selection uses the *youngest* edge by that order.
    - Birth edges must have an existing u->v path using strictly-earlier edges (row-order),
      otherwise they are skipped (prevents MST edges from birthing cycles).
    - Loops with persistence <= min_pers are skipped.
    """
    # --- 0) apply epsilon cap (to match Gudhi’s max_edge_length) ---
    t_masked = np.asarray(t_edge, float).copy()
    if max_eps is not None:
        bad = t_masked > float(max_eps)
        t_masked[bad] = np.inf

    M = len(pairs)
    edge_ids = np.arange(M, dtype=np.int32)

    # --- 1) global row order for edges: sort by (time, id) ---
    row_order = np.lexsort((edge_ids, t_masked))  # sort by (t_edge, edge_id)
    rank = np.empty(M, dtype=np.int32)
    rank[row_order] = np.arange(M, dtype=np.int32)

    def younger_key(eid: int):
        return (t_masked[eid], eid)

    def argmax_youngest(eid_set: set[int]) -> int:
        # "max" by filtration order (i.e., the youngest edge)
        return max(eid_set, key=lambda e: younger_key(e))

    # --- 2) build triangle list (only fully finite triangles) ---
    tab = -np.ones((n, n), dtype=np.int32)
    for eid, (i, j) in enumerate(pairs):
        tab[i, j] = eid

    tris = []  # each = ( (a,b,c), [e_ab,e_bc,e_ac], t_tri, last_edge )
    for a in range(n - 2):
        for b in range(a + 1, n - 1):
            e_ab = tab[a, b]
            if e_ab < 0 or not np.isfinite(t_masked[e_ab]):  # respect cap
                continue
            for c in range(b + 1, n):
                e_ac = tab[a, c]; e_bc = tab[b, c]
                if e_ac < 0 or e_bc < 0:
                    continue
                if not (np.isfinite(t_masked[e_ac]) and np.isfinite(t_masked[e_bc])):
                    continue
                tri_edges = [e_ab, e_bc, e_ac]
                # triangle appears when all 3 edges exist → time = max (by filtration order)
                last_edge = argmax_youngest(set(tri_edges))
                t_tri = float(t_masked[last_edge])
                tris.append(((a, b, c), tri_edges, t_tri, last_edge))

    # sort triangles by (t_tri, vertices) for deterministic reduction
    tris.sort(key=lambda x: (x[2], x[0]))

    # --- 3) Z2 column reduction over edges; store pivot edge → death info ---
    pivot_col: dict[int, set[int]] = {}  # pivot edge → reduced column (set of edges)
    pivot_time: dict[int, float] = {}    # pivot edge → death time (triangle time)
    pivot_last: dict[int, int] = {}      # pivot edge → the triangle's youngest edge (a convenience)

    for (_abc, tri_edges, t_tri, last_e) in tris:
        col = set(tri_edges)
        while col:
            p = argmax_youngest(col)
            if p in pivot_col:
                col ^= pivot_col[p]
            else:
                pivot_col[p] = set(col)
                pivot_time[p] = t_tri
                pivot_last[p] = last_e
                break

    # --- 4) births from pivot edges; require a prior u->v path; drop tiny bars ---
    loops = []

    # helpers to build adjacency by rank threshold
    from collections import deque

    def build_adj_until(rank_thresh: int, inclusive: bool, exclude_edge: int | None = None):
        adj = [[] for _ in range(n)]
        for eid, (x, y) in enumerate(pairs):
            if not np.isfinite(t_masked[eid]):
                continue
            if exclude_edge is not None and eid == exclude_edge:
                continue
            if (rank[eid] < rank_thresh) or (inclusive and rank[eid] <= rank_thresh):
                adj[x].append(y); adj[y].append(x)
        return adj

    def reachable_nodes(seed: int, adj):
        seen = {seed}; q = deque([seed])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); q.append(v)
        return seen

    # iterate pivot edges in row order for determinism
    for e_idx in sorted(pivot_time.keys(), key=lambda e: (t_masked[e], e)):
        b = float(t_masked[e_idx]); d = float(pivot_time[e_idx])
        if not (np.isfinite(b) and np.isfinite(d)):
            continue
        if (d - b) <= float(min_pers):
            continue  # remove ~zero persistence

        u, v = map(int, pairs[e_idx])

        # Require a u→v path using strictly earlier rows (rank < rank[e_idx])
        adj_b = build_adj_until(rank[e_idx], inclusive=False, exclude_edge=e_idx)
        # BFS
        prev = [-1] * n; q = deque([u]); prev[u] = u
        while q:
            w = q.popleft()
            if w == v:
                break
            for z in adj_b[w]:
                if prev[z] == -1:
                    prev[z] = w; q.append(z)
        if prev[v] == -1:
            # No pre-existing path → this edge is not a 1-cycle birth (likely MST). Skip.
            continue

        # reconstruct a path (loop's vertex order along the path)
        path = [v]; cur = v
        while cur != u:
            cur = prev[cur]
            path.append(cur)
        path.reverse()
        L_vertices = list(map(int, path))

        # components at b (strict) and at death (<= last edge of death triangle)
        last_e = int(pivot_last[e_idx])
        adj_comp_b = build_adj_until(rank[e_idx], inclusive=False, exclude_edge=None)
        seen_b = reachable_nodes(L_vertices[0], adj_comp_b)

        adj_comp_d = build_adj_until(rank[last_e], inclusive=True, exclude_edge=None)
        seen_d = reachable_nodes(L_vertices[0], adj_comp_d)

        # edge sets within those components and within rank windows
        comp_b_edges = {
            eid for eid, (a, b_) in enumerate(pairs)
            if (a in seen_b and b_ in seen_b) and np.isfinite(t_masked[eid]) and (rank[eid] < rank[e_idx])
        }
        comp_d_edges = {
            eid for eid, (a, b_) in enumerate(pairs)
            if (a in seen_d and b_ in seen_d) and np.isfinite(t_masked[eid]) and (rank[eid] <= rank[last_e])
        }

        # boundary subsets (exclude MST edges & the last edge)
        B1 = {eid for eid in comp_b_edges if (not mst_mask[eid]) and (eid != last_e)}
        B2 = {
            eid for eid in comp_d_edges
            if (rank[eid] > rank[e_idx]) and (not mst_mask[eid]) and (eid != last_e)
        }

        loops.append(dict(
            e_idx=int(e_idx), f_idx=int(last_e),
            e_uv=tuple(map(int, pairs[e_idx])),
            f_uv=tuple(map(int, pairs[last_e])),
            b=b, d=d,
            L_vertices=L_vertices,
            B1_idx=sorted(B1), B2_idx=sorted(B2), B_idx=None
        ))

    # --- 5) make boundary-unique list (B_idx) like before ---
    loops.sort(key=lambda L: L["d"])
    used = set()
    for L in loops:
        tilde = set(L["B1_idx"]) | set(L["B2_idx"])
        uniq = tilde - used
        L["B_idx"] = sorted(uniq)
        used |= tilde

    return loops


def extract_h01_from_D(
    D: np.ndarray,
    divide_by_two: bool = True,
    max_eps: float | None = None,
    min_pers: float = 0.0,
):
    D = np.asarray(D, float)
    n = D.shape[0]
    pairs, _ = _pairs_index(n)
    t_edge = _edge_times_from_D(D, divide_by_two=divide_by_two)

    h0 = _extract_h0_from_index(n, pairs, t_edge)
    loops = _extract_h1_from_index(
        n, pairs, t_edge, h0["mst_mask"],
        max_eps=max_eps, min_pers=min_pers
    )
    return dict(
        pairs=pairs,
        t_edge=t_edge.astype(np.float32),
        divide_by_two=np.uint8(bool(divide_by_two)),
        h0=h0,
        loops=loops,
    )


# ---------- Ragged packing for H1 loops ----------
def _pack_loops_ragged(loops: List[Dict], pairs: np.ndarray):
    L = len(loops)
    L_e_idx = np.array([L_i['e_idx'] for L_i in loops], dtype=np.int32)
    L_f_idx = np.array([L_i['f_idx'] for L_i in loops], dtype=np.int32)
    L_b     = np.array([L_i['b'] for L_i in loops], dtype=np.float32)
    L_d     = np.array([L_i['d'] for L_i in loops], dtype=np.float32)

    # uv pairs aligned with e_idx and f_idx (convenience)
    L_uv = np.zeros((L,2,2), dtype=np.int32)
    for i,L_i in enumerate(loops):
        L_uv[i,0,:] = np.array(L_i['e_uv'], dtype=np.int32)
        L_uv[i,1,:] = np.array(L_i['f_uv'], dtype=np.int32)

    def to_csr(list_of_lists: List[List[int]]):
        indptr=[0]; data=[]
        for arr in list_of_lists:
            data.extend(arr); indptr.append(len(data))
        return np.array(data, dtype=np.int32), np.array(indptr, dtype=np.int32)

    L_vertices_data, L_vertices_indptr = to_csr([L_i['L_vertices'] for L_i in loops])
    B1_data, B1_indptr   = to_csr([L_i['B1_idx'] for L_i in loops])
    B2_data, B2_indptr   = to_csr([L_i['B2_idx'] for L_i in loops])
    Buniq_data, Buniq_indptr = to_csr([L_i['B_idx']  for L_i in loops])

    return dict(
        L_e_idx=L_e_idx, L_f_idx=L_f_idx, L_b=L_b, L_d=L_d, L_uv=L_uv,
        L_vertices_data=L_vertices_data, L_vertices_indptr=L_vertices_indptr,
        B1_data=B1_data, B1_indptr=B1_indptr,
        B2_data=B2_data, B2_indptr=B2_indptr,
        Buniq_data=Buniq_data, Buniq_indptr=Buniq_indptr
    )

def _unpack_loops_ragged(arrs: Dict) -> List[Dict]:
    def from_csr(data, indptr):
        out=[]; idx0=0
        for k in range(len(indptr)-1):
            idx1 = indptr[k+1]
            out.append(list(map(int, data[idx0:idx1])))
            idx0 = idx1
        return out
    L = int(arrs["L_e_idx"].shape[0])
    e_idx = arrs["L_e_idx"]; f_idx=arrs["L_f_idx"]
    b = arrs["L_b"]; d = arrs["L_d"]; L_uv = arrs["L_uv"]
    L_vertices = from_csr(arrs["L_vertices_data"], arrs["L_vertices_indptr"])
    B1 = from_csr(arrs["B1_data"], arrs["B1_indptr"])
    B2 = from_csr(arrs["B2_data"], arrs["B2_indptr"])
    Buniq = from_csr(arrs["Buniq_data"], arrs["Buniq_indptr"])
    loops=[]
    for i in range(L):
        loops.append(dict(
            e_idx=int(e_idx[i]), f_idx=int(f_idx[i]),
            e_uv=tuple(map(int, L_uv[i,0])),
            f_uv=tuple(map(int, L_uv[i,1])),
            b=float(b[i]), d=float(d[i]),
            L_vertices=L_vertices[i],
            B1_idx=B1[i], B2_idx=B2[i], B_idx=Buniq[i]
        ))
    return loops

# ---------- Batch I/O ----------
def build_features_for_sharded(base_dir: str,
                               out_dir: str,
                               divide_by_two: bool = True,
                               prefer_filled: bool = False,
                               prefer_vr: bool = True,      # <-- NEW
                               overwrite: bool = False):
    """
    Read all distance shards from base_dir (created by save_distances_sharded),
    compute shared H0+H1 features, and save to out_dir with mirrored structure.
    If prefer_vr=True, we use 'D_vr' (if present) so infinite entries are already capped
    at 'cap = eps_max + delta' and consistent across subjects.
    """
    base = Path(base_dir); out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # mirror index.csv into output for easy lookup
    index_in = base / "index.csv"
    if not index_in.exists():
        raise FileNotFoundError(f"Missing {index_in}")
    index_out = out / "index.csv"
    if not index_out.exists():
        index_out.write_text(index_in.read_text())

    # load meta.json (to grab eps_max & cap if available)
    meta_in = base / "meta.json"
    vr_eps_max = None
    vr_cap = None
    if meta_in.exists():
        meta = json.loads(meta_in.read_text())
        vr = (meta.get("distance_info", {}) or {}).get("vr", {})
        vr_eps_max = float(vr.get("eps_max")) if "eps_max" in vr else None
        vr_cap = float(vr.get("cap")) if "cap" in vr else None
        # copy meta.json over (nice to have alongside features)
        if not (out/"meta.json").exists():
            (out/"meta.json").write_text(meta_in.read_text())

    # iterate rows
    with index_in.open() as f:
        r = csv.DictReader(f)
        for row in r:
            cid = row["cid"]
            rel = row["file"]      # e.g., "Group1/subj_0123.npz"
            in_npz = base / rel
            out_npz = out / rel.replace(".npz", "_feat.npz")
            out_npz.parent.mkdir(parents=True, exist_ok=True)
            if out_npz.exists() and not overwrite:
                continue

            with np.load(in_npz) as z:
                # priority: D_vr > D_filled > D
                if prefer_vr and ("D_vr" in z.files):
                    D = z["D_vr"]
                elif prefer_filled and ("D_filled" in z.files):
                    D = z["D_filled"]
                else:
                    D = z["D"]
                # fallback: if subject-level cap present in the shard, prefer it
                subj_eps_max = float(z["eps_max"]) if "eps_max" in z.files else vr_eps_max
                subj_cap     = float(z["cap"])     if "cap"     in z.files else vr_cap

            # extract features
            pack = extract_h01_from_D(
                D,
                divide_by_two=divide_by_two,
                max_eps=subj_eps_max,   # from meta/shard; same value you used as Gudhi's max_edge_length
                min_pers=0.0            # set e.g. 1e-9 if you filtered zero bars in Gudhi
            )
            pairs, t_edge = pack["pairs"], pack["t_edge"]
            h0 = pack["h0"]; loops = pack["loops"]
            L_pack = _pack_loops_ragged(loops, pairs)

            # write subject feature file (also stash eps_max/cap for easy filtering later)
            save_dict = dict(
                pairs=pairs,
                t_edge=t_edge,
                divide_by_two=np.array(pack["divide_by_two"], dtype=np.uint8),

                h0_e_idx=h0["e_idx"],
                h0_delta_t=h0["delta_t"],
                h0_E_counts=h0["E_counts"],
                h0_A_weights=h0["A_weights"],
                h0_w_mst_time=h0["w_mst_time"],
                h0_mst_mask=h0["mst_mask"].astype(np.uint8),

                **L_pack
            )
            if subj_eps_max is not None:
                save_dict["vr_eps_max"] = np.array(subj_eps_max, dtype=np.float32)
            if subj_cap is not None:
                save_dict["vr_cap"] = np.array(subj_cap, dtype=np.float32)

            np.savez_compressed(out_npz, **save_dict)

    print(f"[ok] features written to: {out_dir}")


def load_subject_features(feat_path: str) -> Dict:
    """
    Load one *_feat.npz and reconstruct a user-friendly dict.
    """
    with np.load(feat_path, allow_pickle=True) as z:
        pairs = z["pairs"]; t_edge = z["t_edge"]; divide_by_two = bool(z["divide_by_two"].item())
        h0 = dict(
            e_idx=z["h0_e_idx"], delta_t=z["h0_delta_t"],
            E_counts=z["h0_E_counts"], A_weights=z["h0_A_weights"],
            w_mst_time=z["h0_w_mst_time"], mst_mask=z["h0_mst_mask"].astype(bool)
        )
        rag = {k: z[k] for k in
               ["L_e_idx","L_f_idx","L_b","L_d","L_uv",
                "L_vertices_data","L_vertices_indptr",
                "B1_data","B1_indptr","B2_data","B2_indptr","Buniq_data","Buniq_indptr"]}
        loops = _unpack_loops_ragged(rag)
    return dict(
        edge_index=dict(pairs=pairs, t_edge=t_edge, divide_by_two=divide_by_two),
        h0=h0,
        h1=dict(loops=loops)
    )


# ============================================================
# SIMULATION: concentric rings + H0+H1 features
# ============================================================
N_POINTS   = 150
N_SUBJ     = 10
R_INNER    = 1.0
R_OUTER    = 2.0
INNER_FRAC = 0.5          # ~75 inner, 75 outer
JITTER_SD  = 0.05
H1_LOOPS_INNER = 10       # per subject
H1_LOOPS_OUTER = 10       # per subject

FEAT_DIR   = Path("feats/sim_rings_h0h1")
OUT_ROOT   = Path("runs_sim/sim_rings_h0h1")
CLASS_ID   = "RINGS"
RNG        = np.random.default_rng(20250201)

def make_concentric_base(n_points=150,
                         r_inner=1.0,
                         r_outer=2.0,
                         inner_frac=0.5,
                         rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n1 = int(round(n_points * inner_frac))
    n2 = n_points - n1
    theta1 = rng.uniform(0.0, 2*np.pi, size=n1)
    theta2 = rng.uniform(0.0, 2*np.pi, size=n2)
    X1 = np.c_[r_inner * np.cos(theta1), r_inner * np.sin(theta1)]
    X2 = np.c_[r_outer * np.cos(theta2), r_outer * np.sin(theta2)]
    X  = np.vstack([X1, X2])
    return X

def jitter_subject(X0, sigma=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return X0 + rng.normal(scale=sigma, size=X0.shape)

def mst_counts_from_dist(D, pairs):
    """
    Given distance matrix D (n x n) and global edge list pairs (M x 2),
    compute per-edge:
      E[f]:  # times edge f is chosen in MST (0 or 1)
      A[f]:  total exposure sum_t b_t * 1{f in A_t}
    """
    n = D.shape[0]
    M = len(pairs)
    # Distance per edge
    d_vec = np.array([D[i, j] for (i, j) in pairs], dtype=float)
    order = np.argsort(d_vec)

    E = np.zeros(M, dtype=float)
    A = np.zeros(M, dtype=float)

    parent = np.arange(n, dtype=int)
    rank   = np.zeros(n, dtype=int)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    n_comp = n
    for idx in order:
        if n_comp == 1:
            break
        i, j = pairs[idx]
        ri, rj = find(i), find(j)
        if ri == rj:
            continue

        # this edge is chosen at this step
        b_t = d_vec[idx]

        # admissible edges BEFORE union: edges connecting different components
        for f, (u, v) in enumerate(pairs):
            if find(u) != find(v):
                A[f] += b_t

        E[idx] += 1.0
        union(i, j)
        n_comp -= 1

    return E.astype(np.float32), A.astype(np.float32)

def make_h1_loops_for_subject(X, D, pairs, pair_to_idx,
                              n_loops_inner=10,
                              n_loops_outer=10,
                              r_mid=1.5,
                              rng=None):
    """
    Construct simple H1 features:
      - identify inner vs outer ring via radius threshold r_mid
      - along each ring, sort by angle and connect neighbors to form adjacency edges
      - choose some adjacency edges as (e_idx, f_idx) for loops
      - b, d from corresponding edge lengths (ensuring d >= b + small_eps)
      - B1/B2 empty → only the edge term is used in H1 likelihood
    """
    if rng is None:
        rng = np.random.default_rng()

    r = np.linalg.norm(X, axis=1)
    theta = np.arctan2(X[:, 1], X[:, 0])

    inner = np.where(r < r_mid)[0]
    outer = np.where(r >= r_mid)[0]

    # sort each ring by angle to get adjacency
    inner_sorted = inner[np.argsort(theta[inner])] if inner.size > 0 else np.array([], int)
    outer_sorted = outer[np.argsort(theta[outer])] if outer.size > 0 else np.array([], int)

    def ring_adj_indices(sorted_idx):
        Lr = len(sorted_idx)
        if Lr < 2:
            return []
        edges = []
        for k in range(Lr):
            i = sorted_idx[k]
            j = sorted_idx[(k + 1) % Lr]   # wrap-around
            if i > j:
                i, j = j, i
            edges.append(pair_to_idx[(i, j)])
        return edges

    inner_edges_idx = ring_adj_indices(inner_sorted)
    outer_edges_idx = ring_adj_indices(outer_sorted)

    L_inner = min(n_loops_inner, len(inner_edges_idx))
    L_outer = min(n_loops_outer, len(outer_edges_idx))
    L = L_inner + L_outer

    if L == 0:
        # no loops, return empty-like structure
        return dict(
            L_e_idx=np.zeros(0, dtype=np.int32),
            L_f_idx=np.zeros(0, dtype=np.int32),
            L_b=np.zeros(0, dtype=np.float32),
            L_d=np.zeros(0, dtype=np.float32),
            B1_data=np.zeros(0, dtype=np.int32),
            B1_indptr=np.zeros(1, dtype=np.int32),
            B2_data=np.zeros(0, dtype=np.int32),
            B2_indptr=np.zeros(1, dtype=np.int32),
        )

    e_idx = np.zeros(L, dtype=np.int32)
    f_idx = np.zeros(L, dtype=np.int32)
    b     = np.zeros(L, dtype=np.float32)
    d     = np.zeros(L, dtype=np.float32)

    # helper to get edge length from index
    d_vec = np.array([D[i, j] for (i, j) in pairs], dtype=float)

    pos = 0
    # inner loops
    if L_inner > 0:
        chosen_inner = RNG.choice(inner_edges_idx, size=L_inner, replace=False)
        # for f_idx, just rotate them by 1 for some diversity
        f_inner = np.roll(chosen_inner, -1)
        for k in range(L_inner):
            ei = chosen_inner[k]
            fi = f_inner[k]
            e_idx[pos] = ei
            f_idx[pos] = fi
            be = d_vec[ei]
            df = d_vec[fi]
            b[pos] = min(be, df)
            d[pos] = max(be, df) + 0.02  # ensure d > b
            pos += 1

    # outer loops
    if L_outer > 0:
        chosen_outer = RNG.choice(outer_edges_idx, size=L_outer, replace=False)
        f_outer = np.roll(chosen_outer, -1)
        for k in range(L_outer):
            ei = chosen_outer[k]
            fi = f_outer[k]
            e_idx[pos] = ei
            f_idx[pos] = fi
            be = d_vec[ei]
            df = d_vec[fi]
            b[pos] = min(be, df)
            d[pos] = max(be, df) + 0.02
            pos += 1

    # B1/B2: empty data, but CSR indptr must be length L+1
    B1_data   = np.zeros(0, dtype=np.int32)
    B2_data   = np.zeros(0, dtype=np.int32)
    B1_indptr = np.zeros(L + 1, dtype=np.int32)
    B2_indptr = np.zeros(L + 1, dtype=np.int32)

    return dict(
        L_e_idx=e_idx,
        L_f_idx=f_idx,
        L_b=b,
        L_d=d,
        B1_data=B1_data,
        B1_indptr=B1_indptr,
        B2_data=B2_data,
        B2_indptr=B2_indptr,
    )

def build_simulation_features_using_official_extractor():
    FEAT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) base cloud X0 as before
    X0 = make_concentric_base(
        n_points=N_POINTS,
        r_inner=R_INNER,
        r_outer=R_OUTER,
        inner_frac=INNER_FRAC,
        rng=RNG,
    )

    # 2) meta + node names (same as before)
    node_names = [f"v{i}" for i in range(N_POINTS)]
    (FEAT_DIR / "node_names.json").write_text(json.dumps(node_names, indent=2))
    meta = {
        "node_names_file": "node_names.json",
        "distance_info": {"vr": {"cap": 10.0}},  # any cap > max d is fine
    }
    (FEAT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    # 3) index.csv
    with (FEAT_DIR / "index.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sid", "cid", "file"])
        for s in range(N_SUBJ):
            sid  = f"S{s:03d}"
            file = f"subj{s:03d}.npz"
            w.writerow([sid, CLASS_ID, file])

    # 4) per-subject distances → official H0+H1 features
    for s in range(N_SUBJ):
        file_base = f"subj{s:03d}.npz"
        base_path = FEAT_DIR / file_base
        feat_path = FEAT_DIR / file_base.replace(".npz", "_feat.npz")

        Xs = jitter_subject(X0, sigma=JITTER_SD, rng=RNG)
        D  = np.linalg.norm(Xs[:, None, :] - Xs[None, :, :], axis=2)

        # save raw subject D (optional)
        np.savez_compressed(base_path, X=Xs, D=D)

        # ---- HERE is the key line: official extractor ----
        pack = extract_h01_from_D(
            D,
            divide_by_two=True,   # or False, but then keep consistent everywhere
            max_eps=None,         # or some cutoff if you want
            min_pers=0.0,
        )

        pairs   = pack["pairs"]
        t_edge  = pack["t_edge"]
        h0      = pack["h0"]
        loops   = pack["loops"]
        L_pack  = _pack_loops_ragged(loops, pairs)

        # optional cap for later H1 filtering in pooling
        vr_cap = np.array(np.max(t_edge) * 1.1, dtype=np.float32)

        save_dict = dict(
            pairs=pairs,
            t_edge=t_edge.astype(np.float32),
            divide_by_two=np.array(pack["divide_by_two"], dtype=np.uint8),

            h0_e_idx=h0["e_idx"],
            h0_delta_t=h0["delta_t"],
            h0_E_counts=h0["E_counts"],
            h0_A_weights=h0["A_weights"],
            h0_w_mst_time=h0["w_mst_time"],
            h0_mst_mask=h0["mst_mask"].astype(np.uint8),

            **L_pack,
            vr_cap=vr_cap,
        )

        np.savez_compressed(feat_path, **save_dict)

    print(f"[sim] Built OFFICIAL H0+H1 features for {N_SUBJ} subjects in {FEAT_DIR}")

from types import SimpleNamespace

def default_fitcfg():
    """
    Default NUTS configuration for the rings H0+H1 experiment.
    You can override any of these in the notebook by passing your own fitcfg.
    """
    return SimpleNamespace(
        m=4,                # latent rank
        kappa=6.0,          # quadratic prior
        alpha=0.2,          # log-barrier strength
        num_warmup=1000,
        num_samples=2000,
        num_chains=1,
        target_accept=0.9,
        dense_mass=False,
        seed=1234,
    )

def fit_rings_no_save(fitcfg=None, use_h1: bool = True):
    """
    Interactive version for the RINGS H0+H1 experiment.

    - Builds simulated features in FEAT_DIR (small *_feat.npz files)
    - Runs NUTS for the single class CLASS_ID
    - Returns results in memory, without writing draws/Lambda/PNGs.

    Returns:
        Lambda_hat : (n, n) numpy array
        phi_samples: (T, n*m) numpy array
        fit_dict   : the full dict from nuts_fit_class_from_feats
    """
    from .h0h1_rings import (
        FEAT_DIR,
        CLASS_ID,
        build_simulation_features_using_official_extractor,
        nuts_fit_class_from_feats,
    )

    if fitcfg is None:
        fitcfg = default_fitcfg()

    # Ensure features exist (this only writes the small feat files)
    build_simulation_features_using_official_extractor()

    fit = nuts_fit_class_from_feats(
        feat_dir=str(FEAT_DIR),
        cid=CLASS_ID,
        fitcfg=fitcfg,
        use_h1=use_h1,
    )
    # fit keys: 'pairs', 'n', 'iu', 'mcmc', 'phi_samples', 'Lambda_hat', 'elapsed'
    return fit["Lambda_hat"], fit["phi_samples"], fit

def run_rings_experiment(
    feat_dir: str | Path = FEAT_DIR,
    out_root: str | Path = OUT_ROOT,
    fitcfg=None,
    phi_diag_indices: list[tuple[int, int]] | None = None,
    max_lag: int = 100,
    thin_every_for_diag: int = 1,
    use_h1: bool = True,
):
    """
    High-level wrapper for the rings H0+H1 simulation:

      1) Builds simulated features in `feat_dir`
      2) Runs NUTS using H0+H1 likelihood
      3) Saves:
           - per-class Λ̂
           - φ draws
           - trace & ACF pngs for selected φ entries
           - latent 2D embeddings
           - confusion matrix + accuracy

    Returns:
      class_Lambda, C, acc
    """
    feat_dir = Path(feat_dir)
    out_root = Path(out_root)

    # 1) build features
    build_simulation_features_using_official_extractor()

    # 2) fit config
    if fitcfg is None:
        fitcfg = default_fitcfg()

    if phi_diag_indices is None:
        phi_diag_indices = [(0, 0), (10, 0)]

    class_Lambda, C, acc = run_and_save_all(
        feat_dir=str(feat_dir),
        out_root=str(out_root),
        fitcfg=fitcfg,
        phi_diag_indices=phi_diag_indices,
        max_lag=max_lag,
        thin_every_for_diag=thin_every_for_diag,
        use_h1=use_h1,
    )
    return class_Lambda, C, acc


if __name__ == "__main__":
    # Keep a CLI entry point if you ever want to run this as a script:
    run_rings_experiment()







