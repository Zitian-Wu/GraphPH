# graphph/latent_postproc.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, List

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# =====================================================================
# 1. FDR on ROI latent-position differences
# =====================================================================

def fdr_select_rois_from_latentdiff(
    npz_path: str | Path,
    beta: float = 0.9,          # posterior prob threshold Π(H1|y) > beta
    alpha: float | None = None, # target FDR level; default = 1 - beta
    n_grid: int = 201,          # number of epsilon grid points
    eps_min_frac: float = 0.4,  # start near median (~0.60 of max); tweak this
    out_csv: str | Path | None = None,
    verbose: bool = True,
):
    """
    FDR on ROI latent-position differences.

    Inputs
    ------
    npz_path : str or Path
        Output file from latent_violin_from_samples(..., save_draw_diffs_dir=...).
        Must contain arrays:
            - diffs: (T, N) per-draw ROI latent diff norms
            - roi_names: (N,)
            - A_disp, B_disp, ref_disp, rank (scalars / 0-d arrays)

    beta : float
        Posterior prob threshold for calling H1 at ROI-level:
            d_i(eps) = 1{ P(||Δz_i|| > eps | data) > beta }.

    alpha : float or None
        Posterior FDR target level; if None, alpha = 1 - beta.

    n_grid : int
        Number of epsilon grid points between 0 and eps_max.

    out_csv : str or Path or None
        If provided, saves a CSV of ROI-level summary + selection.

    Returns
    -------
    result : dict with keys:
        - eps_star: chosen epsilon
        - alpha, beta
        - fdr_star: FDR at eps_star
        - selected_idx: np.ndarray of selected ROI indices
        - selected_rois: list of selected ROI names
        - summary: pandas DataFrame with ROI-level stats
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)

    diffs = data["diffs"]           # (T, N)
    roi_names = data["roi_names"].astype(str)

    try:
        A_disp = data["A_disp"].item()
    except Exception:
        A_disp = str(data["A_disp"])
    try:
        B_disp = data["B_disp"].item()
    except Exception:
        B_disp = str(data["B_disp"])
    rank = int(data["rank"])

    T, N = diffs.shape
    if alpha is None:
        alpha = 1.0 - beta

    if verbose:
        print(f"[FDR] Loaded {npz_path}")
        print(f"[FDR] Groups: {A_disp} vs {B_disp}, rank={rank}")
        print(f"[FDR] T (draws) = {T}, N (ROIs) = {N}")
        print(f"[FDR] beta = {beta:.3f}, target FDR alpha = {alpha:.3f}")

    T_draws = diffs

    # eps range based on data (you already saw q90 ~ 1.46)
    eps_max = float(np.quantile(T_draws, 0.9))
    if eps_max <= 0:
        eps_max = float(T_draws.max())
    if eps_max <= 0:
        raise ValueError("All ROI differences are zero; nothing to select.")

    # use median-ish starting point: q50/eps_max ≈ 0.60 from your numbers
    eps_min = max(1e-8, eps_min_frac * eps_max)
    if verbose:
        print(f"[FDR] eps_min = {eps_min:.4f}, eps_max = {eps_max:.4f}")

    eps_grid = np.linspace(eps_min, eps_max, n_grid)

    best_eps  = None
    best_mask = None
    best_fdr  = None

    # IMPORTANT: we do NOT break when condition is satisfied;
    # we keep the LARGEST eps with FDR_hat <= alpha.
    for eps in eps_grid:
        p_H1 = (T_draws > eps).mean(axis=0)   # (N,)
        p_H0 = 1.0 - p_H1

        d = (p_H1 > beta).astype(float)
        num_calls = d.sum()

        if num_calls == 0:
            if verbose:
                print(f"  eps={eps:.4f}  calls=  0  (no ROIs exceed beta)")
            continue

        fdr_hat = float((d * p_H0).sum() / num_calls)

        if verbose:
            print(f"  eps={eps:.4f}  calls={int(num_calls):3d}  FDR_hat={fdr_hat:.4f}")

        if fdr_hat <= alpha:
            best_eps  = eps
            best_mask = (d == 1)
            best_fdr  = fdr_hat
            # DO NOT break; we want the largest eps

    if best_eps is None:
        if verbose:
            print("[FDR] No epsilon found with FDR_hat <= alpha; selecting no ROIs.")
        selected_idx = np.array([], dtype=int)
        selected_rois = []
        p_H1_final = (T_draws > eps_min).mean(axis=0)
    else:
        if verbose:
            print(f"[FDR] Chosen eps* = {best_eps:.4f} with FDR_hat = {best_fdr:.4f}")
        selected_idx = np.where(best_mask)[0]
        selected_rois = [roi_names[i] for i in selected_idx]
        p_H1_final = (T_draws > best_eps).mean(axis=0)

    mean_T   = T_draws.mean(axis=0)
    median_T = np.median(T_draws, axis=0)
    q1 = np.percentile(T_draws, 25, axis=0)
    q3 = np.percentile(T_draws, 75, axis=0)

    selected_flag = np.zeros(N, dtype=int)
    selected_flag[selected_idx] = 1

    df = pd.DataFrame({
        "roi": roi_names,
        "mean_norm": mean_T,
        "median_norm": median_T,
        "q1": q1,
        "q3": q3,
        "post_prob_H1": p_H1_final,
        "selected": selected_flag,
    }).sort_values("mean_norm", ascending=False).reset_index(drop=True)

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        if verbose:
            print(f"[FDR] Saved ROI summary with selection → {out_csv}")

    return {
        "eps_star": best_eps,
        "alpha": alpha,
        "beta": beta,
        "fdr_star": best_fdr,
        "selected_idx": selected_idx,
        "selected_rois": selected_rois,
        "summary": df,
        "A_disp": A_disp,
        "B_disp": B_disp,
        "rank": rank,
    }

# =====================================================================
# 2. Latent coords panels (from Λ)
# =====================================================================

def _svd_rank2_coords(L: np.ndarray) -> np.ndarray:
    """Symmetrize, SVD, take top-2 eigen-embedding; return Z in R^{n x 2}."""
    Ls = 0.5 * (L + L.T)
    U, S, _ = np.linalg.svd(Ls, full_matrices=False)
    s2 = np.maximum(S[:2], 0.0)
    Z2 = U[:, :2] * np.sqrt(s2)
    if Z2.shape[1] == 1:  # edge case: only one positive direction
        Z2 = np.hstack([Z2, np.zeros_like(Z2)])
    return Z2


def _frac_energy_rank2(L: np.ndarray) -> float:
    w = np.linalg.eigvalsh(0.5 * (L + L.T))
    w = np.clip(w, 0, None)
    tot = float(w.sum())
    if tot <= 0:
        return 0.0
    idx = np.argsort(w)[::-1]
    return float(w[idx[:2]].sum() / (tot + 1e-12))


def _procrustes_rotate(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Orientation alignment: argmin_R || X R - Y ||_F (R orthogonal)."""
    U, _, Vt = np.linalg.svd(X.T @ Y, full_matrices=False)
    R = U @ Vt
    return X @ R


def _expand_limits(xmin: float, xmax: float, factor: float) -> Tuple[float, float]:
    """
    Expand [xmin, xmax] by 'factor' around its center (factor >= 1).
    Values unchanged; just a looser window.
    """
    if factor <= 1.0:
        return xmin, xmax
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        return xmin, xmax
    if xmin == xmax:
        eps = 1.0
        xmin, xmax = xmin - eps, xmax + eps
    c = 0.5 * (xmin + xmax)
    r = 0.5 * (xmax - xmin)
    r2 = r * factor
    return (c - r2, c + r2)


def _save_coords_panel(
    Z2: np.ndarray,
    out_path: Path,
    *,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    base_figsize: Tuple[float, float] = (2.4, 2.4),
    x_stretch: float = 1.0,
    dpi: int = 600,
    s: float = 12,
    color: str = "tab:blue",
    show_ticks: bool = True,
    show_labels: bool = True,
    xlabel: str = r"$z_1$",
    ylabel: str = r"$z_2$",
):
    """
    Panel with optional stretched x-axis:
      - x_stretch > 1 enlarges the x axis length (geometry only).
      - Data values, limits and ticks are untouched.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_w, base_h = base_figsize
    fig_w = base_w * x_stretch
    fig_h = base_h

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.scatter(Z2[:, 0], Z2[:, 1], s=s, c=color)

    if xlim is not None and ylim is not None:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.margins(x=0.02, y=0.02)
    else:
        ax.autoscale()
        ax.margins(x=0.08, y=0.08)

    if x_stretch > 1.0:
        ax.set_aspect("auto")
    else:
        ax.set_aspect("equal", adjustable="box")

    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if show_labels:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_latent_coords_panels(
    run_dir: str | Path,
    groups: List[str],
    *,
    align_to: Optional[str] = None,
    out_dir: str | Path = "figs/latent_coords",
    panel_size_in: float = 2.4,
    dpi: int = 600,
    point_size: float = 8,
    point_color: str = "tab:blue",
    use_shared_limits: bool = True,
    expand: float = 1.15,
    x_stretch: float = 1.0,
    bar_label: str = "_bar",
) -> Dict[str, float]:
    """
    Plot 2D latent embeddings from Λ for each group (and optional pooled '_bar').

    Parameters
    ----------
    run_dir : str or Path
        HMC run directory that contains per-class folders with Lambda_hat.npz.
    groups : list of str
        Class ids, e.g. ["Group1","Group2","Group3"].
    align_to : str or None
        If provided, Procrustes-align all groups to this group's embedding.
    out_dir : str or Path
        Root directory for 'latent_coords.png' panels.
    x_stretch : float
        >1 stretches x-axis (visual elongation, no change to underlying values).

    Returns
    -------
    frac_energy : dict
        Mapping cid -> fraction of spectral energy captured by rank-2 embedding.
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)

    Lambda_by: Dict[str, np.ndarray] = {}
    Z2_by: Dict[str, np.ndarray] = {}
    E2_by: Dict[str, float] = {}

    for cid in groups:
        npz = run_dir / cid / "Lambda_hat.npz"
        if not npz.exists():
            raise FileNotFoundError(f"Missing {npz}")
        with np.load(npz) as z:
            L = z["Lambda_hat"].astype(float)
        Lambda_by[cid] = L
        Z2_by[cid] = _svd_rank2_coords(L)
        E2_by[cid] = _frac_energy_rank2(L)

    # optional Procrustes alignment
    if align_to is not None and align_to in Z2_by:
        Zref = Z2_by[align_to]
        for cid in groups:
            if cid == align_to:
                continue
            Z2_by[cid] = _procrustes_rotate(Z2_by[cid], Zref)

    # shared limits if requested
    if use_shared_limits:
        all_pts = np.vstack([Z2_by[cid] for cid in groups])
        xmin, xmax = float(all_pts[:, 0].min()), float(all_pts[:, 0].max())
        ymin, ymax = float(all_pts[:, 1].min()), float(all_pts[:, 1].max())
        XLIM = _expand_limits(xmin, xmax, expand)
        YLIM = _expand_limits(ymin, ymax, expand)
    else:
        XLIM = YLIM = None

    # per-group panels
    for cid in groups:
        if not use_shared_limits:
            x0, x1 = float(Z2_by[cid][:, 0].min()), float(Z2_by[cid][:, 0].max())
            y0, y1 = float(Z2_by[cid][:, 1].min()), float(Z2_by[cid][:, 1].max())
            xlim = _expand_limits(x0, x1, expand)
            ylim = _expand_limits(y0, y1, expand)
        else:
            xlim, ylim = XLIM, YLIM

        out_png = out_dir / cid / "latent_coords.png"
        _save_coords_panel(
            Z2_by[cid],
            out_png,
            xlim=xlim,
            ylim=ylim,
            base_figsize=(panel_size_in, panel_size_in),
            x_stretch=x_stretch,
            dpi=dpi,
            s=point_size,
            color=point_color,
            show_ticks=True,
            show_labels=True,
        )
        print(f"[latent] Saved {out_png} | rank-2 energy={E2_by[cid]:.3f}")

    # optional pooled "_bar"
    bar_npz = run_dir / bar_label / "Lambda_hat.npz"
    if bar_npz.exists():
        with np.load(bar_npz) as z:
            Lbar = z["Lambda_hat"].astype(float)
        Zbar = _svd_rank2_coords(Lbar)
        if align_to is not None and align_to in Z2_by:
            Zbar = _procrustes_rotate(Zbar, Z2_by[align_to])

        if not use_shared_limits:
            x0, x1 = float(Zbar[:, 0].min()), float(Zbar[:, 0].max())
            y0, y1 = float(Zbar[:, 1].min()), float(Zbar[:, 1].max())
            xlim = _expand_limits(x0, x1, expand)
            ylim = _expand_limits(y0, y1, expand)
        else:
            xlim, ylim = XLIM, YLIM

        out_png = out_dir / bar_label / "latent_coords.png"
        _save_coords_panel(
            Zbar,
            out_png,
            xlim=xlim,
            ylim=ylim,
            base_figsize=(panel_size_in, panel_size_in),
            x_stretch=x_stretch,
            dpi=dpi,
            s=point_size,
            color=point_color,
            show_ticks=True,
            show_labels=True,
        )
        e2b = _frac_energy_rank2(Lbar)
        print(f"[latent] Saved {out_png} | rank-2 energy={e2b:.3f}")

    return E2_by


# =====================================================================
# 3. ROI-wise latent-diff violins from HMC samples
# =====================================================================

def _load_blocks(run_dir: Path):
    meta = json.loads((run_dir / "blocks.json").read_text())
    n, m = int(meta["n"]), int(meta["m"])
    classes = list(meta["classes"])
    order = meta["order"]                # ["_bar", <class1>, ...]
    blk = int(meta["block_size"])        # n*m
    return n, m, classes, order, blk


def _split_theta(theta: np.ndarray, blk: int, order: List[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    off = 0
    for key in order:
        out[key] = theta[:, off:off+blk]
        off += blk
    return out


def _phi_to_Z(phi_flat: np.ndarray, n: int, m: int) -> np.ndarray:
    """
    Map flattened Φ to Z via softplus:
        Z = softplus(Φ)  (elementwise).
    """
    Phi = phi_flat.reshape(n, m)
    return np.log1p(np.exp(Phi))


def _svd_rankK_coords(L: np.ndarray, k: int) -> np.ndarray:
    Ls = 0.5 * (L + L.T)
    w, V = np.linalg.eigh(Ls)
    idx = np.argsort(w)[::-1]
    w, V = w[idx], V[:, idx]
    w = np.clip(w, 0.0, None)
    k_eff = min(k, int(np.sum(w > 0)))
    if k_eff == 0:
        return np.zeros((L.shape[0], k))
    Zk = V[:, :k_eff] * np.sqrt(w[:k_eff])
    if k_eff < k:
        Z = np.zeros((L.shape[0], k))
        Z[:, :k_eff] = Zk
        Zk = Z
    return Zk


def _procrustes_to_ref(Z_ref: np.ndarray, Z: np.ndarray) -> np.ndarray:
    C = Z.T @ Z_ref
    U, _, Vt = np.linalg.svd(C, full_matrices=False)
    R = U @ Vt
    return Z @ R


def _latent_diff_draw(ZA: np.ndarray, ZB: np.ndarray) -> np.ndarray:
    return np.linalg.norm(ZA - ZB, axis=1)


def latent_violin_from_samples(
    out_root: str | Path,
    pair: tuple[str, str] = ("Group1", "Group3"),
    rank: int = 5,
    ref_label: str | None = None,
    node_csv: str | None = "data/node_labels.csv",
    top_k: int | None = 20,
    png_path: str | Path | None = None,
    csv_path: str | Path | None = None,
    color: str = "C0",
    label_map: dict[str, str] | None = None,   # {"Group1":"CN","Group2":"MCI","Group3":"AD"}
    sort: bool = False,
    save_draw_diffs_dir: str | Path | None = None,
) -> dict:
    """
    Build ROI-wise latent-position differences from HMC samples and plot violins.

    - Reads: out_root/{theta_samples.npz, blocks.json}
    - Reconstructs per-draw Λ via Φ -> Z = softplus(Φ) -> Λ = ZZᵀ
    - Rank-K eigen-embedding per draw; Procrustes-align to ref group
    - Per-ROI ||z_i^(A) - z_i^(B)||_2 across draws → violin per ROI
    - Optionally saves full per-draw diffs to NPZ for FDR analysis.
    """
    out_root = Path(out_root)
    n, m, classes, order, blk = _load_blocks(out_root)

    label_map = label_map or {}
    inv_map = {v: k for k, v in label_map.items()}  # e.g. CN -> Group1

    def to_internal(k: str) -> str:
        if k in classes:
            return k
        if k in inv_map and inv_map[k] in classes:
            return inv_map[k]
        raise ValueError(f"Unknown class '{k}'. Known: {classes} or {list(inv_map)}")

    def to_display(k: str) -> str:
        return label_map.get(k, k)

    A_int, B_int = to_internal(pair[0]), to_internal(pair[1])
    A_disp, B_disp = to_display(A_int), to_display(B_int)

    if ref_label is None:
        ref_label = A_int
    else:
        ref_label = to_internal(ref_label)
    ref_disp = to_display(ref_label)

    # ---- load samples and split blocks ----
    theta = np.load(out_root / "theta_samples.npz")["theta"]
    parts = _split_theta(theta, blk=blk, order=order)

    phiA, phiB, phiR = parts[A_int], parts[B_int], parts[ref_label]
    T = phiA.shape[0]
    diffs = np.empty((T, n), float)

    for t in range(T):
        ZA = _phi_to_Z(phiA[t], n, m); LA = ZA @ ZA.T
        ZB = _phi_to_Z(phiB[t], n, m); LB = ZB @ ZB.T
        ZR = _phi_to_Z(phiR[t], n, m); LR = ZR @ ZR.T

        ZAk = _svd_rankK_coords(LA, rank)
        ZBk = _svd_rankK_coords(LB, rank)
        ZRk = _svd_rankK_coords(LR, rank)

        ZAk = _procrustes_to_ref(ZRk, ZAk)
        ZBk = _procrustes_to_ref(ZRk, ZBk)

        diffs[t] = _latent_diff_draw(ZAk, ZBk)

    # ---- ROI names ----
    if node_csv is not None and Path(node_csv).exists():
        roi_names: List[str] = []
        with open(node_csv, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip().lstrip("\ufeff")
                if not s or s.lower().startswith("region"):
                    continue
                roi_names.append(s)
        if len(roi_names) != n:
            roi_names = [f"ROI {i+1}" for i in range(n)]
    else:
        roi_names = [f"ROI {i+1}" for i in range(n)]

    # ---- save full diffs for FDR analysis ----
    if save_draw_diffs_dir is not None:
        save_dir = Path(save_draw_diffs_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        base_npz = f"latentdiff_draws_rank{rank}_{A_disp}_vs_{B_disp}.npz"
        npz_path = save_dir / base_npz
        np.savez_compressed(
            npz_path,
            diffs=diffs,
            roi_names=np.array(roi_names),
            A_disp=A_disp,
            B_disp=B_disp,
            ref_disp=ref_disp,
            rank=rank,
        )
        print(f"[FDR] saved per-draw diffs → {npz_path}")

    # ---- selection for violin plot ----
    if top_k is None:
        idx = np.arange(n)
    else:
        if sort:
            med_all = np.median(diffs, axis=0)
            idx = np.argsort(med_all)[::-1][:min(top_k, n)]
        else:
            idx = np.arange(min(top_k, n))

    sel_names = [roi_names[i] for i in idx]
    sel_diffs = diffs[:, idx]
    K = len(idx)

    # ---- output paths ----
    base = f"violin_latentdiff_rank{rank}_{A_disp}_vs_{B_disp}_" + ("all" if K == n else f"top{K}")
    if png_path is None:
        png_path = out_root / f"{base}.png"
    if csv_path is None:
        csv_path = out_root / f"{base}.csv"
    png_path = Path(png_path)
    csv_path = Path(csv_path)

    # ---- CSV summary ----
    med_sel = np.median(sel_diffs, axis=0)
    q1  = np.percentile(sel_diffs, 25, axis=0)
    q3  = np.percentile(sel_diffs, 75, axis=0)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("roi,median,q1,q3\n")
        for nm, m_, a, b in zip(sel_names, med_sel, q1, q3):
            f.write(f"{nm},{m_:.6g},{a:.6g},{b:.6g}\n")

    # ---- violin plot ----
    width = max(12.0, 0.18 * K)
    fig, ax = plt.subplots(figsize=(width, 4.8), dpi=450)
    parts = ax.violinplot(
        [sel_diffs[:, j] for j in range(K)],
        positions=np.arange(1, K + 1),
        showmeans=False, showmedians=False, showextrema=False,
    )
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.25)
        body.set_linewidth(1.0)

    ax.scatter(
        np.arange(1, K + 1),
        np.median(sel_diffs, axis=0),
        s=18, c="black", zorder=3,
    )

    ax.set_xlim(0.5, K + 0.5)
    label_tail = "all ROIs" if K == n else (f"top-{K} ROIs" + (" (sorted)" if sort else " (original order)"))
    ax.set_xlabel(f"{A_disp} vs {B_disp} — {label_tail}")
    ax.set_ylabel(r"$\|z_i^{(" + A_disp + r")}-z_i^{(" + B_disp + r")}\|_2$")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.set_xticks(np.arange(1, K + 1))
    ax.set_xticklabels(sel_names, rotation=60, ha="right")

    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[violin] saved figure → {png_path}")
    print(f"[violin] saved summary → {csv_path}")

    return dict(
        png=str(png_path),
        csv=str(csv_path),
        top_roi=sel_names,
        npz=str(save_draw_diffs_dir) if save_draw_diffs_dir is not None else None,
    )


# =====================================================================
# 4. Clean replot + FDR-annotated violin from saved .npz
# =====================================================================

def _norm_roi_name(s: str) -> str:
    return (
        str(s).strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
    )


def _norm_roi_full(roi: str, hemi: str | None = None) -> str:
    """
    Normalize ROI name, optionally combining hemi + base region
    so that 'L rostralmiddlefrontal' in node_labels and
    ('rostralmiddlefrontal','L') in CSV map to the same key.
    """
    roi = str(roi).strip()
    if hemi is not None and hemi != "" and not roi.lower().startswith(("l ", "r ")):
        roi = f"{hemi.upper()} {roi}"
    return _norm_roi_name(roi)


def _set_manuscript_rc():
    mpl.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })


def replot_latent_violin_from_npz(
    npz_path: str | Path,
    top_k: int | None = None,          # None = all ROIs
    sort: bool = False,                # sort by median diff (desc)
    fig_width: float | None = None,    # in inches
    fig_height: float | None = None,   # in inches
    color: str = "C0",
    png_path: str | Path | None = None,
    csv_path: str | Path | None = None,
    fdr_csv: str | Path | None = None,
    label_only_fdr: bool = False,      # (kept for API compatibility, but x labels off)
    annotate_fdr_points: bool = False, # put ROI labels near median points for FDR ROIs
    set_rc: bool = True,
) -> dict:
    """
    Re-plot latent diff violins from a saved latentdiff_draws_*.npz,
    optionally annotating FDR-selected ROIs using an external CSV.
    """
    if set_rc:
        _set_manuscript_rc()

    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)

    diffs = data["diffs"]
    roi_names_arr = data["roi_names"]
    roi_names = [str(x) for x in roi_names_arr.tolist()]

    A_disp = str(data["A_disp"].item())
    B_disp = str(data["B_disp"].item())
    rank   = int(data["rank"].item())

    T, n = diffs.shape

    # selection
    if top_k is None:
        idx = np.arange(n)
    else:
        top_k = min(top_k, n)
        if sort:
            med_all = np.median(diffs, axis=0)
            idx = np.argsort(med_all)[::-1][:top_k]
        else:
            idx = np.arange(top_k)

    sel_names = [roi_names[i] for i in idx]
    sel_diffs = diffs[:, idx]
    K = len(idx)

    # figure size
    if fig_width is None:
        fig_width = min(max(0.16 * K, 4.0), 6.5)
    if fig_height is None:
        fig_height = 2.4

    # output paths
    base = f"violin_replot_rank{rank}_{A_disp}_vs_{B_disp}_" + ("all" if K == n else f"top{K}")
    out_dir = npz_path.parent

    if png_path is None:
        png_path = out_dir / f"{base}.png"
    if csv_path is None:
        csv_path = out_dir / f"{base}.csv"
    png_path = Path(png_path)
    csv_path = Path(csv_path)

    # CSV summary
    med_sel = np.median(sel_diffs, axis=0)
    q1  = np.percentile(sel_diffs, 25, axis=0)
    q3  = np.percentile(sel_diffs, 75, axis=0)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("roi,median,q1,q3\n")
        for nm, m_, a, b in zip(sel_names, med_sel, q1, q3):
            f.write(f"{nm},{m_:.6g},{a:.6g},{b:.6g}\n")

    # FDR ROIs
    fdr_sig_rois: set[str] = set()
    if (label_only_fdr or annotate_fdr_points) and fdr_csv is not None and Path(fdr_csv).exists():
        try:
            df_fdr = pd.read_csv(fdr_csv)

            if "roi" not in df_fdr.columns:
                print("[FDR] 'roi' column not found in CSV; skipping FDR-based labels.")
            else:
                pair_label = f"{A_disp}_vs_{B_disp}"
                if "pair" in df_fdr.columns:
                    df_fdr = df_fdr[df_fdr["pair"].astype(str).str.strip() == pair_label]

                if df_fdr.empty:
                    print(f"[FDR] No FDR rows for pair {pair_label}; labels will be full.")
                else:
                    if "selected" in df_fdr.columns:
                        df_fdr = df_fdr[df_fdr["selected"] == 1]

                    norm_sel = {_norm_roi_full(nm): nm for nm in sel_names}

                    hems = df_fdr["hemi"].astype(str) if "hemi" in df_fdr.columns else [None] * len(df_fdr)
                    for r, h in zip(df_fdr["roi"].astype(str), hems):
                        key = _norm_roi_full(r, hemi=h)
                        if key in norm_sel:
                            fdr_sig_rois.add(norm_sel[key])

                    if not fdr_sig_rois:
                        print("[FDR] No overlapping FDR ROIs found for this comparison; labels will be full.")
        except Exception as e:
            print(f"[FDR] error reading FDR CSV: {e}")

    # plot
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=450)

    parts = ax.violinplot(
        [sel_diffs[:, j] for j in range(K)],
        positions=np.arange(1, K + 1),
        showmeans=False, showmedians=False, showextrema=False,
    )
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.25)
        body.set_linewidth(1.0)

    medians = np.median(sel_diffs, axis=0)
    ax.scatter(
        np.arange(1, K + 1),
        medians,
        s=18, c="black", zorder=3,
    )

    ax.set_xlim(0.5, K + 0.5)
    label_tail = "all ROIs" if K == n else (f"top-{K} ROIs" + (" (sorted)" if sort else " (original order)"))
    ax.set_xlabel(f"{A_disp} vs {B_disp} — {label_tail}")
    ax.set_ylabel(r"$\|z_i^{(" + A_disp + r")}-z_i^{(" + B_disp + r")}\|_2$")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # hide x-axis ROI names (your manuscript version)
    ax.set_xticks(np.arange(1, K + 1))
    ax.set_xticklabels([""] * K)
    ax.tick_params(axis="x", length=0)

    # optional FDR labels above points
    if annotate_fdr_points and fdr_sig_rois:
        texts = []
        xs = []
        ys = []

        value_range = medians.max() - medians.min()
        if value_range <= 0:
            value_range = max(abs(medians.max()), 1.0)

        base_dy = 0.04 * value_range

        for j, nm in enumerate(sel_names):
            if nm in fdr_sig_rois:
                x = j + 1
                y = medians[j]
                xs.append(x)
                ys.append(y)
                txt = ax.text(
                    x,
                    y + base_dy,
                    nm,
                    fontsize=11,
                    ha="center",
                    va="bottom",
                )
                texts.append(txt)

        try:
            from adjustText import adjust_text  # type: ignore[import]
            adjust_text(
                texts,
                x=xs,
                y=[y + base_dy for y in ys],
                ax=ax,
                arrowprops=None,
                only_move={"points": "y", "texts": "y"},
                expand_points=(1.1, 1.3),
                expand_text=(1.2, 1.4),
                force_text=0.9,
                lim=1000,
            )
        except ImportError:
            # simple fallback: vertical spreading
            min_gap = 0.04 * value_range
            used_y: List[float] = []
            for t in texts:
                x, y = t.get_position()
                while any(abs(y - yy) < min_gap for yy in used_y):
                    y += min_gap
                    t.set_position((x, y))
                used_y.append(y)
            print(
                "[FDR] adjustText not installed; used simple vertical spreading instead. "
                "For better adjustment, run: pip install adjustText"
            )

    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight", dpi=500)
    plt.close(fig)

    print(f"[replot] saved figure → {png_path}")
    print(f"[replot] saved summary → {csv_path}")

    return dict(
        png=str(png_path),
        csv=str(csv_path),
        top_roi=sel_names,
        A_disp=A_disp,
        B_disp=B_disp,
        rank=rank,
    )


# =====================================================================
# Public API
# =====================================================================

__all__ = [
    "fdr_select_rois_from_latentdiff",
    "plot_latent_coords_panels",
    "latent_violin_from_samples",
    "replot_latent_violin_from_npz",
]
