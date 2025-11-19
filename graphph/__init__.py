"""
graphph

Code supplement for the manuscript:

  Zitian Wu, Arkaprava Roy, Leo L. Duan
  "Graphical Model-based Inference on Persistent Homology"

This lightweight package provides the core functions used in the paper:

  - graphph.h0h1_connectome
      * simulation and model-fitting code for H0 / H0–H1 features
      * connectome and connectome-like experiments
      * MLE and hierarchical NUTS (NumPyro/JAX)

  - graphph.latent_postproc
      * plotting of latent coordinates (Λ → Z)
      * ROI-wise latent-distance violin plots
      * FDR-based selection of ROIs from posterior draws

The Jupyter notebooks in `notebooks/` import these submodules explicitly, e.g.:

    import graphph.h0h1_connectome as gpc
    from graphph.latent_postproc import (
        plot_latent_coords_panels,
        latent_violin_from_samples,
        fdr_select_rois_from_latentdiff,
        replot_latent_violin_from_npz,
    )

No heavy work is done at package import time; all public APIs live in the
submodules above.
"""

# We deliberately do NOT import submodules here, to keep `import graphph`
# fast and side-effect free.  Everything is accessed via:
#   import graphph.h0h1_connectome as gpc
#   from graphph import latent_postproc
__all__: list[str] = []
