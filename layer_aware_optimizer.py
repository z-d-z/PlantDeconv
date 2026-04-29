"""
Differentiation-continuity-aware optimizer for PlantDeconv v2.1.

Extends the base Mapper with three layers of differentiation-continuity
regularisation designed for plant tissues:

* **Layer 1 — Expression-aware spatial continuity**: penalises abrupt
  cell-type composition changes between neighbouring spots, with edge
  weights that combine spatial proximity and expression similarity.
  Boundaries with large expression differences are naturally less
  smoothed.

* **Layer 2 — Anisotropic differentiation gradient**: uses
  differentiation-trajectory-aware neighbour weighting so that expected
  smoothness accounts for how transcriptomically similar two regions are.
  Same-region neighbours are weighted more; cross-boundary neighbours
  whose regions are far apart in expression space are weighted less.

* **Layer 3 — Adaptive per-spot regularisation**: each spot has an
  individual regularisation scaling factor based on local expression
  heterogeneity.  Homogeneous tissue interior → strong smoothing;
  heterogeneous boundary zones → weak smoothing.

These three layers are topology-agnostic and apply to any plant tissue,
complementing the optional region-prior penalties.
"""

import logging

import numpy as np
import torch
from torch.nn.functional import cosine_similarity, softmax

from .optimizer import Mapper

LOGGER = logging.getLogger(__name__)
EPS = 1e-12


class LayerAwareMapper(Mapper):
    """
    Mapper with differentiation-continuity-aware regularisation.

    Penalty terms (all optional — controlled by lambda weights):

    **Region-based (optional — useful when region annotations exist):**

    - ``layer_prior``: KL divergence between predicted and expected region
      distributions per cell type.
    - ``out_of_band``: penalises mapping probability to spots outside a
      cell type's supported regions.
    - ``layer_distance``: penalises mapping proportional to distance from
      supported regions.

    **Differentiation-continuity (core v2.1 — work with any plant tissue):**

    - ``spatial_continuity`` (Layer 1): expression-aware spatial smoothing.
    - ``differentiation_gradient`` (Layer 2): trajectory-aware gradient.
    - Both are modulated by ``adaptive_weights`` (Layer 3): per-spot
      regularisation strength based on local heterogeneity.
    """

    def __init__(
        self,
        *,
        # ---- Region-based inputs (optional) ----
        spot_layer_matrix=None,
        ct_layer_prior=None,
        out_of_band_mask=None,
        ct_spot_distance=None,
        lambda_layer_prior=0.0,
        lambda_out_of_band=0.0,
        lambda_layer_distance=0.0,
        # ---- Differentiation-continuity inputs (core v2.1) ----
        spot_adjacency=None,
        gradient_weights=None,
        adaptive_weights=None,
        lambda_spatial_continuity=0.0,
        lambda_differentiation_gradient=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # --- Region-based (optional) ---
        self.lambda_layer_prior = float(lambda_layer_prior)
        self.lambda_out_of_band = float(lambda_out_of_band)
        self.lambda_layer_distance = float(lambda_layer_distance)
        self._layer_criterion = torch.nn.KLDivLoss(reduction="batchmean")

        self.spot_layer_matrix = None
        if spot_layer_matrix is not None:
            self.spot_layer_matrix = torch.tensor(
                spot_layer_matrix, device=self.device, dtype=torch.float32
            )

        self.ct_layer_prior = None
        if ct_layer_prior is not None:
            self.ct_layer_prior = torch.tensor(
                ct_layer_prior, device=self.device, dtype=torch.float32
            )

        self.out_of_band_mask = None
        if out_of_band_mask is not None:
            self.out_of_band_mask = torch.tensor(
                out_of_band_mask, device=self.device, dtype=torch.float32
            )

        self.ct_spot_distance = None
        if ct_spot_distance is not None:
            self.ct_spot_distance = torch.tensor(
                ct_spot_distance, device=self.device, dtype=torch.float32
            )

        # --- Differentiation-continuity (core v2.1) ---
        self.lambda_spatial_continuity = float(lambda_spatial_continuity)
        self.lambda_differentiation_gradient = float(lambda_differentiation_gradient)

        self.spot_adjacency = None
        if spot_adjacency is not None:
            self.spot_adjacency = torch.tensor(
                spot_adjacency, device=self.device, dtype=torch.float32
            )

        self.gradient_weights = None
        if gradient_weights is not None:
            self.gradient_weights = torch.tensor(
                gradient_weights, device=self.device, dtype=torch.float32
            )

        # Layer 3: adaptive per-spot regularisation strength
        self.adaptive_weights = None
        if adaptive_weights is not None:
            self.adaptive_weights = torch.tensor(
                adaptive_weights, device=self.device, dtype=torch.float32
            )

    def _loss_fn(self, verbose=True):
        G = self.G_train
        S = self.S_train
        M_probs = softmax(self.M, dim=1)
        G_pred = torch.matmul(M_probs.t(), S)

        gv_term = self.lambda_g1 * cosine_similarity(G_pred, G, dim=0).mean()
        vg_term = self.lambda_g2 * cosine_similarity(G_pred, G, dim=1).mean()
        expression_term = gv_term + vg_term
        main_loss = (gv_term / self.lambda_g1).tolist()
        vg_reg = (vg_term / self.lambda_g2).tolist()

        if self.target_density_enabled:
            if self.source_density_enabled:
                d_pred = torch.log(self.d_source @ M_probs + EPS)
            else:
                d_pred = torch.log(M_probs.sum(dim=0) / self.M.shape[0] + EPS)
            density_term = self.lambda_d * self._density_criterion(d_pred, self.d)
            kl_reg = (density_term / self.lambda_d).tolist()
        else:
            density_term, kl_reg = 0, np.nan

        entropy_term = self.lambda_r * -(torch.log(M_probs + EPS) * M_probs).sum()
        entropy_reg = (entropy_term / self.lambda_r).tolist()

        if self.lambda_spot_entropy > 0:
            spot_ct = M_probs / (M_probs.sum(dim=0, keepdim=True) + EPS)
            spot_entropy_term = self.lambda_spot_entropy * -(
                torch.log(spot_ct + EPS) * spot_ct
            ).sum(dim=0).mean()
            spot_entropy_reg = (spot_entropy_term / self.lambda_spot_entropy).tolist()
        else:
            spot_entropy_term, spot_entropy_reg = 0, np.nan

        l1_term = self.lambda_l1 * self.M.abs().sum()
        l1_reg = (l1_term / self.lambda_l1).tolist()
        l2_term = self.lambda_l2 * (self.M ** 2).sum()
        l2_reg = (l2_term / self.lambda_l2).tolist()

        if self.lambda_neighborhood_g1 > 0:
            gv_neighborhood_term = self.lambda_neighborhood_g1 * cosine_similarity(
                self.voxel_weights @ G_pred, self.voxel_weights @ G, dim=0,
            ).mean()
            gv_neighborhood_sim = (gv_neighborhood_term / self.lambda_neighborhood_g1).tolist()
        else:
            gv_neighborhood_term, gv_neighborhood_sim = 0, np.nan

        if self.lambda_ct_islands > 0:
            ct_map = M_probs.T @ self.ct_encode
            ct_island_term = self.lambda_ct_islands * torch.max(
                ct_map - (self.neighborhood_filter @ ct_map),
                torch.tensor([0], dtype=torch.float32, device=self.device),
            ).mean()
            ct_island_penalty = (ct_island_term / self.lambda_ct_islands).tolist()
        else:
            ct_island_term, ct_island_penalty = 0, np.nan

        getis_ord_G_star_pred, moran_I_pred, gearys_C_pred = self._spatial_local_indicators(G_pred)

        getis_ord_term, moran_term, gearys_term = 0, 0, 0
        getis_ord_sim, moran_sim, gearys_sim = np.nan, np.nan, np.nan
        if self.lambda_getis_ord > 0:
            getis_ord_term = self.lambda_getis_ord * cosine_similarity(
                self.getis_ord_G_star_ref, getis_ord_G_star_pred, dim=0,
            ).mean()
            getis_ord_sim = (getis_ord_term / self.lambda_getis_ord).tolist()
        if self.lambda_moran > 0:
            moran_term = self.lambda_moran * cosine_similarity(
                self.moran_I_ref, moran_I_pred, dim=0,
            ).mean()
            moran_sim = (moran_term / self.lambda_moran).tolist()
        if self.lambda_geary > 0:
            gearys_term = self.lambda_geary * cosine_similarity(
                self.gearys_C_ref, gearys_C_pred, dim=0,
            ).mean()
            gearys_sim = (gearys_term / self.lambda_geary).tolist()

        # ---- Region-based penalties (optional) ----
        if (
            self.lambda_layer_prior > 0
            and self.spot_layer_matrix is not None
            and self.ct_layer_prior is not None
        ):
            pred_layer = M_probs @ self.spot_layer_matrix
            pred_layer = pred_layer / (pred_layer.sum(dim=1, keepdim=True) + EPS)
            layer_prior_term = self.lambda_layer_prior * self._layer_criterion(
                torch.log(pred_layer + EPS), self.ct_layer_prior,
            )
            layer_prior_reg = (layer_prior_term / self.lambda_layer_prior).tolist()
        else:
            layer_prior_term, layer_prior_reg = 0, np.nan

        if self.lambda_out_of_band > 0 and self.out_of_band_mask is not None:
            out_of_band_term = self.lambda_out_of_band * (
                M_probs * self.out_of_band_mask
            ).sum(dim=1).mean()
            out_of_band_reg = (out_of_band_term / self.lambda_out_of_band).tolist()
        else:
            out_of_band_term, out_of_band_reg = 0, np.nan

        if self.lambda_layer_distance > 0 and self.ct_spot_distance is not None:
            layer_distance_term = self.lambda_layer_distance * (
                M_probs * (self.ct_spot_distance ** 2)
            ).sum(dim=1).mean()
            layer_distance_reg = (layer_distance_term / self.lambda_layer_distance).tolist()
        else:
            layer_distance_term, layer_distance_reg = 0, np.nan

        # ---- Differentiation-continuity penalties (core v2.1) ----

        # Layer 1: Expression-aware spatial continuity
        # With Layer 3 adaptive modulation
        if self.lambda_spatial_continuity > 0 and self.spot_adjacency is not None:
            spot_props = M_probs.T  # (n_spots, n_ct)
            neighbor_props = self.spot_adjacency @ spot_props  # (n_spots, n_ct)
            per_spot_diff = ((spot_props - neighbor_props) ** 2).sum(dim=1)  # (n_spots,)

            # Apply adaptive per-spot weights (Layer 3)
            if self.adaptive_weights is not None:
                per_spot_diff = per_spot_diff * self.adaptive_weights

            continuity_diff = per_spot_diff.mean()
            spatial_continuity_term = self.lambda_spatial_continuity * continuity_diff
            spatial_continuity_reg = (spatial_continuity_term / self.lambda_spatial_continuity).tolist()
        else:
            spatial_continuity_term, spatial_continuity_reg = 0, np.nan

        # Layer 2: Anisotropic differentiation gradient
        # With Layer 3 adaptive modulation
        if self.lambda_differentiation_gradient > 0 and self.gradient_weights is not None:
            spot_props = M_probs.T  # (n_spots, n_ct)
            grad_neighbor_props = self.gradient_weights @ spot_props
            per_spot_grad = ((spot_props - grad_neighbor_props) ** 2).sum(dim=1)

            # Apply adaptive per-spot weights (Layer 3)
            if self.adaptive_weights is not None:
                per_spot_grad = per_spot_grad * self.adaptive_weights

            grad_diff = per_spot_grad.mean()
            gradient_term = self.lambda_differentiation_gradient * grad_diff
            gradient_reg = (gradient_term / self.lambda_differentiation_gradient).tolist()
        else:
            gradient_term, gradient_reg = 0, np.nan

        total_loss = (
            -expression_term
            + density_term + entropy_term + spot_entropy_term
            + l1_term + l2_term
            + ct_island_term - gv_neighborhood_term
            - getis_ord_term - moran_term - gearys_term
            + layer_prior_term + out_of_band_term + layer_distance_term
            + spatial_continuity_term + gradient_term
        )

        if verbose:
            term_numbers = [
                main_loss, vg_reg, kl_reg, entropy_reg, spot_entropy_reg,
                l1_reg, l2_reg, gv_neighborhood_sim, ct_island_penalty,
                getis_ord_sim, moran_sim, gearys_sim,
                layer_prior_reg, out_of_band_reg, layer_distance_reg,
                spatial_continuity_reg, gradient_reg,
            ]
            term_names = [
                "Gene-voxel score", "Voxel-gene score", "Cell densities reg",
                "Entropy reg", "Spot entropy reg", "L1 reg", "L2 reg",
                "Spatial weighted score", "Cell type islands penalty",
                "Getis-Ord score", "Moran score", "Geary score",
                "Layer prior reg", "Out-of-band reg", "Layer distance reg",
                "Spatial continuity reg (L1+L3)", "Differentiation gradient reg (L2+L3)",
            ]
            clean_dict = {
                key: value
                for key, value in zip(term_names, term_numbers)
                if not np.isnan(value)
            }
            msg = [f"{key}: {value:.3f}" for key, value in clean_dict.items()]
            print(", ".join(msg))

        return (
            total_loss, main_loss, vg_reg, kl_reg, entropy_reg,
            layer_prior_reg, out_of_band_reg, layer_distance_reg,
            spatial_continuity_reg, gradient_reg,
        )

    def train(self, num_epochs, learning_rate=0.1, print_each=100, val_each=None):
        """Run the differentiation-continuity-aware optimizer."""
        if self.random_state:
            torch.manual_seed(seed=self.random_state)
        optimizer = torch.optim.Adam([self.M], lr=learning_rate)

        if print_each:
            logging.info(f"Printing scores every {print_each} epochs.")

        keys = [
            "total_loss", "main_loss", "vg_reg", "kl_reg", "entropy_reg",
            "layer_prior_reg", "out_of_band_reg", "layer_distance_reg",
            "spatial_continuity_reg", "gradient_reg",
        ]
        val_keys = [
            "val_total_loss", "val_gene_sim",
            "val_sp_sparsity_weighted_sim", "val_entropy",
        ]
        training_history = {key: [] for key in keys + val_keys}

        for t in range(num_epochs):
            if print_each is None or t % print_each != 0:
                run_loss = self._loss_fn(verbose=False)
            else:
                run_loss = self._loss_fn(verbose=True)

            loss = run_loss[0]
            training_history[keys[0]].append(run_loss[0].clone().detach().cpu().numpy())
            for i in range(1, len(keys)):
                training_history[keys[i]].append(run_loss[i])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if val_each is not None:
                with torch.no_grad():
                    if t % val_each == 0:
                        val_loss = self._val_loss_fn(verbose=False)
                        for i in range(len(val_keys)):
                            training_history[val_keys[i]].append(val_loss[i])

        with torch.no_grad():
            output = softmax(self.M, dim=1).cpu().numpy()
            return output, training_history
