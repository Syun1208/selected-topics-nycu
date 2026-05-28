from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import networkx as nx
import numpy as np
import torch
import tqdm

from data import FilePath
from pipelines.scene import Scene
from retrievers.factory import MASt3RRetrievalASMKRetriever
from shortlists.base import ShortlistGenerator, get_all_pairs
from shortlists.config import (
    AdaptiveExpansionShortlistGeneratorConfig,
    ShortlistGeneratorConfig,
)
from shortlists.global_descriptor import (
    create_global_descriptor_extractor,
    extract_global_features,
)
from workspace import log


class AdaptiveExpansionShortlistGenerator(ShortlistGenerator):
    """Adaptive shortlist:

    1. Compute the union of pairs from base retrievers (e.g. ASMK + SPoC + DINOv2 + ISC).
    2. Build a retrieval graph and compute per-image degree.
    3. For low-degree images, expand with:
        - 2-hop neighbors via the existing graph
        - Reciprocal top-k pairs from each expansion retriever
        - Extra ASMK top-k pairs (k bumped, e.g. 25 -> 40/60)
    4. If the scene graph splits into many small connected components, add
       cross-component bridge pairs based on DINOv2/ISC similarity.

    Motivation: empirical observation that on this pipeline the dominant
    error mode is *missing useful pairs* rather than mismatches. Instead of
    uniformly raising top-k everywhere, we only spend extra pair budget on
    under-connected images and bridge isolated mini-clusters.
    """

    def __init__(
        self,
        conf: AdaptiveExpansionShortlistGeneratorConfig,
        base_generators: list[ShortlistGenerator],
        device: Optional[torch.device] = None,
    ):
        self.conf = conf
        self.base_generators = base_generators
        self.device = device

    @torch.inference_mode()
    def __call__(
        self,
        scene: Scene,
        progress_bar: Optional[tqdm.tqdm] = None,
        **kwargs,
    ) -> list[tuple[int, int]]:
        image_paths = scene.image_paths
        N = len(image_paths)

        if N <= self.conf.all_pairs_fallback_threshold:
            log(
                f"[AdaptiveExpansion] N={N} <= "
                f"{self.conf.all_pairs_fallback_threshold} -> all_pairs"
            )
            pairs = get_all_pairs(image_paths)
            scene.update_shortlist(pairs)
            return pairs

        # ------------------------------------------------------------------
        # 1) Base union pairs
        # ------------------------------------------------------------------
        base_pairs_set: set[tuple[int, int]] = set()
        per_gen_stats: list[tuple[str, int]] = []
        for gen in self.base_generators:
            ps = gen(scene, progress_bar=progress_bar, **kwargs)
            for i, j in ps:
                a, b = (int(i), int(j)) if i < j else (int(j), int(i))
                if a != b:
                    base_pairs_set.add((a, b))
            per_gen_stats.append((gen.__class__.__name__, len(ps)))

        log("[AdaptiveExpansion] base generator stats:")
        for name, n in per_gen_stats:
            log(f"  - {name}: {n}")
        log(f"[AdaptiveExpansion] base union pairs: {len(base_pairs_set)}")

        # ------------------------------------------------------------------
        # 2) Graph + degrees
        # ------------------------------------------------------------------
        G = nx.Graph()
        G.add_nodes_from(range(N))
        G.add_edges_from(base_pairs_set)
        degrees = dict(G.degree())
        deg_arr = np.fromiter(degrees.values(), dtype=np.int32, count=N)
        log(
            f"[AdaptiveExpansion] degrees: min={int(deg_arr.min())} "
            f"max={int(deg_arr.max())} mean={float(deg_arr.mean()):.2f} "
            f"med={int(np.median(deg_arr))}"
        )

        # ------------------------------------------------------------------
        # 3) Identify low-degree nodes
        # ------------------------------------------------------------------
        low_deg = [int(n) for n, d in degrees.items() if d < self.conf.low_degree_threshold]
        log(
            f"[AdaptiveExpansion] low-degree nodes (<{self.conf.low_degree_threshold}): "
            f"{len(low_deg)} / {N}"
        )

        added: set[tuple[int, int]] = set()

        # ------------------------------------------------------------------
        # 4a) 2-hop neighbors (graph-only, cheap)
        # ------------------------------------------------------------------
        if low_deg:
            two_hop_added = 0
            cap = self.conf.two_hop_max_per_node
            for u in low_deg:
                d_map = nx.single_source_shortest_path_length(G, u, cutoff=2)
                two_hop = [v for v, d in d_map.items() if d == 2][:cap]
                for v in two_hop:
                    a, b = (u, v) if u < v else (v, u)
                    if (a, b) not in base_pairs_set and (a, b) not in added:
                        added.add((a, b))
                        two_hop_added += 1
            log(f"[AdaptiveExpansion] 2-hop added: {two_hop_added}")

        # ------------------------------------------------------------------
        # 4b/c) Build distance matrices for expansion retrievers
        # ------------------------------------------------------------------
        dist_mats: list[tuple[str, np.ndarray]] = []
        if self.conf.expansion_dinov2 is not None:
            D = self._extract_global_dist_matrix(self.conf.expansion_dinov2, image_paths)
            dist_mats.append(("dinov2", D))
        if self.conf.expansion_isc is not None:
            D = self._extract_global_dist_matrix(self.conf.expansion_isc, image_paths)
            dist_mats.append(("isc", D))

        asmk_sim: Optional[np.ndarray] = None
        if self.conf.expansion_asmk is not None:
            asmk_sim = self._extract_asmk_sim_matrix(self.conf.expansion_asmk, image_paths)
            dist_mats.append(("asmk", -asmk_sim))

        # ------------------------------------------------------------------
        # 4b) Reciprocal top-k pairs
        # ------------------------------------------------------------------
        if low_deg and dist_mats:
            recip_added = 0
            for name, D in dist_mats:
                k = min(self.conf.reciprocal_topk + 1, N)
                topk = _topk_argsort(D, k)
                topk_sets = [set(r.tolist()) for r in topk]
                for u in low_deg:
                    for v in topk[u]:
                        v = int(v)
                        if v == u:
                            continue
                        if u in topk_sets[v]:
                            a, b = (u, v) if u < v else (v, u)
                            if (a, b) not in base_pairs_set and (a, b) not in added:
                                added.add((a, b))
                                recip_added += 1
            log(f"[AdaptiveExpansion] reciprocal added: {recip_added}")

        # ------------------------------------------------------------------
        # 4c) Extra ASMK top-k for low-degree nodes
        # ------------------------------------------------------------------
        if low_deg and asmk_sim is not None:
            k = min(self.conf.asmk_expansion_topk + 1, N)
            asmk_topk = _topk_argsort(-asmk_sim, k)
            asmk_added = 0
            for u in low_deg:
                for v in asmk_topk[u]:
                    v = int(v)
                    if v == u:
                        continue
                    a, b = (u, v) if u < v else (v, u)
                    if (a, b) not in base_pairs_set and (a, b) not in added:
                        added.add((a, b))
                        asmk_added += 1
            log(f"[AdaptiveExpansion] extra ASMK added: {asmk_added}")

        # ------------------------------------------------------------------
        # 5) Cross-component bridge pairs
        # ------------------------------------------------------------------
        if self.conf.enable_cross_component_bridges and dist_mats:
            G2 = nx.Graph()
            G2.add_nodes_from(range(N))
            G2.add_edges_from(base_pairs_set | added)
            ccs = [sorted(cc) for cc in nx.connected_components(G2)]
            small_ccs = [cc for cc in ccs if len(cc) <= self.conf.small_component_size]
            log(
                f"[AdaptiveExpansion] CCs: total={len(ccs)} "
                f"small(<={self.conf.small_component_size})={len(small_ccs)}"
            )
            if len(small_ccs) >= self.conf.min_small_components:
                D_combined = np.zeros((N, N), dtype=np.float32)
                for _, D in dist_mats:
                    D32 = D.astype(np.float32)
                    rng = float(D32.max() - D32.min() + 1e-9)
                    D_combined += (D32 - float(D32.min())) / rng
                k_per_pair = self.conf.bridge_topk_per_cc_pair
                bridge_added = 0
                for ci in range(len(ccs)):
                    cc_i = np.asarray(ccs[ci], dtype=np.int64)
                    small_i = len(cc_i) <= self.conf.small_component_size
                    for cj in range(ci + 1, len(ccs)):
                        cc_j = np.asarray(ccs[cj], dtype=np.int64)
                        small_j = len(cc_j) <= self.conf.small_component_size
                        if self.conf.bridge_only_touching_small_cc and not (small_i or small_j):
                            continue
                        sub = D_combined[cc_i[:, None], cc_j[None, :]]
                        flat = sub.ravel()
                        n_take = min(k_per_pair, flat.size)
                        if n_take <= 0:
                            continue
                        kth = n_take - 1
                        idx = np.argpartition(flat, kth=kth)[: kth + 1]
                        ncols = sub.shape[1]
                        for f in idx:
                            ui = int(cc_i[int(f) // ncols])
                            vj = int(cc_j[int(f) % ncols])
                            a, b = (ui, vj) if ui < vj else (vj, ui)
                            if (a, b) not in base_pairs_set and (a, b) not in added:
                                added.add((a, b))
                                bridge_added += 1
                log(f"[AdaptiveExpansion] bridges added: {bridge_added}")

        # ------------------------------------------------------------------
        # 6) Finalize
        # ------------------------------------------------------------------
        all_pairs = sorted(base_pairs_set | added)
        log(
            f"[AdaptiveExpansion] final pairs: {len(all_pairs)} "
            f"(base={len(base_pairs_set)}, added={len(added)})"
        )
        scene.update_shortlist(all_pairs)
        return all_pairs

    # ----------------------------------------------------------------------
    # helpers
    # ----------------------------------------------------------------------

    def _extract_global_dist_matrix(
        self,
        conf: ShortlistGeneratorConfig,
        image_paths: Sequence[FilePath],
    ) -> np.ndarray:
        extractor = create_global_descriptor_extractor(conf, device=self.device)
        feats = extract_global_features(
            list(image_paths),
            extractor,
            conf.global_desc_batch_size,
            num_workers=conf.global_desc_num_workers,
        )
        D = torch.cdist(feats, feats, p=2).detach().cpu().float().numpy()
        del feats, extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return D

    def _extract_asmk_sim_matrix(
        self,
        conf: ShortlistGeneratorConfig,
        image_paths: Sequence[FilePath],
    ) -> np.ndarray:
        assert conf.mast3r_retrieval_asmk is not None
        retriever = MASt3RRetrievalASMKRetriever(
            conf.mast3r_retrieval_asmk, device=self.device
        )
        S = retriever.build(list(image_paths)).get_sim_matrix(list(image_paths))
        del retriever
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return S.astype(np.float32)


def _topk_argsort(D: np.ndarray, k: int) -> np.ndarray:
    """Return the indices of the top-k smallest entries per row, sorted."""
    n = D.shape[1]
    k = max(1, min(k, n))
    if k >= n:
        return np.argsort(D, axis=1)
    part = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
    row_idx = np.arange(D.shape[0])[:, None]
    order = np.argsort(D[row_idx, part], axis=1)
    return part[row_idx, order]
