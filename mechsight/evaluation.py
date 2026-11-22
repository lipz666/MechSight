"""Formal evaluation metrics and statistical inference (SI Sections S1.3 and S3.1).

Implements the exact set-theoretic definitions of:
  - Canonical species keys: ``strict`` (sigma-graph / FixedH) and ``equiv`` (14-char InChIKey).
  - Four hierarchical levels: product-reaching rate, single-step accuracy, intermediate recall
    (and stratum micro-recall), and complete-pathway accuracy.
  - Anti-padding guards: intermediate precision, F1, greedy ordered intermediate fraction,
    and strict in-order complete-pathway accuracy.
  - Non-parametric cluster percentile bootstrap 95% CIs and exact paired sign-flip
    permutation tests.
"""
from __future__ import annotations

import random
from typing import Any, Sequence

from .chemistry import equiv_connectivity_keys, fixed_h_inchi_keys, sigma_graph_keys


def _keys_for_tier(smiles: str | None, strict_mode: str = "sigma") -> tuple[set[str], set[str]]:
    if not smiles:
        return set(), set()
    strict_keys = fixed_h_inchi_keys(smiles) if strict_mode == "fixed_h" else sigma_graph_keys(smiles)
    equiv_keys = equiv_connectivity_keys(smiles, allow_unsanitized=True)
    return strict_keys, equiv_keys


def _largest_novel_product(reactants: list[str], products: list[str], strict_mode: str = "sigma") -> str | None:
    from rdkit import Chem

    r_keys: set[str] = set()
    for r in reactants:
        r_keys |= _keys_for_tier(r, strict_mode)[0]
    candidates: list[tuple[int, str]] = []
    for p in products:
        for frag in (p or "").split("."):
            mol = Chem.MolFromSmiles(frag)
            if mol is None:
                continue
            fk = _keys_for_tier(frag, strict_mode)[0]
            if fk and not (fk <= r_keys):
                candidates.append((mol.GetNumHeavyAtoms(), frag))
    if not candidates:
        return max((p for p in products if p), key=len, default=None)
    return max(candidates, key=lambda x: x[0])[1]


def evaluate_pathway(
    *,
    reactants: list[str],
    products: list[str],
    reference_intermediates: list[str],
    predicted_steps: list[dict[str, Any]] | None,
    strict_mode: str = "sigma",
) -> dict[str, dict[str, float]]:
    """Compute SI Section S3.1 pathway metrics under ``strict`` and ``equiv`` tiers."""
    if not predicted_steps:
        zero = {
            "parsed": 0.0,
            "product_reached": 0.0,
            "intermediate_recall": 0.0,
            "complete_pathway": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "ordered_fraction": 0.0,
            "in_order_pathway": 0.0,
        }
        return {"strict": dict(zero), "equiv": dict(zero)}

    pred_smis = [str(s.get("intermediate_smiles") or "") for s in predicted_steps]

    # Precompute endpoint key sets
    end_strict: set[str] = set()
    end_equiv: set[str] = set()
    for ep in reactants + products:
        ks, ke = _keys_for_tier(ep, strict_mode)
        end_strict |= ks
        end_equiv |= ke

    pred_strict = [_keys_for_tier(s, strict_mode)[0] for s in pred_smis]
    pred_equiv = [_keys_for_tier(s, strict_mode)[1] for s in pred_smis]
    union_strict = set().union(*pred_strict) if pred_strict else set()
    union_equiv = (set().union(*pred_equiv) if pred_equiv else set()) | end_equiv

    ref_pairs = [(_keys_for_tier(g, strict_mode), g) for g in reference_intermediates]
    ref_pairs = [pair for pair in ref_pairs if pair[0][0] or pair[0][1]]

    if not ref_pairs:
        hits_s: list[bool] = []
        hits_e: list[bool] = []
    else:
        hits_s = [bool(ks and ks <= union_strict) for (ks, _), _ in ref_pairs]
        hits_e = [
            bool((ks and ks <= union_strict) or (ke and ke <= union_equiv))
            for (ks, ke), _ in ref_pairs
        ]

    # Product-reaching (Eq. S2)
    target_prod = _largest_novel_product(reactants, products, strict_mode)
    if target_prod:
        ps_k, pe_k = _keys_for_tier(target_prod, strict_mode)
        prod_s = float(bool(ps_k and ps_k <= union_strict))
        prod_e = float(bool((ps_k and ps_k <= union_strict) or (pe_k and pe_k <= union_equiv)))
    else:
        prod_s = prod_e = 0.0

    # Recall and complete pathway (Eqs. S4-S5)
    m_count = len(ref_pairs)
    rec_s = sum(hits_s) / m_count if m_count else 1.0
    rec_e = sum(hits_e) / m_count if m_count else 1.0
    comp_s = float(all(hits_s)) if m_count else 1.0
    comp_e = float(all(hits_e)) if m_count else 1.0

    # Precision & F1 excluding endpoints (Eq. S6)
    ref_union_s = set().union(*(ks for (ks, _), _ in ref_pairs)) if ref_pairs else set()
    ref_union_e = (set().union(*(ke for (_, ke), _ in ref_pairs)) if ref_pairs else set()) | end_equiv

    non_end_s = [ks for ks in pred_strict if ks and not (ks <= end_strict)]
    non_end_e = [
        (ks, ke) for ks, ke in zip(pred_strict, pred_equiv) if (ks or ke) and not (ks <= end_strict)
    ]
    prec_s = sum(ks <= ref_union_s for ks in non_end_s) / len(non_end_s) if non_end_s else 0.0
    prec_e = (
        sum((ks and ks <= ref_union_s) or (ke and ke <= ref_union_e) for ks, ke in non_end_e) / len(non_end_e)
        if non_end_e
        else 0.0
    )
    f1_s = (2 * prec_s * rec_s / (prec_s + rec_s)) if (prec_s + rec_s) else 0.0
    f1_e = (2 * prec_e * rec_e / (prec_e + rec_e)) if (prec_e + rec_e) else 0.0

    # Greedy monotone ordered matching (Eq. S7)
    cur_s = ord_s = 0
    for (ks, _), _ in ref_pairs:
        idx = next((i for i in range(cur_s, len(pred_strict)) if ks and ks <= pred_strict[i]), None)
        if idx is not None:
            ord_s += 1
            cur_s = idx + 1
    cur_e = ord_e = 0
    for (ks, ke), _ in ref_pairs:
        idx = next(
            (
                i
                for i in range(cur_e, len(pred_strict))
                if (ks and ks <= pred_strict[i]) or (ke and ke <= (pred_equiv[i] | end_equiv))
            ),
            None,
        )
        if idx is not None:
            ord_e += 1
            cur_e = idx + 1

    ord_frac_s = ord_s / m_count if m_count else 1.0
    ord_frac_e = ord_e / m_count if m_count else 1.0

    return {
        "strict": {
            "parsed": 1.0,
            "product_reached": prod_s,
            "intermediate_recall": rec_s,
            "complete_pathway": comp_s,
            "precision": prec_s,
            "f1": f1_s,
            "ordered_fraction": ord_frac_s,
            "in_order_pathway": float(ord_s == m_count) if m_count else 1.0,
        },
        "equiv": {
            "parsed": 1.0,
            "product_reached": prod_e,
            "intermediate_recall": rec_e,
            "complete_pathway": comp_e,
            "precision": prec_e,
            "f1": f1_e,
            "ordered_fraction": ord_frac_e,
            "in_order_pathway": float(ord_e == m_count) if m_count else 1.0,
        },
    }


def evaluate_single_step(
    *,
    prev_state_smiles: str,
    ref_next_state_smiles: str,
    pred_next_state_smiles: str,
    strict_mode: str = "sigma",
) -> dict[str, bool]:
    """Evaluate single-step prediction given reference state S_{k-1} (SI Eq. S3)."""
    prev_s, prev_e = _keys_for_tier(prev_state_smiles, strict_mode)
    ref_s, ref_e = _keys_for_tier(ref_next_state_smiles, strict_mode)
    pred_s, pred_e = _keys_for_tier(pred_next_state_smiles, strict_mode)

    delta_s = ref_s - prev_s
    delta_e = ref_e - prev_e
    hit_strict = bool(delta_s and delta_s <= pred_s)
    hit_equiv = hit_strict or bool(delta_e and delta_e <= pred_e)
    return {"strict": hit_strict, "equiv": hit_equiv}


def bootstrap_ci(
    values: Sequence[float],
    clusters: Sequence[str] | None = None,
    *,
    n_resamples: int = 4000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Compute mean and 95% cluster-bootstrap percentile interval (SI Section S1.3)."""
    if not values:
        raise ValueError("Cannot compute bootstrap CI on empty sequence.")
    if clusters is None:
        clusters = [str(i) for i in range(len(values))]
    if len(values) != len(clusters):
        raise ValueError("values and clusters must have the same length.")

    by_cluster: dict[str, list[float]] = {}
    for val, cid in zip(values, clusters):
        by_cluster.setdefault(str(cid), []).append(float(val))
    cluster_sums = [(sum(vs), len(vs)) for vs in by_cluster.values()]
    n_clusters = len(cluster_sums)

    total_sum = sum(s for s, _ in cluster_sums)
    total_n = sum(c for _, c in cluster_sums)
    mean = total_sum / total_n

    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_resamples):
        s_acc = 0.0
        n_acc = 0
        for _ in range(n_clusters):
            cs, cn = cluster_sums[rng.randrange(n_clusters)]
            s_acc += cs
            n_acc += cn
        boots.append(s_acc / n_acc if n_acc else mean)
    boots.sort()
    lo = boots[int(0.025 * n_resamples)]
    hi = boots[int(0.975 * n_resamples)]
    return mean, lo, hi


def paired_permutation_pvalue(
    diffs: Sequence[float],
    *,
    n_permutations: int = 10000,
    seed: int = 0,
) -> float:
    """Two-sided paired sign-flip permutation p-value (SI Section S1.3)."""
    if not diffs:
        return 1.0
    n = len(diffs)
    observed = abs(sum(diffs) / n)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_permutations):
        perm_mean = abs(sum(d if rng.random() < 0.5 else -d for d in diffs) / n)
        if perm_mean >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (n_permutations + 1)
