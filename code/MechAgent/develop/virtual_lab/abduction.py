"""Unified VirtualLab facade, Bayesian abductive engine (P2/P3), and Expected Information Gain experiment designer (P4)."""
from __future__ import annotations

import collections
import copy
import math
import re
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from .chem_graph import base_element, connected_molecules, electronic_automorphisms, extract_graph
from .reference_data import TRANSITION_METALS
from .schemas import (
    ExperimentDesignCandidate,
    ExperimentalProbe,
    MechanismHypothesis,
    ObservablePrediction,
)
from .simulators_kinetic import KineticsAndKIESimulator, PhotoredoxElectroSimulator
from .simulators_lfer_control import CompetenceControlSimulator, LFERAndStereoSimulator
from .simulators_structural import (
    CrossoverSimulator,
    IsotopeRayTracer,
    RadicalClockSimulator,
    RadicalTrapSimulator,
)
from .universal_engine import TrajectoryResult, execute_hypothesis


def normalize_bin_token(bin_str: Optional[str]) -> str:
    """Canonicalize outcome_bin strings across evidence_p0/schema.json and shorthand labels."""
    if bin_str is None:
        return "null"
    s = str(bin_str).strip().lower()
    # Strip parenthetical explanatory text e.g. 'suppressed (<10%)' -> 'suppressed', 'primary (>2.5)' -> 'primary'
    s = re.sub(r"\s*\([^)]*\)", "", s).strip()
    return s


class VirtualLab:
    """Full-featured deterministic virtual physical-organic laboratory covering all 17 probe classes."""

    def __init__(self) -> None:
        self._trajectory_cache: Dict[str, TrajectoryResult] = {}

    def clear_cache(self) -> None:
        self._trajectory_cache.clear()

    def trajectory_of(self, hypothesis: MechanismHypothesis) -> TrajectoryResult:
        # Key by id + object id so mutated hypotheses do not collide
        cache_key = f"{hypothesis.hypothesis_id}:{id(hypothesis)}"
        if cache_key not in self._trajectory_cache:
            self._trajectory_cache[cache_key] = execute_hypothesis(hypothesis)
        return self._trajectory_cache[cache_key]

    def predict(
        self,
        hypothesis: MechanismHypothesis,
        probe: ExperimentalProbe,
    ) -> ObservablePrediction:
        """Execute ``probe`` on ``hypothesis`` and return a deterministic ObservablePrediction."""
        traj = self.trajectory_of(hypothesis)
        etype = probe.experiment_type

        if etype == "isotope_label":
            return IsotopeRayTracer.simulate(hypothesis, probe, traj)
        if etype == "crossover":
            return CrossoverSimulator.simulate(hypothesis, probe, traj)
        if etype == "radical_trap":
            return RadicalTrapSimulator.simulate(hypothesis, probe, traj)
        if etype == "radical_clock":
            return RadicalClockSimulator.simulate(hypothesis, probe, traj)
        if etype == "kie":
            return KineticsAndKIESimulator.simulate_kie(hypothesis, probe, traj)
        if etype == "kinetic_order":
            return KineticsAndKIESimulator.simulate_kinetic_order(hypothesis, probe, traj)
        if etype == "quantum_yield":
            return PhotoredoxElectroSimulator.simulate_quantum_yield(hypothesis, probe, traj)
        if etype == "light_on_off":
            return PhotoredoxElectroSimulator.simulate_light_on_off(hypothesis, probe, traj)
        if etype == "stern_volmer":
            return PhotoredoxElectroSimulator.simulate_stern_volmer(hypothesis, probe, traj)
        if etype == "electrochemistry":
            return PhotoredoxElectroSimulator.simulate_electrochemistry(hypothesis, probe, traj)
        if etype == "hammett":
            return LFERAndStereoSimulator.simulate_hammett(hypothesis, probe, traj)
        if etype == "competition":
            return LFERAndStereoSimulator.simulate_competition(hypothesis, probe, traj)
        if etype == "stereochemical_probe":
            return LFERAndStereoSimulator.simulate_stereochemistry(hypothesis, probe, traj)
        if etype == "nonlinear_effect":
            return LFERAndStereoSimulator.simulate_nonlinear_effect(hypothesis, probe, traj)
        if etype == "control":
            return CompetenceControlSimulator.simulate_control(hypothesis, probe, traj)
        if etype == "stoichiometric_intermediate":
            return CompetenceControlSimulator.simulate_stoichiometric_intermediate(hypothesis, probe, traj)
        if etype == "spectroscopy":
            return CompetenceControlSimulator.simulate_spectroscopy(hypothesis, probe, traj)

        return ObservablePrediction(
            hypothesis_id=hypothesis.hypothesis_id,
            experiment_type=etype,
            outcome_bin=None,
            confidence=0.5,
            rationale="Unclassified ('other') probe type requires external/LLM evaluation.",
        )

    def compare_hypotheses(
        self,
        hypotheses: Sequence[MechanismHypothesis],
        probes: Sequence[ExperimentalProbe],
    ) -> List[Dict[str, Any]]:
        """Evaluate a matrix of (hypotheses x probes) and highlight divergent predictions."""
        table: List[Dict[str, Any]] = []
        for probe in probes:
            preds = {h.hypothesis_id: self.predict(h, probe) for h in hypotheses}
            bins = {normalize_bin_token(p.outcome_bin) for p in preds.values()}
            table.append({
                "probe_id": probe.probe_id or probe.experiment_type,
                "experiment_type": probe.experiment_type,
                "perturbation": probe.perturbation,
                "discriminates": len(bins) > 1,
                "predictions": {hid: p.to_dict() for hid, p in preds.items()},
            })
        return table


# ===========================================================================
# Adapter for evidence_p0/schema.json Evidence Cards
# ===========================================================================

def probe_from_evidence_card(card: Dict[str, Any]) -> Tuple[ExperimentalProbe, str, Dict[str, Any]]:
    """Convert an evidence_p0 JSON card into (ExperimentalProbe, normalized_observed_bin, extra_facts)."""
    etype = str(card.get("experiment_type") or "other")
    pert = str(card.get("perturbation") or "")
    obs = str(card.get("observable") or "")
    raw = str(card.get("outcome_raw") or "")
    obs_bin = normalize_bin_token(card.get("outcome_bin"))

    params: Dict[str, Any] = {}
    pert_up = pert.upper()
    if etype == "radical_trap":
        for rname in ("TEMPO", "BHT", "1,1-DPE", "DMPO"):
            if rname in pert_up:
                params["reagent"] = rname
                break
        if "adduct" in obs_bin or "adduct" in obs.lower() or "m/z" in raw.lower() or "hrms" in obs.lower():
            params["observable_mode"] = "adduct"

    # Extract any HRMS m/z values mentioned in outcome_raw (e.g. "m/z 268.2271" or "found 268.2275")
    mz_hits = [float(x) for x in re.findall(r"\b(\d{2,4}\.\d{3,5})\b", raw)]
    extra = {"raw_text": raw, "reported_mz": mz_hits}

    probe = ExperimentalProbe(
        experiment_type=etype if etype in {
            "kie", "radical_trap", "radical_clock", "light_on_off", "quantum_yield",
            "crossover", "isotope_label", "hammett", "kinetic_order", "nonlinear_effect",
            "control", "stern_volmer", "electrochemistry", "spectroscopy",
            "stoichiometric_intermediate", "stereochemical_probe", "competition", "other",
        } else "other",
        perturbation=pert,
        observable=obs,
        parameters=params,
        probe_id=str(card.get("id") or ""),
    )
    return probe, obs_bin, extra


# ===========================================================================
# Bayesian Abductive Engine (Sequential Updating, P2 Leave-One-Out, P3 Flip)
# ===========================================================================

_ADJACENT_BINS: Set[FrozenSet[str]] = {
    frozenset({"suppressed", "reduced"}),
    frozenset({"unaffected", "reduced"}),
    frozenset({"none", "small"}),
    frozenset({"small", "primary"}),
    frozenset({"ring_opened", "mixed"}),
    frozenset({"not_opened", "mixed"}),
}


class AbductiveEngine:
    """Bayesian belief engine scoring competing PGL hypotheses against experimental observations."""

    def __init__(
        self,
        lab: Optional[VirtualLab] = None,
        *,
        match_likelihood: float = 0.88,
        adjacent_likelihood: float = 0.30,
        mismatch_likelihood: float = 0.06,
    ) -> None:
        self.lab = lab or VirtualLab()
        self.match_likelihood = match_likelihood
        self.adjacent_likelihood = adjacent_likelihood
        self.mismatch_likelihood = mismatch_likelihood

    def observation_likelihood(
        self,
        prediction: ObservablePrediction,
        observed_bin: str,
        extra_facts: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute P(y_obs | H_k, probe) combining discrete bin match and structural HRMS m/z verification."""
        pred_bin = normalize_bin_token(prediction.outcome_bin)
        obs_norm = normalize_bin_token(observed_bin)

        if pred_bin == "null" or obs_norm == "null":
            return 0.50

        # Handle colon-prefixed bins (e.g. 'C14:major' vs 'product:major' or 'positive:1-3')
        if pred_bin == obs_norm:
            base_lik = self.match_likelihood
        elif ":" in pred_bin and ":" in obs_norm:
            p_head, p_tail = pred_bin.split(":", 1)
            o_head, o_tail = obs_norm.split(":", 1)
            if p_head == o_head and p_tail == o_tail:
                base_lik = self.match_likelihood
            elif p_head in {"product", o_head} or o_head == "product":
                base_lik = self.match_likelihood if p_tail == o_tail else self.mismatch_likelihood
            elif p_head == o_head:
                # Same sign/component, slightly different magnitude bin
                base_lik = self.adjacent_likelihood
            else:
                base_lik = self.mismatch_likelihood
        elif frozenset({pred_bin, obs_norm}) in _ADJACENT_BINS:
            base_lik = self.adjacent_likelihood
        else:
            base_lik = self.mismatch_likelihood

        # Structural HRMS adduct bonus: if paper reports an exact m/z and H_k predicts an adduct within 0.05 Da!
        if extra_facts and extra_facts.get("reported_mz"):
            adducts = prediction.structural_details.get("trapped_adducts") or []
            pred_mzs = [
                mz_val
                for add in adducts
                for mz_val in (add.get("adduct_hrms_mz") or {}).values()
            ]
            if pred_mzs:
                matched_mz = any(
                    abs(obs_mz - pmz) <= 0.05
                    for obs_mz in extra_facts["reported_mz"]
                    for pmz in pred_mzs
                )
                base_lik = min(0.97, base_lik * 1.10) if matched_mz else max(0.03, base_lik * 0.25)

        return base_lik

    def compute_posteriors(
        self,
        hypotheses: Sequence[MechanismHypothesis],
        observations: Sequence[Tuple[ExperimentalProbe, str, Optional[Dict[str, Any]]]],
    ) -> Dict[str, Any]:
        """Compute sequential posterior trajectory P(H_k | E_{1:t}) for t = 0 .. M."""
        if not hypotheses:
            raise ValueError("At least one MechanismHypothesis is required")

        log_weights = {
            h.hypothesis_id: math.log(max(h.prior, 1e-12))
            for h in hypotheses
        }

        def _normalize(lw: Dict[str, float]) -> Dict[str, float]:
            m = max(lw.values())
            exps = {k: math.exp(v - m) for k, v in lw.items()}
            z = sum(exps.values())
            return {k: exps[k] / z for k in lw}

        trajectory: List[Dict[str, Any]] = [
            {"step": 0, "probe_id": "prior", "posteriors": _normalize(log_weights)}
        ]

        per_evidence_details: List[Dict[str, Any]] = []
        for idx, item in enumerate(observations, start=1):
            probe, obs_bin = item[0], item[1]
            extra = item[2] if len(item) > 2 else None
            step_liks: Dict[str, float] = {}
            step_preds: Dict[str, Dict[str, Any]] = {}
            for h in hypotheses:
                pred = self.lab.predict(h, probe)
                lik = self.observation_likelihood(pred, obs_bin, extra)
                log_weights[h.hypothesis_id] += math.log(max(lik, 1e-12))
                step_liks[h.hypothesis_id] = round(lik, 4)
                step_preds[h.hypothesis_id] = pred.to_dict()

            post = _normalize(log_weights)
            trajectory.append({
                "step": idx,
                "probe_id": probe.probe_id or probe.experiment_type,
                "experiment_type": probe.experiment_type,
                "observed_bin": normalize_bin_token(obs_bin),
                "posteriors": {k: round(v, 5) for k, v in post.items()},
            })
            per_evidence_details.append({
                "probe_id": probe.probe_id or f"E{idx:02d}",
                "experiment_type": probe.experiment_type,
                "observed_bin": normalize_bin_token(obs_bin),
                "likelihoods": step_liks,
                "predictions": step_preds,
            })

        final_post = _normalize(log_weights)
        map_hid = max(final_post, key=final_post.get)
        return {
            "final_posteriors": {k: round(v, 5) for k, v in final_post.items()},
            "map_hypothesis": map_hid,
            "posterior_trajectory": trajectory,
            "evidence_evaluations": per_evidence_details,
        }

    def predict_held_out(
        self,
        hypotheses: Sequence[MechanismHypothesis],
        visible_observations: Sequence[Tuple[ExperimentalProbe, str, Optional[Dict[str, Any]]]],
        target_probe: ExperimentalProbe,
    ) -> Dict[str, Any]:
        """Leave-one-out posterior predictive distribution P(Y_target | E_visible) = sum_k P(Y | H_k) P(H_k | E_visible)."""
        bel = self.compute_posteriors(hypotheses, visible_observations)
        post = bel["final_posteriors"]

        outcome_dist: Dict[str, float] = collections.defaultdict(float)
        by_hyp: Dict[str, ObservablePrediction] = {}
        for h in hypotheses:
            pred = self.lab.predict(h, target_probe)
            by_hyp[h.hypothesis_id] = pred
            b_tok = normalize_bin_token(pred.outcome_bin)
            outcome_dist[b_tok] += post[h.hypothesis_id]

        top_bin = max(outcome_dist, key=outcome_dist.get)
        return {
            "predicted_bin": top_bin,
            "outcome_distribution": {k: round(v, 4) for k, v in sorted(outcome_dist.items(), key=lambda kv: -kv[1])},
            "hypothesis_posteriors": post,
            "hypothesis_predictions": {hid: p.to_dict() for hid, p in by_hyp.items()},
        }

    def counterfactual_flip_test(
        self,
        hypotheses: Sequence[MechanismHypothesis],
        observations: Sequence[Tuple[ExperimentalProbe, str, Optional[Dict[str, Any]]]],
        flip_index: int,
        counterfactual_bin: str,
    ) -> Dict[str, Any]:
        """P3 Diagnostic: Flip observation[flip_index] to counterfactual_bin and quantify posterior shift."""
        orig = self.compute_posteriors(hypotheses, observations)
        cf_obs = list(observations)
        probe, orig_bin = cf_obs[flip_index][0], cf_obs[flip_index][1]
        cf_obs[flip_index] = (probe, counterfactual_bin, None)
        flipped = self.compute_posteriors(hypotheses, cf_obs)

        shifts = {
            hid: round(flipped["final_posteriors"][hid] - orig["final_posteriors"][hid], 5)
            for hid in orig["final_posteriors"]
        }
        return {
            "flipped_probe": probe.probe_id or probe.experiment_type,
            "original_bin": normalize_bin_token(orig_bin),
            "counterfactual_bin": normalize_bin_token(counterfactual_bin),
            "original_map": orig["map_hypothesis"],
            "counterfactual_map": flipped["map_hypothesis"],
            "map_switched": orig["map_hypothesis"] != flipped["map_hypothesis"],
            "posterior_shifts": shifts,
            "original_posteriors": orig["final_posteriors"],
            "counterfactual_posteriors": flipped["final_posteriors"],
        }


# ===========================================================================
# Optimal Discriminating Experiment Designer (Expected Information Gain, P4)
# ===========================================================================

class ExperimentDesigner:
    """Automatically proposes and ranks discriminating experiments across competing PGL hypotheses via Shannon EIG."""

    def __init__(self, engine: Optional[AbductiveEngine] = None) -> None:
        self.engine = engine or AbductiveEngine()
        self.lab = self.engine.lab

    def generate_candidate_probes(
        self,
        hypotheses: Sequence[MechanismHypothesis],
    ) -> List[ExperimentalProbe]:
        """Auto-synthesize a comprehensive candidate experiment pool tailored to the molecular graphs of {H_k}."""
        probes: List[ExperimentalProbe] = []
        if not hypotheses:
            return probes

        h0 = hypotheses[0]
        s0_atoms, _, _ = extract_graph(h0.initial_state)

        # 1. Isotope-labeling probes on reactive heavy atoms (C, O, N) + D2O exchangeable pool
        touched_heavy: Set[str] = set()
        cleaved_xh_owners: Set[str] = set()
        chiral_centers: Set[str] = set()
        component_names: Set[str] = set()

        for h in hypotheses:
            traj = self.lab.trajectory_of(h)
            for aid, meta in h.initial_state.atom_meta.items():
                if meta.stereo_parity in (-1, 1):
                    chiral_centers.add(aid)
            component_names.update(h.catalytic_spec.components.keys())
            for rec in traj.step_records:
                for aid in rec.touched_atoms:
                    if aid in s0_atoms and base_element(s0_atoms[aid].element) in {"C", "O", "N", "S"}:
                        touched_heavy.add(aid)
                for u, v, _ in rec.bonds_broken:
                    if u in s0_atoms and v in s0_atoms:
                        eu, ev = base_element(s0_atoms[u].element), base_element(s0_atoms[v].element)
                        if eu == "H" and ev != "H":
                            cleaved_xh_owners.add(v)
                        elif ev == "H" and eu != "H":
                            cleaved_xh_owners.add(u)

        iso_for_el = {"C": "13C", "O": "18O", "N": "15N", "S": "34S"}
        for aid in sorted(touched_heavy)[:8]:
            el = base_element(s0_atoms[aid].element)
            if el in iso_for_el:
                iso = iso_for_el[el]
                probes.append(ExperimentalProbe(
                    experiment_type="isotope_label",
                    perturbation=f"Site-specific {iso} labeling at {aid}",
                    observable=f"{iso} NMR / HRMS regioselectivity in product",
                    parameters={"labeled_atoms": {aid: iso}, "target_isotope": iso},
                    probe_id=f"iso_{iso}_{aid}",
                ))

        probes.append(ExperimentalProbe(
            experiment_type="isotope_label",
            perturbation="Run reaction in D2O / deuterated protic cosolvent",
            observable="Deuterium incorporation site and %D in product",
            parameters={"solvent_isotope": "2H", "target_isotope": "2H"},
            probe_id="iso_D2O_pool",
        ))

        # 2. Protocol-resolved KIE probes (parallel vs intermolecular vs intramolecular competition)
        kie_targets = sorted(cleaved_xh_owners)[:3] or [None]
        for target_c in kie_targets:
            tag = target_c or "substrate"
            l_param = {"labeled_atoms": [target_c]} if target_c else {}
            for proto in ("parallel", "intermolecular_competition", "intramolecular_competition"):
                probes.append(ExperimentalProbe(
                    experiment_type="kie",
                    perturbation=f"Deuterated {tag} ({proto} KIE)",
                    observable=f"kH/kD via {proto}",
                    parameters={**l_param, "protocol": proto},
                    probe_id=f"kie_{proto}_{tag}",
                ))

        # 3. Double-labeling Crossover
        probes.append(ExperimentalProbe(
            experiment_type="crossover",
            perturbation="1:1 equimolar mixture of doubly substituted analogues A-X and B-Y",
            observable="HRMS / NMR detection of crossover products A-Y and B-X",
            probe_id="crossover_standard",
        ))

        # 4. Radical traps (yield + HRMS adduct) & Radical clocks (standard + hypersensitive)
        probes.append(ExperimentalProbe(
            experiment_type="radical_trap",
            perturbation="Add TEMPO (2.0 equiv)",
            observable="Reaction yield suppression and TEMPO-adduct HRMS m/z",
            parameters={"reagent": "TEMPO", "observable_mode": "adduct"},
            probe_id="trap_TEMPO_adduct",
        ))
        probes.append(ExperimentalProbe(
            experiment_type="radical_clock",
            perturbation="Cyclopropylcarbinyl radical-clock substrate",
            observable="Ratio of ring-opened vs intact cyclopropane product",
            parameters={"clock_type": "cyclopropylcarbinyl"},
            probe_id="clock_cyclopropyl",
        ))
        probes.append(ExperimentalProbe(
            experiment_type="radical_clock",
            perturbation="trans-2-Phenylcyclopropylcarbinyl hypersensitive clock (k_r = 3e11 s^-1)",
            observable="Ring-opened vs unrearranged product (distinguishes cage rebound from concerted)",
            parameters={"clock_type": "trans_2_phenylcyclopropylcarbinyl"},
            probe_id="clock_hypersensitive",
        ))

        # 5. Photochemical quantum yield, light on/off, and Stern-Volmer
        probes.append(ExperimentalProbe(
            experiment_type="quantum_yield",
            perturbation="Ferrioxalate actinometry photon flux measurement",
            observable="Quantum yield Φ",
            probe_id="photochem_quantum_yield",
        ))
        probes.append(ExperimentalProbe(
            experiment_type="light_on_off",
            perturbation="Alternating irradiation and dark intervals",
            observable="Conversion progression during dark periods",
            probe_id="photochem_light_on_off",
        ))
        probes.append(ExperimentalProbe(
            experiment_type="stern_volmer",
            perturbation="Luminescence quenching titration across reaction components",
            observable="Stern-Volmer quenching constant K_SV",
            probe_id="photochem_stern_volmer",
        ))

        # 6. Hammett LFER, Stereochemical probe, and Asymmetric NLE
        probes.append(ExperimentalProbe(
            experiment_type="hammett",
            perturbation="Para-substituted aryl series (p-OMe to p-CF3)",
            observable="Hammett slope ρ from log(k_X/k_H) vs σ",
            probe_id="lfer_hammett",
        ))
        for c_aid in sorted(chiral_centers) or [None]:
            probes.append(ExperimentalProbe(
                experiment_type="stereochemical_probe",
                perturbation=f"Enantioenriched substrate at {c_aid or 'reaction center'}",
                observable="Chiral HPLC configurational outcome (retention / inversion / racemization)",
                parameters={"chiral_atom": c_aid} if c_aid else {},
                probe_id=f"stereo_{c_aid or 'center'}",
            ))
        probes.append(ExperimentalProbe(
            experiment_type="nonlinear_effect",
            perturbation="Vary chiral ligand ee from 0% to 100% (Kagan plot)",
            observable="Product ee vs ligand ee linearity",
            probe_id="asymmetric_nle",
        ))

        # 7. Kinetic orders & Omission controls for all declared components
        for cname in sorted(component_names):
            probes.append(ExperimentalProbe(
                experiment_type="kinetic_order",
                perturbation=f"Variable concentration initial rates / VTNA for {cname}",
                observable=f"Kinetic reaction order in [{cname}]",
                parameters={"component": cname},
                probe_id=f"order_{cname}",
            ))
            probes.append(ExperimentalProbe(
                experiment_type="control",
                perturbation=f"Omit {cname}",
                observable="Product yield relative to standard conditions",
                parameters={"omitted_component": cname},
                probe_id=f"control_no_{cname}",
            ))

        return probes

    @staticmethod
    def _entropy_bits(probs: Iterable[float]) -> float:
        return -sum(p * math.log2(p) for p in probs if p > 1e-12)

    def rank_experiments(
        self,
        hypotheses: Sequence[MechanismHypothesis],
        prior_observations: Sequence[Tuple[ExperimentalProbe, str, Optional[Dict[str, Any]]]] = (),
        candidate_probes: Optional[Sequence[ExperimentalProbe]] = None,
        *,
        top_k: int = 10,
    ) -> List[ExperimentDesignCandidate]:
        """Rank candidate experiments by exact Shannon Expected Information Gain I(H ; Y_e | E)."""
        bel = self.engine.compute_posteriors(hypotheses, prior_observations)
        prior_h = bel["final_posteriors"]
        h_before = self._entropy_bits(prior_h.values())

        pool = list(candidate_probes) if candidate_probes is not None else self.generate_candidate_probes(hypotheses)
        ranked: List[ExperimentDesignCandidate] = []

        for probe in pool:
            preds = {h.hypothesis_id: self.lab.predict(h, probe) for h in hypotheses}
            possible_bins = sorted({
                normalize_bin_token(p.outcome_bin)
                for p in preds.values()
                if p.outcome_bin is not None
            })
            if not possible_bins:
                continue

            # Add an "other_unexpected" noise outcome bucket so probabilities over Y sum to 1
            lik_matrix: Dict[str, Dict[str, float]] = {}
            for h in hypotheses:
                hid = h.hypothesis_id
                raw_l = {
                    y: self.engine.observation_likelihood(preds[hid], y)
                    for y in possible_bins
                }
                if len(possible_bins) == 1:
                    only_y = possible_bins[0]
                    lik_matrix[hid] = {only_y: raw_l[only_y], "__alt__": 1.0 - raw_l[only_y]}
                else:
                    z_l = sum(raw_l.values())
                    lik_matrix[hid] = {y: v / z_l for y, v in raw_l.items()}

            all_y = list(next(iter(lik_matrix.values())).keys())
            p_y: Dict[str, float] = {
                y: sum(prior_h[h.hypothesis_id] * lik_matrix[h.hypothesis_id][y] for h in hypotheses)
                for y in all_y
            }

            expected_h_after = 0.0
            max_bf = 1.0
            for y, py in p_y.items():
                if py <= 1e-12:
                    continue
                post_given_y = [
                    (prior_h[h.hypothesis_id] * lik_matrix[h.hypothesis_id][y]) / py
                    for h in hypotheses
                ]
                expected_h_after += py * self._entropy_bits(post_given_y)
                liks_y = [lik_matrix[h.hypothesis_id][y] for h in hypotheses]
                if min(liks_y) > 0:
                    max_bf = max(max_bf, max(liks_y) / min(liks_y))

            eig = max(0.0, h_before - expected_h_after)
            summary = " vs ".join(
                f"{hid}: {preds[hid].outcome_bin}"
                for hid in sorted(preds)
            )
            ranked.append(
                ExperimentDesignCandidate(
                    probe=probe,
                    expected_info_gain_bits=eig,
                    predictions_by_hypothesis=preds,
                    max_bayes_factor=max_bf,
                    discriminating_summary=summary,
                )
            )

        ranked.sort(key=lambda c: (-c.expected_info_gain_bits, -c.max_bayes_factor, c.probe.probe_id))
        return ranked[:top_k]
