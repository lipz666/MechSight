"""Kinetic, KIE (parallel/inter/intra), photoredox, Stern-Volmer, and electrochemical simulators."""
from __future__ import annotations

import collections
import math
from typing import Any, Dict, List, Optional, Set, Tuple

from .chem_graph import base_element, connected_molecules, extract_graph
from .reference_data import PHOTOCATALYST_DATA, SPECIES_REDOX_LIBRARY, TRANSITION_METALS
from .schemas import AtomMeta, ExperimentalProbe, LabState, MechanismHypothesis, ObservablePrediction
from .universal_engine import StepExecutionRecord, TrajectoryResult, execute_hypothesis


# ===========================================================================
# 5. KIE (Parallel / Intermolecular / Intramolecular) & Rate-Order Simulator
# ===========================================================================

class KineticsAndKIESimulator:
    """Kozuch-Shaik energetic-span rate-law compiler and protocol-aware KIE simulator."""

    @staticmethod
    def _first_irreversible_substrate_step(
        hypothesis: MechanismHypothesis,
        traj: TrajectoryResult,
        substrate_atoms: Set[str],
    ) -> int:
        tdts = hypothesis.resolved_tdts_index()
        for idx, (step, rec) in enumerate(zip(hypothesis.steps, traj.step_records)):
            touches_sub = bool(set(rec.touched_atoms) & substrate_atoms) if substrate_atoms else True
            if touches_sub and (not step.reversible or idx == tdts):
                return idx
        return tdts

    @classmethod
    def simulate_kie(
        cls,
        hypothesis: MechanismHypothesis,
        probe: ExperimentalProbe,
        trajectory: Optional[TrajectoryResult] = None,
    ) -> ObservablePrediction:
        """Simulate Kinetic Isotope Effect (kH/kD) under parallel, intermolecular, or intramolecular protocols.

        Supported ``probe.parameters``:
          - ``protocol``: ``"parallel"`` (default, initial rates), ``"intermolecular_competition"``, or ``"intramolecular_competition"``
          - ``labeled_atoms``: list/set of heavy-atom or H atom IDs bearing D in the KIE experiment
        """
        traj = trajectory or execute_hypothesis(hypothesis)
        if not traj.valid:
            return ObservablePrediction(
                hypothesis_id=hypothesis.hypothesis_id,
                experiment_type="kie",
                outcome_bin=None,
                confidence=0.0,
                rationale=f"Pathway execution failed: {traj.error}",
            )

        params = probe.parameters
        protocol = str(params.get("protocol") or "").lower()
        obs_low = f"{probe.perturbation} {probe.observable}".lower()
        if not protocol:
            if "intramolecular" in obs_low:
                protocol = "intramolecular_competition"
            elif "competition" in obs_low or "one-pot" in obs_low or "intermolecular" in obs_low:
                protocol = "intermolecular_competition"
            else:
                protocol = "parallel"

        queried_atoms = set(params.get("labeled_atoms") or [])
        s0_atoms, _, s0_adj = extract_graph(traj.states[0])

        # Expand heavy-atom labels (e.g. C2 bearing C2-H) to include attached explicit H atoms
        probe_h_atoms: Set[str] = set()
        probe_heavy_atoms: Set[str] = set()
        if queried_atoms:
            for aid in queried_atoms:
                if aid not in s0_atoms:
                    continue
                if base_element(s0_atoms[aid].element) == "H":
                    probe_h_atoms.add(aid)
                    probe_heavy_atoms.update(s0_adj[aid])
                else:
                    probe_heavy_atoms.add(aid)
                    for nb in s0_adj[aid]:
                        if base_element(s0_atoms[nb].element) == "H":
                            probe_h_atoms.add(nb)
        else:
            # Default: inspect all C-H / O-H / N-H bonds present in initial state
            for aid, at in s0_atoms.items():
                if base_element(at.element) == "H":
                    probe_h_atoms.add(aid)
                    probe_heavy_atoms.update(s0_adj[aid])

        tdts = hypothesis.resolved_tdts_index()
        first_irrev = cls._first_irreversible_substrate_step(
            hypothesis, traj, probe_heavy_atoms | probe_h_atoms
        )

        # Find all steps where a probed X-H bond is broken or formed, or where sp2 <-> sp3 hybridization changes
        ch_cleavage_steps: List[int] = []
        hybridization_shift_at_tdts: Optional[str] = None  # "sp3_to_sp2" or "sp2_to_sp3"

        for idx, rec in enumerate(traj.step_records):
            st_before = traj.states[idx]
            st_after = traj.states[idx + 1]
            b_atoms, b_bonds, b_adj = extract_graph(st_before)
            _, a_bonds, a_adj = extract_graph(st_after)

            # Check explicit X-H bond cleavage or formation involving probed H or probed heavy atom
            broke_xh = False
            for u, v, _ in rec.bonds_broken:
                els = {base_element(b_atoms[u].element), base_element(b_atoms[v].element)}
                if "H" in els:
                    if not queried_atoms or ({u, v} & (probe_h_atoms | probe_heavy_atoms)):
                        broke_xh = True
            if broke_xh:
                ch_cleavage_steps.append(idx)

            # Check alpha-secondary hybridization change at TDTS on probed heavy carbon
            if idx == tdts:
                for heavy_c in probe_heavy_atoms:
                    if heavy_c not in b_atoms or base_element(b_atoms[heavy_c].element) != "C":
                        continue
                    max_b_before = max(
                        (float(b_bonds[tuple(sorted((heavy_c, nb)))].order or 1.0) for nb in b_adj[heavy_c]),
                        default=1.0,
                    )
                    max_b_after = max(
                        (float(a_bonds[tuple(sorted((heavy_c, nb)))].order or 1.0) for nb in a_adj[heavy_c]),
                        default=1.0,
                    )
                    if max_b_before >= 1.5 and max_b_after == 1.0:
                        hybridization_shift_at_tdts = "sp2_to_sp3"
                    elif max_b_before == 1.0 and max_b_after >= 1.5:
                        hybridization_shift_at_tdts = "sp3_to_sp2"

        # Evaluate KIE magnitude according to physical-organic protocol!
        kie_val = 1.02
        why = ""
        if ch_cleavage_steps:
            cleave_idx = ch_cleavage_steps[0]
            cleave_step = hypothesis.steps[cleave_idx]
            sc_low = cleave_step.step_class.lower()
            primary_mag = 4.5 if sc_low in {"hat", "cmd_amla", "e2", "hydride_transfer"} else 3.2

            if protocol == "parallel":
                if cleave_idx == tdts:
                    kie_val = primary_mag
                    why = f"X-H bond breaks directly in TDTS (step {tdts + 1}, {cleave_step.step_class}) -> primary parallel KIE ({kie_val:.1f})."
                elif cleave_idx < tdts and all(hypothesis.steps[k].reversible for k in range(cleave_idx, tdts)):
                    kie_val = 1.8
                    why = f"Reversible pre-equilibrium X-H cleavage (step {cleave_idx + 1}) prior to TDTS (step {tdts + 1}) -> Equilibrium Isotope Effect (EIE ~ {kie_val:.1f})."
                else:
                    kie_val = 1.05
                    why = (
                        f"X-H cleavage occurs at step {cleave_idx + 1}, AFTER irreversible TDTS/commitment (step {min(tdts, first_irrev) + 1}); "
                        "parallel initial-rate KIE masks post-RDS C-H cleavage (kH/kD ~ 1.05)."
                    )
            elif protocol == "intermolecular_competition":
                if cleave_idx == first_irrev:
                    kie_val = primary_mag
                    why = f"X-H cleavage IS the first irreversible substrate-committing step (step {cleave_idx + 1}) -> primary intermolecular competition KIE ({kie_val:.1f})."
                elif cleave_idx < first_irrev and all(hypothesis.steps[k].reversible for k in range(cleave_idx, first_irrev + 1)):
                    kie_val = 1.8
                    why = f"Reversible X-H cleavage before substrate commitment -> modest competition KIE/EIE ({kie_val:.1f})."
                else:
                    kie_val = 1.05
                    why = f"Irreversible substrate commitment (step {first_irrev + 1}) precedes X-H cleavage (step {cleave_idx + 1}) -> no intermolecular competition KIE (~1.05)."
            else:  # intramolecular_competition
                kie_val = primary_mag
                why = (
                    f"Intramolecular competition directly probes the product-determining X-H cleavage step "
                    f"(step {cleave_idx + 1}) even after irreversible binding -> primary KIE ({kie_val:.1f})."
                )
        elif hybridization_shift_at_tdts == "sp2_to_sp3":
            kie_val = 0.84
            why = f"No X-H bond broken, but alpha-carbon rehybridizes sp2 -> sp3 at TDTS (step {tdts + 1}) -> inverse secondary KIE ({kie_val:.2f})."
        elif hybridization_shift_at_tdts == "sp3_to_sp2":
            kie_val = 1.22
            why = f"No X-H bond broken, but alpha-carbon rehybridizes sp3 -> sp2 at TDTS (step {tdts + 1}) -> normal secondary KIE ({kie_val:.2f})."
        else:
            kie_val = 1.02
            why = "Probed C-H/X-H bond is neither cleaved nor rehybridized at the rate/product-determining transition state (kH/kD ~ 1.0)."

        if kie_val < 0.90:
            outcome_bin = "inverse (<0.9)"
        elif kie_val <= 1.30:
            outcome_bin = "none (0.9-1.3)"
        elif kie_val <= 2.50:
            outcome_bin = "small (1.3-2.5)"
        else:
            outcome_bin = "primary (>2.5)"

        return ObservablePrediction(
            hypothesis_id=hypothesis.hypothesis_id,
            experiment_type="kie",
            outcome_bin=outcome_bin,
            quantitative_value=round(kie_val, 2),
            structural_details={
                "protocol": protocol,
                "tdts_step": tdts + 1,
                "first_irreversible_step": first_irrev + 1,
                "ch_cleavage_steps": [i + 1 for i in ch_cleavage_steps],
                "estimated_kH_kD": round(kie_val, 2),
            },
            rationale=why,
        )

    @classmethod
    def simulate_kinetic_order(
        cls,
        hypothesis: MechanismHypothesis,
        probe: ExperimentalProbe,
        trajectory: Optional[TrajectoryResult] = None,
    ) -> ObservablePrediction:
        """Derive theoretical kinetic reaction order with respect to a queried component via Energetic Span (TDI -> TDTS)."""
        traj = trajectory or execute_hypothesis(hypothesis)
        if not traj.valid:
            return ObservablePrediction(
                hypothesis_id=hypothesis.hypothesis_id,
                experiment_type="kinetic_order",
                outcome_bin=None,
                confidence=0.0,
                rationale=f"Pathway execution failed: {traj.error}",
            )

        spec = hypothesis.catalytic_spec
        tdi = max(0, min(spec.tdi_index, len(traj.states) - 2))
        tdts = max(tdi, hypothesis.resolved_tdts_index())

        comp_name = str(probe.parameters.get("component") or "substrate").strip()
        comp_info = spec.components.get(comp_name, {})
        comp_atoms = set(comp_info.get("atoms") or probe.parameters.get("component_atoms") or [])
        role = str(comp_info.get("role") or probe.parameters.get("role") or "").lower()

        if comp_name in spec.dissociated_ligands_before_tdts:
            order_label = "inverse"
            order_val: Any = -1.0
            why = f"{comp_name} dissociates from resting state (TDI) prior to TDTS -> inverse (-1) kinetic order."
        elif comp_name in spec.saturation_components:
            order_label = "saturation"
            order_val = "0_to_1_saturation"
            why = f"{comp_name} undergoes reversible Michaelis-Menten pre-equilibrium binding before TDTS -> saturation kinetics."
        elif role == "catalyst" or "cat" in comp_name.lower():
            if spec.catalyst_resting_nuclearity > spec.active_catalyst_nuclearity:
                frac = spec.active_catalyst_nuclearity / spec.catalyst_resting_nuclearity
                order_label = "fractional"
                order_val = round(frac, 2)
                why = (
                    f"Off-cycle catalyst resting reservoir has nuclearity {spec.catalyst_resting_nuclearity} "
                    f"while active TDTS complex has nuclearity {spec.active_catalyst_nuclearity} -> fractional ({order_val}) order."
                )
            elif spec.active_catalyst_nuclearity >= 2 and spec.catalyst_resting_nuclearity == 1:
                order_label = "2"
                order_val = 2
                why = "Bimetallic/homodimeric TDTS assembled from monomeric catalyst resting state -> 2nd-order in [catalyst]."
            else:
                order_label = "1"
                order_val = 1
                why = "Monomeric catalyst participates once in TDTS from monomeric TDI -> 1st-order in [catalyst]."
        else:
            # Count how many distinct initial molecules belonging to comp_atoms enter the reacting complex between TDI and TDTS
            tdi_comps = connected_molecules(traj.states[tdi])
            tdts_after_comps = connected_molecules(traj.states[tdts + 1])

            # Track active core via touched atoms at TDTS
            tdts_touched = set(traj.step_records[tdts].touched_atoms)
            active_in_tdts: Set[str] = set()
            for c in tdts_after_comps:
                if set(c.atom_ids) & tdts_touched:
                    active_in_tdts.update(c.atom_ids)
            # Also include atoms touched in steps tdi .. tdts
            for k in range(tdi, tdts + 1):
                active_in_tdts.update(traj.step_records[k].touched_atoms)

            if comp_atoms:
                # How many separate connected components in S_0 belonging to comp_atoms entered active_in_tdts by step tdts,
                # minus those already bound in TDI?
                s0_comps = [c for c in connected_molecules(traj.states[0]) if set(c.atom_ids) & comp_atoms]
                # Find main catalyst/substrate core in TDI if catalytic
                entered = 0
                for c0 in s0_comps:
                    c0_set = set(c0.atom_ids)
                    participates_by_tdts = any(
                        bool(c0_set & set(traj.step_records[k].touched_atoms))
                        for k in range(tdi, tdts + 1)
                    )
                    if participates_by_tdts:
                        entered += 1
                order_val = min(2, entered)
                order_label = str(order_val)
                why = (
                    f"Between TDI (state {tdi}) and TDTS (step {tdts + 1}), {entered} molecule(s) of {comp_name} "
                    f"enter the rate-determining transition state -> {order_label}-order."
                )
            else:
                order_val = 1
                order_label = "1"
                why = f"Default stoichiometry for {comp_name} entering before/at TDTS -> 1st-order."

        outcome_bin = f"{comp_name}:{order_label}"
        return ObservablePrediction(
            hypothesis_id=hypothesis.hypothesis_id,
            experiment_type="kinetic_order",
            outcome_bin=outcome_bin,
            quantitative_value=order_val,
            structural_details={"component": comp_name, "tdi_state": tdi, "tdts_step": tdts + 1, "order": order_val},
            rationale=why,
        )


# ===========================================================================
# 6. Photoredox, Quantum Yield, Light On/Off, Stern-Volmer & Electrochemistry
# ===========================================================================

class PhotoredoxElectroSimulator:
    """Simulates quantum yields, light on/off behavior, Rehm-Weller Stern-Volmer quenching, and CV electrochemistry."""

    @staticmethod
    def simulate_quantum_yield(
        hypothesis: MechanismHypothesis,
        probe: ExperimentalProbe,
        trajectory: Optional[TrajectoryResult] = None,
    ) -> ObservablePrediction:
        traj = trajectory or execute_hypothesis(hypothesis)
        has_chain = any(s.is_chain_propagation for s in hypothesis.steps) or (
            hypothesis.catalytic_spec.chain_length_estimate > 1.2
        )
        n_photons = sum(max(0, s.photons) for s in hypothesis.steps)
        if has_chain and hypothesis.catalytic_spec.chain_length_estimate >= 1.5:
            phi = round(float(hypothesis.catalytic_spec.chain_length_estimate), 2)
            outcome_bin = "above_1"
            why = f"Self-sustaining radical-chain propagation loop gives Φ ≈ {phi} (> 1 photon-equivalent product per initiation event)."
        elif has_chain:
            phi = 2.5
            outcome_bin = "above_1"
            why = "Pathway contains dark radical-chain propagation steps regenerating the chain-carrying radical -> Φ > 1."
        else:
            phi = round(0.35 / max(1, n_photons), 2)
            outcome_bin = "below_1"
            why = "Closed catalytic/photochemical cycle requires >=1 photon per product turnover with non-unit efficiency -> Φ < 1."

        return ObservablePrediction(
            hypothesis_id=hypothesis.hypothesis_id,
            experiment_type="quantum_yield",
            outcome_bin=outcome_bin,
            quantitative_value=phi,
            structural_details={"has_chain_propagation": has_chain, "photons_per_cycle": n_photons, "estimated_phi": phi},
            rationale=why,
        )

    @staticmethod
    def simulate_light_on_off(
        hypothesis: MechanismHypothesis,
        probe: ExperimentalProbe,
        trajectory: Optional[TrajectoryResult] = None,
    ) -> ObservablePrediction:
        needs_light = any(s.photons > 0 or s.step_class in {"photoexcitation", "ent"} for s in hypothesis.steps) or bool(
            hypothesis.catalytic_spec.photocatalyst
        )
        if not needs_light or hypothesis.catalytic_spec.dark_thermal_initiation:
            outcome_bin = "continues_in_dark"
            why = "Thermal activation / dark initiation sustains conversion without continuous irradiation."
        else:
            outcome_bin = "stops_in_dark"
            has_chain = any(s.is_chain_propagation for s in hypothesis.steps)
            note = (
                " Note (Cismesia-Yoon principle): short-lived radical chains still stop within milliseconds in the dark due to diffusion-limited k_t; combine with quantum yield Φ to distinguish chain vs closed cycle."
                if has_chain
                else ""
            )
            why = f"Photoinitiation/photoredox excitation is required and chains terminate rapidly upon removing light.{note}"

        return ObservablePrediction(
            hypothesis_id=hypothesis.hypothesis_id,
            experiment_type="light_on_off",
            outcome_bin=outcome_bin,
            quantitative_value=outcome_bin,
            rationale=why,
        )

    @staticmethod
    def simulate_stern_volmer(
        hypothesis: MechanismHypothesis,
        probe: ExperimentalProbe,
        trajectory: Optional[TrajectoryResult] = None,
    ) -> ObservablePrediction:
        """Compute Rehm-Weller PET and triplet EnT driving forces to predict Stern-Volmer luminescence quenching."""
        spec = hypothesis.catalytic_spec
        pc_name = str(probe.parameters.get("photocatalyst") or spec.photocatalyst or "Ir(ppy)3")
        pc = dict(PHOTOCATALYST_DATA.get(pc_name, PHOTOCATALYST_DATA["Ir(ppy)3"]))
        if spec.photocatalyst_potentials:
            pc.update(spec.photocatalyst_potentials)

        # Build pool of candidate quenchers
        candidates: Dict[str, Dict[str, float]] = {}
        for s_name, vals in spec.species_redox.items():
            candidates[s_name] = dict(vals)
        for s_name, vals in (probe.parameters.get("candidates_redox") or {}).items():
            candidates[s_name] = dict(vals)

        queried = probe.parameters.get("quencher")
        if not queried and candidates:
            pert_low = probe.perturbation.lower()
            matched_cands = [cname for cname in candidates if cname.lower() in pert_low]
            if len(matched_cands) == 1:
                queried = matched_cands[0]
        if queried and queried not in candidates and queried in SPECIES_REDOX_LIBRARY:
            candidates[queried] = dict(SPECIES_REDOX_LIBRARY[queried])

        scores: Dict[str, Dict[str, Any]] = {}
        for name, rdx in candidates.items():
            # Reductive quenching: *PC + Q -> PC•- + Q•+  => dG_eV = E_ox(Q) - E_red(*PC)
            dg_red_q = (rdx["E_ox"] - pc["E_red_star"]) if "E_ox" in rdx else 9.9
            # Oxidative quenching: *PC + Q -> PC•+ + Q•-  => dG_eV = E_ox(*PC) - E_red(Q)
            dg_ox_q = (pc["E_ox_star"] - rdx["E_red"]) if "E_red" in rdx else 9.9
            # Energy transfer: *PC + Q -> PC + *Q        => dG_kcal = E_T(Q) - E_T(*PC) -> convert to eV (/23.06)
            dg_ent_eV = ((rdx["E_T"] - pc["E_T"]) / 23.06) if ("E_T" in rdx and "E_T" in pc) else 9.9

            best_dg = min(dg_red_q, dg_ox_q, dg_ent_eV)
            if best_dg == dg_red_q:
                mode = "reductive_quenching"
            elif best_dg == dg_ox_q:
                mode = "oxidative_quenching"
            else:
                mode = "energy_transfer"
            scores[name] = {
                "best_dG_eV": round(best_dg, 3),
                "mode": mode,
                "quenches": best_dg <= 0.10,  # <= +0.10 eV (+2.3 kcal/mol) endergonic threshold at rt
            }

        best_quencher = min(scores, key=lambda k: scores[k]["best_dG_eV"]) if scores else None

        if queried and queried in scores:
            q_res = scores[queried]
            outcome_bin = "quenches" if q_res["quenches"] else "no_quenching"
            why = (
                f"Rehm-Weller / EnT check for {queried} with *{pc_name}: "
                f"best mode = {q_res['mode']}, ΔG = {q_res['best_dG_eV']:+.2f} eV -> {outcome_bin}."
            )
        elif best_quencher:
            q_res = scores[best_quencher]
            outcome_bin = "quenches" if q_res["quenches"] else "no_quenching"
            why = (
                f"Among reaction components {list(scores)}, fastest quencher of *{pc_name} is {best_quencher} "
                f"via {q_res['mode']} (ΔG = {q_res['best_dG_eV']:+.2f} eV)."
            )
        else:
            # Fallback to checking mechanistic steps directly
            has_quench_step = any(s.step_class in {"set", "dissociative_set", "ent"} for s in hypothesis.steps)
            outcome_bin = "quenches" if has_quench_step else "no_quenching"
            why = "Evaluated directly from presence of excited-state SET/EnT step in pathway."

        return ObservablePrediction(
            hypothesis_id=hypothesis.hypothesis_id,
            experiment_type="stern_volmer",
            outcome_bin=outcome_bin,
            quantitative_value=best_quencher,
            structural_details={"photocatalyst": pc_name, "component_quenching": scores, "fastest_quencher": best_quencher},
            rationale=why,
        )

    @staticmethod
    def simulate_electrochemistry(
        hypothesis: MechanismHypothesis,
        probe: ExperimentalProbe,
        trajectory: Optional[TrajectoryResult] = None,
    ) -> ObservablePrediction:
        """Analyze anodic/cathodic electron stoichiometry, paired electrolysis, and CV redox matching."""
        anodic_e = sum(-s.electrode_electrons for s in hypothesis.steps if s.electrode_electrons < 0)
        cathodic_e = sum(s.electrode_electrons for s in hypothesis.steps if s.electrode_electrons > 0)
        if anodic_e > 0 and cathodic_e > 0:
            mode = "paired_electrolysis"
        elif anodic_e > 0:
            mode = "net_oxidative_anodic"
        elif cathodic_e > 0:
            mode = "net_reductive_cathodic"
        else:
            mode = "non_faradaic_or_redox_neutral_chemical"

        return ObservablePrediction(
            hypothesis_id=hypothesis.hypothesis_id,
            experiment_type="electrochemistry",
            outcome_bin=mode,
            quantitative_value={"anodic_F_per_mol": anodic_e, "cathodic_F_per_mol": cathodic_e},
            structural_details={
                "electrolysis_mode": mode,
                "anodic_electrons": anodic_e,
                "cathodic_electrons": cathodic_e,
                "net_faradays": anodic_e + cathodic_e,
            },
            rationale=f"Pathway exchanges {anodic_e} anodic e- and {cathodic_e} cathodic e- ({mode}).",
        )
