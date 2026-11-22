"""LFER (Hammett & competition), stereochemistry, nonlinear effects (NLE), omission controls, competence, and spectroscopy."""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .chem_graph import base_element, connected_molecules, extract_graph
from .reference_data import PAULING_ELECTRONEGATIVITY
from .schemas import AtomMeta, ExperimentalProbe, LabState, MechanismHypothesis, ObservablePrediction
from .universal_engine import TrajectoryResult, execute_hypothesis


# ===========================================================================
# 7. LFER (Hammett & Competition), Stereochemistry, and Kagan NLE Simulator
# ===========================================================================

class LFERAndStereoSimulator:
    """Derives Hammett rho, electronic competition ratios, stereochemical outcomes, and asymmetric NLE."""

    @staticmethod
    def _estimate_hammett_rho(
        hypothesis: MechanismHypothesis,
        traj: TrajectoryResult,
        probe_atom: Optional[str] = None,
    ) -> Tuple[float, str]:
        tdts = hypothesis.resolved_tdts_index()
        step = hypothesis.steps[tdts]
        rec = traj.step_records[tdts]

        if step.ts_charge_shift is not None:
            # Negative charge buildup (ts_charge_shift < 0) stabilized by EWG => positive rho!
            rho = -2.0 * float(step.ts_charge_shift)
            return rho, f"Explicit TS charge shift Δq‡ = {step.ts_charge_shift:+.2f} at step {tdts + 1} -> ρ ≈ {rho:+.2f}."

        target_atom = probe_atom or step.aryl_probe_atom
        if target_atom and target_atom in rec.charge_changes:
            q_before, q_after = rec.charge_changes[target_atom]
            dq = q_after - q_before
            if dq < 0:
                return +2.2 * abs(dq), f"Negative charge buildup ({q_before:+d} -> {q_after:+d}) on {target_atom} at TDTS -> positive ρ."
            if dq > 0:
                return -2.4 * abs(dq), f"Positive charge buildup ({q_before:+d} -> {q_after:+d}) on {target_atom} at TDTS -> negative ρ."

        sc_low = step.step_class.lower()
        if sc_low in {"sn1_ionization", "ngp_anchimeric"}:
            return -3.5, f"Carbocationic / anchimeric ionization at TDTS ({step.step_class}) builds strong positive charge -> large negative ρ (~ -3.5)."
        if sc_low in {"electrophilic_aromatic_substitution", "sear", "electrophilic_addition"}:
            return -2.2, f"Electrophilic attack ({step.step_class}) builds positive charge on aryl system -> negative ρ (~ -2.2)."
        if sc_low in {"hat"}:
            return -0.7, "Hydrogen-atom transfer (HAT) develops modest electrophilic partial positive character at benzylic/aryl center -> small negative ρ (~ -0.7)."
        if sc_low in {"oxidative_addition"}:
            return +2.0, "Transition-metal oxidative addition into Ar-X accumulates electron density on the ipso carbon -> positive ρ (~ +2.0)."
        if sc_low in {"snar", "benzylic_deprotonation"}:
            return +3.6, f"Anionic Meisenheimer / carbanion buildup ({step.step_class}) strongly stabilized by EWG -> large positive ρ (~ +3.6)."
        if sc_low in {"addition_elimination", "nucleophilic_addition", "cmd_amla"}:
            return +1.6, f"Nucleophilic addition / base-assisted C-H deprotonation ({step.step_class}) -> moderate positive ρ (~ +1.6)."
        if sc_low in {"sn2"}:
            return +0.6, "Concerted SN2 transition state shows modest positive ρ (~ +0.6)."
        if sc_low in {"pericyclic", "pericyclic_cycloaddition", "sigmatropic", "electrocyclic", "radical_addition"}:
            return +0.3, f"Concerted/nonpolar radical step ({step.step_class}) has minimal charge separation -> small |ρ| (< 1)."

        # Fallback: inspect net formal-charge shift on carbon atoms touched in TDTS
        b_atoms, _, _ = extract_graph(traj.states[tdts])
        c_dq = sum(
            (q1 - q0)
            for aid, (q0, q1) in rec.charge_changes.items()
            if base_element(b_atoms[aid].element) == "C"
        )
        if c_dq > 0:
            return -1.8, f"Net carbon formal charge increase (+{c_dq}) at TDTS -> negative ρ."
        if c_dq < 0:
            return +1.8, f"Net carbon formal charge decrease ({c_dq}) at TDTS -> positive ρ."
        return +0.4, "Minimal net carbon charge buildup at TDTS -> small |ρ| (< 1)."

    @classmethod
    def simulate_hammett(
        cls,
        hypothesis: MechanismHypothesis,
        probe: ExperimentalProbe,
        trajectory: Optional[TrajectoryResult] = None,
    ) -> ObservablePrediction:
        traj = trajectory or execute_hypothesis(hypothesis)
        if not traj.valid:
            return ObservablePrediction(
                hypothesis_id=hypothesis.hypothesis_id,
                experiment_type="hammett",
                outcome_bin=None,
                confidence=0.0,
                rationale=f"Pathway execution failed: {traj.error}",
            )

        rho, why = cls._estimate_hammett_rho(hypothesis, traj, probe.parameters.get("probe_atom"))
        sign_str = "positive" if rho >= 0 else "negative"
        mag = abs(rho)
        if mag < 1.0:
            mag_bin = "<1"
        elif mag <= 3.0:
            mag_bin = "1-3"
        else:
            mag_bin = ">3"

        outcome_bin = f"{sign_str}:{mag_bin}"
        return ObservablePrediction(
            hypothesis_id=hypothesis.hypothesis_id,
            experiment_type="hammett",
            outcome_bin=outcome_bin,
            quantitative_value=round(rho, 2),
            structural_details={"rho_estimate": round(rho, 2), "sign": sign_str, "magnitude_bin": mag_bin},
            rationale=why,
        )

    @classmethod
    def simulate_competition(
        cls,
        hypothesis: MechanismHypothesis,
        probe: ExperimentalProbe,
        trajectory: Optional[TrajectoryResult] = None,
    ) -> ObservablePrediction:
        """Simulate one-pot intermolecular electronic competition between EWG- and EDG-substituted partners."""
        traj = trajectory or execute_hypothesis(hypothesis)
        if not traj.valid:
            return ObservablePrediction(
                hypothesis_id=hypothesis.hypothesis_id,
                experiment_type="competition",
                outcome_bin=None,
                confidence=0.0,
                rationale=f"Pathway execution failed: {traj.error}",
            )

        params = probe.parameters
        ewg_name = str(params.get("ewg_partner") or "EWG")
        edg_name = str(params.get("edg_partner") or "EDG")
        sigma_ewg = float(params.get("sigma_ewg", +0.54))  # e.g. p-CF3
        sigma_edg = float(params.get("sigma_edg", -0.27))  # e.g. p-OMe

        rho, why = cls._estimate_hammett_rho(hypothesis, traj, params.get("probe_atom"))
        log10_ratio = rho * (sigma_ewg - sigma_edg)
        ratio = 10.0 ** abs(log10_ratio)
        preferred = ewg_name if log10_ratio >= 0 else edg_name

        if ratio < 2.0:
            r_bin = "<2"
        elif ratio <= 10.0:
            r_bin = "2-10"
        else:
            r_bin = ">10"

        outcome_bin = f"{preferred}:{r_bin}"
        return ObservablePrediction(
            hypothesis_id=hypothesis.hypothesis_id,
            experiment_type="competition",
            outcome_bin=outcome_bin,
            quantitative_value=round(ratio, 2),
            structural_details={"preferred_partner": preferred, "k_fast_over_k_slow": round(ratio, 2), "rho_used": round(rho, 2)},
            rationale=f"{why} With Δσ = {sigma_ewg - sigma_edg:.2f}, {preferred} reacts {ratio:.1f}x faster ({outcome_bin}).",
        )

    @staticmethod
    def simulate_stereochemistry(
        hypothesis: MechanismHypothesis,
        probe: ExperimentalProbe,
        trajectory: Optional[TrajectoryResult] = None,
    ) -> ObservablePrediction:
        """Predict stereochemical retention, inversion, racemization, or diastereomeric scrambling."""
        traj = trajectory or execute_hypothesis(hypothesis)
        if not traj.valid:
            return ObservablePrediction(
                hypothesis_id=hypothesis.hypothesis_id,
                experiment_type="stereochemical_probe",
                outcome_bin=None,
                confidence=0.0,
                rationale=f"Pathway execution failed: {traj.error}",
            )

        s0 = traj.states[0]
        sf = traj.final_state
        probe_atom: Optional[str] = probe.parameters.get("chiral_atom")

        chiral_atoms = [
            aid for aid, meta in s0.atom_meta.items()
            if meta.stereo_parity in (-1, 1)
        ]
        if probe_atom and probe_atom in s0.atom_meta:
            targets = [probe_atom]
        elif chiral_atoms:
            targets = chiral_atoms
        else:
            targets = []

        if targets:
            aid = targets[0]
            p0 = s0.atom_meta[aid].stereo_parity
            pf = sf.atom_meta[aid].stereo_parity
            step_history = [
                f"step {r.step_index + 1}:{r.stereo_events[aid]}"
                for r in traj.step_records
                if aid in r.stereo_events
            ]
            if pf == 0:
                other_intact = any(
                    sf.atom_meta[o].stereo_parity in (-1, 1)
                    for o in chiral_atoms
                    if o != aid
                )
                outcome_bin = "scrambling" if other_intact else "racemization"
                why = f"Stereocenter {aid} planarizes/racemizes ({', '.join(step_history) or 'planar achiral intermediate'})."
            elif p0 == pf:
                outcome_bin = "retention"
                why = (
                    f"Stereocenter {aid} shows net configuration retention "
                    f"(parity {p0:+d} -> {pf:+d}; events: {', '.join(step_history) or 'untouched/frontside'})."
                )
            else:
                outcome_bin = "inversion"
                why = (
                    f"Stereocenter {aid} undergoes net configuration inversion "
                    f"(parity {p0:+d} -> {pf:+d}; events: {', '.join(step_history)})."
                )
            return ObservablePrediction(
                hypothesis_id=hypothesis.hypothesis_id,
                experiment_type="stereochemical_probe",
                outcome_bin=outcome_bin,
                quantitative_value=outcome_bin,
                structural_details={"probed_atom": aid, "initial_parity": p0, "final_parity": pf, "step_events": step_history},
                rationale=why,
            )

        # Check Woodward-Hoffmann electrocyclic stereochemistry if no sp3 parity atom was set
        wh_modes = [
            r.stereo_events["__electrocyclic__"]
            for r in traj.step_records
            if "__electrocyclic__" in r.stereo_events
        ]
        if wh_modes:
            # Map conrotatory/disrotatory to stereospecific retention/inversion convention or report mode
            mode = wh_modes[0]
            outcome_bin = "retention" if mode == "disrotatory" else "inversion"
            return ObservablePrediction(
                hypothesis_id=hypothesis.hypothesis_id,
                experiment_type="stereochemical_probe",
                outcome_bin=outcome_bin,
                quantitative_value=mode,
                structural_details={"woodward_hoffmann_mode": mode},
                rationale=f"Woodward-Hoffmann stereospecific electrocyclic pathway ({mode}).",
            )

        return ObservablePrediction(
            hypothesis_id=hypothesis.hypothesis_id,
            experiment_type="stereochemical_probe",
            outcome_bin="retention",
            quantitative_value="retention",
            rationale="Concerted stereospecific pathway without planarizing intermediates.",
        )

    @staticmethod
    def simulate_nonlinear_effect(
        hypothesis: MechanismHypothesis,
        probe: ExperimentalProbe,
        trajectory: Optional[TrajectoryResult] = None,
    ) -> ObservablePrediction:
        """Evaluate Kagan's asymmetric nonlinear effect (ee_product vs ee_ligand) at ee_L = 50%."""
        spec = hypothesis.catalytic_spec
        ee_l = float(probe.parameters.get("ee_ligand", 0.50))

        if spec.chiral_ligands_in_active <= 1 and spec.chiral_ligands_in_reservoir <= 1:
            outcome_bin = "linear"
            ee_ratio = 1.0
            why = "Strictly monomeric 1:1 metal-to-chiral-ligand species in both resting state and TDTS -> linear relationship."
        elif spec.chiral_ligands_in_reservoir >= 2 and spec.meso_reservoir_inactive and spec.heterochiral_reservoir_K > 1.2:
            # Kagan reservoir effect: stable inactive heterochiral [M2(L_R)(L_S)] sequesters minor enantiomer -> +NLE!
            k_het = spec.heterochiral_reservoir_K
            ee_eff = min(0.99, ee_l * (1.0 + 0.35 * math.log(k_het)) / (1.0 + 0.35 * math.log(k_het) * ee_l))
            ee_ratio = round(ee_eff / max(ee_l, 1e-6), 3)
            outcome_bin = "positive_NLE"
            why = (
                f"Off-cycle heterochiral meso dimer reservoir (K_hetero/homo = {k_het:.1f}) sequesters the minor ligand enantiomer, "
                f"amplifying monomeric active catalyst ee (+NLE, amplification factor ~ {ee_ratio:.2f}x at 50% ee_L)."
            )
        elif spec.chiral_ligands_in_active >= 2:
            # Active ML2 species: Kagan (1 + beta) / (1 + g * beta) model
            g_react = float(probe.parameters.get("heterochiral_reactivity_g", 0.0 if spec.heterochiral_reservoir_K <= 1.0 else 2.5))
            k_stat = spec.heterochiral_reservoir_K if spec.heterochiral_reservoir_K > 0 else 2.0
            if g_react > 1.05:
                outcome_bin = "negative_NLE"
                ee_ratio = 0.65
                why = "Active ML2 species where heterochiral (R,S) complex turns over faster than homochiral (R,R) -> negative NLE (-NLE)."
            else:
                outcome_bin = "positive_NLE"
                ee_ratio = 1.35
                why = f"Homochiral ML2 active complex with statistical/stabilized meso ML_R L_S state (K={k_stat}) -> positive NLE (+NLE)."
        else:
            outcome_bin = "linear"
            ee_ratio = 1.0
            why = "No higher-order diastereomeric ligand association detected -> linear ee correlation."

        return ObservablePrediction(
            hypothesis_id=hypothesis.hypothesis_id,
            experiment_type="nonlinear_effect",
            outcome_bin=outcome_bin,
            quantitative_value=ee_ratio,
            structural_details={"ee_ligand_test": ee_l, "amplification_factor": ee_ratio},
            rationale=why,
        )


# ===========================================================================
# 8. Omission Controls, Stoichiometric Intermediate Competence & Spectroscopy
# ===========================================================================

class CompetenceControlSimulator:
    """Evaluates causal component omission controls, stoichiometric intermediate competence, and spectroscopic resting states."""

    @staticmethod
    def simulate_control(
        hypothesis: MechanismHypothesis,
        probe: ExperimentalProbe,
        trajectory: Optional[TrajectoryResult] = None,
    ) -> ObservablePrediction:
        """Simulate omitting a reagent, catalyst, base, light, or solvent swap."""
        traj = trajectory or execute_hypothesis(hypothesis)
        if not traj.valid:
            return ObservablePrediction(
                hypothesis_id=hypothesis.hypothesis_id,
                experiment_type="control",
                outcome_bin=None,
                confidence=0.0,
                rationale=f"Pathway execution failed: {traj.error}",
            )

        pert = f"{probe.perturbation} {probe.parameters.get('omitted', '')}".lower()
        spec = hypothesis.catalytic_spec

        # Check light omission ("no light", "absence of light", "in the dark", "without irradiation")
        if any(
            w in pert
            for w in (
                "no light",
                "without light",
                "absence of light",
                "light off",
                "in the dark",
                "dark",
                "without hv",
                "without irradiation",
                "absence of irradiation",
            )
        ):
            needs_hv = any(s.photons > 0 or s.step_class in {"photoexcitation", "ent"} for s in hypothesis.steps) or bool(spec.photocatalyst)
            if needs_hv and not spec.dark_thermal_initiation:
                return ObservablePrediction(
                    hypothesis_id=hypothesis.hypothesis_id,
                    experiment_type="control",
                    outcome_bin="suppressed (<10%)",
                    quantitative_value=0.0,
                    rationale="Pathway requires photochemical excitation; omitting light completely suppresses product formation (<10%).",
                )
            return ObservablePrediction(
                hypothesis_id=hypothesis.hypothesis_id,
                experiment_type="control",
                outcome_bin="unaffected (>=80%)",
                quantitative_value=0.90,
                rationale="Ground-state thermal pathway does not require photons -> yield unaffected in the dark.",
            )

        # Match omitted component against catalytic_spec.components (including metal symbol <-> name aliases)
        _METAL_ALIASES = {
            "iron": "Fe", "fe": "iron",
            "palladium": "Pd", "pd": "palladium",
            "nickel": "Ni", "ni": "nickel",
            "copper": "Cu", "cu": "copper",
            "cobalt": "Co", "co": "cobalt",
            "rhodium": "Rh", "rh": "rhodium",
            "iridium": "Ir", "ir": "iridium",
            "ruthenium": "Ru", "ru": "ruthenium",
            "gold": "Au", "au": "gold",
            "silver": "Ag", "ag": "silver",
            "zinc": "Zn", "zn": "zinc",
            "manganese": "Mn", "mn": "manganese",
        }
        raw_pert = f"{probe.perturbation} {probe.parameters.get('omitted', '')}"
        omitted_comp = probe.parameters.get("omitted_component")
        if not omitted_comp:
            for cname in spec.components:
                c_low = cname.lower()
                if c_low in pert:
                    omitted_comp = cname
                    break
                alias = _METAL_ALIASES.get(c_low)
                if alias:
                    if len(alias) <= 2 and re.search(rf"\b{re.escape(alias)}(?:[^a-zA-Z]|$|[A-Z])", raw_pert):
                        omitted_comp = cname
                        break
                    if len(alias) > 2 and alias.lower() in pert:
                        omitted_comp = cname
                        break

        all_touched: Set[str] = set()
        for rec in traj.step_records:
            all_touched.update(rec.touched_atoms)

        if omitted_comp and omitted_comp in spec.components:
            cinfo = spec.components[omitted_comp]
            catoms = set(cinfo.get("atoms") or [])
            essential = cinfo.get("essential")
            bg_yield = float(cinfo.get("background_yield_fraction", 0.0))

            # If the component's atoms exist in PGL state and NONE are ever touched along H_k => innocent spectator!
            if catoms and catoms.isdisjoint(all_touched) and essential is not True:
                return ObservablePrediction(
                    hypothesis_id=hypothesis.hypothesis_id,
                    experiment_type="control",
                    outcome_bin="unaffected (>=80%)",
                    quantitative_value=max(0.85, bg_yield),
                    structural_details={"omitted_component": omitted_comp, "participates_in_pgl": False},
                    rationale=f"Atoms of {omitted_comp} ({sorted(catoms)}) are untouched spectators in hypothesis {hypothesis.name!r} -> omitting {omitted_comp} leaves yield unaffected (>=80%).",
                )

            if essential is False or 0.10 <= bg_yield < 0.80:
                y_est = bg_yield if bg_yield > 0 else 0.45
                b = "unaffected (>=80%)" if y_est >= 0.80 else "reduced (10-80%)"
                return ObservablePrediction(
                    hypothesis_id=hypothesis.hypothesis_id,
                    experiment_type="control",
                    outcome_bin=b,
                    quantitative_value=y_est,
                    structural_details={"omitted_component": omitted_comp, "participates_in_pgl": True, "essential": False},
                    rationale=f"{omitted_comp} acts as a non-essential promoter/base with background reactivity (~{y_est:.0%}) -> {b}.",
                )

            return ObservablePrediction(
                hypothesis_id=hypothesis.hypothesis_id,
                experiment_type="control",
                outcome_bin="suppressed (<10%)",
                quantitative_value=bg_yield,
                structural_details={"omitted_component": omitted_comp, "participates_in_pgl": True, "essential": True},
                rationale=f"{omitted_comp} is causally required on the productive reaction path -> omitting it suppresses yield (<10%).",
            )

        # Element-symbol check directly on PGL touched atoms (e.g. "No Fe", "Omission of Fe(NO3)3", "No Pd(OAc)2")
        s0_atoms, _, _ = extract_graph(traj.states[0])
        queried_element = probe.parameters.get("omitted_element")
        raw_pert = f"{probe.perturbation} {probe.parameters.get('omitted', '')}"
        if not queried_element:
            for el in sorted({base_element(a.element) for a in s0_atoms.values()}, key=len, reverse=True):
                if el in {"C", "H", "O", "N", "B", "S", "P"}:
                    continue
                if el.lower() in pert.split():
                    queried_element = el
                    break
                # Match case-sensitive element symbol in chemical salt/complex formulas e.g. Fe(NO3)3, PdCl2, Cs2CO3
                if re.search(rf"\b{re.escape(el)}(?:[^a-zA-Z]|$|[A-Z])", raw_pert):
                    if el == "In" and re.match(r"^\s*In\s+", raw_pert):
                        continue
                    queried_element = el
                    break

        if queried_element:
            el_atoms = {aid for aid, a in s0_atoms.items() if base_element(a.element) == queried_element}
            if el_atoms and el_atoms.isdisjoint(all_touched):
                return ObservablePrediction(
                    hypothesis_id=hypothesis.hypothesis_id,
                    experiment_type="control",
                    outcome_bin="unaffected (>=80%)",
                    quantitative_value=0.88,
                    rationale=f"Element {queried_element} never participates in any elementary step of {hypothesis.name!r} -> omitting it leaves yield unaffected.",
                )
            if el_atoms and (el_atoms & all_touched):
                return ObservablePrediction(
                    hypothesis_id=hypothesis.hypothesis_id,
                    experiment_type="control",
                    outcome_bin="suppressed (<10%)",
                    quantitative_value=0.0,
                    rationale=f"Element {queried_element} directly mediates bond-making/breaking steps ({sorted(el_atoms & all_touched)}) -> omitting it suppresses yield (<10%).",
                )

        return ObservablePrediction(
            hypothesis_id=hypothesis.hypothesis_id,
            experiment_type="control",
            outcome_bin="suppressed (<10%)",
            quantitative_value=0.05,
            rationale="Omitted standard reaction component breaks productive pathway closure -> suppressed (<10%).",
        )

    @staticmethod
    def simulate_stoichiometric_intermediate(
        hypothesis: MechanismHypothesis,
        probe: ExperimentalProbe,
        trajectory: Optional[TrajectoryResult] = None,
    ) -> ObservablePrediction:
        """Test whether an isolated putative intermediate is chemically/kinetically competent to form the product."""
        traj = trajectory or execute_hypothesis(hypothesis)
        if not traj.valid:
            return ObservablePrediction(
                hypothesis_id=hypothesis.hypothesis_id,
                experiment_type="stoichiometric_intermediate",
                outcome_bin=None,
                confidence=0.0,
                rationale=f"Pathway execution failed: {traj.error}",
            )

        params = probe.parameters
        query_formula: Optional[str] = params.get("formula")
        query_smiles: Optional[str] = params.get("smiles")
        query_atoms = set(params.get("intermediate_atoms") or [])
        query_bonds = {tuple(sorted(pair)) for pair in (params.get("characteristic_bonds") or [])}

        matched_state_idx: Optional[int] = None
        for idx, st in enumerate(traj.states[:-1]):
            _, st_bonds, _ = extract_graph(st)
            if query_bonds and query_bonds.issubset(set(st_bonds.keys())):
                matched_state_idx = idx
                break
            for comp in connected_molecules(st):
                if query_formula and comp.hill_formula() == query_formula:
                    matched_state_idx = idx
                    break
                if query_smiles and comp.to_smiles(include_isotopes=False) == query_smiles:
                    matched_state_idx = idx
                    break
                if query_atoms and set(comp.atom_ids) == query_atoms:
                    matched_state_idx = idx
                    break
            if matched_state_idx is not None:
                break

        if matched_state_idx is not None:
            outcome_bin = "competent (gives product)"
            why = f"Putative intermediate matches on-path state {matched_state_idx} of {hypothesis.name!r} and converts to final product."
        else:
            outcome_bin = "not_competent"
            why = f"Putative intermediate does not appear along {hypothesis.name!r} (off-cycle or belongs to competing mechanism) -> not_competent."

        return ObservablePrediction(
            hypothesis_id=hypothesis.hypothesis_id,
            experiment_type="stoichiometric_intermediate",
            outcome_bin=outcome_bin,
            quantitative_value=(matched_state_idx is not None),
            structural_details={"matched_state_index": matched_state_idx},
            rationale=why,
        )

    @staticmethod
    def simulate_spectroscopy(
        hypothesis: MechanismHypothesis,
        probe: ExperimentalProbe,
        trajectory: Optional[TrajectoryResult] = None,
    ) -> ObservablePrediction:
        """Predict spectroscopic (NMR/IR/UV-Vis/EPR/HRMS) detectability of a resting state or open-shell intermediate."""
        traj = trajectory or execute_hypothesis(hypothesis)
        if not traj.valid:
            return ObservablePrediction(
                hypothesis_id=hypothesis.hypothesis_id,
                experiment_type="spectroscopy",
                outcome_bin=None,
                confidence=0.0,
                rationale=f"Pathway execution failed: {traj.error}",
            )

        params = probe.parameters
        modality = str(params.get("modality") or probe.observable or "NMR").upper()
        tdi = hypothesis.catalytic_spec.tdi_index
        tdts = hypothesis.resolved_tdts_index()

        if "EPR" in modality:
            has_rad = any(
                comp.total_radicals > 0
                for st in traj.states[1:-1]
                for comp in connected_molecules(st)
            )
            outcome_bin = "observed" if has_rad else "not_observed"
            why = (
                "Paramagnetic open-shell species generated along pathway -> EPR active (observed)."
                if has_rad
                else "Closed-shell diamagnetic pathway throughout -> EPR silent (not_observed)."
            )
            return ObservablePrediction(
                hypothesis_id=hypothesis.hypothesis_id,
                experiment_type="spectroscopy",
                outcome_bin=outcome_bin,
                quantitative_value=has_rad,
                rationale=why,
            )

        # For in-situ NMR / IR / UV-Vis: species at TDI or pre-TDTS equilibrium accumulate; post-TDTS fleeting species do not
        query_bonds = {tuple(sorted(pair)) for pair in (params.get("characteristic_bonds") or [])}
        query_formula: Optional[str] = params.get("formula")

        detected_idx: Optional[int] = None
        fleeting_only_idx: Optional[int] = None
        for idx, st in enumerate(traj.states[:-1]):
            _, st_bonds, _ = extract_graph(st)
            hit = False
            if query_bonds and query_bonds.issubset(set(st_bonds.keys())):
                hit = True
            elif query_formula and any(c.hill_formula() == query_formula for c in connected_molecules(st)):
                hit = True
            if hit:
                if idx == tdi or idx <= tdts:
                    detected_idx = idx
                    break
                fleeting_only_idx = idx

        if detected_idx is not None:
            outcome_bin = "observed"
            why = f"Queried species accumulates at/before TDTS (state {detected_idx}, TDI={tdi}) -> spectroscopically observable."
        elif fleeting_only_idx is not None:
            outcome_bin = "not_observed"
            why = f"Queried species forms only AFTER rate-limiting step (state {fleeting_only_idx} > TDTS {tdts + 1}) and decays rapidly -> below steady-state NMR/IR limit."
        else:
            outcome_bin = "not_observed"
            why = f"Queried species is absent from pathway {hypothesis.name!r} -> not_observed."

        return ObservablePrediction(
            hypothesis_id=hypothesis.hypothesis_id,
            experiment_type="spectroscopy",
            outcome_bin=outcome_bin,
            quantitative_value=(detected_idx is not None),
            structural_details={"accumulated_state": detected_idx, "fleeting_post_rds_state": fleeting_only_idx},
            rationale=why,
        )
