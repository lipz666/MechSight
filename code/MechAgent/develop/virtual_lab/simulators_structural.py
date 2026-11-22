"""Structural & topological experiment simulators: Isotope Ray-Tracing, Crossover, Radical Trapping, and Radical Clocks."""
from __future__ import annotations

import collections
import copy
from typing import Any, Dict, List, Optional, Set, Tuple

from .. import standalone_chemistry as sc
from .chem_graph import (
    MolecularEntity,
    atom_exact_mass,
    automorphism_orbits,
    base_element,
    canonical_isotope_tag,
    connected_molecules,
    electronic_automorphisms,
    extract_graph,
    materialize_implicit_hydrogens,
)
from .reference_data import (
    HRMS_ION_OFFSETS,
    MONOISOTOPIC_MASS,
    PROTIC_HETEROATOMS,
    RADICAL_CLOCK_LIBRARY,
    RADICAL_TRAP_REAGENTS,
    TRANSITION_METALS,
)
from .schemas import AtomMeta, ExperimentalProbe, LabState, MechanismHypothesis, ObservablePrediction
from .universal_engine import TrajectoryResult, execute_hypothesis, execute_step


def _main_organic_product(state: LabState) -> MolecularEntity:
    """Select the primary organic product component (largest carbon-containing component, else largest component)."""
    comps = connected_molecules(state)
    organic = [
        c for c in comps
        if any(base_element(a.element) == "C" for a in c.atoms.values())
        and not any(base_element(a.element) in TRANSITION_METALS for a in c.atoms.values())
    ]
    if organic:
        organic.sort(
            key=lambda c: (
                sum(1 for a in c.atoms.values() if base_element(a.element) == "C"),
                len(c.atom_ids),
            ),
            reverse=True,
        )
        return organic[0]
    return comps[0]


# ===========================================================================
# 1. Isotope Ray-Tracing Simulator (with Exact Symmetry Automorphism Branching)
# ===========================================================================

class IsotopeRayTracer:
    """Propagates isotopic distributions across PGL pathways with exact symmetry branching and protic pool exchange."""

    @staticmethod
    def _apply_protic_exchange(
        iso_map: Dict[str, Optional[str]],
        state: LabState,
        solvent_h_isotope: Optional[str],
    ) -> Dict[str, Optional[str]]:
        """Equilibrate exchangeable O-H / N-H / S-H protons with excess deuterated/protic solvent pool."""
        if not solvent_h_isotope:
            return iso_map
        atoms, _, adj = extract_graph(state)
        updated = dict(iso_map)
        norm_tag = canonical_isotope_tag("H", solvent_h_isotope)
        for aid, atom in atoms.items():
            if base_element(atom.element) != "H":
                continue
            # Free H+ or H bonded to O, N, S, halogen exchanges rapidly with protic solvent
            if int(atom.charge or 0) == 1 and len(adj[aid]) == 0:
                updated[aid] = norm_tag
                continue
            if len(adj[aid]) == 1:
                owner = next(iter(adj[aid]))
                if base_element(atoms[owner].element) in PROTIC_HETEROATOMS:
                    updated[aid] = norm_tag
        return updated

    @staticmethod
    def _branch_by_symmetry(
        branches: Dict[Tuple[Tuple[str, str], ...], float],
        state: LabState,
    ) -> Dict[Tuple[Tuple[str, str], ...], float]:
        """Distribute isotopic label configurations across the electronic automorphism group Aut(G)."""
        autos = electronic_automorphisms(state)
        if len(autos) <= 1:
            return branches
        new_branches: Dict[Tuple[Tuple[str, str], ...], float] = collections.defaultdict(float)
        w_factor = 1.0 / len(autos)
        for frozen_iso, prob in branches.items():
            if prob <= 1e-12:
                continue
            cur_dict = dict(frozen_iso)
            for perm in autos:
                # If atom u moves to perm[u] under molecular symmetry, its carried isotope lands at perm[u]
                permuted = sorted((perm.get(u, u), iso) for u, iso in cur_dict.items())
                new_branches[tuple(permuted)] += prob * w_factor
        return dict(new_branches)

    @classmethod
    def simulate(
        cls,
        hypothesis: MechanismHypothesis,
        probe: ExperimentalProbe,
        trajectory: Optional[TrajectoryResult] = None,
    ) -> ObservablePrediction:
        """Simulate an isotope-labeling experiment.

        Supported ``probe.parameters``:
          - ``labeled_atoms``: dict {atom_id: isotope_str} e.g. ``{"C14": "13C"}`` or ``{"O1": "18O"}``
          - ``solvent_isotope``: e.g. ``"2H"`` / ``"D"`` for D2O / CD3OD exchangeable proton pool
          - ``query_atom``: optional specific product atom ID to report in ``outcome_bin``
          - ``target_isotope``: optional isotope string to summarize (inferred from inputs if omitted)
        """
        params = probe.parameters
        labeled_atoms: Dict[str, str] = dict(params.get("labeled_atoms") or {})
        solvent_iso: Optional[str] = params.get("solvent_isotope")

        # Promote implicit hydrogens if any heavy atom was given a "2H"/"D" label on its implicit H,
        # OR if a deuterated/tritiated protic solvent pool is present and protic heteroatoms carry implicit H.
        base_hyp = hypothesis
        auto_promoted: Dict[str, List[str]] = {}
        init_atoms, _, _ = extract_graph(hypothesis.initial_state)
        heavy_d_requests = {
            aid: iso
            for aid, iso in list(labeled_atoms.items())
            if aid in init_atoms
            and base_element(init_atoms[aid].element) != "H"
            and canonical_isotope_tag("H", iso) in {"2H", "3H"}
        }
        protic_implicit_owners = {
            aid
            for aid, at in init_atoms.items()
            if solvent_iso and base_element(at.element) in PROTIC_HETEROATOMS and int(at.Hcount or 0) > 0
        }
        promote_targets = set(heavy_d_requests.keys()) | protic_implicit_owners
        if promote_targets:
            mat_state, auto_promoted = materialize_implicit_hydrogens(
                hypothesis.initial_state, target_atoms=promote_targets
            )
            for owner, h_list in auto_promoted.items():
                if owner in heavy_d_requests:
                    iso = heavy_d_requests[owner]
                    del labeled_atoms[owner]
                    for hid in h_list:
                        labeled_atoms[hid] = iso
            base_hyp = copy.deepcopy(hypothesis)
            base_hyp.initial_state = mat_state

        traj = trajectory if (trajectory is not None and not auto_promoted) else execute_hypothesis(base_hyp)
        if not traj.valid:
            return ObservablePrediction(
                hypothesis_id=hypothesis.hypothesis_id,
                experiment_type="isotope_label",
                outcome_bin=None,
                confidence=0.0,
                rationale=f"Pathway execution failed: {traj.error}",
            )

        s0 = traj.states[0]
        s0_atoms, _, _ = extract_graph(s0)
        init_iso: Dict[str, Optional[str]] = {}
        for aid, atom in s0_atoms.items():
            existing = canonical_isotope_tag(atom.element, s0.atom_meta.get(aid, AtomMeta()).isotope)
            if aid in labeled_atoms:
                existing = canonical_isotope_tag(
                    "H" if canonical_isotope_tag("H", labeled_atoms[aid]) in {"2H", "3H"} else atom.element,
                    labeled_atoms[aid],
                )
            init_iso[aid] = existing

        init_iso = cls._apply_protic_exchange(init_iso, s0, solvent_iso)
        frozen0 = tuple(sorted((k, v) for k, v in init_iso.items() if v is not None))
        branches: Dict[Tuple[Tuple[str, str], ...], float] = {frozen0: 1.0}

        # Propagate across every intermediate state with symmetry branching + protic exchange
        for idx, st in enumerate(traj.states):
            branches = cls._branch_by_symmetry(branches, st)
            if solvent_iso:
                exchanged_branches: Dict[Tuple[Tuple[str, str], ...], float] = collections.defaultdict(float)
                for f_map, prob in branches.items():
                    ex = cls._apply_protic_exchange(dict(f_map), st, solvent_iso)
                    key = tuple(sorted((k, v) for k, v in ex.items() if v is not None))
                    exchanged_branches[key] += prob
                branches = dict(exchanged_branches)

        final_state = traj.final_state
        prod_entity = _main_organic_product(final_state)
        prod_set = set(prod_entity.atom_ids)
        final_atoms, _, final_adj = extract_graph(final_state)

        # Determine target isotope being tracked
        target_iso = params.get("target_isotope")
        if target_iso:
            target_iso = canonical_isotope_tag("H" if target_iso in {"D", "2H", "T", "3H"} else "", target_iso) or target_iso
        else:
            seen_isos = [iso for f_map in branches for _, iso in f_map]
            if solvent_iso:
                target_iso = canonical_isotope_tag("H", solvent_iso)
            elif seen_isos:
                target_iso = seen_isos[0]
            else:
                target_iso = "2H"

        atom_fraction: Dict[str, float] = collections.defaultdict(float)
        heavy_carrier_fraction: Dict[str, float] = collections.defaultdict(float)
        isotopologue_dist: Dict[int, float] = collections.defaultdict(float)

        for f_map, prob in branches.items():
            iso_dict = dict(f_map)
            n_in_prod = 0
            for aid in prod_entity.atom_ids:
                if iso_dict.get(aid) == target_iso:
                    atom_fraction[aid] += prob
                    n_in_prod += 1
                    # Map explicit H/D back to its heavy-atom owner for NMR regioselectivity reporting
                    if base_element(final_atoms[aid].element) == "H" and len(final_adj[aid]) == 1:
                        owner = next(iter(final_adj[aid]))
                        heavy_carrier_fraction[owner] += prob
                    else:
                        heavy_carrier_fraction[aid] += prob
            isotopologue_dist[n_in_prod] += prob

        site_report = {k: round(v, 4) for k, v in sorted(heavy_carrier_fraction.items()) if v > 1e-6}
        raw_atom_report = {k: round(v, 4) for k, v in sorted(atom_fraction.items()) if v > 1e-6}

        # Group final heavy sites by product electronic symmetry orbits (= distinct NMR chemical shift signals!)
        prod_orbits = automorphism_orbits(final_state)
        nmr_signal_fraction: Dict[str, float] = collections.defaultdict(float)
        for site, val in heavy_carrier_fraction.items():
            if val <= 1e-6:
                continue
            orb = prod_orbits.get(site, (site,))
            orb_key = "/".join(orb)
            nmr_signal_fraction[orb_key] += val
        nmr_report = {k: round(v, 4) for k, v in sorted(nmr_signal_fraction.items()) if v > 1e-6}
        is_scrambled = sum(1 for v in nmr_report.values() if 0.05 < v < 0.95) >= 2

        query_atom = params.get("query_atom")
        if query_atom:
            frac = site_report.get(query_atom, raw_atom_report.get(query_atom, 0.0))
            pos_label = query_atom
        elif nmr_report:
            top_signals = [k for k, v in sorted(nmr_report.items(), key=lambda kv: (-kv[1], kv[0])) if v >= 0.05]
            pos_label = "+".join(top_signals) if is_scrambled else top_signals[0]
            frac = max(nmr_report.values())
        else:
            pos_label = "product"
            frac = 0.0

        if frac < 0.05:
            level = "none"
        elif frac < 0.50 or (is_scrambled and frac <= 0.5001):
            level = "partial (<50%)"
        else:
            level = "major (>=50%)"

        outcome_bin = f"{pos_label}:{level}"

        # Compute labeled product HRMS mass shift
        unlabeled_mass = prod_entity.exact_mass
        mean_labels = sum(k * p for k, p in isotopologue_dist.items())

        rationale = (
            f"Isotope {target_iso} ray-traced across {len(traj.step_records)} steps "
            f"({len(branches)} symmetry/exchange branch(es)): NMR signal fractions = {nmr_report or 'none'}"
            + (" (transient symmetric isotopic scrambling across distinct product signals)." if is_scrambled else ".")
        )
        return ObservablePrediction(
            hypothesis_id=hypothesis.hypothesis_id,
            experiment_type="isotope_label",
            outcome_bin=outcome_bin,
            quantitative_value=nmr_report,
            structural_details={
                "target_isotope": target_iso,
                "nmr_signal_incorporation": nmr_report,
                "site_incorporation": site_report,
                "atom_incorporation": raw_atom_report,
                "isotopologue_distribution": {f"n={k}": round(p, 4) for k, p in sorted(isotopologue_dist.items())},
                "is_scrambled": is_scrambled,
                "mean_labels_incorporated": round(mean_labels, 4),
                "unlabeled_product_exact_mass": unlabeled_mass,
            },
            rationale=rationale,
        )


# ===========================================================================
# 2. Double-Labeling Crossover Simulator
# ===========================================================================

class CrossoverSimulator:
    """Determines intramolecular retention vs intermolecular scrambling in double-labeling crossover experiments."""

    @staticmethod
    def simulate(
        hypothesis: MechanismHypothesis,
        probe: ExperimentalProbe,
        trajectory: Optional[TrajectoryResult] = None,
    ) -> ObservablePrediction:
        """Simulate a crossover experiment between two differentially labeled substrates A-X and B-Y.

        Supported ``probe.parameters``:
          - ``fragment_1_atoms``: iterable of atom IDs forming scaffold half 1 (optional; auto-inferred from cleaved/reformed product bonds if omitted)
          - ``fragment_2_atoms``: iterable of atom IDs forming migrating/coupling half 2
        """
        traj = trajectory or execute_hypothesis(hypothesis)
        if not traj.valid:
            return ObservablePrediction(
                hypothesis_id=hypothesis.hypothesis_id,
                experiment_type="crossover",
                outcome_bin=None,
                confidence=0.0,
                rationale=f"Pathway execution failed: {traj.error}",
            )

        s0 = traj.states[0]
        sf = traj.final_state
        prod = _main_organic_product(sf)
        prod_atoms = set(prod.atom_ids)

        params = probe.parameters
        frag1 = set(params.get("fragment_1_atoms") or []) & prod_atoms
        frag2 = set(params.get("fragment_2_atoms") or []) & prod_atoms

        # Auto-detect the two intramolecularly rearranging fragments if not explicitly passed:
        # Look for two subsets of product atoms that started in the SAME initial molecule,
        # had a bond broken between them, and formed a new bond between them!
        init_comps = connected_molecules(s0)
        dissociated_step: Optional[int] = None
        tight_cage_only = False

        if frag1 and frag2:
            for step_idx, st in enumerate(traj.states):
                comps = connected_molecules(st)
                comp_of: Dict[str, int] = {aid: c_i for c_i, comp in enumerate(comps) for aid in comp.atom_ids}
                c1_set = {comp_of[u] for u in frag1 if u in comp_of}
                c2_set = {comp_of[v] for v in frag2 if v in comp_of}
                if c1_set and c2_set and c1_set.isdisjoint(c2_set):
                    if step_idx == 0:
                        dissociated_step = 0
                        break
                    creating_step = hypothesis.steps[step_idx - 1]
                    if creating_step.cage_escaped is False:
                        tight_cage_only = True
                    else:
                        dissociated_step = step_idx
                        break
        else:
            # Automatic double-labeling mode: check whether ANY initial substrate component splits
            # into >=2 disconnected intermediate fragments that subsequently recombine in the product.
            s0_atoms_dict, _, _ = extract_graph(s0)
            for init_c in init_comps:
                shared = [
                    u for u in init_c.atom_ids
                    if u in prod_atoms and base_element(s0_atoms_dict[u].element) != "H"
                ]
                if len(shared) < 2:
                    continue
                for step_idx, st in enumerate(traj.states[1:-1], start=1):
                    comps = connected_molecules(st)
                    comp_of = {aid: ci for ci, c in enumerate(comps) for aid in c.atom_ids}
                    distinct_mid_comps = {comp_of[u] for u in shared}
                    if len(distinct_mid_comps) >= 2:
                        creating_step = hypothesis.steps[step_idx - 1]
                        if creating_step.cage_escaped is False:
                            tight_cage_only = True
                        else:
                            dissociated_step = step_idx
                            break
                if dissociated_step is not None:
                    break

        if dissociated_step is not None:
            outcome_bin = "crossover"
            crossover_fraction = 0.50
            dist = {"AA": 0.25, "BB": 0.25, "AB": 0.25, "BA": 0.25}
            why = (
                f"Fragments exist as disconnected diffusible species at state {dissociated_step}, "
                "giving statistical intermolecular recombination (1:1:1:1 AA/BB/AB/BA)."
            )
        elif tight_cage_only:
            outcome_bin = "no_crossover"
            crossover_fraction = 0.03
            dist = {"AA": 0.485, "BB": 0.485, "AB": 0.015, "BA": 0.015}
            why = "Fragments cleave only within an unescaped tight solvent/metal cage; in-cage recombination suppresses crossover (<5%)."
        else:
            outcome_bin = "no_crossover"
            crossover_fraction = 0.0
            dist = {"AA": 0.50, "BB": 0.50, "AB": 0.0, "BA": 0.0}
            why = "Strictly intramolecular pathway: reacting fragments remain within one connected molecular entity across all steps."

        return ObservablePrediction(
            hypothesis_id=hypothesis.hypothesis_id,
            experiment_type="crossover",
            outcome_bin=outcome_bin,
            quantitative_value=crossover_fraction,
            structural_details={
                "product_distribution": dist,
                "dissociated_at_state": dissociated_step,
                "tight_cage": tight_cage_only,
            },
            rationale=why,
        )


# ===========================================================================
# 3. Radical Trapping & Adduct HRMS Generator
# ===========================================================================

class RadicalTrapSimulator:
    """Detects open-shell intermediates along H_k, predicts yield suppression, and builds trapped adduct formulas & HRMS m/z."""

    @staticmethod
    def _build_adduct_details(
        entity: MolecularEntity,
        radical_atom_id: str,
        trap_name: str,
    ) -> Dict[str, Any]:
        trap_info = RADICAL_TRAP_REAGENTS.get(trap_name, RADICAL_TRAP_REAGENTS["TEMPO"])
        frag_counts = entity.element_counts(separate_isotopes=False)
        trap_counts = dict(trap_info["formula"])
        combined: Dict[str, int] = collections.Counter(frag_counts)
        for el, cnt in trap_counts.items():
            combined[el] += cnt

        # 1,1-DPE oxidative radical trapping loses 1 H• (R• + CH2=CPh2 -> R-CH=CPh2); also compute hydroalkylated adduct
        if trap_name == "1,1-DPE":
            # trap_info["formula"] already accounts for C14H11 (-H substitution onto alkene)
            pass

        # Hill formula string
        parts: List[str] = []
        c_copy = dict(combined)
        if "C" in c_copy:
            n_c = c_copy.pop("C")
            parts.append("C" if n_c == 1 else f"C{n_c}")
            if "H" in c_copy:
                n_h = c_copy.pop("H")
                parts.append("H" if n_h == 1 else f"H{n_h}")
        for sym in sorted(c_copy):
            n_s = c_copy[sym]
            parts.append(sym if n_s == 1 else f"{sym}{n_s}")
        formula_str = "".join(parts)

        trap_exact_mass = sum(MONOISOTOPIC_MASS[el] * cnt for el, cnt in trap_counts.items())
        adduct_exact_mass = round(entity.exact_mass + trap_exact_mass, 5)
        hrms = {
            ion: round((adduct_exact_mass + offset) / z, 5)
            for ion, (offset, z) in HRMS_ION_OFFSETS.items()
        }
        return {
            "trap_reagent": trap_name,
            "trapped_radical_atom": radical_atom_id,
            "trapped_radical_element": base_element(entity.atoms[radical_atom_id].element),
            "radical_smiles": entity.to_smiles(),
            "adduct_formula": formula_str,
            "adduct_exact_mass": adduct_exact_mass,
            "adduct_hrms_mz": hrms,
        }

    @classmethod
    def simulate(
        cls,
        hypothesis: MechanismHypothesis,
        probe: ExperimentalProbe,
        trajectory: Optional[TrajectoryResult] = None,
    ) -> ObservablePrediction:
        """Simulate addition of a radical scavenger (TEMPO, BHT, 1,1-DPE, DMPO).

        Supported ``probe.parameters``:
          - ``reagent``: ``"TEMPO"`` (default), ``"BHT"``, ``"1,1-DPE"``, or ``"DMPO"``
          - ``observable_mode``: ``"yield"`` (default -> unaffected/reduced/suppressed) or ``"adduct"`` (-> adduct_detected/adduct_not_detected)
        """
        traj = trajectory or execute_hypothesis(hypothesis)
        if not traj.valid:
            return ObservablePrediction(
                hypothesis_id=hypothesis.hypothesis_id,
                experiment_type="radical_trap",
                outcome_bin=None,
                confidence=0.0,
                rationale=f"Pathway execution failed: {traj.error}",
            )

        reagent = str(probe.parameters.get("reagent") or "TEMPO").upper()
        if "DPE" in reagent or "DIPHENYLETHYLENE" in reagent:
            reagent = "1,1-DPE"
        elif "BHT" in reagent:
            reagent = "BHT"
        elif "DMPO" in reagent:
            reagent = "DMPO"
        else:
            reagent = "TEMPO"

        obs_text = f"{probe.observable} {probe.parameters.get('observable_mode', '')}".lower()
        want_adduct_bin = any(k in obs_text for k in ("adduct", "hrms", "lc-ms", "gc-ms", "epr", "detected"))

        adducts: List[Dict[str, Any]] = []
        seen_rad_signatures: Set[Tuple[str, str]] = set()
        has_diffusible_radical = False
        has_tight_cage_radical = False

        # Inspect intermediate states S_1 .. S_{L-1}
        for s_idx, st in enumerate(traj.states[1:-1], start=1):
            creating_step = hypothesis.steps[s_idx - 1]
            comps = connected_molecules(st)
            for comp in comps:
                for rad_aid in comp.radical_sites:
                    el = base_element(comp.atoms[rad_aid].element)
                    if el in TRANSITION_METALS:
                        continue
                    sig = (comp.hill_formula(), rad_aid)
                    if creating_step.cage_escaped is False:
                        has_tight_cage_radical = True
                    else:
                        has_diffusible_radical = True
                        if sig not in seen_rad_signatures:
                            seen_rad_signatures.add(sig)
                            adducts.append(cls._build_adduct_details(comp, rad_aid, reagent))

        has_chain = any(s.is_chain_propagation for s in hypothesis.steps) or (
            hypothesis.catalytic_spec.chain_length_estimate > 1.5 and has_diffusible_radical
        )

        if has_diffusible_radical:
            yield_bin = "suppressed (<10%)" if (has_chain or reagent in {"TEMPO", "1,1-DPE"}) else "reduced (10-80%)"
            adduct_bin = "adduct_detected"
            why = (
                f"Pathway generates {len(adducts)} diffusible organic radical intermediate(s) "
                f"trappable by {reagent} (predicted HRMS [M+H]+ m/z: "
                f"{[a['adduct_hrms_mz']['[M+H]+'] for a in adducts]})."
            )
        elif has_tight_cage_radical:
            yield_bin = "reduced (10-80%)"
            adduct_bin = "adduct_not_detected"
            why = f"Radical pair remains inside a tight solvent/metal coordination cage; {reagent} only partially intercepts cage escape."
        else:
            yield_bin = "unaffected (>=80% of standard yield)"
            adduct_bin = "adduct_not_detected"
            why = f"Purely closed-shell/concerted pathway with zero organic radical intermediates; {reagent} does not intercept productive turnover."

        outcome_bin = adduct_bin if want_adduct_bin else yield_bin
        return ObservablePrediction(
            hypothesis_id=hypothesis.hypothesis_id,
            experiment_type="radical_trap",
            outcome_bin=outcome_bin,
            quantitative_value=adducts[0]["adduct_hrms_mz"]["[M+H]+"] if adducts else None,
            structural_details={
                "yield_bin": yield_bin,
                "adduct_bin": adduct_bin,
                "trapped_adducts": adducts,
                "has_radical_chain": has_chain,
            },
            rationale=why,
        )


# ===========================================================================
# 4. Radical Clock Competition Simulator
# ===========================================================================

class RadicalClockSimulator:
    """Computes kinetic partitioning between unimolecular radical clock rearrangement (k_clock) and trapping/rebound."""

    @staticmethod
    def simulate(
        hypothesis: MechanismHypothesis,
        probe: ExperimentalProbe,
        trajectory: Optional[TrajectoryResult] = None,
    ) -> ObservablePrediction:
        """Simulate a radical-clock substrate probe.

        Supported ``probe.parameters``:
          - ``clock_type``: key in ``RADICAL_CLOCK_LIBRARY`` (default ``"cyclopropylcarbinyl"``)
          - ``k_clock_s``: optional custom unimolecular rearrangement rate in s^-1
          - ``clock_atom``: optional specific PGL atom ID where the radical would reside
        """
        traj = trajectory or execute_hypothesis(hypothesis)
        if not traj.valid:
            return ObservablePrediction(
                hypothesis_id=hypothesis.hypothesis_id,
                experiment_type="radical_clock",
                outcome_bin=None,
                confidence=0.0,
                rationale=f"Pathway execution failed: {traj.error}",
            )

        params = probe.parameters
        clock_name = str(params.get("clock_type") or "cyclopropylcarbinyl")
        clock_info = RADICAL_CLOCK_LIBRARY.get(clock_name, RADICAL_CLOCK_LIBRARY["cyclopropylcarbinyl"])
        k_clock = float(params.get("k_clock_s") or clock_info["k_rearr_s"])
        clock_atom: Optional[str] = params.get("clock_atom")

        # Check if pathway explicitly executes beta-scission ring opening on the clock already
        explicit_beta_scission = any(
            s.step_class.lower() in {"beta_scission", "radical_ring_opening"}
            for s in hypothesis.steps
        )

        # Find radical lifetime at clock_atom (or any organic carbon radical intermediate)
        competing_rates: List[float] = []
        for s_idx, st in enumerate(traj.states[1:-1], start=1):
            atoms, _, _ = extract_graph(st)
            next_step = hypothesis.steps[s_idx] if s_idx < len(hypothesis.steps) else hypothesis.steps[-1]
            creating_step = hypothesis.steps[s_idx - 1]
            for aid, atom in atoms.items():
                if clock_atom and aid != clock_atom:
                    continue
                if base_element(atom.element) in TRANSITION_METALS:
                    continue
                if st.atom_meta.get(aid, AtomMeta()).radicals > 0:
                    if creating_step.rebound_rate_s is not None:
                        k_comp = float(creating_step.rebound_rate_s)
                    elif next_step.rebound_rate_s is not None:
                        k_comp = float(next_step.rebound_rate_s)
                    elif creating_step.cage_escaped is False:
                        k_comp = 5.0e10  # picosecond in-cage rebound
                    else:
                        k_comp = float(next_step.bimolecular_trap_rate_s or 1.0e6)
                    competing_rates.append(k_comp)

        if explicit_beta_scission:
            f_open = 1.0
        elif not competing_rates:
            f_open = 0.0
        else:
            k_comp = min(competing_rates)
            f_open = k_clock / (k_clock + k_comp)

        if f_open >= 0.85:
            outcome_bin = "ring_opened"
        elif f_open <= 0.15:
            outcome_bin = "not_opened"
        else:
            outcome_bin = "mixed"

        rationale = (
            f"Clock {clock_name!r} (k_r = {k_clock:.2e} s^-1): "
            + (
                f"competes against k_trap/rebound = {min(competing_rates):.2e} s^-1 -> fraction opened = {f_open:.1%} ({outcome_bin})."
                if competing_rates
                else (
                    "pathway explicitly performs radical ring-opening beta-scission -> ring_opened."
                    if explicit_beta_scission
                    else "no radical intermediate generated at probe carbon -> ring stays intact (not_opened)."
                )
            )
        )
        return ObservablePrediction(
            hypothesis_id=hypothesis.hypothesis_id,
            experiment_type="radical_clock",
            outcome_bin=outcome_bin,
            quantitative_value=round(f_open, 4),
            structural_details={
                "clock_type": clock_name,
                "k_clock_s": k_clock,
                "k_compete_s": min(competing_rates) if competing_rates else None,
                "fraction_opened": round(f_open, 4),
            },
            rationale=rationale,
        )
