"""Comprehensive unit & integration suite for MechAgent.develop.virtual_lab.

Covers:
  1. Graph algorithms, exact IUPAC masses/HRMS, WL+backtracking symmetry automorphisms, SMILES, and conservation laws.
  2. Polar 2e (SN2 vs spiro-phenonium NGP vs SN1): 13C scrambling, stereo retention/inversion/racemization, Hammett & competition.
  3. Pericyclic reactions ([3,3] sigmatropic Claisen crossover vs tight-cage vs dissociative; 4pi/6pi Woodward-Hoffmann).
  4. Radical 1e fishhooks, photoredox closed cycles vs ATRA radical chains (quantum yield, light on/off, Stern-Volmer, TEMPO HRMS, radical clocks).
  5. Transition-metal organometallic cycles (Pd OA/CMD/RE: parallel vs inter vs intra KIE, inverse secondary KIE, kinetic orders, Kagan NLE, competence, NMR/EPR spectroscopy).
  6. Electrosynthesis (anodic, cathodic, paired electrolysis) and D2O protic-pool deuterium ray-tracing.
  7. Real 2026 literature abduction (PMC13039371 Fe-innocent EDA discovery), P2 leave-one-out, P3 counterfactual flip, P4 Shannon EIG design, and real core/ PGL dataset replay.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[3]
MECH_AGENT_DIR = Path(__file__).resolve().parents[2]
if str(MECH_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(MECH_AGENT_DIR))

from develop import standalone_chemistry as sc
from develop.virtual_lab import (
    EXPERIMENT_TYPES,
    AbductiveEngine,
    AtomMeta,
    CatalyticSpec,
    ExperimentDesigner,
    ExperimentalProbe,
    LabState,
    MechanismHypothesis,
    MechanisticConservationError,
    MechanisticStep,
    VirtualLab,
    automorphism_orbits,
    connected_molecules,
    electronic_automorphisms,
    execute_hypothesis,
    execute_step,
    probe_from_evidence_card,
    woodward_hoffmann_mode,
)


def _species(role: str, sub_name: str, atoms: dict, bonds: list) -> sc.Species:
    return sc.Species(
        role=role,
        state="stable",
        molecule=sc.Molecule(
            name="",
            substructures=[sc.Substructure(name=sub_name, atomdict=atoms, bondlist=bonds)],
        ),
    )


class TestGraphSymmetryAndConservation(unittest.TestCase):
    def test_all_17_plus_other_experiment_types_registered(self) -> None:
        self.assertEqual(len(EXPERIMENT_TYPES), 18)
        schema_path = MECH_AGENT_DIR / "experiments" / "evidence_p0" / "schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        enum_types = tuple(schema["properties"]["experiment_type"]["enum"])
        self.assertEqual(EXPERIMENT_TYPES, enum_types)

    def test_exact_monoisotopic_mass_isotopes_and_hrms(self) -> None:
        # Ethanol C2H6O: 2*12.000000 + 6*1.0078250322 + 15.9949146196 = 46.041865 Da
        st = LabState.from_pgl(
            sc.ReactionState(
                species=[
                    _species(
                        "reactant",
                        "etoh",
                        {
                            "C1": sc.Atom("C", Hcount=3),
                            "C2": sc.Atom("C", Hcount=2),
                            "O1": sc.Atom("O", Hcount=1),
                        },
                        [sc.Bond("C1", "C2", "single"), sc.Bond("C2", "O1", "single")],
                    )
                ]
            )
        )
        ent = connected_molecules(st)[0]
        self.assertEqual(ent.hill_formula(), "C2H6O")
        self.assertAlmostEqual(ent.exact_mass, 46.041865, places=5)
        hrms = ent.hrms_mz()
        self.assertAlmostEqual(hrms["[M+H]+"], 47.04914, places=4)
        self.assertTrue(len(ent.to_smiles()) >= 3)

        # 13C1-labeled ethanol increases mass by 13.003355 - 12.000000 = +1.003355 Da
        st_13c = st.clone()
        st_13c.atom_meta["C1"].isotope = "13C"
        ent_13c = connected_molecules(st_13c)[0]
        self.assertAlmostEqual(ent_13c.exact_mass - ent.exact_mass, 1.003355, places=5)
        self.assertIn("13", ent_13c.to_smiles(include_isotopes=True))

    def test_benzene_and_spiro_phenonium_automorphisms(self) -> None:
        # Kekule / aromatic benzene ring (6 carbons with 1.5 order bonds) -> D6h = 12 automorphisms
        atoms = {f"C{i}": sc.Atom("C", Hcount=1) for i in range(1, 7)}
        bonds = [
            sc.Bond(f"C{i}", f"C{(i % 6) + 1}", "aromatic", order=1.5)
            for i in range(1, 7)
        ]
        bz = LabState.from_pgl(sc.ReactionState(species=[_species("reactant", "bz", atoms, bonds)]))
        autos = electronic_automorphisms(bz)
        self.assertEqual(len(autos), 12)
        orbits = automorphism_orbits(bz)
        self.assertEqual(orbits["C1"], ("C1", "C2", "C3", "C4", "C5", "C6"))

    def test_conservation_laws_catch_atom_charge_and_spin_violations(self) -> None:
        st = LabState.from_pgl(
            sc.ReactionState(
                species=[
                    _species(
                        "reactant",
                        "eth",
                        {"C1": sc.Atom("C", Hcount=3), "Br1": sc.Atom("Br")},
                        [sc.Bond("C1", "Br1", "single")],
                    )
                ]
            )
        )
        # 1. Unbalanced formal charge without electrode_electrons
        with self.assertRaises(MechanisticConservationError):
            execute_step(st, MechanisticStep(step_class="polar", charge_changes={"C1": +1}))

        # 2. Closed-circuit non-metal spin parity violation (0 -> 1 unpaired electron)
        with self.assertRaises(MechanisticConservationError):
            execute_step(st, MechanisticStep(step_class="radical", radical_changes={"C1": +1}))


class TestPolarVsAnchimericNGP(unittest.TestCase):
    """Compare direct SN2, symmetric phenonium NGP, and dissociative SN1 producing identical unlabeled Ph-CH2-CH2-OAc."""

    def setUp(self) -> None:
        self.lab = VirtualLab()
        # Substrate: C_ipso(Ph ring C_ipso-C_o1-C_m1-C_p-C_m2-C_o2) - C_alpha(H2) - C_beta(H2) - Br1 + O_nu(AcO-)
        ring_atoms = {
            "C_ipso": sc.Atom("C", Hcount=0),
            "C_o1": sc.Atom("C", Hcount=1),
            "C_m1": sc.Atom("C", Hcount=1),
            "C_p": sc.Atom("C", Hcount=1),
            "C_m2": sc.Atom("C", Hcount=1),
            "C_o2": sc.Atom("C", Hcount=1),
            "C_alpha": sc.Atom("C", Hcount=2),
            "C_beta": sc.Atom("C", Hcount=2),
            "Br1": sc.Atom("Br", Hcount=0),
        }
        ring_bonds = [
            sc.Bond("C_ipso", "C_o1", "aromatic", order=1.5),
            sc.Bond("C_o1", "C_m1", "aromatic", order=1.5),
            sc.Bond("C_m1", "C_p", "aromatic", order=1.5),
            sc.Bond("C_p", "C_m2", "aromatic", order=1.5),
            sc.Bond("C_m2", "C_o2", "aromatic", order=1.5),
            sc.Bond("C_o2", "C_ipso", "aromatic", order=1.5),
            sc.Bond("C_ipso", "C_alpha", "single"),
            sc.Bond("C_alpha", "C_beta", "single"),
            sc.Bond("C_beta", "Br1", "single"),
        ]
        nu_atoms = {
            "O_nu": sc.Atom("O", charge=-1),
            "C_ac": sc.Atom("C", Hcount=3),
        }
        nu_bonds = [sc.Bond("O_nu", "C_ac", "single")]

        init_pgl = sc.ReactionState(
            species=[
                _species("reactant", "sub", ring_atoms, ring_bonds),
                _species("reactant", "nu", nu_atoms, nu_bonds),
            ]
        )
        init_meta = {"C_beta": AtomMeta(stereo_parity=+1)}
        self.init_state = LabState.from_pgl(init_pgl, atom_meta=init_meta)

        # Hypothesis 1: Direct concerted SN2 at C_beta
        sn2_action = sc.ReactionTransformation(
            transformation_type="reaction",
            electron_transfers=[
                sc.ElectronTransfer(("O_nu", "lone_pair"), ("O_nu", "C_beta", "new_sigma_bond"), 2, "bond_forming"),
                sc.ElectronTransfer(("C_beta", "Br1", "sigma_bond"), ("Br1", "lone_pair"), 2, "bond_breaking"),
            ],
        )
        self.h_sn2 = MechanismHypothesis(
            hypothesis_id="H_SN2",
            name="Direct Concerted SN2",
            initial_state=self.init_state,
            steps=[MechanisticStep(step_class="sn2", name="sn2_attack", pgl_action=sn2_action, is_rds=True)],
        )

        # Hypothesis 2: Anchimeric Neighboring-Group Participation via Symmetric Spiro Phenonium Ion!
        # Step 1: C_ipso forms bond to C_beta, Br1 leaves, C_p gains +1 formal charge -> C_alpha and C_beta become C2v symmetric!
        # Step 2: O_nu attacks C_beta, breaking C_ipso-C_beta and restoring neutrality on C_p.
        self.h_ngp = MechanismHypothesis(
            hypothesis_id="H_NGP",
            name="Anchimeric Spiro Phenonium Pathway",
            initial_state=self.init_state,
            steps=[
                MechanisticStep(
                    step_class="ngp_anchimeric",
                    name="phenonium_formation",
                    bonds_formed=[("C_ipso", "C_beta", 1.0)],
                    bonds_broken=[("C_beta", "Br1")],
                    charge_changes={"C_p": +1, "Br1": -1},
                    is_rds=True,
                ),
                MechanisticStep(
                    step_class="ngp_opening",
                    name="phenonium_opening",
                    bonds_formed=[("O_nu", "C_beta", 1.0)],
                    bonds_broken=[("C_ipso", "C_beta")],
                    charge_changes={"O_nu": +1, "C_p": -1},
                ),
            ],
        )

        # Hypothesis 3: Stepwise SN1 via open primary/benzylic-like planar carbocation at C_beta
        self.h_sn1 = MechanismHypothesis(
            hypothesis_id="H_SN1",
            name="Stepwise Open Carbocation SN1",
            initial_state=self.init_state,
            steps=[
                MechanisticStep(
                    step_class="sn1_ionization",
                    name="heterolysis",
                    pgl_action=sc.ReactionTransformation(
                        "reaction",
                        [sc.ElectronTransfer(("C_beta", "Br1", "sigma_bond"), ("Br1", "lone_pair"), 2, "bond_breaking")],
                    ),
                    is_rds=True,
                ),
                MechanisticStep(
                    step_class="nucleophilic_capture",
                    name="capture",
                    pgl_action=sc.ReactionTransformation(
                        "reaction",
                        [sc.ElectronTransfer(("O_nu", "lone_pair"), ("O_nu", "C_beta", "new_sigma_bond"), 2, "bond_forming")],
                    ),
                ),
            ],
        )

    def test_identical_unlabeled_products_across_all_three_hypotheses(self) -> None:
        smiles_set = set()
        for h in (self.h_sn2, self.h_ngp, self.h_sn1):
            traj = execute_hypothesis(h)
            self.assertTrue(traj.valid, traj.error)
            prods = sorted(c.to_smiles(include_isotopes=False) for c in connected_molecules(traj.final_state))
            smiles_set.add(tuple(prods))
        self.assertEqual(len(smiles_set), 1, f"Expected identical unlabeled products, got {smiles_set}")

    def test_13c_isotope_ray_tracing_separates_sn2_from_symmetric_phenonium(self) -> None:
        probe = ExperimentalProbe(
            experiment_type="isotope_label",
            perturbation="Label alpha-carbon with 13C",
            parameters={"labeled_atoms": {"C_alpha": "13C"}, "target_isotope": "13C"},
        )
        p_sn2 = self.lab.predict(self.h_sn2, probe)
        p_ngp = self.lab.predict(self.h_ngp, probe)

        self.assertFalse(p_sn2.structural_details["is_scrambled"])
        self.assertEqual(p_sn2.structural_details["nmr_signal_incorporation"], {"C_alpha": 1.0})
        self.assertEqual(p_sn2.outcome_bin, "C_alpha:major (>=50%)")

        # Symmetric phenonium intermediate scrambles 13C 50:50 across C_alpha and C_beta!
        self.assertTrue(p_ngp.structural_details["is_scrambled"])
        self.assertEqual(p_ngp.structural_details["nmr_signal_incorporation"], {"C_alpha": 0.5, "C_beta": 0.5})
        self.assertIn("C_alpha+C_beta:partial", p_ngp.outcome_bin)

    def test_stereochemistry_separates_inversion_retention_and_racemization(self) -> None:
        probe = ExperimentalProbe(experiment_type="stereochemical_probe", parameters={"chiral_atom": "C_beta"})
        self.assertEqual(self.lab.predict(self.h_sn2, probe).outcome_bin, "inversion")
        # Double inversion (NGP step 1 inversion + step 2 inversion) => net retention!
        self.assertEqual(self.lab.predict(self.h_ngp, probe).outcome_bin, "retention")
        # Planar 3-coordinate C_beta+ intermediate => racemization!
        self.assertEqual(self.lab.predict(self.h_sn1, probe).outcome_bin, "racemization")

    def test_hammett_and_competition_separate_polar_ts_charge(self) -> None:
        ham = ExperimentalProbe(experiment_type="hammett")
        self.assertEqual(self.lab.predict(self.h_sn2, ham).outcome_bin, "positive:<1")
        self.assertEqual(self.lab.predict(self.h_ngp, ham).outcome_bin, "negative:>3")

        comp = ExperimentalProbe(
            experiment_type="competition",
            parameters={"ewg_partner": "p-CF3", "edg_partner": "p-OMe"},
        )
        self.assertTrue(self.lab.predict(self.h_ngp, comp).outcome_bin.startswith("p-OMe:>10"))
        self.assertTrue(self.lab.predict(self.h_sn2, comp).outcome_bin.startswith("p-CF3:"))


class TestPericyclicAndCrossover(unittest.TestCase):
    """Test [3,3] sigmatropic Claisen rearrangement vs tight-cage vs escaped dissociative crossover, plus Woodward-Hoffmann."""

    def test_claisen_crossover_concerted_vs_cage_vs_dissociative(self) -> None:
        # Allyl vinyl ether: C1(=C2) - O1 - C3 - C4(=C5)  ->  O1=C1 - C2 - C5 - C4=C3 (pent-4-enal)
        ave_atoms = {
            "C1": sc.Atom("C", Hcount=1),
            "C2": sc.Atom("C", Hcount=2),
            "O1": sc.Atom("O", Hcount=0),
            "C3": sc.Atom("C", Hcount=2),
            "C4": sc.Atom("C", Hcount=1),
            "C5": sc.Atom("C", Hcount=2),
        }
        ave_bonds = [
            sc.Bond("C1", "C2", "double"),
            sc.Bond("C1", "O1", "single"),
            sc.Bond("O1", "C3", "single"),
            sc.Bond("C3", "C4", "single"),
            sc.Bond("C4", "C5", "double"),
        ]
        init = LabState.from_pgl(sc.ReactionState(species=[_species("reactant", "ave", ave_atoms, ave_bonds)]))

        # 1. Concerted [3,3] sigmatropic curly-arrow cycle
        concerted = MechanismHypothesis(
            hypothesis_id="H_claisen_33",
            name="Concerted [3,3] Claisen",
            initial_state=init,
            steps=[
                MechanisticStep(
                    step_class="sigmatropic",
                    pgl_action=sc.ReactionTransformation(
                        "reaction",
                        [
                            sc.ElectronTransfer(("O1", "C3", "sigma_bond"), ("C1", "O1", "new_pi_bond"), 2, "bond_breaking_or_pi_shift"),
                            sc.ElectronTransfer(("C1", "C2", "pi_bond"), ("C2", "C5", "new_sigma_bond"), 2, "bond_forming"),
                            sc.ElectronTransfer(("C4", "C5", "pi_bond"), ("C3", "C4", "new_pi_bond"), 2, "bond_breaking_or_pi_shift"),
                        ],
                    ),
                )
            ],
        )

        # 2. Dissociative ion-pair fragmentation (escaped vs tight cage)
        het_break = sc.ReactionTransformation(
            "reaction",
            [sc.ElectronTransfer(("O1", "C3", "sigma_bond"), ("O1", "lone_pair"), 2, "bond_breaking")],
        )
        recomb = MechanisticStep(
            step_class="polar",
            bonds_formed=[("C2", "C5", 1.0)],
            bond_order_changes=[("C1", "O1", 2.0), ("C1", "C2", 1.0), ("C4", "C5", 1.0), ("C3", "C4", 2.0)],
            charge_changes={"O1": +1, "C3": -1},
        )
        tight_cage = MechanismHypothesis(
            hypothesis_id="H_tight_cage",
            name="Tight Solvent-Cage Ion Pair",
            initial_state=init,
            steps=[MechanisticStep(step_class="polar", pgl_action=het_break, cage_escaped=False), recomb],
        )
        escaped = MechanismHypothesis(
            hypothesis_id="H_escaped",
            name="Diffusible Fragmentation-Recombination",
            initial_state=init,
            steps=[MechanisticStep(step_class="polar", pgl_action=het_break, cage_escaped=True), recomb],
        )

        lab = VirtualLab()
        probe = ExperimentalProbe(experiment_type="crossover")
        p_conc = lab.predict(concerted, probe)
        p_cage = lab.predict(tight_cage, probe)
        p_esc = lab.predict(escaped, probe)

        self.assertEqual(p_conc.outcome_bin, "no_crossover")
        self.assertEqual(p_conc.quantitative_value, 0.0)
        self.assertEqual(p_cage.outcome_bin, "no_crossover")
        self.assertLess(p_cage.quantitative_value, 0.05)
        self.assertEqual(p_esc.outcome_bin, "crossover")
        self.assertEqual(p_esc.structural_details["product_distribution"], {"AA": 0.25, "BB": 0.25, "AB": 0.25, "BA": 0.25})

    def test_woodward_hoffmann_electrocyclization_rules(self) -> None:
        self.assertEqual(woodward_hoffmann_mode(4, photochemical=False), "conrotatory")
        self.assertEqual(woodward_hoffmann_mode(4, photochemical=True), "disrotatory")
        self.assertEqual(woodward_hoffmann_mode(6, photochemical=False), "disrotatory")
        self.assertEqual(woodward_hoffmann_mode(6, photochemical=True), "conrotatory")

        # Execute 6-pi hexatriene -> cyclohexadiene thermal vs photochemical
        atoms = {
            "C1": sc.Atom("C", Hcount=2),
            "C2": sc.Atom("C", Hcount=1),
            "C3": sc.Atom("C", Hcount=1),
            "C4": sc.Atom("C", Hcount=1),
            "C5": sc.Atom("C", Hcount=1),
            "C6": sc.Atom("C", Hcount=2),
        }
        bonds = [
            sc.Bond("C1", "C2", "double"),
            sc.Bond("C2", "C3", "single"),
            sc.Bond("C3", "C4", "double"),
            sc.Bond("C4", "C5", "single"),
            sc.Bond("C5", "C6", "double"),
        ]
        init = LabState.from_pgl(sc.ReactionState(species=[_species("reactant", "hex", atoms, bonds)]))
        h_therm = MechanismHypothesis(
            hypothesis_id="H_6pi_therm",
            name="Thermal 6pi Electrocyclization",
            initial_state=init,
            steps=[
                MechanisticStep(
                    step_class="electrocyclic",
                    bonds_formed=[("C1", "C6", 1.0)],
                    bond_order_changes=[("C1", "C2", 1.0), ("C2", "C3", 2.0), ("C3", "C4", 1.0), ("C4", "C5", 2.0), ("C5", "C6", 1.0)],
                    pericyclic_electrons=6,
                    photons=0,
                )
            ],
        )
        h_photo = MechanismHypothesis(
            hypothesis_id="H_6pi_photo",
            name="Photochemical 6pi Electrocyclization",
            initial_state=init,
            steps=[
                MechanisticStep(
                    step_class="electrocyclic",
                    bonds_formed=[("C1", "C6", 1.0)],
                    bond_order_changes=[("C1", "C2", 1.0), ("C2", "C3", 2.0), ("C3", "C4", 1.0), ("C4", "C5", 2.0), ("C5", "C6", 1.0)],
                    pericyclic_electrons=6,
                    photons=1,
                )
            ],
        )
        lab = VirtualLab()
        st_probe = ExperimentalProbe(experiment_type="stereochemical_probe")
        self.assertEqual(lab.predict(h_therm, st_probe).structural_details["woodward_hoffmann_mode"], "disrotatory")
        self.assertEqual(lab.predict(h_photo, st_probe).structural_details["woodward_hoffmann_mode"], "conrotatory")


class TestRadicalPhotoredoxChain(unittest.TestCase):
    """Test 1e fishhooks, ATRA chain vs closed photoredox cycle, Stern-Volmer, TEMPO HRMS m/z, and radical clocks."""

    def setUp(self) -> None:
        self.lab = VirtualLab()
        # ATRA reaction: CH3-Br (C1-Br1) + CH2=CH2 (C2=C3) -> CH3-CH2-CH2-Br
        init_pgl = sc.ReactionState(
            species=[
                _species("reactant", "rbr", {"C1": sc.Atom("C", Hcount=3), "Br1": sc.Atom("Br")}, [sc.Bond("C1", "Br1", "single")]),
                _species("reactant", "alk", {"C2": sc.Atom("C", Hcount=2), "C3": sc.Atom("C", Hcount=2)}, [sc.Bond("C2", "C3", "double")]),
            ]
        )
        self.init_state = LabState.from_pgl(init_pgl)

        # True 1-electron fishhook steps!
        # Step 1: Homolytic photolysis C1-Br1 -> C1• + Br1•
        step_homolysis = MechanisticStep(
            step_class="homolysis",
            photons=1,
            pgl_action=sc.ReactionTransformation(
                "reaction",
                [
                    sc.ElectronTransfer(("C1", "Br1", "sigma_bond"), ("C1", "radical"), 1, "homolysis"),
                    sc.ElectronTransfer(("C1", "Br1", "sigma_bond"), ("Br1", "radical"), 1, "homolysis"),
                ],
            ),
        )
        # Step 2: Radical addition C1• + C2=C3 -> C1-C2-C3•
        step_rad_add = MechanisticStep(
            step_class="radical_addition",
            is_chain_propagation=True,
            pgl_action=sc.ReactionTransformation(
                "reaction",
                [
                    sc.ElectronTransfer(("C1", "radical"), ("C1", "C2", "new_sigma_bond"), 1, "bond_forming"),
                    sc.ElectronTransfer(("C2", "C3", "pi_bond"), ("C1", "C2", "new_sigma_bond"), 1, "bond_forming"),
                    sc.ElectronTransfer(("C2", "C3", "pi_bond"), ("C3", "radical"), 1, "pi_to_atom"),
                ],
            ),
        )
        # Step 3: Radical coupling / chain termination C3• + Br1• -> C3-Br1
        step_recomb = MechanisticStep(
            step_class="radical_coupling",
            pgl_action=sc.ReactionTransformation(
                "reaction",
                [
                    sc.ElectronTransfer(("C3", "radical"), ("C3", "Br1", "new_sigma_bond"), 1, "bond_forming"),
                    sc.ElectronTransfer(("Br1", "radical"), ("C3", "Br1", "new_sigma_bond"), 1, "bond_forming"),
                ],
            ),
        )

        self.h_chain = MechanismHypothesis(
            hypothesis_id="H_ATRA_chain",
            name="Self-Sustaining Radical Chain ATRA",
            initial_state=self.init_state,
            steps=[step_homolysis, step_rad_add, step_recomb],
            catalytic_spec=CatalyticSpec(
                photocatalyst="Ir(ppy)3",
                chain_length_estimate=18.5,
                species_redox={
                    "alkyl_bromide": {"E_ox": +2.20, "E_red": -1.62, "E_T": 70.0},
                    "ethylene": {"E_ox": +2.60, "E_red": -2.90, "E_T": 82.0},
                },
            ),
        )

        # Closed photoredox cycle (no chain propagation, chain_length = 1.0)
        step_rad_add_closed = MechanisticStep(
            step_class="radical_addition",
            is_chain_propagation=False,
            pgl_action=step_rad_add.pgl_action,
        )
        self.h_closed_pr = MechanismHypothesis(
            hypothesis_id="H_closed_photoredox",
            name="Closed Photoredox Cycle",
            initial_state=self.init_state,
            steps=[step_homolysis, step_rad_add_closed, step_recomb],
            catalytic_spec=CatalyticSpec(
                photocatalyst="Ir(ppy)3",
                chain_length_estimate=1.0,
                species_redox={
                    "alkyl_bromide": {"E_ox": +2.20, "E_red": -1.62, "E_T": 70.0},
                    "ethylene": {"E_ox": +2.60, "E_red": -2.90, "E_T": 82.0},
                },
            ),
        )

        # Concerted closed-shell 2-electron polar/pericyclic insertion control (zero radicals)
        self.h_polar = MechanismHypothesis(
            hypothesis_id="H_polar_insertion",
            name="Concerted Closed-Shell Insertion",
            initial_state=self.init_state,
            steps=[
                MechanisticStep(
                    step_class="pericyclic",
                    bonds_formed=[("C1", "C2", 1.0), ("C3", "Br1", 1.0)],
                    bonds_broken=[("C1", "Br1")],
                    bond_order_changes=[("C2", "C3", 1.0)],
                )
            ],
        )

    def test_1e_fishhooks_track_spin_parity_and_converge_to_closed_shell_product(self) -> None:
        traj = execute_hypothesis(self.h_chain)
        self.assertTrue(traj.valid, traj.error)
        # State 1: C1• (1) and Br1• (1)
        self.assertEqual(traj.states[1].atom_meta["C1"].radicals, 1)
        self.assertEqual(traj.states[1].atom_meta["Br1"].radicals, 1)
        # State 2: C1 paired (0), C3• (1), Br1• (1)
        self.assertEqual(traj.states[2].atom_meta["C1"].radicals, 0)
        self.assertEqual(traj.states[2].atom_meta["C3"].radicals, 1)
        # Final state: all closed-shell (0)
        self.assertEqual(sum(m.radicals for m in traj.final_state.atom_meta.values()), 0)

    def test_quantum_yield_and_cismesia_yoon_light_on_off(self) -> None:
        qy = ExperimentalProbe(experiment_type="quantum_yield")
        self.assertEqual(self.lab.predict(self.h_chain, qy).outcome_bin, "above_1")
        self.assertAlmostEqual(self.lab.predict(self.h_chain, qy).quantitative_value, 18.5)
        self.assertEqual(self.lab.predict(self.h_closed_pr, qy).outcome_bin, "below_1")

        # Both photo-driven mechanisms stop in the dark (Cismesia-Yoon principle!), whereas thermal continues
        loo = ExperimentalProbe(experiment_type="light_on_off")
        self.assertEqual(self.lab.predict(self.h_chain, loo).outcome_bin, "stops_in_dark")
        self.assertEqual(self.lab.predict(self.h_closed_pr, loo).outcome_bin, "stops_in_dark")
        self.assertEqual(self.lab.predict(self.h_polar, loo).outcome_bin, "continues_in_dark")

    def test_stern_volmer_rehm_weller_identifies_true_quencher(self) -> None:
        # *Ir(ppy)3 E_ox* = -1.73 V vs SCE reduces alkyl_bromide (E_red = -1.62 V, dG = -0.11 eV), NOT ethylene (-2.90 V)
        sv_rbr = ExperimentalProbe(experiment_type="stern_volmer", perturbation="Titrate alkyl_bromide with Ir(ppy)3")
        sv_eth = ExperimentalProbe(experiment_type="stern_volmer", perturbation="Titrate ethylene with Ir(ppy)3")
        p_rbr = self.lab.predict(self.h_chain, sv_rbr)
        p_eth = self.lab.predict(self.h_chain, sv_eth)
        self.assertEqual(p_rbr.outcome_bin, "quenches")
        self.assertEqual(p_eth.outcome_bin, "no_quenching")

    def test_tempo_radical_trap_computes_exact_adduct_hrms_mz_and_boosts_bayes(self) -> None:
        trap = ExperimentalProbe(
            experiment_type="radical_trap",
            perturbation="Add TEMPO (2 equiv)",
            observable="HRMS detection of TEMPO adduct",
            parameters={"reagent": "TEMPO", "observable_mode": "adduct"},
        )
        p_rad = self.lab.predict(self.h_chain, trap)
        p_pol = self.lab.predict(self.h_polar, trap)
        self.assertEqual(p_rad.outcome_bin, "adduct_detected")
        self.assertEqual(p_pol.outcome_bin, "adduct_not_detected")

        # Methyl radical CH3• (15.02348 Da) + TEMPO (C9H18NO = 156.13884 Da) -> C10H21NO (exact mass 171.16231 Da, [M+H]+ = 172.16959)
        adducts = p_rad.structural_details["trapped_adducts"]
        formulas = {a["adduct_formula"]: a["adduct_hrms_mz"]["[M+H]+"] for a in adducts}
        self.assertIn("C10H21NO", formulas)
        self.assertAlmostEqual(formulas["C10H21NO"], 172.16959, places=3)

        # Verify Bayesian HRMS structural likelihood bonus vs penalty
        eng = AbductiveEngine(self.lab)
        lik_exact = eng.observation_likelihood(p_rad, "adduct_detected", {"reported_mz": [172.1696]})
        lik_wrong_mz = eng.observation_likelihood(p_rad, "adduct_detected", {"reported_mz": [310.1120]})
        self.assertGreater(lik_exact, 0.90)
        self.assertLess(lik_wrong_mz, 0.30)

    def test_radical_clock_standard_vs_hypersensitive_against_metal_rebound(self) -> None:
        # Standard cyclopropylcarbinyl clock: free radical opens, closed-shell polar stays intact
        std_clock = ExperimentalProbe(experiment_type="radical_clock", parameters={"clock_type": "cyclopropylcarbinyl"})
        self.assertEqual(self.lab.predict(self.h_chain, std_clock).outcome_bin, "ring_opened")
        self.assertEqual(self.lab.predict(self.h_polar, std_clock).outcome_bin, "not_opened")

        # Ultrafast inner-sphere cage rebound (k_reb = 5e10 s^-1): outruns standard cyclopropylcarbinyl (1e8 s^-1),
        # BUT is caught by Newcomb's hypersensitive trans-2-phenylcyclopropylcarbinyl clock (3e11 s^-1)!
        fast_rebound_hyp = MechanismHypothesis(
            hypothesis_id="H_fast_rebound",
            name="Ultrafast Metal Oxygen Rebound",
            initial_state=self.init_state,
            steps=[
                MechanisticStep(step_class="hat", pgl_action=self.h_chain.steps[0].pgl_action, rebound_rate_s=5.0e10),
                MechanisticStep(
                    step_class="radical_coupling",
                    pgl_action=sc.ReactionTransformation(
                        "reaction",
                        [
                            sc.ElectronTransfer(("C1", "radical"), ("C1", "Br1", "new_sigma_bond"), 1, "bond_forming"),
                            sc.ElectronTransfer(("Br1", "radical"), ("C1", "Br1", "new_sigma_bond"), 1, "bond_forming"),
                        ],
                    ),
                ),
            ],
        )
        hyper_clock = ExperimentalProbe(
            experiment_type="radical_clock",
            parameters={"clock_type": "trans_2_phenylcyclopropylcarbinyl"},
        )
        self.assertEqual(self.lab.predict(fast_rebound_hyp, std_clock).outcome_bin, "not_opened")
        self.assertEqual(self.lab.predict(fast_rebound_hyp, hyper_clock).outcome_bin, "ring_opened")


class TestTransitionMetalOrganometallicCycle(unittest.TestCase):
    """Test Pd-catalyzed C-H arylation: 3-protocol KIE, inverse secondary KIE, kinetic orders, Kagan NLE, competence & spectroscopy."""

    def setUp(self) -> None:
        self.lab = VirtualLab()
        # Model Pd0 + C_ar1-Br1 + C_ar2-H1 + O_base(-) -> C_ar1-C_ar2 + Pd0 + H1-O_base + Br1(-)
        pgl0 = sc.ReactionState(
            species=[
                _species("catalyst", "pd", {"Pd1": sc.Atom("Pd")}, []),
                _species("reactant", "arx", {"C_ar1": sc.Atom("C", Hcount=3), "Br1": sc.Atom("Br")}, [sc.Bond("C_ar1", "Br1", "single")]),
                _species("reactant", "arh", {"C_ar2": sc.Atom("C", Hcount=3), "H1": sc.Atom("H")}, [sc.Bond("C_ar2", "H1", "single")]),
                _species("reactant", "base", {"O_base": sc.Atom("O", Hcount=1, charge=-1)}, []),
            ]
        )
        self.init_state = LabState.from_pgl(pgl0)

        step_oa = MechanisticStep(
            step_class="oxidative_addition",
            name="Pd_OA",
            bonds_formed=[("Pd1", "C_ar1", 1.0), ("Pd1", "Br1", 1.0)],
            bonds_broken=[("C_ar1", "Br1")],
            oxidation_state_changes={"Pd1": +2},
            reversible=False,
        )
        step_cmd = MechanisticStep(
            step_class="cmd_amla",
            name="Pd_CMD",
            bonds_formed=[("Pd1", "C_ar2", 1.0), ("O_base", "H1", 1.0)],
            bonds_broken=[("C_ar2", "H1"), ("Pd1", "Br1")],
            charge_changes={"O_base": +1, "Br1": -1},
            reversible=False,
        )
        step_re = MechanisticStep(
            step_class="reductive_elimination",
            name="Pd_RE",
            bonds_formed=[("C_ar1", "C_ar2", 1.0)],
            bonds_broken=[("Pd1", "C_ar1"), ("Pd1", "C_ar2")],
            oxidation_state_changes={"Pd1": -2},
            reversible=False,
        )

        # Hypothesis A: OA is irreversible RDS (tdts=0); CMD C-H cleavage occurs AFTER RDS (step 1)
        # Off-cycle dimeric [Pd2] resting reservoir + PPh3 dissociation before OA + heterochiral meso reservoir (+NLE)
        self.h_oa_rds = MechanismHypothesis(
            hypothesis_id="H_OA_RDS",
            name="OA-Rate-Limiting Dimeric Pd Cycle",
            initial_state=self.init_state,
            steps=[step_oa, step_cmd, step_re],
            catalytic_spec=CatalyticSpec(
                is_catalytic=True,
                tdi_index=0,
                tdts_step_index=0,
                catalyst_resting_nuclearity=2,
                active_catalyst_nuclearity=1,
                dissociated_ligands_before_tdts=["PPh3"],
                chiral_ligands_in_active=1,
                chiral_ligands_in_reservoir=2,
                heterochiral_reservoir_K=12.0,
                meso_reservoir_inactive=True,
                components={
                    "Pd_cat": {"atoms": ["Pd1"], "role": "catalyst", "essential": True},
                    "ArBr": {"atoms": ["C_ar1", "Br1"], "role": "substrate", "essential": True},
                    "Arene_H": {"atoms": ["C_ar2", "H1"], "role": "substrate", "essential": True},
                    "Base": {"atoms": ["O_base"], "role": "base", "essential": True},
                    "PPh3": {"atoms": [], "role": "ligand", "essential": False, "background_yield_fraction": 0.40},
                },
            ),
        )

        # Hypothesis B: Reversible OA (step 0) followed by rate-limiting CMD C-H cleavage (tdts=1); monomeric 1:1 ML catalyst
        step_oa_rev = MechanisticStep(
            step_class="oxidative_addition",
            name="Pd_OA_rev",
            bonds_formed=step_oa.bonds_formed,
            bonds_broken=step_oa.bonds_broken,
            oxidation_state_changes=step_oa.oxidation_state_changes,
            reversible=True,
        )
        self.h_cmd_rds = MechanismHypothesis(
            hypothesis_id="H_CMD_RDS",
            name="CMD-Rate-Limiting Monomeric Pd Cycle",
            initial_state=self.init_state,
            steps=[step_oa_rev, step_cmd, step_re],
            catalytic_spec=CatalyticSpec(
                is_catalytic=True,
                tdi_index=1,  # Resting state is the oxidative-addition complex Pd(Ar)(Br)!
                tdts_step_index=1,
                catalyst_resting_nuclearity=1,
                active_catalyst_nuclearity=1,
                chiral_ligands_in_active=1,
                chiral_ligands_in_reservoir=1,
                components={
                    "Pd_cat": {"atoms": ["Pd1"], "role": "catalyst", "essential": True},
                    "ArBr": {"atoms": ["C_ar1", "Br1"], "role": "substrate", "essential": True},
                    "Arene_H": {"atoms": ["C_ar2", "H1"], "role": "substrate", "essential": True},
                    "Base": {"atoms": ["O_base"], "role": "base", "essential": True},
                },
            ),
        )

    def test_three_kie_protocols_distinguish_all_simmons_hartwig_regimes(self) -> None:
        # Simmons-Hartwig (Angew. Chem. Int. Ed. 2012, 51, 3066) Case II:
        # Under H_OA_RDS, Ar-Br OA is TDTS (step 1) and Arene-H enters irreversibly AFTER RDS (step 2):
        # -> parallel = none (~1.05), intermolecular competition = primary (4.5), intramolecular = primary (4.5)!
        for proto, expected_bin in (
            ("parallel", "none (0.9-1.3)"),
            ("intermolecular_competition", "primary (>2.5)"),
            ("intramolecular_competition", "primary (>2.5)"),
        ):
            pred = self.lab.predict(
                self.h_oa_rds,
                ExperimentalProbe("kie", parameters={"protocol": proto, "labeled_atoms": ["C_ar2"]}),
            )
            self.assertEqual(pred.outcome_bin, expected_bin, f"Case II failed for {proto}: {pred}")

        # Simmons-Hartwig Case I (H_CMD_RDS: C-H cleavage IS TDTS):
        # -> ALL THREE protocols give primary (>2.5)!
        for proto in ("parallel", "intermolecular_competition", "intramolecular_competition"):
            pred = self.lab.predict(
                self.h_cmd_rds,
                ExperimentalProbe("kie", parameters={"protocol": proto, "labeled_atoms": ["C_ar2"]}),
            )
            self.assertEqual(pred.outcome_bin, "primary (>2.5)", f"Case I failed for {proto}: {pred}")

        # Simmons-Hartwig Case III: Irreversible directing-group / pi-binding of Arene-H to Pd (step 1, TDTS)
        # precedes C-H cleavage (step 2) -> parallel = none, intermolecular = none, intramolecular = primary!
        h_irrev_bind = MechanismHypothesis(
            hypothesis_id="H_Irrev_Bind",
            name="Irreversible Substrate Coordination Before C-H Cleavage",
            initial_state=self.init_state,
            steps=[
                MechanisticStep(
                    step_class="ligand_association",
                    bonds_formed=[("Pd1", "C_ar2", 1.0)],
                    reversible=False,
                    is_rds=True,
                ),
                MechanisticStep(
                    step_class="cmd_amla",
                    bonds_formed=[("O_base", "H1", 1.0)],
                    bonds_broken=[("C_ar2", "H1")],
                    charge_changes={"O_base": +1, "C_ar2": -1},
                    reversible=False,
                ),
            ],
        )
        for proto, expected_bin in (
            ("parallel", "none (0.9-1.3)"),
            ("intermolecular_competition", "none (0.9-1.3)"),
            ("intramolecular_competition", "primary (>2.5)"),
        ):
            pred = self.lab.predict(
                h_irrev_bind,
                ExperimentalProbe("kie", parameters={"protocol": proto, "labeled_atoms": ["C_ar2"]}),
            )
            self.assertEqual(pred.outcome_bin, expected_bin, f"Case III failed for {proto}: {pred}")

    def test_inverse_secondary_kie_on_migratory_insertion_sp2_to_sp3(self) -> None:
        # Pd-H migratory insertion across C1=C2 (D-labeled at C1, without C-D bond cleavage!)
        st = LabState.from_pgl(
            sc.ReactionState(
                species=[
                    _species("catalyst", "pdh", {"Pd1": sc.Atom("Pd"), "H_pd": sc.Atom("H")}, [sc.Bond("Pd1", "H_pd", "single")]),
                    _species("reactant", "alk", {"C1": sc.Atom("C", Hcount=1), "H_alk": sc.Atom("H"), "C2": sc.Atom("C", Hcount=2)}, [sc.Bond("C1", "C2", "double"), sc.Bond("C1", "H_alk", "single")]),
                ]
            )
        )
        hyp_mi = MechanismHypothesis(
            hypothesis_id="H_MI",
            name="Alkene Migratory Insertion",
            initial_state=st,
            steps=[
                MechanisticStep(
                    step_class="migratory_insertion",
                    bonds_formed=[("Pd1", "C1", 1.0), ("C2", "H_pd", 1.0)],
                    bonds_broken=[("Pd1", "H_pd")],
                    bond_order_changes=[("C1", "C2", 1.0)],
                    is_rds=True,
                )
            ],
        )
        pred = self.lab.predict(
            hyp_mi,
            ExperimentalProbe("kie", parameters={"protocol": "parallel", "labeled_atoms": ["H_alk"]}),
        )
        self.assertEqual(pred.outcome_bin, "inverse (<0.9)")
        self.assertLess(pred.quantitative_value, 0.90)

    def test_kinetic_orders_nle_competence_and_spectroscopy(self) -> None:
        # Kinetic orders on H_OA_RDS (TDI=0, TDTS=0):
        # PPh3 -> inverse (-1); Pd_cat -> fractional (0.5); ArBr -> 1; Arene_H (post-RDS!) -> 0
        self.assertEqual(
            self.lab.predict(self.h_oa_rds, ExperimentalProbe("kinetic_order", parameters={"component": "PPh3"})).outcome_bin,
            "PPh3:inverse",
        )
        self.assertEqual(
            self.lab.predict(self.h_oa_rds, ExperimentalProbe("kinetic_order", parameters={"component": "Pd_cat"})).outcome_bin,
            "Pd_cat:fractional",
        )
        self.assertEqual(
            self.lab.predict(self.h_oa_rds, ExperimentalProbe("kinetic_order", parameters={"component": "ArBr"})).outcome_bin,
            "ArBr:1",
        )
        self.assertEqual(
            self.lab.predict(self.h_oa_rds, ExperimentalProbe("kinetic_order", parameters={"component": "Arene_H"})).outcome_bin,
            "Arene_H:0",
        )

        # Nonlinear effect: +NLE for heterochiral meso dimer reservoir vs linear for monomeric 1:1
        self.assertEqual(self.lab.predict(self.h_oa_rds, ExperimentalProbe("nonlinear_effect")).outcome_bin, "positive_NLE")
        self.assertEqual(self.lab.predict(self.h_cmd_rds, ExperimentalProbe("nonlinear_effect")).outcome_bin, "linear")

        # Stoichiometric intermediate competence: Pd(C_ar1)(Br1) complex is on-path -> competent
        comp_probe = ExperimentalProbe(
            "stoichiometric_intermediate",
            parameters={"characteristic_bonds": [("Pd1", "C_ar1"), ("Pd1", "Br1")]},
        )
        self.assertEqual(self.lab.predict(self.h_cmd_rds, comp_probe).outcome_bin, "competent (gives product)")

        # In-situ 31P/1H NMR spectroscopy: Pd(Ar)(Br) is TDI in H_CMD_RDS (observed),
        # whereas in H_OA_RDS it forms AFTER TDTS (step 0) and is consumed rapidly (not_observed)!
        spec_probe = ExperimentalProbe(
            "spectroscopy",
            observable="31P NMR resting state",
            parameters={"modality": "NMR", "characteristic_bonds": [("Pd1", "C_ar1"), ("Pd1", "Br1")]},
        )
        self.assertEqual(self.lab.predict(self.h_cmd_rds, spec_probe).outcome_bin, "observed")
        self.assertEqual(self.lab.predict(self.h_oa_rds, spec_probe).outcome_bin, "not_observed")


class TestElectrosynthesisAndProticDeuteration(unittest.TestCase):
    def test_anodic_cathodic_and_paired_electrolysis(self) -> None:
        lab = VirtualLab()
        # Paired electrolysis: anodic oxidation of R-CO2- (-1e-) + cathodic reduction of Ar-CN (+1e-)
        st = LabState.from_pgl(
            sc.ReactionState(
                species=[
                    _species("reactant", "sub1", {"O1": sc.Atom("O", charge=-1), "C1": sc.Atom("C", Hcount=3)}, [sc.Bond("O1", "C1", "single")]),
                    _species("reactant", "sub2", {"C2": sc.Atom("C", Hcount=4)}, []),
                ]
            )
        )
        hyp = MechanismHypothesis(
            hypothesis_id="H_paired",
            name="Paired Electrolysis",
            initial_state=st,
            steps=[
                MechanisticStep(
                    step_class="anodic_oxidation",
                    charge_changes={"O1": +1},
                    radical_changes={"O1": +1},
                    electrode_electrons=-1,
                ),
                MechanisticStep(
                    step_class="cathodic_reduction",
                    charge_changes={"C2": -1},
                    radical_changes={"O1": -1},  # paired spin closure
                    electrode_electrons=+1,
                ),
            ],
        )
        traj = execute_hypothesis(hyp)
        self.assertTrue(traj.valid, traj.error)
        el_pred = lab.predict(hyp, ExperimentalProbe("electrochemistry"))
        self.assertEqual(el_pred.outcome_bin, "paired_electrolysis")
        self.assertEqual(el_pred.structural_details["anodic_electrons"], 1)
        self.assertEqual(el_pred.structural_details["cathodic_electrons"], 1)

    def test_d2o_solvent_pool_exchanges_implicit_oh_and_deuterates_alpha_carbon(self) -> None:
        # Enol CH2=CH-OH (where OH is initially an IMPLICIT Hcount=1 on O1!) tautomerizes in D2O -> D-CH2-CH=O!
        enol = LabState.from_pgl(
            sc.ReactionState(
                species=[
                    _species(
                        "reactant",
                        "enol",
                        {
                            "C_alpha": sc.Atom("C", Hcount=2),
                            "C_carb": sc.Atom("C", Hcount=1),
                            "O1": sc.Atom("O", Hcount=1),  # implicit exchangeable O-H!
                        },
                        [sc.Bond("C_alpha", "C_carb", "double"), sc.Bond("C_carb", "O1", "single")],
                    )
                ]
            )
        )
        # Note: materialize_implicit_hydrogens promotes O1's implicit H into H_imp_1 automatically,
        # or if we write explicit H_oh on O1 for tautomerism curly arrows:
        enol_exp = LabState.from_pgl(
            sc.ReactionState(
                species=[
                    _species(
                        "reactant",
                        "enol",
                        {
                            "C_alpha": sc.Atom("C", Hcount=2),
                            "C_carb": sc.Atom("C", Hcount=1),
                            "O1": sc.Atom("O", Hcount=0),
                            "H_oh": sc.Atom("H"),
                        },
                        [
                            sc.Bond("C_alpha", "C_carb", "double"),
                            sc.Bond("C_carb", "O1", "single"),
                            sc.Bond("O1", "H_oh", "single"),
                        ],
                    )
                ]
            )
        )
        taut_hyp = MechanismHypothesis(
            hypothesis_id="H_enol_keto",
            name="Enol-Keto Tautomerization in D2O",
            initial_state=enol_exp,
            steps=[
                MechanisticStep(
                    step_class="proton_transfer",
                    pgl_action=sc.ReactionTransformation(
                        "reaction",
                        [
                            sc.ElectronTransfer(("O1", "H_oh", "sigma_bond"), ("C_carb", "O1", "new_pi_bond"), 2, "bond_breaking_or_pi_shift"),
                            sc.ElectronTransfer(("C_alpha", "C_carb", "pi_bond"), ("C_alpha", "H_oh", "new_sigma_bond"), 2, "bond_forming"),
                        ],
                    ),
                )
            ],
        )
        lab = VirtualLab()
        d2o_probe = ExperimentalProbe(
            "isotope_label",
            perturbation="Run in D2O",
            parameters={"solvent_isotope": "D"},
        )
        pred = lab.predict(taut_hyp, d2o_probe)
        self.assertEqual(pred.outcome_bin, "C_alpha:major (>=50%)")
        self.assertEqual(pred.structural_details["site_incorporation"], {"C_alpha": 1.0})

        # Also verify implicit O-H on ethanol exchanges cleanly in D2O without arrows
        noop_hyp = MechanismHypothesis(hypothesis_id="H_oh_ex", name="Alcohol Exchange", initial_state=enol, steps=[])
        pred_oh = lab.predict(noop_hyp, d2o_probe)
        self.assertEqual(pred_oh.structural_details["site_incorporation"], {"O1": 1.0})


class TestRealLiteratureAndPGLIntegration(unittest.TestCase):
    """End-to-end validation on real 2026 PMC13039371 evidence cards, P2/P3/P4 abduction, and real core/ PGL paths."""

    def test_pmc13039371_fe_innocent_abductive_discovery_flip_and_eig_design(self) -> None:
        json_path = MECH_AGENT_DIR / "outputs" / "evidence_p0_20261101" / "gemini-3.5-flash-lite" / "PMC13039371.json"
        data = json.loads(json_path.read_text(encoding="utf-8"))
        cards_by_id = {c["id"]: c for c in data["cards"]}

        # Model the 3 competing mechanisms of PMC13039371 (BCP boronate synthesis from R-CO2H + B2cat2 + Fe + Cs2CO3 + DMSO + hv):
        # Reactants in PGL: R(C1)-CO2(C2,O1,O2)-H1 + B1-B2 + Fe1 (spectator or active!)
        pgl0 = sc.ReactionState(
            species=[
                _species(
                    "reactant",
                    "acid",
                    {"C1": sc.Atom("C", Hcount=3), "C2": sc.Atom("C"), "O1": sc.Atom("O"), "O2": sc.Atom("O", Hcount=1)},
                    [sc.Bond("C1", "C2", "single"), sc.Bond("C2", "O1", "double"), sc.Bond("C2", "O2", "single")],
                ),
                _species("reactant", "diboron", {"B1": sc.Atom("B", Hcount=2), "B2": sc.Atom("B", Hcount=2)}, [sc.Bond("B1", "B2", "single")]),
                _species("catalyst", "iron", {"Fe1": sc.Atom("Fe")}, []),
            ]
        )
        init_st = LabState.from_pgl(pgl0)

        # H1: Author's initial working hypothesis — Iron-catalyzed LMCT decarboxylative borylation (Fe1 is essential!)
        h1_fe = MechanismHypothesis(
            hypothesis_id="H1_Fe_LMCT",
            name="Fe-Catalyzed LMCT Decarboxylation",
            initial_state=init_st,
            steps=[
                MechanisticStep(
                    step_class="lmct_homolysis",
                    photons=1,
                    bonds_formed=[("Fe1", "O2", 1.0)],
                    bonds_broken=[("C1", "C2")],
                    radical_changes={"C1": +1},
                ),
                MechanisticStep(
                    step_class="radical_borylation",
                    bonds_formed=[("C1", "B1", 1.0), ("C2", "B2", 1.0)],
                    bonds_broken=[("B1", "B2"), ("Fe1", "O2")],
                    radical_changes={"C1": -1},
                ),
            ],
            catalytic_spec=CatalyticSpec(
                is_catalytic=True,
                components={
                    "iron": {"atoms": ["Fe1"], "role": "catalyst", "essential": True},
                    "Cs2CO3": {"atoms": [], "role": "base", "essential": False, "background_yield_fraction": 0.58},
                    "DMSO": {"atoms": [], "role": "solvent", "essential": False, "background_yield_fraction": 0.31},
                },
            ),
        )

        # H2: True surprising mechanism — Iron-innocent DMSO/B2cat2 EDA photoinduced radical decarboxylation!
        # Notice Fe1 is present in init_st, NEVER touched in any step -> VirtualLab automatically detects Fe1 as innocent spectator!
        h2_eda = MechanismHypothesis(
            hypothesis_id="H2_Fe_Innocent_EDA",
            name="Fe-Innocent Photoinduced B2cat2-DMSO EDA Radical Pathway",
            initial_state=init_st,
            steps=[
                MechanisticStep(
                    step_class="homolysis",
                    photons=1,
                    bonds_broken=[("C1", "C2")],
                    radical_changes={"C1": +1, "C2": +1},
                ),
                MechanisticStep(
                    step_class="radical_borylation",
                    is_chain_propagation=True,
                    bonds_formed=[("C1", "B1", 1.0), ("C2", "B2", 1.0)],
                    bonds_broken=[("B1", "B2")],
                    radical_changes={"C1": -1, "C2": -1},
                ),
            ],
            catalytic_spec=CatalyticSpec(
                chain_length_estimate=6.0,
                components={
                    "iron": {"atoms": ["Fe1"], "role": "catalyst", "essential": False, "background_yield_fraction": 0.85},
                    "Cs2CO3": {"atoms": [], "role": "base", "essential": False, "background_yield_fraction": 0.58},
                    "DMSO": {"atoms": [], "role": "solvent", "essential": False, "background_yield_fraction": 0.31},
                },
            ),
        )

        # H3: Dark closed-shell polar boronation (neither light nor radical intermediates)
        h3_dark_polar = MechanismHypothesis(
            hypothesis_id="H3_Dark_Polar",
            name="Thermal Closed-Shell Polar Decarboxylation",
            initial_state=init_st,
            steps=[
                MechanisticStep(
                    step_class="polar",
                    bonds_formed=[("C1", "B1", 1.0), ("C2", "B2", 1.0)],
                    bonds_broken=[("C1", "C2"), ("B1", "B2")],
                )
            ],
            catalytic_spec=CatalyticSpec(
                components={
                    "iron": {"atoms": ["Fe1"], "role": "catalyst", "essential": False, "background_yield_fraction": 0.85},
                    "Cs2CO3": {"atoms": [], "role": "base", "essential": False, "background_yield_fraction": 0.58},
                    "DMSO": {"atoms": [], "role": "solvent", "essential": False, "background_yield_fraction": 0.31},
                },
            ),
        )

        hypotheses = [h1_fe, h2_eda, h3_dark_polar]
        selected_ids = ["E01", "E02", "E03", "E05", "E06", "E07", "E10", "E11"]
        observations = [probe_from_evidence_card(cards_by_id[cid]) for cid in selected_ids]

        engine = AbductiveEngine()
        result = engine.compute_posteriors(hypotheses, observations)

        # 1. Sequential Bayes identifies H2_Fe_Innocent_EDA with >93% posterior (exact 14.7:1 odds over H1, >40000:1 over H3)!
        self.assertEqual(result["map_hypothesis"], "H2_Fe_Innocent_EDA")
        self.assertGreater(result["final_posteriors"]["H2_Fe_Innocent_EDA"], 0.93)
        self.assertLess(result["final_posteriors"]["H1_Fe_LMCT"], 0.07)
        self.assertLess(result["final_posteriors"]["H3_Dark_Polar"], 0.001)

        # 2. P2 Leave-One-Out: holding out E06 (radical_clock), visible evidence already locks onto H2 -> predicts 'ring_opened'!
        vis = [obs for obs in observations if obs[0].probe_id != "E06"]
        e06_probe = [obs[0] for obs in observations if obs[0].probe_id == "E06"][0]
        held_out = engine.predict_held_out(hypotheses, vis, e06_probe)
        self.assertEqual(held_out["predicted_bin"], "ring_opened")
        self.assertGreater(held_out["outcome_distribution"]["ring_opened"], 0.95)

        # 3. P3 Counterfactual Flip Test: if E02 ("No iron and no Cs2CO3") and E11 ("Omission of Fe(NO3)3")
        #    on the core 5-card subset had been 'suppressed' instead of 'unaffected', MAP switches from H2 to H1_Fe_LMCT!
        core5 = [probe_from_evidence_card(cards_by_id[cid]) for cid in ("E01", "E02", "E03", "E05", "E06")]
        flip = engine.counterfactual_flip_test(hypotheses, core5, 1, "suppressed")
        self.assertEqual(flip["original_map"], "H2_Fe_Innocent_EDA")
        self.assertEqual(flip["counterfactual_map"], "H1_Fe_LMCT")
        self.assertTrue(flip["map_switched"])

        # 4. P4 Optimal Experiment Design (Shannon EIG): starting from uniform prior over H1 vs H2 vs H3,
        #    rank_experiments automatically surfaces high-information probes (control_no_iron, radical_trap, quantum_yield, light_on_off)!
        designer = ExperimentDesigner(engine)
        ranked = designer.rank_experiments(hypotheses, prior_observations=(), top_k=8)
        self.assertTrue(len(ranked) >= 5)
        self.assertGreater(ranked[0].expected_info_gain_bits, 0.45)
        top_ids = {c.probe.probe_id for c in ranked[:6]}
        self.assertTrue(
            bool(top_ids & {"control_no_iron", "photochem_quantum_yield", "photochem_light_on_off", "trap_TEMPO_adduct", "clock_cyclopropyl"}),
            f"Unexpected top EIG probes: {top_ids}",
        )

    def test_real_core_named_reactions_pgl_replay(self) -> None:
        core_jsonl = REPO_ROOT / "core" / "named_reactions_val_pgl_paths.core.jsonl"
        rows = [json.loads(line) for line in core_jsonl.read_text(encoding="utf-8").splitlines()[:5] if line.strip()]
        lab = VirtualLab()
        replayed = 0
        for row in rows:
            rpath = sc.parse_pgl_literal(row["ReactionPath"])
            states = [x for x in rpath.sequence if isinstance(x, sc.ReactionState)]
            # Only replay atom-balanced entries (some raw entries drop catalytic In1 on ts_collapse)
            a0 = set(dict(sc.pgl.iter_atoms(states[0])).keys())
            if not all(set(dict(sc.pgl.iter_atoms(s)).keys()) == a0 for s in states):
                continue
            hyp = MechanismHypothesis.from_pgl_path(
                rpath,
                hypothesis_id=row["mechanism_id"],
                name=row["mechanism_name"],
            )
            traj = lab.trajectory_of(hyp)
            self.assertTrue(traj.valid, f"{row['mechanism_id']} failed: {traj.error}")
            # Run multiple probes across real curated named reactions (including generic Atom('R') templates!)
            trap_pred = lab.predict(hyp, ExperimentalProbe("radical_trap"))
            self.assertIn("unaffected", trap_pred.outcome_bin)
            cross_pred = lab.predict(hyp, ExperimentalProbe("crossover"))
            self.assertIn(cross_pred.outcome_bin, {"crossover", "no_crossover"})
            replayed += 1

        self.assertGreaterEqual(replayed, 3)

    def test_all_extracted_literature_evidence_cards_execute_without_error(self) -> None:
        out_dir = MECH_AGENT_DIR / "outputs" / "evidence_p0_20261101" / "gemini-3.5-flash-lite"
        if not out_dir.exists():
            self.skipTest("evidence_p0_20261101 output directory not present")
        core_jsonl = REPO_ROOT / "core" / "named_reactions_val_pgl_paths.core.jsonl"
        rows = [json.loads(line) for line in core_jsonl.read_text(encoding="utf-8").splitlines()[:3] if line.strip()]
        hyp = MechanismHypothesis.from_pgl_path(
            rows[1]["ReactionPath"],
            hypothesis_id="H_chugaev_smoke",
            name="Chugaev Elimination",
        )
        lab = VirtualLab()
        total_cards = 0
        for fp in sorted(out_dir.glob("*.json")):
            doc = json.loads(fp.read_text(encoding="utf-8"))
            for card in doc.get("cards", []):
                probe, obs_bin, extra = probe_from_evidence_card(card)
                pred = lab.predict(hyp, probe)
                self.assertEqual(pred.experiment_type, probe.experiment_type)
                self.assertIsInstance(obs_bin, str)
                self.assertIn("reported_mz", extra)
                total_cards += 1
        self.assertGreaterEqual(total_cards, 500)


if __name__ == "__main__":
    unittest.main()
