"""Physical-organic reference constants, isotopic masses, redox scales, radical clocks, and traps.

Pure Python stdlib; zero external dependencies required at runtime.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

# ---------------------------------------------------------------------------
# Exact IUPAC Monoisotopic Masses (in Daltons, u) of Most Abundant Isotopes
# ---------------------------------------------------------------------------
MONOISOTOPIC_MASS: Dict[str, float] = {
    "H": 1.0078250322,
    "D": 2.0141017781,
    "T": 3.0160492819,
    "He": 4.0026032541,
    "Li": 7.0160034366,
    "Be": 9.012183065,
    "B": 11.00930536,
    "C": 12.0000000000,
    "N": 14.0030740044,
    "O": 15.9949146196,
    "F": 18.9984031627,
    "Na": 22.9897692820,
    "Mg": 23.985041697,
    "Al": 26.98153853,
    "Si": 27.9769265346,
    "P": 30.9737619984,
    "S": 31.9720711744,
    "Cl": 34.968852682,
    "K": 38.9637064864,
    "Ca": 39.962590863,
    "Sc": 44.95590828,
    "Ti": 47.94794198,
    "V": 50.94395704,
    "Cr": 51.94050623,
    "Mn": 54.93804391,
    "Fe": 55.93493633,
    "Co": 58.93319429,
    "Ni": 57.93534241,
    "Cu": 62.92959772,
    "Zn": 63.92914201,
    "Ga": 68.9255735,
    "Ge": 73.921177761,
    "As": 74.92159457,
    "Se": 79.9165218,
    "Br": 78.9183376,
    "Rb": 84.9117897379,
    "Sr": 87.9056125,
    "Y": 88.9058403,
    "Zr": 89.9046977,
    "Mo": 97.90540482,
    "Ru": 101.9043441,
    "Rh": 102.9054980,
    "Pd": 105.9034804,
    "Ag": 106.9050916,
    "Cd": 113.90336509,
    "In": 114.903878776,
    "Sn": 119.90220163,
    "Sb": 120.903812,
    "Te": 129.906222748,
    "I": 126.9044719,
    "Cs": 132.905451961,
    "Ba": 137.9052470,
    "La": 138.9063563,
    "Ce": 139.9054431,
    "Yb": 173.9388664,
    "Hf": 179.9465570,
    "Ta": 180.9479958,
    "W": 183.95093092,
    "Re": 186.9557501,
    "Os": 191.9614770,
    "Ir": 192.9629216,
    "Pt": 194.9647917,
    "Au": 196.96656879,
    "Hg": 201.9706434,
    "Tl": 204.9744278,
    "Pb": 207.9766525,
    "Bi": 208.9803987,
}

# Nominal substituent masses for generic PGL template pseudoatoms (e.g. Atom('R'), Atom('Ar'))
PSEUDOATOM_NOMINAL_MASS: Dict[str, float] = {
    "R": 15.02347510,   # -CH3 nominal placeholder
    "Ar": 77.03912516,  # -C6H5 nominal placeholder
    "Ph": 77.03912516,
    "Me": 15.02347510,
    "Et": 29.03912516,
    "X": 34.96885268,   # Generic halide
    "Nu": 17.00273965,
    "LG": 34.96885268,
    "EWG": 45.99290424,
    "EDG": 31.01838972,
    "L": 0.0,
    "M": 105.9034804,
}

# Exact isotopic masses keyed by canonical isotope tag (e.g. "2H", "13C", "18O")
ISOTOPE_EXACT_MASS: Dict[str, float] = {
    "1H": 1.0078250322,
    "2H": 2.0141017781,
    "D": 2.0141017781,
    "3H": 3.0160492819,
    "T": 3.0160492819,
    "6Li": 6.0151228874,
    "7Li": 7.0160034366,
    "10B": 10.01293695,
    "11B": 11.00930536,
    "12C": 12.0000000000,
    "13C": 13.0033548351,
    "14C": 14.0032419884,
    "14N": 14.0030740044,
    "15N": 15.0001088989,
    "16O": 15.9949146196,
    "17O": 16.9991317565,
    "18O": 17.9991596129,
    "19F": 18.9984031627,
    "28Si": 27.9769265346,
    "29Si": 28.9764946649,
    "31P": 30.9737619984,
    "32S": 31.9720711744,
    "33S": 32.9714589098,
    "34S": 33.967867004,
    "35Cl": 34.968852682,
    "37Cl": 36.965902602,
    "79Br": 78.9183376,
    "81Br": 80.9162897,
    "127I": 126.9044719,
}

ELECTRON_MASS: float = 0.0005485799

TRANSITION_METALS = frozenset({
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd",
    "La", "Ce", "Yb", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
})

MAIN_GROUP_METALS = frozenset({
    "Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba",
    "Al", "Ga", "In", "Tl", "Sn", "Pb", "Bi",
})

PROTIC_HETEROATOMS = frozenset({"O", "N", "S", "Se", "F", "Cl", "Br", "I"})

PAULING_ELECTRONEGATIVITY: Dict[str, float] = {
    "H": 2.20, "Li": 0.98, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
    "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16,
    "K": 0.82, "Fe": 1.83, "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65,
    "Se": 2.55, "Br": 2.96, "Rh": 2.28, "Pd": 2.20, "Ag": 1.93, "Sn": 1.96,
    "I": 2.66, "Cs": 0.79, "Ir": 2.20, "Pt": 2.28, "Au": 2.54, "Hg": 2.00,
}

# ---------------------------------------------------------------------------
# Photocatalyst Excited-State Redox Potentials (V vs SCE in MeCN) & E_T (kcal/mol)
# ---------------------------------------------------------------------------
PHOTOCATALYST_DATA: Dict[str, Dict[str, Any]] = {
    "Ir(ppy)3": {
        "E_ox_star": -1.73,   # E1/2(PC•+ / *PC): excited state as reductant
        "E_red_star": +0.31,  # E1/2(*PC / PC•-): excited state as oxidant
        "E_ox_ground": +0.77, # E1/2(PC•+ / PC)
        "E_red_ground": -2.19,# E1/2(PC / PC•-)
        "E_T": 55.2,          # Triplet energy (kcal/mol)
    },
    "[Ir(dF(CF3)ppy)2(dtbbpy)]+": {
        "E_ox_star": -0.89,
        "E_red_star": +1.21,
        "E_ox_ground": +1.69,
        "E_red_ground": -1.37,
        "E_T": 60.8,
    },
    "[Ir(ppy)2(dtbbpy)]+": {
        "E_ox_star": -0.96,
        "E_red_star": +0.66,
        "E_ox_ground": +1.21,
        "E_red_ground": -1.51,
        "E_T": 49.2,
    },
    "[Ru(bpy)3]2+": {
        "E_ox_star": -0.81,
        "E_red_star": +0.77,
        "E_ox_ground": +1.29,
        "E_red_ground": -1.33,
        "E_T": 46.5,
    },
    "4CzIPN": {
        "E_ox_star": -1.18,
        "E_red_star": +1.43,
        "E_ox_ground": +1.49,
        "E_red_ground": -1.24,
        "E_T": 58.3,
    },
    "Mes-Acr+": {
        "E_ox_star": -0.57,
        "E_red_star": +2.06,
        "E_ox_ground": +2.15,
        "E_red_ground": -0.57,
        "E_T": 44.7,
    },
    "Eosin_Y": {
        "E_ox_star": -1.11,
        "E_red_star": +0.83,
        "E_ox_ground": +0.78,
        "E_red_ground": -1.06,
        "E_T": 44.0,
    },
    "TBADT": {
        "E_ox_star": -0.20,
        "E_red_star": +2.44,
        "E_ox_ground": +2.50,
        "E_red_ground": -0.90,
        "E_T": 62.0,
        "hat_active": True,
    },
    "thioxanthone": {
        "E_ox_star": -1.10,
        "E_red_star": +1.18,
        "E_ox_ground": +1.70,
        "E_red_ground": -1.65,
        "E_T": 65.5,
    },
}

# Typical substrate / additive ground-state redox potentials (V vs SCE) & triplet E_T (kcal/mol)
SPECIES_REDOX_LIBRARY: Dict[str, Dict[str, float]] = {
    "tertiary_amine": {"E_ox": +0.78, "E_red": -3.00, "E_T": 75.0},   # Et3N, DIPEA
    "hantzsch_ester": {"E_ox": +0.89, "E_red": -2.30, "E_T": 58.0},
    "carboxylate": {"E_ox": +1.25, "E_red": -2.80, "E_T": 78.0},      # RCO2-
    "katritzy_salt": {"E_ox": +2.20, "E_red": -0.95, "E_T": 55.0},
    "nhpi_ester": {"E_ox": +2.30, "E_red": -1.26, "E_T": 57.0},       # Redox-active ester
    "aryl_iodide": {"E_ox": +2.10, "E_red": -1.90, "E_T": 64.0},
    "aryl_bromide": {"E_ox": +2.20, "E_red": -2.30, "E_T": 68.0},
    "alkyl_iodide": {"E_ox": +2.25, "E_red": -1.60, "E_T": 72.0},
    "thiolate": {"E_ox": +0.25, "E_red": -2.90, "E_T": 70.0},
    "styrene": {"E_ox": +1.90, "E_red": -2.45, "E_T": 61.7},
    "enone": {"E_ox": +2.35, "E_red": -1.85, "E_T": 54.0},
    "diaryliodonium": {"E_ox": +2.40, "E_red": -0.45, "E_T": 56.0},
    "sulfonyl_chloride": {"E_ox": +2.50, "E_red": -0.90, "E_T": 65.0},
    "oxygen_O2": {"E_ox": +2.80, "E_red": -0.82, "E_T": 22.5},        # 3O2 -> 1O2 (22.5 kcal/mol)
}

# ---------------------------------------------------------------------------
# Calibrated Unimolecular Radical Clock Rate Constants at 298 K (s^-1)
# References: Newcomb, Tetrahedron 1993, 49, 1151; Newcomb, Encyclopedia of Radicals 2012.
# ---------------------------------------------------------------------------
RADICAL_CLOCK_LIBRARY: Dict[str, Dict[str, Any]] = {
    "5_hexenyl": {
        "k_rearr_s": 2.3e5,
        "mode": "5-exo-trig_cyclization",
        "description": "5-hexen-1-yl -> cyclopentylmethyl radical",
    },
    "cyclopropylcarbinyl": {
        "k_rearr_s": 1.0e8,
        "mode": "beta_scission_ring_opening",
        "description": "cyclopropylmethyl -> but-3-en-1-yl radical",
    },
    "trans_2_phenylcyclopropylcarbinyl": {
        "k_rearr_s": 3.0e11,
        "mode": "hypersensitive_ring_opening",
        "description": "trans-2-phenylcyclopropylmethyl -> 4-phenylbut-3-en-1-yl radical (hypersensitive probe)",
    },
    "2_2_diphenylcyclopropylcarbinyl": {
        "k_rearr_s": 5.0e11,
        "mode": "hypersensitive_ring_opening",
        "description": "2,2-diphenylcyclopropylmethyl -> 4,4-diphenylbut-3-en-1-yl radical",
    },
    "oxiranylcarbinyl": {
        "k_rearr_s": 4.0e10,
        "mode": "epoxide_ring_opening",
        "description": "oxiranylmethyl -> allyloxy radical",
    },
    "bicyclo_1_1_1_pentyl": {
        "k_rearr_s": 1.5e2,
        "mode": "strain_ring_opening",
        "description": "bicyclo[1.1.1]pent-1-yl -> methylenecyclobutyl radical (slow clock)",
    },
}

# ---------------------------------------------------------------------------
# Standard Radical Trapping Agents (Formula & Exact Mass of Reactive Fragment)
# ---------------------------------------------------------------------------
RADICAL_TRAP_REAGENTS: Dict[str, Dict[str, Any]] = {
    "TEMPO": {
        "name": "2,2,6,6-tetramethylpiperidin-1-oxyl",
        "formula": {"C": 9, "H": 18, "N": 1, "O": 1},
        "coupling_mode": "radical_recombination",  # R• + •O-NC9H18 -> R-O-NC9H18
        "reactive_element": "O",
        "smiles_prefix": "CC1(C)CCCC(C)(C)N1O",
        "k_trap_M_s": 1.0e9,
    },
    "BHT": {
        "name": "2,6-di-tert-butyl-4-methylphenol",
        "formula": {"C": 15, "H": 23, "O": 1},     # BHT• fragment after HAT or dearomative coupling
        "coupling_mode": "hat_then_recombination", # R• + BHT-H -> R-H + BHT• (or R-BHT adduct C15H23O-R)
        "reactive_element": "C",
        "smiles_prefix": "CC1=CC(C(C)(C)C)=C(O)C(C(C)(C)C)=C1",
        "k_trap_M_s": 1.0e5,
    },
    "1,1-DPE": {
        "name": "1,1-diphenylethylene",
        "formula": {"C": 14, "H": 11},             # Heck-like / oxidative R-CH=CPh2 adduct (-H)
        "addition_formula": {"C": 14, "H": 12},    # Direct hydroalkylated R-CH2-CHPh2 adduct
        "coupling_mode": "alkene_addition_elimination",
        "reactive_element": "C",
        "smiles_prefix": "C(=C(c1ccccc1)c2ccccc2)",
        "k_trap_M_s": 3.0e6,
    },
    "DMPO": {
        "name": "5,5-dimethyl-1-pyrroline N-oxide",
        "formula": {"C": 6, "H": 11, "N": 1, "O": 1},
        "coupling_mode": "nitrone_spin_trap",      # R• + DMPO -> persistent nitroxide adduct (EPR + HRMS)
        "reactive_element": "C",
        "smiles_prefix": "CC1(C)CCC([N+]([O-])=1)",
        "k_trap_M_s": 2.0e7,
    },
}

# Exact adduct ion mass offsets (in Da) for electrospray / EI HRMS
HRMS_ION_OFFSETS: Dict[str, Tuple[float, int]] = {
    "[M]+.": (-ELECTRON_MASS, 1),
    "[M+H]+": (MONOISOTOPIC_MASS["H"] - ELECTRON_MASS, 1),
    "[M+Na]+": (MONOISOTOPIC_MASS["Na"] - ELECTRON_MASS, 1),
    "[M+K]+": (MONOISOTOPIC_MASS["K"] - ELECTRON_MASS, 1),
    "[M-H]-": (-MONOISOTOPIC_MASS["H"] + ELECTRON_MASS, 1),
}
