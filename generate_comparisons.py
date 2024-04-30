import os
import random
import pandas as pd
from pathlib import Path
from itertools import combinations
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import TanimotoSimilarity, TverskySimilarity, CosineSimilarity, DiceSimilarity, SokalSimilarity, \
    RusselSimilarity, KulczynskiSimilarity, McConnaugheySimilarity
BASE_DIR = Path(__file__).resolve().parent
# random.seed(10)

FRACTION = 0.05
ORIG_FILE = "data_files/ocelot_d3tales_CLEAN.pkl"
FP_FILE = "data_files/ocelot_d3tales_fps.pkl"
SIM_FILE = "data_files/combo_sims_{:02d}perc.csv".format(round(FRACTION*100))

# Electronic Props to compare
ELEC_PROPS = ['aie', 'aea', 'hl', 'lumo', 'homo', 's0s1']
# Fingerprint Generators (both bit and count forms)
FP_GENS = {

  "mfpReg": rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint,
  "mfpSCnt": rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, countSimulation=True).GetFingerprint,
  # "mfpCnt": rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetCountFingerprint,

  "rdkReg": rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048).GetFingerprint,
  "rdkSCnt": rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, countSimulation=True).GetFingerprint,
  # "rdkCnt": rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048).GetCountFingerprint,

  "aprReg": rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048).GetFingerprint,
  "aprSCnt": rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048, countSimulation=True).GetFingerprint,
  # "aprCnt": rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048).GetCountFingerprint,

  "ttrReg": rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048).GetFingerprint,
  "ttrSCnt": rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048, countSimulation=True).GetFingerprint,
  # "ttrCnt": rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048).GetCountFingerprint,
}
# Similarity metrics
SIM_METRICS = {
    "Tanimoto": TanimotoSimilarity,
    "Cosine": CosineSimilarity,
    # "Tversky": TverskySimilarity,  # requires alpha and beta params
    "Dice": DiceSimilarity,
    "Sokal": SokalSimilarity,
    "McConnaughey": McConnaugheySimilarity,
    "Russel": RusselSimilarity,
    "Kulczynski": KulczynskiSimilarity,
}


def add_fingerprints(data_df, fp_dict=FP_GENS):
    # Add fingerprints
    for fp_name, fpgen in fp_dict.items():
        print(f"FP Generation Method: {fp_name}")
        data_df[fp_name] = data_df.mol.apply(lambda x: fpgen(x))
    return data_df


def create_compare_df(ref_df, frac=FRACTION, elec_props=None, sim_metrics=None, fp_gens=None, random_seed=1, verbose=1):
    # Get reference data
    elec_props = elec_props or ELEC_PROPS
    sim_metrics = sim_metrics or SIM_METRICS
    fp_gens = fp_gens or FP_GENS

    all_ids = list(ref_df.sample(frac=frac, random_state=random_seed).index)
    print("Num of IDs Used: ", len(all_ids)) if verbose > 0 else None
    with open("ids_used_{:02d}perc.txt".format(round(frac*100)), 'w') as fn:
        fn.write(",".join(all_ids))
    all_id_pairs = list(combinations(all_ids, 2))
    print("Num of Comps: ", len(all_id_pairs)) if verbose > 0 else None
    with open("id_pairs_{:02d}perc.txt".format(round(frac*100)), 'w') as fn:
        fn.write(str(all_id_pairs))
    # Create comparison DF
    comp_df = pd.DataFrame(all_id_pairs, columns=["id_1", "id_2"])

    # Add electronic property differences
    fp_dict = ref_df.to_dict()
    print("Calculating differences in electronic properties") if verbose > 0 else None
    for ep in elec_props:
        print("\t", ep) if verbose > 1 else None
        comp_df["diff_" + ep] = comp_df.apply(lambda x: fp_dict[ep][x.id_1] - fp_dict[ep][x.id_2], axis=1)

    # Add similarity metrics
    for fp in fp_gens.keys():
        print(f"FP Generation Method: {fp}") if verbose > 0 else None
        for sim, SimCalc in sim_metrics.items():
            print(f"\tSimilarity {sim}") if verbose > 1 else None
            comp_df[f"{fp}_{sim.lower()}"] = comp_df.apply(lambda x: SimCalc(fp_dict[fp][x.id_1], fp_dict[fp][x.id_2]), axis=1)
    return comp_df


def get_all_d():
    fp_path = os.path.join(BASE_DIR, FP_FILE)
    if os.path.isfile(fp_path):
        all_d = pd.read_pickle(fp_path)
    else:
        # Get Dataset
        all_d = pd.read_pickle(os.path.join(BASE_DIR, ORIG_FILE))
        all_d = add_fingerprints(all_d)
        all_d.to_pickle(fp_path)
        print("Fingerprint file saved to ", fp_path)
    return all_d


if __name__ == "__main__":
    all_df = get_all_d()
    compare_df = create_compare_df(all_df)
    compare_df.to_csv(os.path.join(BASE_DIR, SIM_FILE))
    print("Comparison file saved to ", SIM_FILE)
