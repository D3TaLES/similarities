import os
import pandas as pd
from pathlib import Path
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import TanimotoSimilarity, TverskySimilarity, CosineSimilarity, DiceSimilarity, SokalSimilarity, \
    RusselSimilarity, KulczynskiSimilarity, McConnaugheySimilarity


BASE_DIR = Path(__file__).resolve().parent

DATA_PERCENT = 0.05
TOP_PERCENT = 0.10
NUM_TRIALS = 30

ORIG_FILE = "../data_files/ocelot_d3tales_CLEAN.pkl"
FP_FILE = "../data_files/ocelot_d3tales_fps.pkl"
SIM_FILE = "composite_data/combo_sims_{:02d}perc.csv".format(round(DATA_PERCENT * 100))
FP_DICT = pd.read_pickle(os.path.join(BASE_DIR, FP_FILE)).to_dict() if os.path.isfile(os.path.join(BASE_DIR, FP_FILE)) else None

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
