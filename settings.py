from pathlib import Path
from rdkit import DataStructs as rdk_d
from rdkit.Chem import rdFingerprintGenerator as rdk_gen


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data_files'
PLOT_DIR = BASE_DIR / 'plots'

# SQLite DB
DB_FILE = DATA_DIR / "similarities.db"
# MongoDB
MONGO_CONNECT = "mongodb://10.33.30.17:23771/"
MONGO_DB = "random"
MONGO_MOL_COLL = "molecules"
MONGO_PAIRS_COLL = "mol_pairs"


# Electronic Props to compare
ELEC_PROPS = ['aie', 'aea', 'hl', 'lumo', 'homo', 's0s1']

# Fingerprint Generators (both bit and simulated count forms)
FP_GENS = {

  "mfpReg": rdk_gen.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint,
  "mfpSCnt": rdk_gen.GetMorganGenerator(radius=2, fpSize=2048, countSimulation=True).GetFingerprint,
  # "mfpCnt": rdk_gen.GetMorganGenerator(radius=2, fpSize=2048).GetCountFingerprint,

  "rdkReg": rdk_gen.GetRDKitFPGenerator(fpSize=2048).GetFingerprint,
  "rdkSCnt": rdk_gen.GetRDKitFPGenerator(fpSize=2048, countSimulation=True).GetFingerprint,
  # "rdkCnt": rdk_gen.GetRDKitFPGenerator(fpSize=2048).GetCountFingerprint,

  "aprReg": rdk_gen.GetAtomPairGenerator(fpSize=2048).GetFingerprint,
  "aprSCnt": rdk_gen.GetAtomPairGenerator(fpSize=2048, countSimulation=True).GetFingerprint,
  # "aprCnt": rdk_gen.GetAtomPairGenerator(fpSize=2048).GetCountFingerprint,

  "ttrReg": rdk_gen.GetTopologicalTorsionGenerator(fpSize=2048).GetFingerprint,
  "ttrSCnt": rdk_gen.GetTopologicalTorsionGenerator(fpSize=2048, countSimulation=True).GetFingerprint,
  # "ttrCnt": rdk_gen.GetTopologicalTorsionGenerator(fpSize=2048).GetCountFingerprint,
}

# Similarity metrics
SIM_METRICS = {
    "Tanimoto": rdk_d.TanimotoSimilarity,
    "Cosine": rdk_d.CosineSimilarity,
    # "Tversky": rdk_d.TverskySimilarity,  # requires alpha and beta params
    "Dice": rdk_d.DiceSimilarity,
    "Sokal": rdk_d.SokalSimilarity,
    # "McConnaughey": rdk_d.McConnaugheySimilarity,
    # "Russel": rdk_d.RusselSimilarity,  # range depends on number of set bits
    "Kulczynski": rdk_d.KulczynskiSimilarity,
}
