import os
import multiprocessing
from functools import partial
import pymongo.errors
import tqdm
import joblib

import pandas as pd
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import TanimotoSimilarity, TverskySimilarity, CosineSimilarity, DiceSimilarity, SokalSimilarity, \
    RusselSimilarity, KulczynskiSimilarity, McConnaugheySimilarity

from d3tales_api.D3database.d3database import D3Database

BASE_DIR = Path(__file__).resolve().parent

ORIG_FILE = "data_files/ocelot_d3tales_CLEAN.pkl"
FP_FILE = "data_files/ocelot_d3tales_fps.pkl"
SIM_FILE = "data_files/combo_sims_100perc.csv"
FP_DICT = pd.read_pickle(os.path.join(BASE_DIR, FP_FILE)).to_dict() if os.path.isfile(os.path.join(BASE_DIR, FP_FILE)) else None
SIM_DB = D3Database(database='random', collection_name='similarities')

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


def get_incomplete_ids(ref_df, expected_num=26583):
    all_ids = list(ref_df.index)
    incomplete_ids = []
    for i in tqdm.tqdm(all_ids):
        if SIM_DB.coll.count_documents({'$or': [{"id_1": i}, {"id_2": i}]}) < expected_num:
            incomplete_ids.append(i)
    return incomplete_ids


def add_db_idx(ref_df, skip_existing=False, id_list=None):
    sim_db = D3Database(database='random', collection_name='similarities')
    all_ids = list(ref_df.index)
    print("Num of IDs Used: ", len(all_ids))

    for i in tqdm.tqdm(id_list or all_ids):
        if skip_existing:
            try:
                sim_db.coll.insert_one({"_id": i + "_" + i, "id_1": i, "id_2": i})
            except pymongo.errors.DuplicateKeyError:
                print("Base ID Skipped: ", i)
                continue
        print("Base ID: ", i)
        insert_data = []
        for j in all_ids:
            id_set = sorted([str(i), str(j)])
            insert_data.append({"_id": "_".join(id_set), "id_1": id_set[0], "id_2": id_set[1]})
        try:
            sim_db.coll.insert_many(insert_data, ordered=False)
        except pymongo.errors.BulkWriteError:
            continue


def create_all_idx():
    sim_db = D3Database(database='random', collection_name='similarities')
    all_props = sim_db.coll.find_one({"diff_homo": {"$exists": True}}).keys()
    for prop in [p for p in all_props if "id" not in p]:
        sim_db.coll.create_index(prop)
        print("Index created for ", prop)


def insert_db_data(_id, sim_db=None, fp_dict=None, elec_props=None, sim_metrics=None, fp_gens=None, verbose=1):
    # Get reference data
    elec_props = elec_props or ELEC_PROPS
    sim_metrics = sim_metrics or SIM_METRICS
    fp_gens = fp_gens or FP_GENS
    fp_dict = fp_dict or FP_DICT
    sim_db = sim_db or SIM_DB

    # Query database
    db_data = sim_db.coll.find_one({"_id": _id})
    if db_data.get("diff_" + elec_props[-1]):
        print("SKIPPED") if verbose else None
        return

    insert_data = {}
    # Add electronic property differences
    for ep in elec_props:
        insert_data["diff_" + ep] = fp_dict[ep][db_data["id_1"]] - fp_dict[ep][db_data["id_2"]]

    # Add similarity metrics
    for fp in fp_gens.keys():
        print(f"FP Generation Method: {fp}") if verbose > 1 else None
        for sim, SimCalc in sim_metrics.items():
            print(f"\tSimilarity {sim}") if verbose > 1 else None
            insert_data[f"{fp}_{sim.lower()}"] = SimCalc(fp_dict[fp][db_data["id_1"]], fp_dict[fp][db_data["id_2"]])

    sim_db.coll.update_one({"_id": _id}, {"$set": insert_data})
    print("ID {} updated with fingerprint and similarity info".format(_id)) if verbose else None


def create_compare_df(limit=1000, sim_db=None, **kwargs):
    sim_db = sim_db or SIM_DB

    test_prop = "diff_homo"
    ids = [d.get("_id") for d in sim_db.coll.find({test_prop: {"$exists": False}}, {"_id": 1}).limit(limit)]

    while ids:
        for i in ids:
            insert_db_data(i, **kwargs)
        print("making next query of {}...".format(limit))
        ids = [d["_id"] for d in sim_db.coll.find({test_prop: {"$exists": False}}, {"_id": 1}).limit(limit)]


def create_compare_df_parallel(limit=1000, sim_db=None, **kwargs):
    sim_db = sim_db or SIM_DB
    test_prop = "diff_homo"
    ids = [d.get("_id") for d in sim_db.coll.find({test_prop: {"$exists": False}}, {"_id": 1}).limit(limit)]

    # Use multiprocessing to parallelize the processing of database data
    while ids:
        print("Starting multiprocessing with {} CPUs".format(multiprocessing.cpu_count()))
        with ThreadPoolExecutor(max_workers=64) as executor:
            executor.map(partial(insert_db_data, **kwargs), ids)
        print("making next query of {}...".format(limit))
        ids = [d["_id"] for d in sim_db.coll.find({test_prop: {"$exists": False}}, {"_id": 1}).limit(limit)]


if __name__ == "__main__":

    # fp_path = os.path.join(BASE_DIR, FP_FILE)
    # if os.path.isfile(fp_path):
    #     all_d = pd.read_pickle(fp_path)
    # else:
    #     # Create Dataset
    #     all_d = pd.read_pickle(os.path.join(BASE_DIR, ORIG_FILE))
    #     all_d = add_fingerprints(all_d)
    #     all_d.to_pickle(fp_path)
    #     print("Fingerprint file saved to ", fp_path)

    # add_db_idx(all_d, skip_existing=False, id_list=['csd_EWENOJ', 'csd_KUDSIM'])
    # create_compare_df(verbose=1)
    create_compare_df_parallel(limit=100000, verbose=1)
    # create_all_idx()
    # print(get_incomplete_ids(all_d))
