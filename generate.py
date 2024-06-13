import os
import tqdm
import pandas as pd
import pymongo.errors
from rdkit import Chem
import multiprocessing
from functools import partial
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor


from similarities.settings import *

def generate_molecules_df(orig_df: pd.DataFrame = None, smiles_pickle: str = None, fp_dict: dict = FP_GENS):
    if not orig_df:
        if not os.path.isfile(smiles_pickle):
            raise IOError("No DF pickle file found at {}. This function requires either an original_df argument or"
                          "a valid DF pickle file location".format(smiles_pickle))
        orig_df = pd.read_pickle(smiles_pickle)

    if 'mol' not in orig_df.columns:
        if 'smiles' not in orig_df.columns:
            raise KeyError(f'Column "smiles" not found in {smiles_pickle} columns')
        orig_df['mol'] = orig_df.smiles.apply(lambda x: Chem.MolFromSmiles(x))
    for fp_name, fpgen in fp_dict.items():
        print(f"FP Generation Method: {fp_name}")
        if fp_name not in orig_df.columns:
            orig_df[fp_name] = orig_df.mol.apply(lambda x: ''.join(map(str, fpgen(x))))
        else:
            orig_df[fp_name] = orig_df[fp_name].apply(lambda x: ''.join(map(str, x)))
    return orig_df


def load_mols_db(smiles_pickle, fp_dict=FP_GENS, mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB, mongo_coll="molecules"):
    # Get Dataset
    all_d = generate_molecules_df(smiles_pickle, fp_dict=fp_dict)

    # Push to DB
    with MongoClient(mongo_uri) as client:
        client[mongo_db][mongo_coll].insert_many(all_d.to_dict('records'))
        print("Fingerprint database saved.")
    return all_d


def add_pairs_db_idx(skip_existing=False, id_list=None, mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB):
    with MongoClient(mongo_uri) as client:
        all_ids = list(client[mongo_db]["molecules"].distinct("_id"))
        print("Num of IDs Used: ", len(all_ids))

        for i in tqdm.tqdm(id_list or all_ids):
            if skip_existing:
                try:
                    client[mongo_db]["mol_pairs"].insert_one({"_id": i + "_" + i, "id_1": i, "id_2": i})
                except pymongo.errors.DuplicateKeyError:
                    print("Base ID Skipped: ", i)
                    continue
            print("Base ID: ", i)
            insert_data = []
            for j in all_ids:
                id_set = sorted([str(i), str(j)])
                insert_data.append({"_id": "_".join(id_set), "id_1": id_set[0], "id_2": id_set[1]})
            try:
                client[mongo_db]["mol_pairs"].insert_many(insert_data, ordered=False)
            except pymongo.errors.BulkWriteError:
                continue


def insert_pair_data(_id, db_conn, elec_props=ELEC_PROPS, sim_metrics=SIM_METRICS, fp_gens=FP_GENS,
                     verbose=1, insert=True):

    # Query database
    db_data = db_conn["mol_pairs"].find_one({"_id": _id})
    id_1_dict = db_conn["molecules"].find_one({"_id": db_data["id_1"]})
    id_2_dict = db_conn["molecules"].find_one({"_id": db_data["id_2"]})
    if db_data.get("diff_" + elec_props[-1]):
        print("SKIPPED") if verbose else None
        return

    insert_data = {}
    # Add electronic property differences
    for ep in elec_props:
        insert_data["diff_" + ep] = id_1_dict[ep] - id_2_dict[ep]

    # Add similarity metrics
    for fp in fp_gens.keys():
        print(f"FP Generation Method: {fp}") if verbose > 1 else None
        for sim, SimCalc in sim_metrics.items():
            print(f"\tSimilarity {sim}") if verbose > 1 else None
            insert_data[f"{fp}_{sim.lower()}"] = SimCalc(id_1_dict[fp], id_2_dict[fp])

    if insert:
        db_conn["mol_pairs"].update_one({"_id": _id}, {"$set": insert_data})
        print("ID {} updated with fingerprint and similarity info".format(_id)) if verbose else None
    insert_data.update({"_id": _id})
    return insert_data


def create_pairs_db_parallel(limit=1000, mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB, mongo_coll="mol_pairs",
                             **kwargs):
    test_prop = "diff_homo"
    with MongoClient(mongo_uri) as client:
        ids = [d.get("_id") for d in client[mongo_db][mongo_coll].find({test_prop: {"$exists": False}}, {"_id": 1}).limit(limit)]

        # Use multiprocessing to parallelize the processing of database data
        while ids:
            print("Starting multiprocessing with {} CPUs".format(multiprocessing.cpu_count()))
            with ThreadPoolExecutor(max_workers=64) as executor:
                executor.map(partial(insert_pair_data, client[mongo_db], **kwargs), ids)
            print("making next query of {}...".format(limit))
            ids = [d["_id"] for d in client[mongo_db][mongo_coll].find({test_prop: {"$exists": False}}, {"_id": 1}).limit(limit)]


def index_all(mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB, mongo_coll="mol_pairs"):
    with MongoClient(mongo_uri) as client:
        all_props = client[mongo_db][mongo_coll].find_one({"diff_homo": {"$exists": True}}).keys()
        sim_metrics = [p for p in all_props if "Reg_" in p or "SCnt" in p]
        print(f"Creating index for {', '.join(sim_metrics)}...")
        for prop in sim_metrics:
            print(f"Starting index creation for {prop}...")
            client[mongo_db][mongo_coll].create_index(prop)
            print("--> Success!")


def get_incomplete_ids(ref_df, expected_num=26583, mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB, mongo_coll="mol_pairs"):
    all_ids = list(ref_df.index)
    incomplete_ids = []
    with MongoClient(mongo_uri) as client:
        for i in tqdm.tqdm(all_ids):
            if client[mongo_db][mongo_coll].count_documents({'$or': [{"id_1": i}, {"id_2": i}]}) < expected_num:
                incomplete_ids.append(i)
    return incomplete_ids


if __name__ == "__main__":
    d_percent, a_percent, t_percent = 0.10, 0.10, 0.10

    # all_d = load_mols_db(DATA_DIR / "ocelot_d3tales_CLEAN.pkl")

    # # Comparison DF
    # pairs_df = pd.read_csv(DATA_DIR / f"combo_sims_{int(d_percent*100):02d}perc.csv", index_col=0)
    # sim_reg_cols = [c for c in pairs_df.columns if "Reg_" in c]
    # sim_scnt_cols = [c for c in pairs_df.columns if "SCnt_" in c]
    # sim_cols = sim_reg_cols + sim_scnt_cols
    # prop_cols = [c for c in pairs_df.columns if (c not in sim_cols and "id_" not in c)]
    # print("Num Instances: ", pairs_df.shape[0])