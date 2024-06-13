import tqdm
import pandas as pd
import pymongo.errors
import multiprocessing
from rdkit import Chem
from functools import partial
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor

from similarities.settings import *


def load_mols_db(smiles_pickle, fp_dict=FP_GENS, replace=False, 
                 mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB, mongo_coll="molecules"):
    # Get Dataset
    all_d = pd.read_pickle(smiles_pickle)
    if 'mol' not in all_d.columns:
        if 'smiles' not in all_d.columns:
            raise KeyError(f'Column "smiles" not found in {smiles_pickle} columns')
        all_d['mol'] = all_d.smiles.apply(lambda x: Chem.MolFromSmiles(x))
    for fp_name, fpgen in fp_dict.items():
        print(f"FP Generation Method: {fp_name}")
        if fp_name not in all_d.columns:
            all_d[fp_name] = all_d.mol.apply(lambda x: fpgen(x))

    # Push to DB
    with MongoClient(mongo_uri)[mongo_db][mongo_coll] as db_coll: 
        db_coll.insert_many(all_d.to_dict('records'))
        print("Fingerprint database saved.")
    return all_d


def get_incomplete_ids(ref_df, expected_num=26583, mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB, mongo_coll="molecules"):
    all_ids = list(ref_df.index)
    incomplete_ids = []
    with MongoClient(mongo_uri)[mongo_db][mongo_coll] as db_coll: 
        for i in tqdm.tqdm(all_ids):
            if db_coll.count_documents({'$or': [{"id_1": i}, {"id_2": i}]}) < expected_num:
                incomplete_ids.append(i)
    return incomplete_ids


def add_db_idx(skip_existing=False, id_list=None, mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB):
    with MongoClient(mongo_uri)[mongo_db] as db_conn: 
        all_ids = list(db_conn["molecules"].distinct("_id"))
        print("Num of IDs Used: ", len(all_ids))
    
        for i in tqdm.tqdm(id_list or all_ids):
            if skip_existing:
                try:
                    db_conn["mol_pairs"].insert_one({"_id": i + "_" + i, "id_1": i, "id_2": i})
                except pymongo.errors.DuplicateKeyError:
                    print("Base ID Skipped: ", i)
                    continue
            print("Base ID: ", i)
            insert_data = []
            for j in all_ids:
                id_set = sorted([str(i), str(j)])
                insert_data.append({"_id": "_".join(id_set), "id_1": id_set[0], "id_2": id_set[1]})
            try:
                db_conn["mol_pairs"].insert_many(insert_data, ordered=False)
            except pymongo.errors.BulkWriteError:
                continue


def create_all_idx(db_coll=SIM_DB["mol_pairs"], mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB, mongo_coll="molecules"):
    all_props = db_coll.find_one({"diff_homo": {"$exists": True}}).keys()
    sim_metrics = [p for p in all_props if "Reg_" in p or "SCnt" in p]
    print(f"Creating index for {', '.join(sim_metrics)}...")
    for prop in sim_metrics:
        print(f"Starting index creation for {prop}...")
        db_coll.create_index(prop)
        print("--> Success!")


def insert_db_data(_id, db_conn, db_conn=SIM_DB, elec_props=ELEC_PROPS, sim_metrics=SIM_METRICS, fp_gens=FP_GENS, verbose=1, insert=True, 
                   mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB):
    # Get reference data

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


def batch_insert_db_data(ids, batch_size=100, mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB, mongo_coll="molecules", **kwargs):
    chunks = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
    with MongoClient(mongo_uri)[mongo_db] as db_conn:
        for chunk in chunks:
            insert_data = [insert_db_data(i, db_conn, insert=False) for i in chunk]
            try:
                db_conn[mongo_coll].insert_many(insert_data, ordered=False)
            except pymongo.errors.BulkWriteError:
                continue


def create_db_compare_df_parallel(limit=1000, mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB, mongo_coll="molecules", **kwargs):
    test_prop = "diff_homo"
    with MongoClient(mongo_uri)[mongo_db] as db_conn: 
        ids = [d.get("_id") for d in db_conn[mongo_coll].find({test_prop: {"$exists": False}}, {"_id": 1}).limit(limit)]
    
        # Use multiprocessing to parallelize the processing of database data
        while ids:
            print("Starting multiprocessing with {} CPUs".format(multiprocessing.cpu_count()))
            with ThreadPoolExecutor(max_workers=64) as executor:
                executor.map(partial(insert_db_data, [db_conn], **kwargs), ids)
            print("making next query of {}...".format(limit))
            ids = [d["_id"] for d in db_conn[mongo_coll].find({test_prop: {"$exists": False}}, {"_id": 1}).limit(limit)]


if __name__ == "__main__":
    d_percent, a_percent, t_percent = 0.10, 0.10, 0.10

    all_d = load_mols_db()

    # # Comparison DF
    # compare_df = pd.read_csv(DATA_DIR / f"combo_sims_{int(d_percent*100):02d}perc.csv", index_col=0)
    # sim_reg_cols = [c for c in compare_df.columns if "Reg_" in c]
    # sim_scnt_cols = [c for c in compare_df.columns if "SCnt_" in c]
    # sim_cols = sim_reg_cols + sim_scnt_cols
    # prop_cols = [c for c in compare_df.columns if (c not in sim_cols and "id_" not in c)]
    # print("Num Instances: ", compare_df.shape[0])