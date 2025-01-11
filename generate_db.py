import os
import random
import time
import tqdm
import pandas as pd
import pymongo.errors
from rdkit import Chem
import multiprocessing
from functools import partial
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from similarities.settings import *


def generate_molecules_df(orig_df: pd.DataFrame = None, smiles_pickle: str = None, fp_dict: dict = FP_GENS):
    """
    Generates a DataFrame of molecules with specified fingerprints from either an original DataFrame or a pickle file
    containing SMILES strings.

    Parameters:
    orig_df (pd.DataFrame): Original DataFrame containing SMILES strings and/or molecular objects.
    smiles_pickle (str): Path to a pickle file containing a DataFrame with SMILES strings.
    fp_dict (dict): Dictionary of fingerprint generation methods.

    Returns:
    pd.DataFrame: DataFrame containing molecules with generated fingerprints.
        """
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
            orig_df[fp_name] = orig_df.mol.apply(lambda x: fpgen(x).ToBase64())
        else:
            orig_df[fp_name] = orig_df[fp_name].apply(lambda x: x.ToBase64())
    return orig_df


def load_mols_db(smiles_pickle, fp_dict=FP_GENS, mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB, mongo_coll=MONGO_MOL_COLL):
    """
    Loads molecules from a SMILES pickle file, generates fingerprints, and stores them in a MongoDB database.

    Parameters:
    smiles_pickle (str): Path to a pickle file containing a DataFrame with SMILES strings.
    fp_dict (dict): Dictionary of fingerprint generation methods.
    mongo_uri (str): MongoDB connection URI.
    mongo_db (str): MongoDB database name.
    mongo_coll (str): MongoDB collection name for molecules.

    Returns:
    pd.DataFrame: DataFrame containing molecules with generated fingerprints.
    """
    # Get Dataset
    all_d = generate_molecules_df(smiles_pickle=smiles_pickle, fp_dict=fp_dict)
    all_d.drop(columns=['mol'], inplace=True)
    all_d["_id"] = all_d.index

    # Push to DB
    with MongoClient(mongo_uri) as client:
        client[mongo_db][mongo_coll].insert_many(all_d.to_dict('records'))
        print("Fingerprint database saved.")
    return all_d


def add_new_pairs_db_idx(new_ids, mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB, mol_coll=MONGO_MOL_COLL, pair_coll=MONGO_PAIRS_COLL):
    """
    Adds pairs of molecule IDs to a MongoDB collection, ensuring unique combinations.

    Parameters:
    new_ids (list):
    mongo_uri (str): MongoDB connection URI.
    mongo_db (str): MongoDB database name.
    """
    with MongoClient(mongo_uri) as client:
        all_ids = list(client[mongo_db][mol_coll].distinct("_id"))
        old_ids = [i for i in all_ids if i not in new_ids]
        print("Num of IDs Used: ", len(all_ids))

        for i_1, id_1 in tqdm.tqdm(enumerate(new_ids)):
            # Generate insert data with new ids and old ids
            id_sets = []
            for id_2 in [id_ for i_2, id_ in enumerate(new_ids) if i_2 > i_1] + old_ids:
                id_sets.append(sorted([str(id_1), str(id_2)]))

            insert_data = [{"_id": "_".join(s), "id_1": s[0], "id_2": s[1]} for s in id_sets]

            # Insert ids into database
            if insert_data:
                try:
                    print("Inserting {} documents...".format(len(insert_data)))
                    client[mongo_db][pair_coll].insert_many(insert_data, ordered=False)
                except pymongo.errors.BulkWriteError:
                    continue


def add_pairs_db_idx(mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB, mol_coll=MONGO_MOL_COLL, pair_coll=MONGO_PAIRS_COLL):
    """
    Adds pairs of molecule IDs to a MongoDB collection, ensuring unique combinations.

    Parameters:
    mongo_uri (str): MongoDB connection URI.
    mongo_db (str): MongoDB database name.
    """
    with MongoClient(mongo_uri) as client:
        all_ids = list(client[mongo_db][mol_coll].distinct("_id"))
        print("Num of IDs Used: ", len(all_ids))

        for i_1, id_1 in tqdm.tqdm(enumerate(all_ids)):
            # Generate insert data
            insert_data = []
            for id_2 in [id_ for i_2, id_ in enumerate(all_ids) if i_2 > i_1]:
                id_set = sorted([str(id_1), str(id_2)])
                insert_data.append({"_id": "_".join(id_set), "id_1": id_set[0], "id_2": id_set[1]})

            # Insert ids into database
            if insert_data:
                try:
                    print("Inserting {} documents...".format(len(insert_data)))
                    client[mongo_db][pair_coll].insert_many(insert_data, ordered=False)
                except pymongo.errors.BulkWriteError:
                    continue


def delete_pairs_db_idx(new_ids, mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB, mol_coll=MONGO_MOL_COLL, pair_coll="mol_pairs_old"):
    """
    Adds pairs of molecule IDs to a MongoDB collection, ensuring unique combinations.

    Parameters:
    mongo_uri (str): MongoDB connection URI.
    mongo_db (str): MongoDB database name.
    """
    with MongoClient(mongo_uri) as client:
        all_ids = list(client[mongo_db][mol_coll].distinct("_id"))
        old_ids = [i for i in all_ids if i not in new_ids]
        print("Num of IDs Used: ", len(all_ids))

        for i_1, id_1 in tqdm.tqdm(enumerate(new_ids)):
            # Generate insert data with new ids and old ids
            for id_2 in [id_ for i_2, id_ in enumerate(new_ids) if i_2 > i_1] + old_ids:
                _id = "_".join(sorted([str(id_1), str(id_2)]))
                client[mongo_db][pair_coll].delete_one({"_id": _id})
                print("Deleted {} document...".format(_id))
            print("Deleted {} documents...".format(id_1))


def insert_pair_data(_id, elec_props=ELEC_PROPS, sim_metrics=SIM_METRICS, fp_gens=FP_GENS,
                     mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB, mol_coll=MONGO_MOL_COLL, pair_coll=MONGO_PAIRS_COLL,
                     verbose=1, insert=True, sim_min=None, sim_min_metric="mfpReg_tanimoto"):
    """
    Inserts electronic properties and similarity metrics for a pair of molecules into a MongoDB collection.

    Parameters:
    _id (str): ID of the pair of molecules.
    db_conn (MongoClient): MongoDB client instance connected to the database.
    elec_props (list): List of electronic properties to calculate differences.
    sim_metrics (dict): Dictionary of similarity metrics to calculate.
    fp_gens (dict): Dictionary of fingerprint generation methods.
    verbose (int): Verbosity level (0 = silent, 1 = minimal output, 2 = detailed output).
    insert (bool): If True, inserts data into MongoDB collection. Default is True.

    Returns:
    dict: Inserted data dictionary if `insert` is False, otherwise None.
    """
    # Query database
    with MongoClient(mongo_uri) as client:
        db_conn = client[mongo_db]
        db_data = db_conn[pair_coll].find_one({"_id": _id})
        id_1_dict = db_conn[mol_coll].find_one({"_id": db_data["id_1"]})
        id_2_dict = db_conn[mol_coll].find_one({"_id": db_data["id_2"]})
        if db_data.get("diff_" + elec_props[-1]):
            print("SKIPPED") if verbose > 2 else None
            return

        insert_data = {}
        # Add electronic property differences
        for ep in elec_props:
            insert_data["diff_" + ep] = id_1_dict[ep] - id_2_dict[ep]

        # Add similarity metrics
        for fp in fp_gens.keys():
            print(f"FP Generation Method: {fp}") if verbose > 2 else None
            for sim, SimCalc in sim_metrics.items():
                print(f"\tSimilarity {sim}") if verbose > 2 else None
                metric_name = f"{fp}_{sim.lower()}"
                fp1 = ExplicitBitVect(2048)
                fp1.FromBase64(id_1_dict[fp])
                fp2 = ExplicitBitVect(2048)
                fp2.FromBase64(id_2_dict[fp])
                similarity = SimCalc(fp1, fp2)
                if sim_min is not None and metric_name==sim_min_metric and (similarity < sim_min):
                    print(f"{_id} insertion skipped because {sim_min_metric} value {similarity} < {sim_min}")
                    return None
                insert_data[f"{metric_name}"] = similarity
        if insert:
            db_conn[pair_coll].update_one({"_id": _id}, {"$set": insert_data})
            print("{} updated with fingerprint and similarity info".format(_id)) if verbose > 1 else None
        insert_data.update({"_id": _id})
        return insert_data


def create_pairs_db_parallel(limit=10000, mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB, mongo_coll=MONGO_PAIRS_COLL,
                             **kwargs):
    """
    Creates pairs of molecule IDs and calculates electronic properties and similarity metrics in parallel using multiprocessing.

    Parameters:
    limit (int): Limit of pairs to process in each query.
    mongo_uri (str): MongoDB connection URI.
    mongo_db (str): MongoDB database name.
    mongo_coll (str): MongoDB collection name. Default is MONGO_PAIRS_COLL.
    **kwargs: Additional keyword arguments passed to `insert_pair_data`.
    """
    test_prop = "diff_homo"
    with MongoClient(mongo_uri) as client:
        query = {test_prop: {"$exists": False}}
        if "sim_min" in kwargs:
            low_prop_ids = [d["_id"] for d in client[mongo_db]["low_props"].find({}).limit(500000)]
            query.update({"_id": {"$nin": low_prop_ids}})
        ids = [d.get("_id") for d in client[mongo_db][mongo_coll].find(query, {"_id": 1}).limit(limit)]

        # Use multiprocessing to parallelize the processing of database data
        while ids:
            print(f"Starting multiprocessing with {multiprocessing.cpu_count()} CPUs to insert props for {len(ids)} ids")
            with ThreadPoolExecutor(max_workers=64) as executor:
                executor.map(partial(insert_pair_data, **kwargs), ids)
            print("making next query of {}...".format(limit))
            ids = [d["_id"] for d in client[mongo_db][mongo_coll].find({test_prop: {"$exists": False}}, {"_id": 1}).limit(limit)]


def create_pairs_db_newID(new_ids, mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB, mol_coll=MONGO_MOL_COLL,
                          pairs_coll=MONGO_PAIRS_COLL, **kwargs):
    """
    Adds pairs of molecule IDs to a MongoDB collection, ensuring unique combinations.

    Parameters:
    new_ids (list):
    mongo_uri (str): MongoDB connection URI.
    mongo_db (str): MongoDB database name.
    """
    with MongoClient(mongo_uri) as client:
        all_ids = list(client[mongo_db][mol_coll].distinct("_id"))
        old_ids = [i for i in all_ids if i not in new_ids]
        print("Num of IDs Used: ", len(all_ids))

        random.shuffle(new_ids)
        for i_1, id_1 in tqdm.tqdm(enumerate(new_ids)):
            # Generate insert data with new ids and old ids

            already_updated = client[mongo_db]["updated_ids"].find_one({"_id": id_1})
            if already_updated:
                print("Already updated ", id_1)
                continue
            print("Starting un-updated query for {}...".format(id_1))
            potential_ids = ["_".join(sorted([str(id_1), str(id_2)])) for id_2 in [id_ for i_2, id_ in enumerate(new_ids) if i_2 > i_1] + old_ids]
            query = {"_id": {"$in": potential_ids}, "diff_homo": {"$exists": False}}
            ids = [d["_id"] for d in client[mongo_db][pairs_coll].find(query, {"_id": 1})]

            print(f"Inserting data for {len(ids)} ids...")

            with ThreadPoolExecutor(max_workers=64) as executor:
                executor.map(partial(insert_pair_data, insert=True, **kwargs), ids)

                client[mongo_db]["updated_ids"].insert_one({"_id": id_1}, {})
                print(f"\n--------------------------\nAll pairs with {id_1} updated.\n--------------------------\n")


def create_pairs_db_batch(limit=2, mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB, mongo_coll=MONGO_PAIRS_COLL,
                          **kwargs):
    """
    Creates pairs of molecule IDs and calculates electronic properties and similarity metrics in parallel using multiprocessing.

    Parameters:
    limit (int): Limit of pairs to process in each query.
    mongo_uri (str): MongoDB connection URI.
    mongo_db (str): MongoDB database name.
    mongo_coll (str): MongoDB collection name. Default is MONGO_PAIRS_COLL.
    **kwargs: Additional keyword arguments passed to `insert_pair_data`.
    """
    test_prop = "diff_homo"
    with MongoClient(mongo_uri) as client:
        query = {test_prop: {"$exists": False}}
        ids = [d["_id"] for d in client[mongo_db][mongo_coll].find(query, {"_id": 1}).limit(limit)]

    # Use multiprocessing to parallelize the processing of database data
    while ids:
        print(f"Starting multiprocessing with {multiprocessing.cpu_count()} CPUs to insert props for {len(ids)} ids")
        initial_pair_data_partial = partial(insert_pair_data, insert=False, **kwargs)
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = [executor.submit(initial_pair_data_partial, _id) for _id in ids]
            results = [future.result() for future in as_completed(futures)]
            results = [r for r in results if r is not None]
        print(f"Inserting pair data for {len(results)} docs...")
        with MongoClient(mongo_uri) as client:
            result_ids = [r["_id"] for r in results]
            print(result_ids)
            client[mongo_db][mongo_coll].insert_many(results)
        old_ids = ids

        print("making next query of {}...".format(limit))
        with MongoClient(mongo_uri) as client:
            ids = [d["_id"] for d in client[mongo_db][mongo_coll].find(query, {"_id": 1}).limit(limit) if d["_id"] not in old_ids]


def index_all(mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB, mongo_coll=MONGO_PAIRS_COLL):
    """
    Creates indexes for specific properties in a MongoDB collection.

    Parameters:
    mongo_uri (str): MongoDB connection URI.
    mongo_db (str): MongoDB database name.
    mongo_coll (str): MongoDB collection name. Default is MONGO_PAIRS_COLL.
    """
    with MongoClient(mongo_uri) as client:
        all_props = client[mongo_db][mongo_coll].find_one({"diff_homo": {"$exists": True}}).keys()
        sim_metrics = [p for p in all_props if "Reg_" in p or "SCnt" in p]
        print(f"Creating index for {', '.join(sim_metrics)}...")
        for prop in sim_metrics:
            print(f"Starting index creation for {prop}...")
            client[mongo_db][mongo_coll].create_index(prop)
            print("--> Success!")


def get_incomplete_ids(ref_df, expected_num=26583, mongo_uri=MONGO_CONNECT, mongo_db=MONGO_DB, mongo_coll=MONGO_PAIRS_COLL):
    """
    Retrieves IDs from a reference DataFrame that have fewer documents associated with them than expected in a MongoDB collection.

    Parameters:
    ref_df (DataFrame): Reference DataFrame containing index IDs to check.
    expected_num (int): Expected number of associated documents for each ID. Default is 26583.
    mongo_uri (str): MongoDB connection URI.
    mongo_db (str): MongoDB database name.
    mongo_coll (str): MongoDB collection name. Default is MONGO_PAIRS_COLL.

    Returns:
    list: List of IDs from `ref_df` that have fewer associated documents than `expected_num`.
    """
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
    new_ids = list(pd.read_json(DATA_DIR / "new_ids.json", typ="series").index)
    # add_new_pairs_db_idx(new_ids=new_ids)
    # pairs_df = pd.read_csv(DATA_DIR / f"combo_sims_{int(d_percent*100):02d}perc.csv", index_col=0)
    # sim_reg_cols = [c for c in pairs_df.columns if "Reg_" in c]
    # sim_scnt_cols = [c for c in pairs_df.columns if "SCnt_" in c]
    # sim_cols = sim_reg_cols + sim_scnt_cols
    # prop_cols = [c for c in pairs_df.columns if (c not in sim_cols and "id_" not in c)]
    # print("Num Instances: ", pairs_df.shape[0])

    # create_pairs_db_parallel(verbose=2, sim_min=0.15)

    create_pairs_db_parallel(verbose=2)
    # create_pairs_db_newID(new_ids, verbose=2, sim_min=0.15)

    # delete_pairs_db_idx(new_ids=new_ids)
