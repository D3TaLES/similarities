import tqdm
import pymongo.errors
import multiprocessing
from functools import partial
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor
from settings import *

from d3tales_api.D3database.d3database import D3Database
SIM_DB = D3Database(database='random', collection_name='similarities')


def add_fingerprints(data_df, fp_dict=FP_GENS):
    # Add fingerprints
    for fp_name, fpgen in fp_dict.items():
        print(f"FP Generation Method: {fp_name}")
        data_df[fp_name] = data_df.mol.apply(lambda x: fpgen(x))
    return data_df


def create_compare_df(ref_df, frac=DATA_PERCENT, elec_props=None, sim_metrics=None, fp_gens=None, random_seed=1, verbose=1):
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


# ------------- SIMILARITIES DATABASE --------------------

def get_incomplete_ids(ref_df, expected_num=26583):
    all_ids = list(ref_df.index)
    incomplete_ids = []
    for i in tqdm.tqdm(all_ids):
        if SIM_DB.coll.count_documents({'$or': [{"id_1": i}, {"id_2": i}]}) < expected_num:
            incomplete_ids.append(i)
    return incomplete_ids


def add_db_idx(ref_df, skip_existing=False, id_list=None):
    all_ids = list(ref_df.index)
    print("Num of IDs Used: ", len(all_ids))

    for i in tqdm.tqdm(id_list or all_ids):
        if skip_existing:
            try:
                SIM_DB.coll.insert_one({"_id": i + "_" + i, "id_1": i, "id_2": i})
            except pymongo.errors.DuplicateKeyError:
                print("Base ID Skipped: ", i)
                continue
        print("Base ID: ", i)
        insert_data = []
        for j in all_ids:
            id_set = sorted([str(i), str(j)])
            insert_data.append({"_id": "_".join(id_set), "id_1": id_set[0], "id_2": id_set[1]})
        try:
            SIM_DB.coll.insert_many(insert_data, ordered=False)
        except pymongo.errors.BulkWriteError:
            continue


def create_all_idx():
    all_props = SIM_DB.coll.find_one({"diff_homo": {"$exists": True}}).keys()
    for prop in [p for p in all_props if "id" not in p]:
        SIM_DB.coll.create_index(prop)
        print("Index created for ", prop)


def insert_db_data(_id, sim_db=None, fp_dict=None, elec_props=None, sim_metrics=None, fp_gens=None, verbose=1, insert=True):
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

    if insert:
        sim_db.coll.update_one({"_id": _id}, {"$set": insert_data})
        print("ID {} updated with fingerprint and similarity info".format(_id)) if verbose else None
    insert_data.update({"_id": _id})
    return insert_data


def batch_insert_db_data(ids, batch_size=100, sim_db=None, **kwargs):
    sim_db = sim_db or SIM_DB
    chunks = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
    for chunk in chunks:
        insert_data = [insert_db_data(i, insert=False) for i in chunk]
        try:
            sim_db.coll.insert_many(insert_data, ordered=False)
        except pymongo.errors.BulkWriteError:
            continue


def create_db_compare_df_parallel(limit=1000, sim_db=None, **kwargs):
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

    # # Get fingerprint DF
    # all_d = get_all_d()
    #
    # # Create comparison DF CSV
    # compare_df = create_compare_df(all_d)
    # compare_df.to_csv(os.path.join(BASE_DIR, SIM_FILE))
    # print("Comparison file saved to ", SIM_FILE)
    #
    # # Generate Database indices
    # all_df = get_all_d()
    # add_db_idx(all_d, skip_existing=False, id_list=['csd_EWENOJ', 'csd_KUDSIM'])
    # create_compare_df(verbose=1)
    # create_all_idx()
    # print(get_incomplete_ids(all_d))

    # Add comparison properties to DB
    create_db_compare_df_parallel(limit=100000, verbose=1)

