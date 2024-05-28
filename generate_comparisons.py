from itertools import combinations
from similarities.settings import *


def add_fingerprints(data_df, fp_dict=FP_GENS):
    # Add fingerprints
    for fp_name, fpgen in fp_dict.items():
        print(f"FP Generation Method: {fp_name}")
        data_df[fp_name] = data_df.mol.apply(lambda x: fpgen(x))
    return data_df


def create_compare_df(ref_df, frac=DATA_FRACTION, elec_props=None, sim_metrics=None, fp_gens=None, random_seed=1, verbose=1):
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
    # import random
    # random.seed(10)

    all_df = get_all_d()
    compare_df = create_compare_df(all_df)
    compare_df.to_csv(os.path.join(BASE_DIR, SIM_FILE))
    print("Comparison file saved to ", SIM_FILE)
