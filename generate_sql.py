import tqdm
import sqlite3

from similarities.utils import *
from similarities.settings import *


def load_mols_db(smiles_pickle, fp_dict=FP_GENS, elec_props=ELEC_PROPS, db_file=DB_FILE, replace=False):
    table_name = "molecules"
    with closing(sqlite3.connect(db_file)) as conn:

        if replace or not table_exists(conn, table_name):
            # Create the table if it doesn't exist
            create_text_table(conn, table_name, txt_columns=["smiles"] + list(fp_dict.keys()), float_columns=elec_props)

            # Get Dataset
            all_d = generate_molecules_df(smiles_pickle, fp_dict=fp_dict)

            # Push to DB
            all_d[["smiles"] + list(fp_dict.keys()) + elec_props].to_sql(table_name, conn, if_exists='append', index_label='_id', index=True)
            print("Fingerprint database saved.")

        return pd.read_sql(f'SELECT * FROM {table_name}', conn, index_col='_id')


def pair_db_from_json(json_file, db_file=DB_FILE, elec_props=ELEC_PROPS, sim_metrics=SIM_METRICS, fp_gens=FP_GENS,
                   replace=False, chunk_size=5000, verbose=1, start_num=0):
    table_name = "mol_pairs"
    print(f"Loading {json_file} from {db_file}") if verbose else None
    with closing(sqlite3.connect(db_file)) as conn:

        if replace and table_exists(conn, table_name):
            print(f"Table {table_name} already exists...replacing it") if verbose else None
            # Create the MolPairs table if it doesn't exist
            prop_cols = ["diff_" + ep for ep in elec_props]
            metric_cols = [f"{fp}_{sim.lower()}" for sim in sim_metrics for fp in fp_gens]
            create_text_table(conn, table_name, txt_columns=["id_1", "id_2"], float_columns=prop_cols + metric_cols)

        # Read JSON file into DB table
        print("Dividing JSON into chunks...") if verbose else None
        chunks = pd.read_json(json_file, lines=True, chunksize=chunk_size)
        count = 0
        for c in tqdm.tqdm(chunks):
            count += chunk_size
            if count < start_num:
                continue
            try:
                if row_exists(conn, c._id.iloc[0]):
                    continue
                non_existing_df = c[c._id.apply(lambda x: not row_exists(conn, x))]
                if not non_existing_df.empty:
                    non_existing_df.to_sql(table_name, conn, if_exists='append', index=False)
                    print(f"Successfully inserted {non_existing_df} entries") if verbose else None

            except Exception as e:
                print("ERROR inserting chunk! ", e) if verbose else None
                break

        conn.commit()

def create_pair_db(db_file=DB_FILE, elec_props=ELEC_PROPS, sim_metrics=SIM_METRICS, fp_gens=FP_GENS,
                   replace=False, frac=1, random_seed=1, verbose=1):
    table_name = "mol_pairs" + "{:02d}_{:02d}".format(frac*100, random_seed) if frac < 1 else ""
    with closing(sqlite3.connect(db_file)) as conn:

        if replace and table_exists(conn, table_name):
            # Create the MolPairs table if it doesn't exist
            prop_cols = ["diff_" + ep for ep in elec_props]
            metric_cols = [f"{fp}_{sim.lower()}" for sim in sim_metrics for fp in fp_gens]
            create_text_table(conn, table_name, txt_columns=["id_1", "id_2"], float_columns=prop_cols + metric_cols)

        # Get ids
        all_ids = list(pd.read_sql('SELECT id FROM moleclues', conn, index_col='id').sample(frac=frac, random_state=random_seed).index)
        print("Num of IDs Used: ", len(all_ids)) if verbose > 0 else None

        # Add data to db
        for i_1, id_1 in tqdm.tqdm(enumerate(all_ids)):
            for id_2 in [id_ for i_2, id_ in enumerate(all_ids) if i_2 >= i_1]:
                id_1, id_2 = sorted(map(str, [id_1, id_2]))
                pair_dict = {"_id": "_".join([id_1, id_2]), "id_1": id_1, "id_2": id_2}

                with closing(conn.cursor()) as cursor:
                    # Add electronic property differences
                    print("Calculating differences in electronic properties") if verbose > 0 else None
                    for ep in elec_props:
                        print("\t", ep) if verbose > 1 else None
                        [[ep1], [ep2]] = cursor.execute(f"SELECT {ep} FROM molecules WHERE id IN ('{id_1}', '{id_2}')").fetchall()[0]
                        pair_dict["diff_" + ep] = ep1 - ep2

                    # Add similarity metrics
                    for fp in fp_gens.keys():
                        print(f"FP Generation Method: {fp}") if verbose > 0 else None
                        for sim, SimCalc in sim_metrics.items():
                            print(f"\tSimilarity {sim}") if verbose > 1 else None
                            [[fp1], [fp2]] = cursor.execute(f"SELECT {ep} FROM molecules WHERE id IN ('{id_1}', '{id_2}')").fetchall()[0]
                            pair_dict[f"{fp}_{sim.lower()}"] = SimCalc( list(map(int, fp1)),  list(map(int, fp2)))

                    sql = 'INSERT INTO {} ({}) VALUES ({})'.format(table_name, ', '.join(pair_dict.keys()),
                                                                   ', '.join('?' * len(pair_dict)))
                    cursor.execute(sql, list(pair_dict.values()))
            conn.commit()


if __name__ == "__main__":

    # # Generate molecules DB
    # orig_file = DATA_DIR / "ocelot_d3tales_fps.pkl"
    # print(load_mols_db(orig_file, replace=True).head())

    # Generate pairs DB
    pair_db_from_json(DATA_DIR / "similarities.json", replace=False, verbose=2, start_num=230052000)

    # with closing(sqlite3.connect(DB_FILE)) as conn:
    #     q_df = pd.read_sql("SELECT COUNT(*) FROM mol_pairs LIMIT 100;", conn)
    # print(q_df)

