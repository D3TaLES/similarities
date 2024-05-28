import pandas as pd
from rdkit import Chem

d_data = pd.read_csv("data_files/d3tales.csv")
d_data = d_data[d_data["groundState.homo_lumo_gap"].notna()]
o_data = pd.read_csv("data_files/ocelot_chromophore_v1.csv")

d2o_names = {
    "_id": "identifier", "smiles": "smiles",
    "vertical_ionization_energy": "vie", "adiabatic_ionization_energy": "aie",
    "vertical_electron_affinity": "vea", "adiabatic_electron_affinity": "aea",
    "groundState.homo_lumo_gap": "hl", "groundState.lumo": "lumo", "groundState.homo": "homo",
    "groundState.singlet_states": "s0s1", "groundState.triplet_states": "s0t1",
    "hole_reorganization_energy": "hr", "electron_reorganization_energy": "er"
}

merge_d_data = d_data.rename(columns=d2o_names)[d2o_names.values()].assign(source="d3tales")
merge_d_data.set_index("identifier", inplace=True)
print("D3TaLES Data: ", merge_d_data.shape[0])
merge_o_data = o_data[d2o_names.values()].assign(source="ocelot")
merge_o_data.set_index("identifier", inplace=True)
print("OCELOT Data: ", merge_o_data.shape[0])

merge_data = pd.concat([merge_d_data, merge_o_data])
merge_data['mol'] = merge_data.smiles.apply(lambda x: Chem.MolFromSmiles(x))
print("Merge Data: ", merge_data.shape[0])
print("Merge Data CLEAN: ", merge_data.dropna(axis=0).shape[0])

merge_data.to_pickle("composite_data/ocelot_d3tales.pkl")
merge_data.dropna(axis=0).to_pickle("composite_data/ocelot_d3tales_CLEAN.pkl")

