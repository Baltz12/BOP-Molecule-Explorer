# quantum_dashboard_smiles_combobox.py

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Quantum Molecule Explorer", layout="wide")

st.title("🧬 BOP Quantum Molecule Explorer")
st.caption("Compare molecules, simulate quantum states, and expose pharma patterns.")

# --- Known SMILES Strings ---
smiles_options = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Acetaminophen": "CC(=O)NC1=CC=C(O)C=C1",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Penicillin G": "CC1(C)S[C@@H](N2C(=O)CSC2=O)[C@H](C(=O)O)N1",
    "Morphine": "CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
    "Methamphetamine": "CC(CC1=CC=CC=C1)N",
    "LSD": "CN(C)C1CCC2=C1C3=C(C=C2)C4=CNC5=C4C3=CC=C5",
    "Ethanol": "CCO",
    "Glucose": "C(C1C(C(C(C(O1)O)O)O)O)O",
    "Water": "O",
    "Carbon Dioxide": "O=C=O",
    "Methane": "C",
    "Benzene": "c1ccccc1",
    "Toluene": "Cc1ccccc1",
    "Formaldehyde": "C=O"
}

smiles_labels = ["Select...", "Custom"] + list(smiles_options.keys())

# --- Compound Inputs ---
st.subheader("🔍 Compare Two Compounds")

col1, col2 = st.columns(2)
with col1:
    selected_1 = st.selectbox("Compound 1", smiles_labels, index=0)
    if selected_1 == "Custom":
        smiles_1 = st.text_input("Enter custom SMILES for Compound 1")
    elif selected_1 != "Select...":
        smiles_1 = smiles_options[selected_1]
    else:
        smiles_1 = None

with col2:
    selected_2 = st.selectbox("Compound 2", smiles_labels, index=0)
    if selected_2 == "Custom":
        smiles_2 = st.text_input("Enter custom SMILES for Compound 2")
    elif selected_2 != "Select...":
        smiles_2 = smiles_options[selected_2]
    else:
        smiles_2 = None

# --- Molecule Previews ---
st.subheader("🧪 Individual Molecule Previews")

mol1 = Chem.MolFromSmiles(smiles_1) if smiles_1 else None
mol2 = Chem.MolFromSmiles(smiles_2) if smiles_2 else None

col1, col2 = st.columns(2)
with col1:
    if mol1:
        label_1 = selected_1 if selected_1 != "Custom" else "Manual SMILES"
        st.image(Draw.MolToImage(mol1, size=(300, 300)))
        st.markdown(f"**Compound 1:** {label_1}")
        st.text(f"SMILES: {smiles_1}")
with col2:
    if mol2:
        label_2 = selected_2 if selected_2 != "Custom" else "Manual SMILES"
        st.image(Draw.MolToImage(mol2, size=(300, 300)))
        st.markdown(f"**Compound 2:** {label_2}")
        st.text(f"SMILES: {smiles_2}")

# --- Combined Molecule Preview ---
st.subheader("🔗 Combined Molecule Preview")

if smiles_1 and smiles_2:
    combined_smiles = f"{smiles_1}.{smiles_2}"
    combined_mol = Chem.MolFromSmiles(combined_smiles)
    if combined_mol:
        st.image(Draw.MolToImage(combined_mol, size=(400, 400)))
        st.caption("Combined Molecule (non-bonded)")
        st.text(f"Combined SMILES: {combined_smiles}")
    else:
        st.warning("⚠️ Could not render combined molecule.")
else:
    st.info("Enter both SMILES to generate combined preview.")

# --- Quantum Simulation with VQE ---
st.subheader("⚛️ Quantum Simulation (VQE Approximation)")

def run_vqe(atom_count):
    wires = min(max(atom_count, 2), 4)
    dev = qml.device("default.qubit", wires=wires)

    def ansatz(params):
        for i in range(wires):
            qml.RY(params[i], wires=i)
        for i in range(wires - 1):
            qml.CNOT(wires=[i, i + 1])

    @qml.qnode(dev)
    def cost_fn(params):
        ansatz(params)
        return qml.expval(qml.PauliZ(0))

    init_params = np.random.uniform(0, np.pi, wires)
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    steps = 30
    params = init_params

    energy_track = []
    for _ in range(steps):
        params = opt.step(cost_fn, params)
        energy = cost_fn(params)
        energy_track.append(energy)

    return energy_track

def plot_energy_curve(energy_track, label):
    fig, ax = plt.subplots()
    ax.plot(range(len(energy_track)), energy_track, marker='o')
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("Estimated Energy")
    ax.set_title(f"VQE Energy Curve: {label}")
    st.pyplot(fig)

with st.expander("Run Quantum Circuit"):
    if smiles_1:
        atom_count_1 = mol1.GetNumAtoms()
        energy_track_1 = run_vqe(atom_count_1)
        plot_energy_curve(energy_track_1, selected_1 if selected_1 != "Custom" else "Compound 1")
    if smiles_2:
        atom_count_2 = mol2.GetNumAtoms()
        energy_track_2 = run_vqe(atom_count_2)
        plot_energy_curve(energy_track_2, selected_2 if selected_2 != "Custom" else "Compound 2")

# --- Pharma Watchdog Placeholder ---
st.subheader("🚨 Pharma Watchdog")
st.markdown("Flag suspicious compounds or corporate behaviors here (coming soon).")

# --- Footer ---
st.markdown("---")
st.caption("Built by Mitchell • Powered by outrage and quantum logic ⚛️")
