from counterstone.counterstone import explain
import counterstone
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi


def test_version():
    assert counterstone.__version__


def test_randomize_smiles():
    si = 'N#CC=CC(C(=O)NCC1=CC=CC=C1C(=O)N)(C)CC2=CC=C(F)C=C2CC'
    m = smi2mol(si)
    so = counterstone.stoned.randomize_smiles(m)
    assert si != so


def test_sanitize_smiles():
    si = 'N#CC=CC(C(=O)NCC1=CC=CC=C1C(=O)N)(C)CC2=CC=C(F)C=C2CC'
    result = counterstone.stoned.sanitize_smiles(si)
    assert result[1] is not None

# TODO let STONED people write these when they finish their repo


def test_run_stones():
    result = counterstone.run_stoned(
        'N#CC=CC(C(=O)NCC1=CC=CC=C1C(=O)N)(C)CC2=CC=C(F)C=C2CC',
        num_samples=10, max_mutations=1)
    assert len(result[0]) == 10


def test_run_stones_callback():
    result = counterstone.run_stoned(
        'N#CC=CC(C(=O)NCC1=CC=CC=C1C(=O)N)(C)CC2=CC=C(F)C=C2CC',
        num_samples=100, max_mutations=1, stop_callback=lambda x, y: True)
    # check that no redundants
    assert len(result) == len(set([r[0] for r in result]))


def test_explain():
    def model(s, se):
        return 'N' in s
    explanation, _ = counterstone.explain(
        'CCCC', model, max_k=3, cluster=False, batched=False)
    # check that no redundants
    assert len(explanation) == len(set([e.smiles for e in explanation]))


def test_cluster_explain():
    def model(s, se):
        return 'N' in s
    explanation, _ = counterstone.explain(
        'CCCC', model, max_k=3, batched=False)


def test_plot():
    def model(s, se):
        return 'N' in s
    explanation, _ = counterstone.explain(
        'CCCC', model, max_k=3, batched=False)
    counterstone.plot_explanation(explanation)


def test_compare_img():
    smi1 = 'CCCC'
    smi2 = 'CCN'
    m1 = smi2mol(smi1)
    m2 = smi2mol(smi2)
    r, _ = counterstone.moldiff(m1, m2)
    assert len(r) > 0
