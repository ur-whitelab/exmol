import exmol
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi


def test_version():
    assert exmol.__version__


def test_example():
    e = exmol.Example("CC", "", 0, 0, 0, is_origin=True)
    print(e)


def test_randomize_smiles():
    si = "N#CC=CC(C(=O)NCC1=CC=CC=C1C(=O)N)(C)CC2=CC=C(F)C=C2CC"
    m = smi2mol(si)
    so = exmol.stoned.randomize_smiles(m)
    assert si != so


def test_sanitize_smiles():
    si = "N#CC=CC(C(=O)NCC1=CC=CC=C1C(=O)N)(C)CC2=CC=C(F)C=C2CC"
    result = exmol.stoned.sanitize_smiles(si)
    assert result[1] is not None


# TODO let STONED people write these when they finish their repo


def test_run_stones():
    result = exmol.run_stoned(
        "N#CC=CC(C(=O)NCC1=CC=CC=C1C(=O)N)(C)CC2=CC=C(F)C=C2CC",
        num_samples=10,
        max_mutations=1,
    )
    # Can get duplicates
    assert len(result[0]) >= 0


def test_run_stones_alphabet():
    result = exmol.run_stoned(
        "N#CC=CC(C(=O)NCC1=CC=CC=C1C(=O)N)(C)CC2=CC=C(F)C=C2CC",
        num_samples=10,
        max_mutations=1,
        alphabet=["[C]", "[O]"],
    )
    # Can get duplicates
    assert len(result[0]) >= 0


def test_sample():
    def model(s, se):
        return int("N" in s)

    explanation = exmol.sample_space("CCCC", model, batched=False)
    # check that no redundants
    assert len(explanation) == len(set([e.smiles for e in explanation]))


def test_sample_preset():
    def model(s, se):
        return int("N" in s)

    explanation = exmol.sample_space("CCCC", model, preset="narrow", batched=False)
    # check that no redundants
    assert len(explanation) == len(set([e.smiles for e in explanation]))


def test_cf_explain():
    def model(s, se):
        return int("N" in s)

    samples = exmol.sample_space("CCCC", model, batched=False)
    exps = exmol.cf_explain(samples, 3)
    assert len(exps) == 4  # +1 for base


def test_rcf_explain():
    def model(s, se):
        return len(s)

    samples = exmol.sample_space("CCCC", model, batched=False)
    exps = exmol.rcf_explain(samples)
    assert len(exps) == 5
    exps = exmol.rcf_explain(samples, delta=(None, 1))
    assert len(exps) == 3


def test_plot():
    def model(s, se):
        return int("N" in s)

    samples = exmol.sample_space("CCCC", model, batched=False)
    exps = exmol.cf_explain(samples, 3)
    exmol.plot_cf(exps)
    exmol.plot_space(samples, exps)


def test_plot_clusters():
    def model(s, se):
        return int("N" in s)

    samples = exmol.sample_space("CCCC", model, batched=False)
    exps = exmol.cf_explain(samples, 3)
    exmol.plot_cf(exps)
    exmol.plot_space(samples, exps, highlight_clusters=True)


def test_empty_plot():
    def model(s, se):
        return int("N" in s)

    samples = exmol.sample_space("CCCC", model, batched=False)
    exps = exmol.cf_explain(samples, 3)
    exmol.plot_space(samples, [])


def test_compare_img():
    smi1 = "CCCC"
    smi2 = "CCN"
    m1 = smi2mol(smi1)
    m2 = smi2mol(smi2)
    r, _ = exmol.moldiff(m1, m2)
    assert len(r) > 0
