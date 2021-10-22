from PIL.ImageChops import offset
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
import exmol
import io


def test_insert_svg():
    def model(s, se):
        return int("N" in s)

    samples = exmol.sample_space("CCCC", model, batched=False)
    exps = exmol.cf_explain(samples, 3)
    exmol.plot_cf(exps)
    exmol.plot_space(samples, exps)
    svg = exmol.insert_svg(exps)
    with open("test-space.svg", "w") as f:
        f.write(svg)


def test_insert_svg_long():
    def model(s, se):
        return int("N" in s)

    samples = exmol.sample_space(
        "O=C(COCCCCCCCCCC)CC(C)C1CCCC1C2=CC=CC(F)=C2 ", model, batched=False
    )
    exps = exmol.cf_explain(samples, 3)
    exmol.plot_cf(exps)
    exmol.plot_space(samples, exps, mol_size=(300, 200))
    svg = exmol.insert_svg(exps, mol_size=(300, 200))
    with open("test-long.svg", "w") as f:
        f.write(svg)


def test_insert_svg_grid():
    def model(s, se):
        return int("N" in s)

    samples = exmol.sample_space(
        "O=C(COCCCCCCCCCC)CC(C)C1CCCC1C2=CC=CC(F)=C2 ", model, batched=False
    )
    exps = exmol.cf_explain(samples, 3)
    exmol.plot_cf(exps)
    svg = exmol.insert_svg(exps)
    with open("test-cf.svg", "w") as f:
        f.write(svg)
