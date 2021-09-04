from PIL.ImageChops import offset
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
import exmol
import io


def draw_svg(mol, width=300, height=300):
    drawer = Chem.Draw.rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().bgColor = None
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg


def _test_replace_svg():
    from matplotlib.patches import Rectangle
    from matplotlib.offsetbox import (DrawingArea,
                                      AnnotationBbox)
    mol = Chem.MolFromSmiles(
        'O=C(NCC1CCCCC1N)C2=CC=CC=C2C3=CC=C(F)C=C3C(=O)NC4CCCCC4')
    msvg = draw_svg(mol, 300, 300)

    fig, ax = plt.subplots()
    p = Rectangle((0, 0), 50, 50)
    offsetbox = DrawingArea(50, 50)
    offsetbox.add_artist(p)

    ab = AnnotationBbox(offsetbox, (0.5, 0.5))

    ax.add_artist(ab)
    p.set_gid('offset_box_0')

    svg = exmol.plot_utils.mpl2svg()
    svg = exmol.plot_utils.rewrite_svg(svg, {'offset_box_0': (msvg, 50 / 300)})


def _test_replace_svg_img():
    from rdkit.Chem.Draw import MolToImage as mol2img
    from matplotlib.patches import Rectangle
    from matplotlib.offsetbox import (OffsetImage,
                                      AnnotationBbox)
    mol = Chem.MolFromSmiles(
        'O=C(NCC1CCCCC1N)C2=CC=CC=C2C3=CC=C(F)C=C3C(=O)NC4CCCCC4')
    msvg = draw_svg(mol, 300, 300)
    img = mol2img(mol, width=300, height=300)

    fig, ax = plt.subplots()
    offsetbox = OffsetImage(img, zoom=50 / 300)

    ab = AnnotationBbox(offsetbox, (0.5, 0.5))

    ax.add_artist(ab)
    # differrent for images
    offsetbox.properties()['children'][0].set_gid('offset_box_0')

    svg = exmol.plot_utils.mpl2svg()
    svg = exmol.plot_utils.rewrite_svg(svg, {'offset_box_0': (msvg, 50 / 300)})


def _test_insert_svg():
    def model(s, se):
        return int("N" in s)

    samples = exmol.sample_space("CCCC", model, batched=False)
    exps = exmol.cf_explain(samples, 3)
    exmol.plot_cf(exps)
    exmol.plot_space(samples, exps)
    svg = exmol.insert_svg(exps)


def _test_insert_svg_long():
    def model(s, se):
        return int("N" in s)

    samples = exmol.sample_space(
        "O=C(COCCCCCCCCCC)CC(C)C1CCCC1C2=CC=CC(F)=C2 ", model, batched=False)
    exps = exmol.cf_explain(samples, 3)
    exmol.plot_cf(exps)
    exmol.plot_space(samples, exps, mol_size=(300, 200))
    svg = exmol.insert_svg(exps, mol_size=(300, 200))
    # with open('test.svg', 'w') as f:
    #    f.write(svg)


def _test_insert_svg_grid():
    def model(s, se):
        return int("N" in s)

    samples = exmol.sample_space(
        "O=C(COCCCCCCCCCC)CC(C)C1CCCC1C2=CC=CC(F)=C2 ", model, batched=False)
    exps = exmol.cf_explain(samples, 3)
    exmol.plot_cf(exps)
    svg = exmol.insert_svg(exps)
    # with open('test.svg', 'w') as f:
    #    f.write(svg)


def test_insert_svg_grid_d():
    def model(s, se):
        return int("N" in s)

    samples = exmol.sample_space(
        "CCCC", model, batched=False)
    exps = exmol.cf_explain(samples, 3)
    exmol.plot_cf(exps)
    svg = exmol.insert_svg(exps, descriptors=True)
    with open('test.svg', 'w') as f:
        f.write(svg)
