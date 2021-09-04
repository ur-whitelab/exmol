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

    svg = exmol.mpl2svg()
    svg = exmol.rewrite_svg(svg, {'offset_box_0': msvg}, 50 / 300)


def test_replace_svg_img():
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

    svg = exmol.mpl2svg()
    svg = exmol.rewrite_svg(svg, {'offset_box_0': msvg}, 50 / 300)
    with open('test.svg', 'w') as f:
        f.write(svg)
