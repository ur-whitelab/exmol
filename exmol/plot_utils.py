from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea, VPacker
from typing import *
import xml.etree.ElementTree as ET
import io
import matplotlib.pyplot as plt
import numpy as np
from rdkit.Chem import rdFMCS as MCS
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem.Draw import MolToImage as mol2img
import rdkit.Chem
import matplotlib.pyplot as plt
import matplotlib as mpl
from .data import *

delete_color = mpl.colors.to_rgb("#F06060")
modify_color = mpl.colors.to_rgb("#1BBC9B")


def _extract_loc(e):
    path = e.attrib['d']
    spath = path.split()
    x, y = [], []
    a1, a2 = x, y
    for s in spath:
        try:
            a1.append(float(s))
            a1, a2 = a2, a1
        except ValueError:
            continue
    return min(x), min(y), max(x) - min(x), max(y) - min(y)


def rewrite_svg(svg, rdict):
    ns = "http://www.w3.org/2000/svg"
    root, idmap = ET.XMLID(svg)
    parent_map = {c: p for p in root.iter() for c in p}
    for rk, rvtup in rdict.items():
        rv, size = rvtup
        if rk in idmap:
            e = idmap[rk]
            # try to use id width/height
            # case when we have image
            if 'width' in e.attrib:
                x, y = e.attrib['x'], -float(e.attrib['y'])
                # make new node
                # to hold things
                new_e = ET.SubElement(
                    parent_map[e], f'{{{ns}}}g', {'id': f'{rk}-g'})
                parent_map[e].remove(e)
                #dx, dy = float(e.attrib['width']), float(e.attrib['height'])
                dx, dy = size
                e = new_e
            else:
                # relying on there being a path object inside to give clue
                # to size
                c = list(e)[0]
                x, y, dx, dy = _extract_loc(c)
                e.remove(c)
            # set attributes on SVG so loc and width/height are correct
            rr = ET.fromstring(rv)
            rr.attrib['x'] = str(x)
            rr.attrib['y'] = str(y)
            rr.attrib['width'] = str(dx)
            rr.attrib['height'] = str(dy)
            e.insert(0, rr)
        else:
            print('Warning, could not find', rk)
            print(list(idmap.keys()))
    ET.register_namespace("", ns)
    return ET.tostring(root, encoding="unicode", method='xml')


def insert_svg(exps: List[Example],
               mol_size: Tuple[int, int] = (200, 200)) -> str:
    """Replace rasterized image files with SVG versions of molecules

    :param exps: The molecules for which images should be replaced. Typically just counterfactuals or some small set
    :param mol_size: If mol_size was specified, it needs to be re-specified here
    :return: SVG string that can be saved or displayed in juypter notebook
    """
    size = mol_size
    mol_svgs = _mol_images(exps, mol_size, 12, True)
    svg = mpl2svg(bbox_inches='tight')
    rewrites = {f'rdkit-img-{i}': (v, size) for i, v in enumerate(mol_svgs)}
    return rewrite_svg(svg, rewrites)


def mpl2svg(**kwargs):
    with io.BytesIO() as output:
        plt.savefig(output, format='svg', **kwargs)
        return output.getvalue()


def trim(im):
    """Implementation of whitespace trim

    credit: https://stackoverflow.com/a/10616717

    :param im: PIL image
    :return: PIL image
    """
    from PIL import Image, ImageChops

    # https://stackoverflow.com/a/10616717
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def _nearest_spiral_layout(x, y, offset):
    # make spiral
    angles = np.linspace(-np.pi, np.pi, len(x) + 1 + offset)[offset:]
    coords = np.stack((np.cos(angles), np.sin(angles)), -1)
    order = np.argsort(np.arctan2(y, x))
    return coords[order]


def _image_scatter(x, y, imgs, subtitles, colors, ax, offset):
    box_coords = _nearest_spiral_layout(x, y, offset)
    bbs = []
    for i, (x0, y0, im, t, c) in enumerate(zip(x, y, imgs, subtitles, colors)):
        # TODO Figure out how to put this back
        #im = trim(im)
        img_data = np.asarray(im)
        img_box = OffsetImage(img_data)
        title_box = TextArea(t)
        packed = VPacker(children=[img_box, title_box],
                         pad=0, sep=4, align="center")
        bb = AnnotationBbox(
            packed,
            (x0, y0),
            frameon=True,
            xybox=box_coords[i] + 0.5,
            arrowprops=dict(arrowstyle="->", edgecolor="black"),
            pad=0.3,
            boxcoords="axes fraction",
            bboxprops=dict(edgecolor=c),
        )
        ax.add_artist(bb)

        # add gid in case svg-rewrite
        img_box.properties()['children'][0].set_gid(f'rdkit-img-{i}')

        bbs.append(bb)
    return bbs


def _mol2svg(m, size, options, **kwargs):
    d = rdkit.Chem.Draw.rdMolDraw2D.MolDraw2DSVG(*size)
    d.SetDrawOptions(options)
    d.DrawMolecule(m, **kwargs)
    d.FinishDrawing()
    return d.GetDrawingText()


def _mol_images(exps, mol_size, fontsize, svg=False):
    if len(exps) == 0:
        return [], []
    # get aligned images
    ms = [smi2mol(e.smiles) for e in exps]
    dos = rdkit.Chem.Draw.MolDrawOptions()
    dos.useBWAtomPalette()
    dos.minFontSize = fontsize
    rdkit.Chem.AllChem.Compute2DCoords(ms[0])
    imgs = []
    for m in ms[1:]:
        rdkit.Chem.AllChem.GenerateDepictionMatching2DStructure(
            m, ms[0], acceptFailure=True
        )
        aidx, bidx = moldiff(ms[0], m)
        if not svg:
            imgs.append(
                mol2img(
                    m,
                    size=mol_size,
                    options=dos,
                    highlightAtoms=aidx,
                    highlightBonds=bidx,
                    highlightColor=modify_color if len(
                        bidx) > 0 else delete_color,
                )
            )
        else:
            imgs.append(
                _mol2svg(
                    m,
                    size=mol_size,
                    options=dos,
                    highlightAtoms=aidx,
                    highlightBonds=bidx,
                    highlightAtomColors={k: modify_color for k in aidx} if len(
                        bidx) > 0 else {k: delete_color for k in aidx},
                    highlightBondColors={k: modify_color for k in bidx} if len(
                        bidx) > 0 else {k: delete_color for k in bidx},
                )
            )

    rdkit.Chem.AllChem.GenerateDepictionMatching2DStructure(
        ms[0], ms[1], acceptFailure=True
    )
    if svg:
        imgs.insert(0, _mol2svg(ms[0], size=mol_size, options=dos))
        imgs = _cleanup_rdkit_svgs(imgs)
    else:
        imgs.insert(0, mol2img(ms[0], size=mol_size, options=dos))
    return imgs


def _cleanup_rdkit_svgs(svgs):
    for i in range(len(svgs)):
        # simple approach
        svgs[i] = svgs[i].replace('stroke-width:2.0px;', '')
    return svgs


def moldiff(template, query) -> Tuple[List[int], List[int]]:
    """Compare the two rdkit molecules.

    :param template: template molecule
    :param query: query molecule
    :return: list of modified atoms in query, list of modified bonds in query
    """
    r = MCS.FindMCS([template, query])
    substructure = rdkit.Chem.MolFromSmarts(r.smartsString)
    raw_match = query.GetSubstructMatches(substructure)
    template_match = template.GetSubstructMatches(substructure)
    # flatten it
    match = list(raw_match[0])
    template_match = list(template_match[0])

    # need to invert match to get diffs
    inv_match = [i for i in range(query.GetNumAtoms()) if i not in match]

    # get bonds
    bond_match = []
    for b in query.GetBonds():
        if b.GetBeginAtomIdx() in inv_match or b.GetEndAtomIdx() in inv_match:
            bond_match.append(b.GetIdx())

    # now get bonding changes from deletion

    def neigh_hash(a):
        return "".join(sorted([n.GetSymbol() for n in a.GetNeighbors()]))

    for ti, qi in zip(template_match, match):
        if neigh_hash(template.GetAtomWithIdx(ti)) != neigh_hash(
            query.GetAtomWithIdx(qi)
        ):
            inv_match.append(qi)

    return inv_match, bond_match
