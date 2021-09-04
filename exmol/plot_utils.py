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
    return min(x), min(y)


def rewrite_svg(svg, rdict, scale):
    ns = "http://www.w3.org/2000/svg"
    root, idmap = ET.XMLID(svg)
    parent_map = {c: p for p in root.iter() for c in p}
    for rk, rv in rdict.items():
        if rk in idmap:
            e = idmap[rk]
            # try to use id width/height
            # case when we have image
            if 'width' in e.attrib:
                x, y = e.attrib['x'], -float(e.attrib['y'])
                # make new node
                # to hold things (so we can transform)
                new_e = ET.SubElement(
                    parent_map[e], f'{{{ns}}}g', {'id': f'{rk}-g'})
                parent_map[e].remove(e)
                e = new_e
            else:
                # relying on there being a path object inside to give clue
                # to size
                c = list(e)[0]
                x, y = _extract_loc(c)
                e.remove(c)
            # now set-up enclosing element transform for image
            e.attrib['transform'] = f'translate({x}, {y}) scale({scale}, {scale})'
            rr = ET.fromstring(rv)
            e.insert(0, rr)
        else:
            print('Warning, could not find', rk)
            print(list(idmap.keys()))
    ET.register_namespace("", ns)
    return ET.tostring(root, encoding="unicode", method='xml')


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
        # add transparency
        im = trim(im)
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
        bbs.append(bb)
    return bbs


def _mol_images(exps, mol_size, fontsize):
    if len(exps) == 0:
        return []
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
        # if it is too large, we ignore it
        # if len(aidx) > 8:
        #    aidx = []
        #    bidx = []
        imgs.append(
            mol2img(
                m,
                size=mol_size,
                options=dos,
                highlightAtoms=aidx,
                highlightBonds=bidx,
                highlightColor=modify_color if len(bidx) > 0 else delete_color,
            )
        )
    rdkit.Chem.AllChem.GenerateDepictionMatching2DStructure(
        ms[0], ms[1], acceptFailure=True
    )
    imgs.insert(0, mol2img(ms[0], size=mol_size, options=dos))
    return imgs


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
