from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea, VPacker  # type: ignore
from typing import *
import xml.etree.ElementTree as ET  # type: ignore
import io  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from rdkit.Chem import rdFMCS as MCS  # type: ignore
from rdkit.Chem import MolFromSmiles as smi2mol  # type: ignore
from rdkit.Chem.Draw import MolToImage as mol2img  # type: ignore
from rdkit.Chem import AllChem  # type: ignore
from rdkit.Chem.Draw import SimilarityMaps  # type:ignore
from rdkit.Chem import FindAtomEnvironmentOfRadiusN  # type: ignore
import rdkit.Chem  # type: ignore
import matplotlib as mpl  # type: ignore
from .data import *
import skunk  # type: ignore

delete_color = mpl.colors.to_rgb("#F06060")
modify_color = mpl.colors.to_rgb("#1BBC9B")


def _bit2atoms(m, bitInfo, key):
    # get atom id and radius
    i, r = bitInfo[key][0]  # just take first matching atom
    # taken from rdkit drawing code
    bitPath = FindAtomEnvironmentOfRadiusN(m, r, i)

    # get the atoms for highlighting
    atoms = set((i,))
    for b in bitPath:
        a = m.GetBondWithIdx(b).GetBeginAtomIdx()
        atoms.add(a)
        a = m.GetBondWithIdx(b).GetEndAtomIdx()
        atoms.add(a)
    return atoms


def _imgtext2mpl(txt):
    from PIL import Image  # type: ignore
    from io import BytesIO  # type: ignore

    plt.imshow(Image.open(BytesIO(txt)))
    plt.axis("off")


def _extract_loc(e):
    path = e.attrib["d"]
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


def insert_svg(
    exps: List[Example],
    mol_size: Tuple[int, int] = (200, 200),
    mol_fontsize: int = 10,
) -> str:
    """Replace rasterized image files with SVG versions of molecules

    :param exps: The molecules for which images should be replaced. Typically just counterfactuals or some small set
    :param mol_size: If mol_size was specified, it needs to be re-specified here
    :return: SVG string that can be saved or displayed in juypter notebook
    """
    size = mol_size
    mol_svgs = _mol_images(exps, mol_size, mol_fontsize, True)
    svg = skunk.pltsvg(bbox_inches="tight")
    rewrites = {f"rdkit-img-{i}": v for i, v in enumerate(mol_svgs)}
    return skunk.insert(rewrites, svg=svg)


def trim(im):
    """Implementation of whitespace trim

    credit: https://stackoverflow.com/a/10616717

    :param im: PIL image
    :return: PIL image
    """
    from PIL import Image, ImageChops  # type: ignore

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
        # im = trim(im)
        img_data = np.asarray(im)
        img_box = skunk.ImageBox(f"rdkit-img-{i}", img_data)
        title_box = TextArea(t)
        packed = VPacker(children=[img_box, title_box], pad=0, sep=4, align="center")
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
                    highlightColor=modify_color if len(bidx) > 0 else delete_color,
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
                    highlightAtomColors={k: modify_color for k in aidx}
                    if len(bidx) > 0
                    else {k: delete_color for k in aidx},
                    highlightBondColors={k: modify_color for k in bidx}
                    if len(bidx) > 0
                    else {k: delete_color for k in bidx},
                )
            )

    if len(ms) > 1:
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
        svgs[i] = svgs[i].replace("stroke-width:2.0px;", "")
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


def similarity_map_using_tstats(
    example: Example, mol_size: Tuple[int, int] = (300, 200), return_svg: bool = False
) -> Optional[str]:
    """Create similarity map for example molecule using descriptor t-statistics.
    Only works for ECFP descriptors

    :param example: Example object
    :param mol_size: size of molecule image
    :param return_svg: return svg instead of saving to file
    :return: svg if return_svg is True, else None
    """
    assert (
        example.descriptors.descriptor_type.lower() == "ecfp"
    ), "Similarity maps can only be drawn for ECFP descriptors"
    # get necessary info for mol
    mol = smi2mol(example.smiles)
    tstat_dict = {
        a: b
        for a, b in zip(
            example.descriptors.descriptor_names, example.descriptors.tstats
        )
    }
    tstat_dict = dict(
        sorted(tstat_dict.items(), key=lambda item: abs(item[1]), reverse=True)
    )
    bi = {}  # type: Dict[Any, Any]
    fp = AllChem.GetMorganFingerprint(mol, 3, bitInfo=bi)
    # Get contributions for atoms
    contribs = {atom: [0] for atom in range(mol.GetNumAtoms())}  # type: Dict[Any,Any]
    for b in bi:
        for atom in _bit2atoms(mol, bi, b):
            if b in tstat_dict:
                contribs[atom].append(tstat_dict[b])
    weights = [max(contribs[a], key=abs) for a in range(mol.GetNumAtoms())]
    # use max or threshold of significance t-value
    # threshold significance of 0.1 --> t >= |1.645|
    cut = min(max(abs(min(weights)), abs(max(weights))), 1.645)
    weights = [b if abs(b) >= cut else 0 for b in weights]
    dos = rdkit.Chem.Draw.MolDrawOptions()
    dos.useBWAtomPalette()
    dos.drawMolsSameScale = True
    if return_svg:
        d = rdkit.Chem.Draw.MolDraw2DSVG(*mol_size)
    else:
        d = rdkit.Chem.Draw.MolDraw2DCairo(*mol_size)
    d.SetDrawOptions(dos)
    rdkit.Chem.Draw.SimilarityMaps.GetSimilarityMapFromWeights(
        mol, weights=weights, draw2d=d, contourLines=0, colorMap="bwr_r"
    )
    d.FinishDrawing()
    text = d.GetDrawingText()
    if return_svg:
        return text
    _imgtext2mpl(text)


def plot_space_by_fit(
    examples: List[Example],
    exps: List[Example],
    beta: List,
    mol_size: Tuple[int, int] = (200, 200),
    mol_fontsize: int = 8,
    offset: int = 0,
    ax: Any = None,
    figure_kwargs: Dict = None,
    cartoon: bool = False,
    rasterized: bool = False,
):
    """Plot chemical space around example by LIME fit and annotate given examples.
    Adapted from :func:`plot_space`.

    :param examples: Large list of :obj:Example which make-up points
    :param exps: Small list of :obj:Example which will be annotated
    :param beta: beta output from :func:`lime_explain`
    :param mol_size: size of rdkit molecule rendering, in pixles
    :param mol_fontsize: minimum font size passed to rdkit
    :param offset: offset annotations to allow colorbar or other elements to fit into plot.
    :param ax: axis onto which to plot
    :param figure_kwargs: kwargs to pass to :func:`plt.figure<matplotlib.pyplot.figure>`
    :param cartoon: do cartoon outline on points?
    :param rasterized: raster the scatter?
    """
    imgs = _mol_images(exps, mol_size, mol_fontsize)
    if figure_kwargs is None:
        figure_kwargs = {"figsize": (12, 8)}
    base_color = "gray"
    if ax is None:
        ax = plt.figure(**figure_kwargs).gca()

    yhat = np.array([e.yhat for e in examples])
    yhat -= np.mean(yhat)
    x_mat = np.array([list(e.descriptors.descriptors) for e in examples]).reshape(
        len(examples), -1
    )
    y = x_mat @ beta
    # use resids as colors
    colors = (yhat - y) ** 2
    normalizer = plt.Normalize(min(colors), max(colors))
    cmap = "PuBu_r"

    space_x = [e.position[0] for e in examples]
    space_y = [e.position[1] for e in examples]
    if cartoon:
        # plot shading, lines, front
        ax.scatter(space_x, space_y, 50, "0.0", lw=2, rasterized=rasterized)
        ax.scatter(space_x, space_y, 50, "1.0", lw=0, rasterized=rasterized)
        ax.scatter(
            space_x,
            space_y,
            50,
            c=normalizer(colors),
            cmap=cmap,
            lw=2,
            alpha=0.1,
            rasterized=rasterized,
        )
    else:
        im = ax.scatter(
            space_x,
            space_y,
            40,
            c=normalizer(colors),
            cmap=cmap,
            edgecolors="grey",
            linewidth=0.25,
        )
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")
    cbar = plt.colorbar(im, orientation="horizontal", aspect=35, pad=0.05)
    cbar.set_label("squared error")

    # now plot cfs/annotated points
    ax.scatter(
        [e.position[0] for e in exps],
        [e.position[1] for e in exps],
        c=normalizer([e.yhat for e in exps]),
        cmap=cmap,
        edgecolors="black",
    )

    x = [e.position[0] for e in exps]
    y = [e.position[1] for e in exps]
    titles = []
    colors = []
    for e in exps:
        if not e.is_origin:
            titles.append(f"Similarity = {e.similarity:.2f}\nf(x)={e.yhat:.3f}")
            colors.append(cast(Any, base_color))
        else:
            titles.append(f"Base \nf(x)={e.yhat:.3f}")
            colors.append(cast(Any, base_color))
    _image_scatter(x, y, imgs, titles, colors, ax, offset=offset)
    ax.axis("off")
    ax.set_aspect("auto")
