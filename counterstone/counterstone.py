from rdkit.Chem import rdFMCS as MCS
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import selfies as sf
import itertools
import math
from . import stoned
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem.Draw import MolToImage as mol2img
import rdkit.Chem
import matplotlib.pyplot as plt


@dataclass
class Examples:
    smiles: str
    selfies: str
    similarity: float
    yhat: float
    index: int
    position: np.ndarray = None
    is_origin: bool = False
    cluster: int = 0
    label: str = None


def change_classification(yhats, y):
    return (yhats - y).astype(bool)


def _fp_dist_matrix(smiles, fp_type='ECFP4'):
    mols = [smi2mol(s) for s in smiles]
    fp = [stoned.get_fingerprint(m, fp_type) for m in mols]
    # 1 - Ts because we want distance
    dist = list(1 - stoned.TanimotoSimilarity(x, y)
                for x, y in itertools.product(fp, repeat=2))
    return np.array(dist).reshape(len(mols), len(mols))


def run_stoned(
        s, fp_type='ECFP4', num_samples=2000,
        max_mutations=2, alphabet=None):
    '''Run ths STONED SELFIES algorithm
    '''
    if alphabet is None:
        alphabet = list(sf.get_semantic_robust_alphabet())
    num_mutation_ls = list(range(1, max_mutations + 1))

    mol = smi2mol(s)
    if mol == None:
        raise Exception('Invalid starting structure encountered')

    randomized_smile_orderings = [stoned.randomize_smiles(
        mol) for _ in range(num_samples)]

    # Convert all the molecules to SELFIES
    selfies_ls = [sf.encoder(x) for x in randomized_smile_orderings]

    all_smiles_collect = []
    all_selfies_collect = []
    for num_mutations in num_mutation_ls:
        # Mutate the SELFIES:
        selfies_mut = stoned.get_mutated_SELFIES(
            selfies_ls.copy(), num_mutations=num_mutations, alphabet=alphabet)
        # Convert back to SMILES:
        smiles_back = [sf.decoder(x) for x in selfies_mut]
        all_smiles_collect = all_smiles_collect + smiles_back
        all_selfies_collect = all_selfies_collect + selfies_mut
        print('STONED Round Complete with', len(smiles_back))

    # Work on:  all_smiles_collect
    canon_smi_ls = []
    for item in all_smiles_collect:
        mol, smi_canon, did_convert = stoned.sanitize_smiles(item)
        if mol == None or smi_canon == '' or did_convert == False:
            raise Exception('Invalid smile string found')
        canon_smi_ls.append(smi_canon)

    # remove redundant/non-unique/duplicates
    # in a way to keep the selfies
    canon_smi_ls = list(set(canon_smi_ls))

    canon_smi_ls_scores = stoned.get_fp_scores(
        canon_smi_ls, target_smi=s, fp_type=fp_type)
    # NOTE Do not think of returning selfies. They have duplicates
    return canon_smi_ls, canon_smi_ls_scores


def sample_space(origin_smiles, f, batched=True, preset='medium', stoned_kwargs=None):
    batched_f = f
    if not batched:
        def batched_f(sm, se): return np.array(
            [f(smi, sei) for smi, sei in zip(sm, se)])
    smi_yhat = batched_f([origin_smiles], [sf.encoder(origin_smiles)])
    try:
        iter(smi_yhat)
    except TypeError:
        raise ValueError('Your model function does not appear to be batched')
    smi_yhat = np.squeeze(smi_yhat[0])

    if stoned_kwargs is None:
        stoned_kwargs = {}
        if preset == 'medium':
            stoned_kwargs['num_samples'] = 1500
            stoned_kwargs['max_mutations'] = 2
        elif preset == 'narrow':
            stoned_kwargs['num_samples'] = 3000
            stoned_kwargs['max_mutations'] = 1
        elif preset == 'wide':
            stoned_kwargs['num_samples'] = 600
            stoned_kwargs['max_mutations'] = 5
        else:
            raise ValueError(f'Unknown preset "{preset}"')

    # STONED
    smiles, scores = run_stoned(origin_smiles, **stoned_kwargs)
    selfies = [sf.encoder(s) for s in smiles]
    fxn_values = batched_f(smiles, selfies)

    # pack them into data structure with filtering out identical
    # and nan
    exps = [
        Examples(origin_smiles, sf.encoder(origin_smiles),
                 1.0, smi_yhat, index=0, is_origin=True)
    ] +\
        [
        Examples(sm, se, s, np.squeeze(y), index=0) for i, (sm, se, s, y) in
        enumerate(zip(smiles, selfies, scores, fxn_values)) if s < 1.0 and np.squeeze(y)
    ]
    for i, e in enumerate(exps):
        e.index = i

    # compute distance matrix
    full_dmat = _fp_dist_matrix([e.smiles for e in exps],
                                stoned_kwargs['fp_type'] if ('fp_type' in stoned_kwargs) else 'ECFP4')

    # compute PCA
    pca = PCA(n_components=2)
    proj_dmat = pca.fit_transform(full_dmat)
    for e in exps:
        e.position = proj_dmat[e.index, :]

    # do clustering everwhere (maybe do counter/same separately?)
    # clustering = AgglomerativeClustering(
    #    n_clusters=max_k, affinity='precomputed', linkage='complete').fit(full_dmat)
    # Just do it on projected so it looks prettier.
    clustering = DBSCAN(eps=0.15, min_samples=5).fit(proj_dmat)

    for i, e in enumerate(exps):
        e.cluster = clustering.labels_[i]

    return exps


def _select_examples(cond, examples, nmols):
    result = []

    # similarit filtered by if cluster/counter
    def cluster_score(e, i):
        return (e.cluster == i) * cond(e) * e.similarity
    clusters = set([e.cluster for e in examples])
    for i in clusters:
        close_counter = max(examples, key=lambda e,
                            i=i: cluster_score(e, i))
        # check if actually is (since call could have been off)
        if cluster_score(close_counter, i):
            result.append(close_counter)

    # trim, in case we had too many cluster
    result = sorted(result, key=lambda v: v.similarity *
                    cond(v), reverse=True)[:nmols]

    # fill in remaining
    ncount = sum([cond(e) for e in result])
    fill = max(0, nmols - ncount)
    result.extend(sorted(examples, key=lambda v: v.similarity * cond(v),
                         reverse=True)[:fill])

    return result


def counterfactual_explain(examples, nmols=3):

    def is_counter(e):
        return e.yhat != examples[0].yhat

    result = _select_examples(is_counter, examples[1:], nmols)
    for r in result:
        r.label = 'Counterfactual'

    return examples[:1] + result


def regression_explain(examples, delta=(-1, 1), nmols=4):
    if type(delta) is float:
        delta = (-delta, delta)

    def is_high(e):
        return e.yhat + delta[0] >= examples[0].yhat

    def is_low(e):
        return e.yhat + delta[1] <= examples[0].yhat

    hresult = [] if delta[0] is None else _select_examples(
        is_high, examples[1:], nmols // 2)
    for h in hresult:
        h.label = 'Increase'
    lresult = [] if delta[1] is None else _select_examples(
        is_low, examples[1:], nmols // 2)
    for h in lresult:
        h.label = 'Decrease'
    return examples[:1] + lresult + hresult


def _mol_images(exps, mol_size):
    # get aligned images
    ms = [smi2mol(e.smiles) for e in exps]
    dos = rdkit.Chem.Draw.MolDrawOptions()
    dos.useBWAtomPalette()
    rdkit.Chem.AllChem.Compute2DCoords(ms[0])
    imgs = []
    for m in ms[1:]:
        rdkit.Chem.AllChem.GenerateDepictionMatching2DStructure(
            m, ms[0], acceptFailure=True)
        aidx, bidx = moldiff(ms[0], m)
        imgs.append(mol2img(m, size=mol_size, options=dos,
                            highlightAtoms=aidx, highlightBonds=bidx))
    rdkit.Chem.AllChem.GenerateDepictionMatching2DStructure(
        ms[0], ms[1], acceptFailure=True)
    imgs.insert(0, mol2img(ms[0], size=mol_size, options=dos))
    return imgs


def plot_space(examples, exps, figure_kwargs=None, mol_size=(200, 200)):
    imgs = _mol_images(exps, mol_size)
    if figure_kwargs is None:
        figure_kwargs = {'figsize': (12, 8)}
    base_color = 'gray'
    plt.figure(**figure_kwargs)
    yhats = [e.yhat for e in examples]
    normalizer = plt.Normalize(min(yhats), max(yhats))
    plt.scatter(
        [e.position[0] for e in examples],
        [e.position[1] for e in examples],
        c=normalizer(yhats), cmap='viridis',
        alpha=0.5, edgecolors='none')
    plt.scatter(
        [e.position[0] for e in exps],
        [e.position[1] for e in exps],
        c=normalizer([e.yhat for e in exps]), cmap='viridis',
        edgecolors='black')

    x = [e.position[0] for e in exps]
    y = [e.position[1] for e in exps]
    titles = []
    colors = []
    for e in exps:
        if not e.is_origin:
            titles.append(f'Similarity = {e.similarity:.2f}\n{e.label}')
            colors.append(base_color)
        else:
            titles.append('Base')
            colors.append(base_color)
    _image_scatter(x, y, imgs, titles, colors, plt.gca())
    plt.axis('off')
    plt.gca().set_aspect('auto')


def _nearest_spiral_layout(x, y):
    # make spiral
    angles = np.linspace(-np.pi, np.pi, len(x) + 1)
    coords = np.stack((np.cos(angles), np.sin(angles)), -1)
    order = np.argsort(np.arctan2(y, x))
    return coords[order]


def _image_scatter(x, y, imgs, subtitles, colors, ax):
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea, VPacker

    box_coords = _nearest_spiral_layout(x, y)
    bbs = []
    for i, (x0, y0, im, t, c) in enumerate(zip(x, y, imgs, subtitles, colors)):
        # add transparency
        im = trim(im)
        img_data = np.asarray(im)
        img_box = OffsetImage(img_data)
        title_box = TextArea(t)
        packed = VPacker(children=[img_box, title_box],
                         pad=0, sep=4, align='center')
        bb = AnnotationBbox(
            packed, (x0, y0),
            alpha=0.0, frameon=True, xybox=box_coords[i] + 0.5,
            arrowprops=dict(arrowstyle="->", edgecolor='black'), pad=0.3,
            boxcoords='axes fraction',
            bboxprops=dict(edgecolor=c))
        ax.add_artist(bb)
        bbs.append(bb)
    return bbs


def plot_explanation(exps, figure_kwargs=None, mol_size=(200, 200)):
    imgs = _mol_images(exps, mol_size)
    if figure_kwargs is None:
        figure_kwargs = {'figsize': (12, 8)}
    C = math.ceil(math.sqrt(len(imgs)))
    R = math.ceil(len(imgs) / C)
    fig, axs = plt.subplots(R, C, **figure_kwargs)
    axs = axs.flatten()
    for i, (img, e) in enumerate(zip(imgs, exps)):
        title = 'Base' if e.is_origin else f'Similarity = {e.similarity:.2f}\n{e.label}'
        title += f'\nf(x) = {e.yhat:.3f}'
        axs[i].set_title(title)
        axs[i].imshow(np.asarray(img))
        axs[i].axis('off')
    for j in range(i, C * R):
        axs[j].axis('off')
        axs[j].set_facecolor('white')
    plt.tight_layout()


def moldiff(template, query):
    r = MCS.FindMCS([template, query])
    substructure = rdkit.Chem.MolFromSmarts(r.smartsString)
    raw_match = query.GetSubstructMatches(substructure)
    # flatten it
    match = list(itertools.chain(*raw_match))
    # need to invert match to get diffs
    inv_match = [i for i in range(
        query.GetNumAtoms()) if i not in match]
    # get bonds
    bond_match = []
    for b in query.GetBonds():
        if b.GetBeginAtomIdx() in inv_match or b.GetEndAtomIdx() in inv_match:
            bond_match.append(b.GetIdx())
    return inv_match, bond_match


def trim(im):
    from PIL import Image, ImageChops
    # https://stackoverflow.com/a/10616717
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
