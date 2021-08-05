from dataclasses import dataclass
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import selfies as sf
from counterstone.stoned.stoned import get_fingerprint
import itertools
import math
from typing import Type
from . import stoned
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem.Draw import MolToImage as mol2img
import rdkit.Chem


@dataclass
class Explanation:
    smiles: str
    selfies: str
    similarity: float
    position: np.ndarray = None
    is_counter: bool = True
    is_base: bool = False


def trim(im):
    from PIL import Image, ImageChops
    # https://stackoverflow.com/a/10616717
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def _fp_dist_matrix(smiles, fp_type='ECFP4'):
    mols = [smi2mol(s) for s in smiles]
    fp = [stoned.get_fingerprint(m, fp_type) for m in mols]
    # 1 - Ts because we want distance
    dist = list(1 - stoned.TanimotoSimilarity(x, y)
                for x, y in itertools.product(fp, repeat=2))
    return np.array(dist).reshape(len(mols), len(mols))


def _draw_svg(smi, size=(400, 200)):
    m = smi2mol(smi)
    rdkit.Chem.rdDepictor.Compute2DCoords(m)

    drawer = rdkit.Chem.Draw.rdMolDraw2D.MolDraw2DSVG(*size)
    drawer.drawOptions().bgColor = None
    drawer.DrawMolecule(m)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:', '')
    return svg


def run_stoned(
        s, fp_type='ECFP4', num_samples=1000,
        max_mutations=3, stop_callback=None, fp_matrix=False):
    '''Run ths STONED SELFIES algorithm
    '''
    num_mutation_ls = list(range(1, max_mutations + 1))
    if stop_callback is None:
        def stop_callback(x, y): return False

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
            selfies_ls.copy(), num_mutations=num_mutations)
        # Convert back to SMILES:
        smiles_back = [sf.decoder(x) for x in selfies_mut]
        all_smiles_collect = all_smiles_collect + smiles_back
        all_selfies_collect = all_selfies_collect + selfies_mut
        if stop_callback(smiles_back, selfies_mut):
            break

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


def explain(smi, f, batched=True, max_k=10,  cluster=True, stoned_kwargs=None, min_similarity=0.5):

    # ARGUMENT PROCESSING
    batched_f = f
    if not batched:
        def batched_f(sm, se): return [f(smi, sei) for smi, sei in zip(sm, se)]
    if stoned_kwargs is None:
        stoned_kwargs = {}

    def callback(sm, se):
        try:
            complete = sum(batched_f(sm, se))
        except TypeError as e:
            print('Maybe you forgot to indicate your function is not batched')
            raise e
        return complete
    stoned_kwargs['stop_callback'] = callback

    # STONED
    smiles, scores = run_stoned(smi, **stoned_kwargs)
    selfies = [sf.decoder(s) for s in smiles]
    switched = batched_f(smiles, selfies)
    if not sum(switched):
        raise ValueError(
            'Failed to find counterfactual. Try adjusting stoned_kwargs')
    max_k = min(max_k, sum(switched))

    # GET PROJECTED COORDINATES
    if cluster:
        # compute distance matrix
        from sklearn.decomposition import PCA
        full_dmat = _fp_dist_matrix(smiles,
                                    stoned_kwargs['fp_type'] if ('fp_type' in stoned_kwargs) else 'ECFP4')
        # compute positions
        pca = PCA(n_components=2)
        proj_dmat = pca.fit_transform(full_dmat)

    # PROCESSING COUNTERFACTUALS
    # reduce to subset
    c_smiles = [s for s, l in zip(smiles, switched) if l]
    c_scores = [s for s, l in zip(scores, switched) if l]
    c_selfies = [s for s, l in zip(selfies, switched) if l]
    if cluster and len(c_smiles) >= max_k:
        # compute distance matrix Again
        # TODO: maybe do not do twice?
        from sklearn.decomposition import PCA
        dmat = _fp_dist_matrix(c_smiles,
                               stoned_kwargs['fp_type'] if ('fp_type' in stoned_kwargs) else 'ECFP4')
        # compute positions
        pca = PCA(n_components=2)
        proj_dmat = pca.fit_transform(dmat)

        # do clustering
        clustering = AgglomerativeClustering(
            n_clusters=max_k, affinity='precomputed', linkage='complete').fit(dmat)
        # get highest in each label
        result = []
        for i in range(max_k):
            ci = [Explanation(sm, se, s, proj_dmat[j, :]) for j, (sm, se, s) in enumerate(
                zip(c_smiles, c_selfies, c_scores)) if clustering.labels_[j] == i]
            result.append(sorted(ci, key=lambda k: k.similarity)[-1])
    else:
        result = [Explanation(sm, se, s)
                  for sm, se, s in zip(c_smiles, c_selfies, c_scores)]
        result = sorted(result, key=lambda v: v.similarity,
                        reverse=True)[:max_k]

    # PROCESSING NEARBY NON-COUNTERFACTUALS
    # TODO
    #  nc_scores = [s for]

    # apply final filter
    result = [r for r in result if r.similarity > min_similarity]
    # add base smiles to output
    return [Explanation(smi, sf.decoder(smi), 1.0, np.array([0, 0]), is_base=True)] + result


def plot_explanation(exps, figure_kwargs=None):
    # get aligned images
    ms = [smi2mol(e.smiles) for e in exps]
    rdkit.Chem.AllChem.Compute2DCoords(ms[0])
    for m in ms[1:]:
        rdkit.Chem.AllChem.GenerateDepictionMatching2DStructure(
            m, ms[0], acceptFailure=True)
    if exps[-1].position is None:
        _grid_plot_explanation(exps, ms, figure_kwargs)
    else:
        _project_plot_explanation(exps, ms, figure_kwargs)


def _project_plot_explanation(exps, mols, figure_kwargs=None, mol_size=(200, 200)):
    import matplotlib.pyplot as plt
    if figure_kwargs is None:
        figure_kwargs = {'figsize': (12, 8)}
    plt.figure(**figure_kwargs)
    imgs = [mol2img(m, size=mol_size) for m in mols]
    clabel = False
    plabel = False
    for i, e in enumerate(exps[1:]):
        if e.is_counter:
            plt.plot([0, e.position[0]], [0, e.position[1]],
                     marker='o', color='C0', label='Counterfactual' if not clabel else None)
            clabel = True
        else:
            plt.plot([0, e.position[0]], [0, e.position[1]],
                     marker='o', color='C1', label='Counterfactual' if not plabel else None)
            plabel = True
    plt.gca().set_facecolor('white')
    plt.axis('off')
    plt.gca().set_aspect('equal')
    x = [e.position[0] for e in exps]
    y = [e.position[1] for e in exps]
    titles = [f'Similarity = {e.similarity:.2f}' for e in exps]
    _image_scatter(x, y, imgs, titles, plt.gca())
    plt.legend()


def _image_scatter(x, y, imgs, subtitles, ax):
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea, VPacker

    ax.scatter(x, y)

    for x0, y0, i, t in zip(x, y, imgs, subtitles):
        # add transparency
        i = trim(i)
        img_data = np.asarray(i)
        img_box = OffsetImage(img_data)
        title_box = TextArea(t)
        packed = VPacker(children=[img_box, title_box],
                         pad=0, sep=4, align='center')
        bb = AnnotationBbox(packed,
                            (x0, y0),  alpha=0.0, frameon=True)
        ax.add_artist(bb)


def _grid_plot_explanation(exps, mols, figure_kwargs=None):
    import matplotlib.pyplot as plt
    imgs = [mol2img(m, size=(150, 100)) for m in mols]
    if figure_kwargs is None:
        figure_kwargs = {'figsize': (12, 8)}
    C = math.ceil(math.sqrt(len(imgs)))
    R = math.ceil(len(imgs) / C)
    fig, axs = plt.subplots(R, C, **figure_kwargs)
    axs = axs.flatten()
    for i, (img, e) in enumerate(zip(imgs, exps)):
        axs[i].set_title('Base' if e.is_base else f'{e.similarity:.2f}')
        axs[i].imshow(np.asarray(img))
        axs[i].axis('off')

    plt.tight_layout()
