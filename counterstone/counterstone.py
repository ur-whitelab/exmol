from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import selfies as sf
import itertools
import math
from . import stoned
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem.Draw import MolToImage as mol2img
import rdkit.Chem


@dataclass
class Explanation:
    smiles: str
    selfies: str
    similarity: float
    index: int
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


def run_stoned(
        s, fp_type='ECFP4', num_samples=2000,
        max_mutations=2, stop_callback=None, fp_matrix=False):
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
        print('Round Complete with', len(smiles_back))
        # TODO: GEt rid of callback -- to slow and we do not actually use it
        # OR start using it
        # if stop_callback(smiles_back, selfies_mut):
        #    break

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


def explain(smi, f, batched=True, max_k=3, preset='medium', cluster=True, stoned_kwargs=None, min_similarity=0.5):

    # ARGUMENT PROCESSING
    batched_f = f
    if not batched:
        def batched_f(sm, se): return [f(smi, sei) for smi, sei in zip(sm, se)]
    if stoned_kwargs is None:
        stoned_kwargs = {}
        if preset is 'medium':
            stoned_kwargs['num_samples'] = 1500
            stoned_kwargs['max_mutations'] = 2
        elif preset is 'narrow':
            stoned_kwargs['num_samples'] = 3000
            stoned_kwargs['max_mutations'] = 1
        elif preset is 'wide':
            stoned_kwargs['num_samples'] = 600
            stoned_kwargs['max_mutations'] = 5
        else:
            raise ValueError(f'Unknown preset "{preset}"')

    def callback(sm, se):
        try:
            complete = sum(batched_f(sm, se)) > max_k
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
    print('Adjusting max_k to', max_k)
    # only cluster if 3
    cluster = cluster and max_k > 2

    # pack them into data structure with filtering
    exps = [
        Explanation(sm, se, s, index=0, is_counter=l) for i, (sm, se, s, l) in
        enumerate(zip(smiles, selfies, scores, switched)) if s < 1.0
    ]
    # add 1 to leave space for base
    for i, e in enumerate(exps):
        e.index = i + 1

    print('Starting with', len(exps), 'explanations')
    result = []
    # GET PROJECTED COORDINATES
    if cluster:
        # compute distance matrix
        full_dmat = _fp_dist_matrix([smi] + [e.smiles for e in exps],
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
        clustering = KMeans(n_clusters=max_k).fit(proj_dmat)

        # get highest in each label using helper function
        def cluster_score(e, i, c):
            return (clustering.labels_[e.index] == i) * (e.is_counter == c) * e.similarity
        for i in range(max_k):
            close_counter = max(exps, key=lambda e,
                                i=i: cluster_score(e, i, True))
            close_same = close_counter = max(
                exps, key=lambda e, i=i: cluster_score(e, i, False))
            # check if actually is (since call could have been off)
            # when inserting delete from exps for purposes of final step
            if cluster_score(close_counter, i, True):
                result.append(close_counter)
                del exps[exps.index(close_counter)]
            if cluster_score(close_same, i, False):
                result.append(close_same)
                del exps[exps.index(close_same)]
    # fill in remaining
    ncount = sum([e.is_counter for e in result])
    fill = max(0, max_k - ncount)
    result.extend(sorted(exps, key=lambda v: v.similarity * v.is_counter,
                         reverse=True)[:fill])
    fill = max(0, max_k - (len(result) - ncount))
    result.extend(sorted(exps, key=lambda v: v.similarity * (~v.is_counter),
                         reverse=True)[:fill])
    # remove from original array so we do not get duplicates
    for r in result:
        if r in exps:
            del exps[exps.index(r)]
    # sort to have counterfactuals first
    result = sorted(result, key=lambda v: v.similarity +
                    max_k * v.is_counter, reverse=True)
    # add base smiles to output
    return [Explanation(smi, sf.decoder(smi), 1.0, index=0, position=proj_dmat[0, :] if cluster else None, is_base=True)] + result, exps


def plot_explanation(exps, space=None, show_para=False, figure_kwargs=None):
    # optionally filter out para
    if not show_para:
        exps = [e for e in exps if e.is_counter or e.is_base]
    # get aligned images
    ms = [smi2mol(e.smiles) for e in exps]
    rdkit.Chem.AllChem.Compute2DCoords(ms[0])
    for m in ms[1:]:
        rdkit.Chem.AllChem.GenerateDepictionMatching2DStructure(
            m, ms[0], acceptFailure=True)
    if space is None:
        _grid_plot_explanation(exps, ms, figure_kwargs)
    else:
        _project_plot_explanation(exps, space, ms, figure_kwargs)


def _project_plot_explanation(exps, space, mols, figure_kwargs=None, mol_size=(200, 200)):
    import matplotlib.pyplot as plt
    if figure_kwargs is None:
        figure_kwargs = {'figsize': (12, 8)}
    base_color = 'gray'
    plt.figure(**figure_kwargs)
    imgs = [mol2img(m, size=mol_size) for m in mols]
    plt.scatter([], [], label='Counterfactual')
    plt.scatter([], [], label='Parafactual')
    plt.scatter(
        [e.position[0] for e in space],
        [e.position[1] for e in space],
        c=['C0' if e.is_counter else 'C1' for e in space],
        alpha=0.5, edgecolors='none')
    plt.scatter(
        [e.position[0] for e in exps[1:]],
        [e.position[1] for e in exps[1:]],
        c=['C0' if e.is_counter else 'C1' for e in exps[1:]],
        alpha=1.0)
    plt.scatter(*exps[0].position, color=base_color)
    plt.gca().set_facecolor('white')
    plt.axis('off')
    plt.gca().set_aspect('equal')
    x = [e.position[0] for e in exps]
    y = [e.position[1] for e in exps]
    titles = []
    colors = []
    for e in exps:
        if not e.is_base:
            titles.append(f'Similarity = {e.similarity:.2f}')
            colors.append('C0' if e.is_counter else 'C1')
        else:
            titles.append('Base')
            colors.append(base_color)
    _image_scatter(x, y, imgs, titles, colors, plt.gca())
    plt.legend()
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


def _grid_plot_explanation(exps, mols, figure_kwargs=None):
    import matplotlib.pyplot as plt
    imgs = [mol2img(m, size=(250, 200)) for m in mols]
    if figure_kwargs is None:
        figure_kwargs = {'figsize': (12, 8)}
    C = math.ceil(math.sqrt(len(imgs)))
    R = math.ceil(len(imgs) / C)
    fig, axs = plt.subplots(R, C, **figure_kwargs)
    axs = axs.flatten()
    for i, (img, e) in enumerate(zip(imgs, exps)):
        title = 'Base' if e.is_base else f'Similarity = {e.similarity:.2f}'
        if not e.is_base:
            title += '\nCounterfactual' if e.is_counter else '\nParafactual'
        axs[i].set_title(title)
        axs[i].imshow(np.asarray(img))
        axs[i].axis('off')
    for j in range(i, C * R):
        axs[j].axis('off')
        axs[j].set_facecolor('white')
    plt.tight_layout()
