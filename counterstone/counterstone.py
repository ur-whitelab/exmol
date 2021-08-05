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


@dataclass
class Explanation:
    smiles: str
    selfies: str
    similarity: float
    position: np.ndarray = None
    is_counter: bool = True
    is_base: bool = False


def _fp_dist_matrix(smiles, fp_type='ECFP4'):
    mols = [smi2mol(s) for s in smiles]
    fp = [stoned.get_fingerprint(m, fp_type) for m in mols]
    # 1 - Ts because we want distance
    dist = list(1 - stoned.TanimotoSimilarity(x, y)
                for x, y in itertools.product(fp, repeat=2))
    return np.array(dist).reshape(len(mols), len(mols))


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
    print('NUMBER OF UNIQUE SMILES', len(canon_smi_ls))

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

    # PROCESSING COUNTERFACTUALS
    # reduce to subset
    c_smiles = [s for s, l in zip(smiles, switched) if l]
    c_scores = [s for s, l in zip(scores, switched) if l]
    c_selfies = [s for s, l in zip(selfies, switched) if l]
    if cluster and len(c_smiles) >= max_k:
        # compute distance matrix
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
        print(clustering.labels_)
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
    # nc_scores = [s for ]

    # apply final filter
    result = [r for r in result if r.similarity > min_similarity]
    # add base smiles to output
    return [Explanation(smi, sf.decoder(smi), 1.0, np.array([0, 0]), is_base=True)] + result


def plot_explanation(exps, figure_kwargs=None):
    if exps[-1].position is None:
        _grid_plot_explanation(exps, figure_kwargs)
    else:
        _project_plot_explanation(exps, figure_kwargs)
    


def _project_plot_explanation(exps, figure_kwargs=None):
    import matplotlib.pyplot as plt
    imgs = [mol2img(smi2mol(e.smiles), size=(150, 100)) for e in exps]
    if figure_kwargs is None:
        figure_kwargs = {'figsize': (12, 8)}
    plt.axes('off')


def _grid_plot_explanation(exps, figure_kwargs=None):
    import matplotlib.pyplot as plt
    imgs = [mol2img(smi2mol(e.smiles), size=(150, 100)) for e in exps]
    if figure_kwargs is None:
        figure_kwargs = {'figsize': (12, 8)}
    C = math.ceil(math.sqrt(len(imgs)))
    R = len(imgs) // C
    fig, axs = plt.subplots(R, C, **figure_kwargs)
    axs = axs.flatten()
    for i, (img, e) in enumerate(zip(imgs, exps)):
        axs[i].set_title('Base' if e.is_base else f'{e.similarity:.2f}')
        axs[i].imshow(np.asarray(img))
        axs[i].axis('off')

    plt.tight_layout()
