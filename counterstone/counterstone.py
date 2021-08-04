from counterstone.stoned.stoned import get_fingerprint
import itertools
from typing import Type
from . import stoned
from rdkit.Chem import MolFromSmiles as smi2mol
import selfies
from sklearn.cluster import AgglomerativeClustering
import numpy as np


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
        def stop_callback(x): return False

    mol = smi2mol(s)
    if mol == None:
        raise Exception('Invalid starting structure encountered')

    randomized_smile_orderings = [stoned.randomize_smiles(
        mol) for _ in range(num_samples)]

    # Convert all the molecules to SELFIES
    selfies_ls = [selfies.encoder(x) for x in randomized_smile_orderings]

    all_smiles_collect = []
    for num_mutations in num_mutation_ls:
        # Mutate the SELFIES:
        selfies_mut = stoned.get_mutated_SELFIES(
            selfies_ls.copy(), num_mutations=num_mutations)
        # Convert back to SMILES:
        smiles_back = [selfies.decoder(x) for x in selfies_mut]
        all_smiles_collect = all_smiles_collect + smiles_back
        if stop_callback(smiles_back):
            break

    # Work on:  all_smiles_collect
    canon_smi_ls = []
    for item in all_smiles_collect:
        mol, smi_canon, did_convert = stoned.sanitize_smiles(item)
        if mol == None or smi_canon == '' or did_convert == False:
            raise Exception('Invalid smile string found')
        canon_smi_ls.append(smi_canon)
    canon_smi_ls = list(set(canon_smi_ls))

    canon_smi_ls_scores = stoned.get_fp_scores(
        canon_smi_ls, target_smi=s, fp_type=fp_type)
    return canon_smi_ls, canon_smi_ls_scores


def explain(smi, f, batched=True, top_k=10,  cluster=True, stoned_kwargs=None):
    batched_f = f
    if not batched:
        def batched_f(s): return [f(si) for si in s]
    if stoned_kwargs is None:
        stoned_kwargs = {}

    def callback(s):
        try:
            complete = sum(batched_f(s))
        except TypeError as e:
            print('Maybe you forgot to indicate your function is not batched')
            raise e
        return complete
    stoned_kwargs['stop_callback'] = callback
    smiles, scores = run_stoned(smi, **stoned_kwargs)
    switched = batched_f(smiles)
    if not sum(switched):
        raise ValueError(
            'Failed to find counterfactual. Try adjusting stoned_kwargs')
    # reduce to subset
    smiles = [s for s, l in zip(smiles, switched) if l]
    scores = [s for s, l in zip(scores, switched) if l]
    if cluster and len(smiles) >= top_k:
        # compute distance matrix
        dmat = _fp_dist_matrix(smiles,
                               stoned_kwargs['fp_type'] if ('fp_type' in stoned_kwargs) else 'ECFP4')

        # do clustering
        clustering = AgglomerativeClustering(
            n_clusters=top_k, affinity='precomputed', linkage='complete').fit(dmat)
        # get highest in each label
        result = []
        for i in range(top_k):
            ci = [(sm, s) for i, (sm, s) in enumerate(
                zip(smiles, scores)) if clustering.labels_[i]]
            result.append(sorted(ci, key=lambda k: k[1])[-1])
    else:
        result = [(sm, s) for sm, s in zip(smiles, scores)]
        result = sorted(result, key=lambda v: v[1], reverse=True)[:top_k]
    return result
