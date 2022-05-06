from typing import *

import itertools
import math
from xml.sax.handler import feature_external_ges
import requests  # type: ignore
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.patches import Rectangle, FancyBboxPatch  # type: ignore
from matplotlib.offsetbox import AnnotationBbox  # type: ignore
import matplotlib as mpl  # type: ignore
import selfies as sf  # type: ignore
import tqdm  # type: ignore
import textwrap  # type: ignore
import skunk  # type: ignore

from ratelimit import limits, sleep_and_retry  # type: ignore
from sklearn.cluster import DBSCAN  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
import scipy.stats as ss  # type: ignore
from rdkit.Chem import MolFromSmiles as smi2mol  # type: ignore
from rdkit.Chem import MolFromSmarts  # type: ignore
from rdkit.Chem import MolToSmiles as mol2smi  # type: ignore
from rdkit.Chem import rdchem, MACCSkeys, AllChem  # type: ignore
from rdkit.Chem.Draw import MolToImage as mol2img, DrawMorganBit  # type: ignore
from rdkit.Chem import rdchem  # type: ignore
from rdkit.Chem import rdFMCS as MCS  # type: ignore
from rdkit import DataStructs  # type: ignore


from . import stoned
from .plot_utils import _mol_images, _image_scatter
from .data import *


def _fp_dist_matrix(smiles, fp_type, _pbar):
    mols = [(smi2mol(s), _pbar.update(0.5))[0] for s in smiles]
    # Sorry about the one-line. Just sneaky insertion of progressbar update
    fp = [(stoned.get_fingerprint(m, fp_type), _pbar.update(0.5))[0] for m in mols]
    M = np.array([DataStructs.BulkTanimotoSimilarity(f, fp) for f in fp])
    # 1 - similarity because we want distance
    return 1 - M


def _calculate_rdkit_descriptors(mol):
    from rdkit.ML.Descriptors import MoleculeDescriptors  # type: ignore

    dlist = [
        "NumHDonors",
        "NumHAcceptors",
        "MolLogP",
        "NumHeteroatoms",
        "RingCount",
        "NumRotatableBonds",
    ]  # , 'NumHeteroatoms']
    c = MoleculeDescriptors.MolecularDescriptorCalculator(dlist)
    d = c.CalcDescriptors(mol)

    def calc_aromatic_bonds(mol):
        return sum(1 for b in mol.GetBonds() if b.GetIsAromatic())

    def _create_smarts(SMARTS):
        s = ",".join("$(" + s + ")" for s in SMARTS)
        _mol = MolFromSmarts("[" + s + "]")
        return _mol

    def calc_acid_groups(mol):
        acid_smarts = (
            "[O;H1]-[C,S,P]=O",
            "[*;-;!$(*~[*;+])]",
            "[NH](S(=O)=O)C(F)(F)F",
            "n1nnnc1",
        )
        pat = _create_smarts(acid_smarts)
        return len(mol.GetSubstructMatches(pat))

    def calc_basic_groups(mol):
        basic_smarts = (
            "[NH2]-[CX4]",
            "[NH](-[CX4])-[CX4]",
            "N(-[CX4])(-[CX4])-[CX4]",
            "[*;+;!$(*~[*;-])]",
            "N=C-N",
            "N-C=N",
        )
        pat = _create_smarts(basic_smarts)
        return len(mol.GetSubstructMatches(pat))

    def calc_apol(mol, includeImplicitHs=True):
        # atomic polarizabilities available here:
        # https://github.com/mordred-descriptor/mordred/blob/develop/mordred/data/polarizalibity78.txt
        from importlib_resources import files  # type: ignore
        import exmol.lime_data  # type: ignore

        ap = files(exmol.lime_data).joinpath("atom_pols.txt")
        with open(ap, "r") as f:
            atom_pols = [float(x) for x in next(f).split(",")]
        res = 0.0
        for atom in mol.GetAtoms():
            anum = atom.GetAtomicNum()
            if anum <= len(atom_pols):
                apol = atom_pols[anum]
                if includeImplicitHs:
                    apol += atom_pols[1] * atom.GetTotalNumHs(includeNeighbors=False)
                res += apol
            else:
                raise ValueError(f"atomic number {anum} not found")
        return res

    d = d + (
        calc_aromatic_bonds(mol),
        calc_acid_groups(mol),
        calc_basic_groups(mol),
        calc_apol(mol),
    )
    return d


def add_descriptors(
    examples: List[Example], descriptor_type: str = "MACCS", mols: List[Any] = None
) -> List[Example]:
    """Add descriptors to passed examples

    :param examples: List of example
    :param descriptor_type: Kind of descriptors to return, choose between 'Classic', 'ECFP', or 'MACCS'. Default is 'MACCS'.
    :param mols: Can be used if you already have rdkit Mols computed.
    :return: List of examples with added descriptors
    """
    from importlib_resources import files
    import exmol.lime_data

    if mols is None:
        mols = [smi2mol(m.smiles) for m in examples]
    if descriptor_type == "Classic":
        names = tuple(
            [
                "number of hydrogen bond donor",
                "number of hydrogen bond acceptor",
                "Wildman-Crippen LogP",
                "number of heteroatoms",
                "ring count",
                "number of rotatable bonds",
                "aromatic bonds count",
                "acidic group count",
                "basic group count",
                "atomic polarizability",
            ]
        )
        for e, m in zip(examples, mols):
            descriptors = _calculate_rdkit_descriptors(m)
            descriptor_names = names
            e.descriptors = Descriptors(
                descriptor_type=descriptor_type,
                descriptors=descriptors,
                descriptor_names=descriptor_names,
            )
        return examples
    elif descriptor_type == "MACCS":
        mk = files(exmol.lime_data).joinpath("MACCSkeys.txt")
        with open(str(mk), "r") as f:
            names = tuple([x.strip().split("\t")[-1] for x in f.readlines()[1:]])
        for e, m in zip(examples, mols):
            # rdkit sets fps[0] to 0 and starts keys at 1!
            fps = list(MACCSkeys.GenMACCSKeys(m).ToBitString())
            descriptors = tuple(int(i) for i in fps)
            descriptor_names = names
            e.descriptors = Descriptors(
                descriptor_type=descriptor_type,
                descriptors=descriptors,
                descriptor_names=descriptor_names,
            )
        return examples
    elif descriptor_type == "ECFP":
        # get reference
        bi = {}  # type: Dict[Any, Any]
        ref_fp = AllChem.GetMorganFingerprint(mols[0], 3, bitInfo=bi)
        descriptor_names = tuple(bi.keys())
        for e, m in zip(examples, mols):
            # Now compare to reference and get other fp vectors
            b = {}  # type: Dict[Any, Any]
            temp_fp = AllChem.GetMorganFingerprint(m, 3, bitInfo=b)
            descriptors = tuple([1 if x in b.keys() else 0 for x in descriptor_names])
            e.descriptors = Descriptors(
                descriptor_type=descriptor_type,
                descriptors=descriptors,
                descriptor_names=descriptor_names,
            )
        return examples
    else:
        raise ValueError(
            "Invalid descriptor string. Valid descriptor strings are 'Classic', 'ECFP', or 'MACCS'."
        )


def get_basic_alphabet() -> Set[str]:
    """Returns set of interpretable SELFIES tokens

    Generated by removing P and most ionization states from :func:`selfies.get_semantic_robust_alphabet`

    :return: Set of interpretable SELFIES tokens
    """
    a = sf.get_semantic_robust_alphabet()
    # remove cations/anions except oxygen anion
    to_remove = []
    for ai in a:
        if "+1" in ai:
            to_remove.append(ai)
        elif "-1" in ai:
            to_remove.append(ai)
    # remove [P],[#P],[=P]
    to_remove.extend(["[P]", "[#P]", "[=P]"])

    a -= set(to_remove)
    a.add("[O-1]")
    return a


def run_stoned(
    s: str,
    fp_type: str = "ECFP4",
    num_samples: int = 2000,
    max_mutations: int = 2,
    min_mutations: int = 1,
    alphabet: Union[List[str], Set[str]] = None,
    _pbar: Any = None,
) -> Tuple[List[str], List[float]]:
    """Run ths STONED SELFIES algorithm. Typically not used, call :func:`sample_space` instead.

    :param s: SMILES string to start from
    :param fp_type: Fingerprint type
    :param num_samples: Number of total molecules to generate
    :param max_mutations: Maximum number of mutations
    :param min_mutations: Minimum number of mutations
    :param alphabet: Alphabet to use for mutations, typically from :func:`get_basic_alphabet()`
    :return: SMILES and SCORES generated
    """
    if alphabet is None:
        alphabet = list(sf.get_semantic_robust_alphabet())
    if type(alphabet) == set:
        alphabet = list(alphabet)
    num_mutation_ls = list(range(min_mutations, max_mutations + 1))

    mol = smi2mol(s)
    if mol == None:
        raise Exception("Invalid starting structure encountered")

    # want it so after sampling have num_samples
    randomized_smile_orderings = [
        stoned.randomize_smiles(mol) for _ in range(num_samples // len(num_mutation_ls))
    ]

    # Convert all the molecules to SELFIES
    selfies_ls = [sf.encoder(x) for x in randomized_smile_orderings]

    all_smiles_collect: List[str] = []
    all_selfies_collect: List[str] = []
    for num_mutations in num_mutation_ls:
        # Mutate the SELFIES:
        if _pbar:
            _pbar.set_description(f"🥌STONED🥌 Mutations: {num_mutations}")
        selfies_mut = stoned.get_mutated_SELFIES(
            selfies_ls.copy(), num_mutations=num_mutations, alphabet=alphabet
        )
        # Convert back to SMILES:
        smiles_back = [sf.decoder(x) for x in selfies_mut]
        all_smiles_collect = all_smiles_collect + smiles_back
        all_selfies_collect = all_selfies_collect + selfies_mut
        if _pbar:
            _pbar.update(len(smiles_back))

    # Work on:  all_smiles_collect
    if _pbar:
        _pbar.set_description(f"🥌STONED🥌 Done")
    canon_smi_ls = []
    for item in all_smiles_collect:
        mol, smi_canon, did_convert = stoned.sanitize_smiles(item)
        if mol == None or smi_canon == "" or did_convert == False:
            raise Exception("Invalid smiles string found")
        canon_smi_ls.append(smi_canon)

    # remove redundant/non-unique/duplicates
    # in a way to keep the selfies
    canon_smi_ls = list(set(canon_smi_ls))

    canon_smi_ls_scores = stoned.get_fp_scores(
        canon_smi_ls, target_smi=s, fp_type=fp_type
    )
    # NOTE Do not think of returning selfies. They have duplicates
    return canon_smi_ls, canon_smi_ls_scores


@sleep_and_retry
@limits(calls=2, period=30)
def run_chemed(
    origin_smiles: str,
    num_samples: int,
    similarity: float = 0.1,
    fp_type: str = "ECFP4",
    _pbar: Any = None,
) -> Tuple[List[str], List[float]]:
    """
    This method is similar to STONED but works by quering PubChem

    :param origin_smiles: Base SMILES
    :param num_samples: Minimum number of returned molecules. May return less due to network timeout or exhausting tree
    :param similarity: Tanimoto similarity to use in query (float between 0 to 1)
    :param fp_type: Fingerprint type
    :return: SMILES and SCORES
    """
    if _pbar:
        _pbar.set_description("⚡CHEMED⚡")
    else:
        print("⚡CHEMED⚡")
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/smiles/{requests.utils.quote(origin_smiles)}/property/CanonicalSMILES/JSON"
    try:
        reply = requests.get(
            url,
            params={"Threshold": int(similarity * 100), "MaxRecords": num_samples},
            headers={"accept": "text/json"},
            timeout=10,
        )
    except requests.exceptions.Timeout:
        print("Pubchem seems to be down right now ️☠️☠️")
        return [], []
    try:
        data = reply.json()
    except:
        return [], []
    smiles = [d["CanonicalSMILES"] for d in data["PropertyTable"]["Properties"]]
    smiles = list(set(smiles))

    if _pbar:
        _pbar.set_description(f"Received {len(smiles)} similar molecules")

    mol0 = smi2mol(origin_smiles)
    mols = [smi2mol(s) for s in smiles]
    fp0 = stoned.get_fingerprint(mol0, fp_type)
    scores = []
    # drop Nones
    smiles = [s for s, m in zip(smiles, mols) if m is not None]
    for m in mols:
        if m is None:
            continue
        fp = stoned.get_fingerprint(m, fp_type)
        scores.append(stoned.TanimotoSimilarity(fp0, fp))
        if _pbar:
            _pbar.update()
    return smiles, scores


def run_custom(
    origin_smiles: str,
    data: List[Union[str, rdchem.Mol]],
    fp_type: str = "ECFP4",
    _pbar: Any = None,
    **kwargs,
) -> Tuple[List[str], List[float]]:
    """
    This method is similar to STONED but uses a custom dataset provided by the user

    :param origin_smiles: Base SMILES
    :param data: List of SMILES or RDKit molecules
    :param fp_type: Fingerprint type
    :return: SMILES and SCORES
    """
    if _pbar:
        _pbar.set_description("⚡CUSTOM⚡")
    else:
        print("⚡CUSTOM⚡")
    mol0 = smi2mol(origin_smiles)
    fp0 = stoned.get_fingerprint(mol0, fp_type)
    scores = []
    smiles = []
    # drop invalid molecules
    for d in data:
        if isinstance(d, str):
            m = smi2mol(d)
        else:
            m = d
        if m is None:
            continue
        smiles.append(mol2smi(m))
        fp = stoned.get_fingerprint(m, fp_type)
        scores.append(stoned.TanimotoSimilarity(fp0, fp))
        if _pbar:
            _pbar.update()
    return smiles, scores


def sample_space(
    origin_smiles: str,
    f: Union[
        Callable[[str, str], List[float]],
        Callable[[str], List[float]],
        Callable[[List[str], List[str]], List[float]],
        Callable[[List[str]], List[float]],
    ],
    batched: bool = True,
    preset: str = "medium",
    data: List[Union[str, rdchem.Mol]] = None,
    method_kwargs: Dict = None,
    num_samples: int = None,
    stoned_kwargs: Dict = None,
    quiet: bool = False,
    use_selfies: bool = False,
) -> List[Example]:
    """Sample chemical space around given SMILES

    This will evaluate the given function and run the :func:`run_stoned` function over chemical space around molecule. ``num_samples`` will be
    set to 3,000 by default if using STONED and 150 if using ``chemed``.

    :param origin_smiles: starting SMILES
    :param f: A function which takes in SMILES or SELFIES and returns predicted value. Assumed to work with lists of SMILES/SELFIES unless `batched = False`
    :param batched: If `f` is batched
    :param preset: Can be wide, medium, or narrow. Determines how far across chemical space is sampled. Try `"chemed"` preset to only sample commerically available compounds.
    :param data: If not None and preset is `"custom"` will use this data instead of generating new ones.
    :param method_kwargs: More control over STONED, CHEMED and CUSTOM can be set here. See :func:`run_stoned`, :func:`run_chemed` and  :func:`run_custom`
    :param num_samples: Number of desired samples. Can be set in `method_kwargs` (overrides) or here. `None` means default for preset
    :param stoned_kwargs: Backwards compatible alias for `methods_kwargs`
    :param quiet: If True, will not print progress bar
    :param use_selfies: If True, will use SELFIES instead of SMILES for `f`
    :return: List of generated :obj:`Example`
    """

    wrapped_f = f

    # if f only takes in 1 arg, wrap it in a function that takes in 2
    if f.__code__.co_argcount == 1:
        if use_selfies:

            def wrapped_f(sm, sf):
                return f(sf)

        else:

            def wrapped_f(sm, sf):
                return f(sm)

    batched_f: Any = wrapped_f
    if not batched:

        def batched_f(sm, se):
            return np.array([wrapped_f(smi, sei) for smi, sei in zip(sm, se)])

    origin_smiles = stoned.sanitize_smiles(origin_smiles)[1]
    if origin_smiles is None:
        raise ValueError("Given SMILES does not appear to be valid")
    smi_yhat = np.asarray(batched_f([origin_smiles], [sf.encoder(origin_smiles)]))
    try:
        iter(smi_yhat)
    except TypeError:
        raise ValueError("Your model function does not appear to be batched")
    smi_yhat = np.squeeze(smi_yhat[0])

    if stoned_kwargs is not None:
        method_kwargs = stoned_kwargs

    if method_kwargs is None:
        method_kwargs = {}
        if preset == "medium":
            method_kwargs["num_samples"] = 3000 if num_samples is None else num_samples
            method_kwargs["max_mutations"] = 2
            method_kwargs["alphabet"] = get_basic_alphabet()
        elif preset == "narrow":
            method_kwargs["num_samples"] = 3000 if num_samples is None else num_samples
            method_kwargs["max_mutations"] = 1
            method_kwargs["alphabet"] = get_basic_alphabet()
        elif preset == "wide":
            method_kwargs["num_samples"] = 3000 if num_samples is None else num_samples
            method_kwargs["max_mutations"] = 5
            method_kwargs["alphabet"] = sf.get_semantic_robust_alphabet()
        elif preset == "chemed":
            method_kwargs["num_samples"] = 150 if num_samples is None else num_samples
        elif preset == "custom" and data is not None:
            method_kwargs["num_samples"] = len(data)
        else:
            raise ValueError(f'Unknown preset "{preset}"')
    try:
        num_samples = method_kwargs["num_samples"]
    except KeyError as e:
        if num_samples is None:
            num_samples = 150
        method_kwargs["num_samples"] = num_samples

    pbar = tqdm.tqdm(total=num_samples, disable=quiet)

    # STONED
    if preset.startswith("chem"):
        smiles, scores = run_chemed(origin_smiles, _pbar=pbar, **method_kwargs)
    elif preset == "custom":
        smiles, scores = run_custom(
            origin_smiles, data=cast(Any, data), _pbar=pbar, **method_kwargs
        )
    else:
        smiles, scores = run_stoned(origin_smiles, _pbar=pbar, **method_kwargs)
    selfies = [sf.encoder(s) for s in smiles]

    pbar.set_description("😀Calling your model function😀")
    fxn_values = batched_f(smiles, selfies)

    # pack them into data structure with filtering out identical
    # and nan
    exps = [
        Example(
            origin_smiles,
            sf.encoder(origin_smiles),
            1.0,
            cast(Any, smi_yhat),
            index=0,
            is_origin=True,
        )
    ] + [
        Example(sm, se, s, cast(Any, np.squeeze(y)), index=0)
        for i, (sm, se, s, y) in enumerate(zip(smiles, selfies, scores, fxn_values))
        if s < 1.0 and np.isfinite(np.squeeze(y))
    ]

    for i, e in enumerate(exps):  # type: ignore
        e.index = i  # type: ignore

    pbar.reset(len(exps))
    pbar.set_description("🔭Projecting...🔭")

    # compute distance matrix
    full_dmat = _fp_dist_matrix(
        [e.smiles for e in exps],
        method_kwargs["fp_type"] if ("fp_type" in method_kwargs) else "ECFP4",
        _pbar=pbar,
    )

    pbar.set_description("🥰Finishing up🥰")

    # compute PCA
    pca = PCA(n_components=2)
    proj_dmat = pca.fit_transform(full_dmat)
    for e in exps:  # type: ignore
        e.position = proj_dmat[e.index, :]  # type: ignore

    # do clustering everwhere (maybe do counter/same separately?)
    # clustering = AgglomerativeClustering(
    #    n_clusters=max_k, affinity='precomputed', linkage='complete').fit(full_dmat)
    # Just do it on projected so it looks prettier.
    clustering = DBSCAN(eps=0.15, min_samples=5).fit(proj_dmat)

    for i, e in enumerate(exps):  # type: ignore
        e.cluster = clustering.labels_[i]  # type: ignore

    pbar.set_description("🤘Done🤘")
    pbar.close()
    return exps


def _select_examples(cond, examples, nmols):
    result = []

    # similarity filtered by if cluster/counter
    def cluster_score(e, i):
        return (e.cluster == i) * cond(e) * e.similarity

    clusters = set([e.cluster for e in examples])
    for i in clusters:
        close_counter = max(examples, key=lambda e, i=i: cluster_score(e, i))
        # check if actually is (since call could have been zero)
        if cluster_score(close_counter, i):
            result.append(close_counter)

    # trim, in case we had too many cluster
    result = sorted(result, key=lambda v: v.similarity * cond(v), reverse=True)[:nmols]

    # fill in remaining
    ncount = sum([cond(e) for e in result])
    fill = max(0, nmols - ncount)
    result.extend(
        sorted(examples, key=lambda v: v.similarity * cond(v), reverse=True)[:fill]
    )

    return list(filter(cond, result))


def lime_explain(
    examples: List[Example],
    descriptor_type: str,
    return_beta: bool = True,
):
    """From given :obj:`Examples<Example>`, find descriptor t-statistics (see
    :doc: `index`)

    :param examples: Output from :func: `sample_space`
    :param descriptor_type: Desired descriptors, choose from 'Classic', 'ECFP' 'MACCS'
    :return_beta: Whether or not the function should return regression coefficient values
    """
    # add descriptors
    examples = add_descriptors(examples, descriptor_type)
    # weighted tanimoto similarities
    w = np.array([1 / (1 + (1 / (e.similarity + 0.000001) - 1) ** 5) for e in examples])
    # Only keep nonzero weights
    non_zero = w > 10 ** (-6)
    nonzero_w = w[non_zero]
    # create a diagonal matrix of w
    N = nonzero_w.shape[0]
    diag_w = np.zeros((N, N))
    np.fill_diagonal(diag_w, nonzero_w)
    # get feature matrix
    x_mat = np.array([list(e.descriptors.descriptors) for e in examples])[
        non_zero
    ].reshape(N, -1)
    # remove zero variance columns
    y = (
        np.array([e.yhat for e in examples])
        .reshape(len(examples))[non_zero]
        .astype(float)
    )
    # remove bias
    y -= np.mean(y)
    # compute least squares fit
    xtinv = np.linalg.pinv(
        (x_mat.T @ diag_w @ x_mat)
        + 0.001 * np.identity(len(examples[0].descriptors.descriptors))
    )
    beta = xtinv @ x_mat.T @ (y * nonzero_w)
    # compute standard error in beta
    yhat = x_mat @ beta
    resids = yhat - y
    SSR = np.sum(resids**2)
    se2_epsilon = SSR / (len(examples) - len(beta))
    se2_beta = se2_epsilon * xtinv
    # now compute t-statistic for existence of coefficients
    tstat = beta * np.sqrt(1 / np.diag(se2_beta))
    # Set tstats for base, to be used later
    examples[0].descriptors.tstats = tstat
    # Return beta (feature weights) which are the fits if asked for
    if return_beta:
        return beta
    else:
        return None


def cf_explain(examples: List[Example], nmols: int = 3) -> List[Example]:
    """From given :obj:`Examples<Example>`, find closest counterfactuals (see :doc:`index`)

    :param examples: Output from :func:`sample_space`
    :param nmols: Desired number of molecules
    """

    def is_counter(e):
        return e.yhat != examples[0].yhat

    result = _select_examples(is_counter, examples[1:], nmols)
    for i, r in enumerate(result):
        r.label = f"Counterfactual {i+1}"

    return examples[:1] + result


def rcf_explain(
    examples: List[Example],
    delta: Union[Any, Tuple[float, float]] = (-1, 1),
    nmols: int = 4,
) -> List[Example]:
    """From given :obj:`Examples<Example>`, find closest counterfactuals (see :doc:`index`)
    This version works with regression, so that a counterfactual is if the given example is higher or
    lower than base.

    :param examples: Output from :func:`sample_space`
    :param delta: float or tuple of hi/lo indicating margin for what is counterfactual
    :param nmols: Desired number of molecules
    """
    if type(delta) is float:
        delta = (-delta, delta)

    def is_high(e):
        return e.yhat + delta[0] >= examples[0].yhat

    def is_low(e):
        return e.yhat + delta[1] <= examples[0].yhat

    hresult = (
        [] if delta[0] is None else _select_examples(is_high, examples[1:], nmols // 2)
    )
    for i, h in enumerate(hresult):
        h.label = f"Increase ({i+1})"
    lresult = (
        [] if delta[1] is None else _select_examples(is_low, examples[1:], nmols // 2)
    )
    for i, l in enumerate(lresult):
        l.label = f"Decrease ({i+1})"
    return examples[:1] + lresult + hresult


def plot_space(
    examples: List[Example],
    exps: List[Example],
    figure_kwargs: Dict = None,
    mol_size: Tuple[int, int] = (200, 200),
    highlight_clusters: bool = False,
    mol_fontsize: int = 8,
    offset: int = 0,
    ax: Any = None,
    cartoon: bool = False,
    rasterized: bool = False,
):
    """Plot chemical space around example and annotate given examples.

    :param examples: Large list of :obj:Example which make-up points
    :param exps: Small list of :obj:Example which will be annotated
    :param figure_kwargs: kwargs to pass to :func:`plt.figure<matplotlib.pyplot.figure>`
    :param mol_size: size of rdkit molecule rendering, in pixles
    :param highlight_clusters: if `True`, cluster indices are rendered instead of :obj:Example.yhat
    :param mol_fontsize: minimum font size passed to rdkit
    :param offset: offset annotations to allow colorbar or other elements to fit into plot.
    :param ax: axis onto which to plot
    :param cartoon: do cartoon outline on points?
    :param rasterized: raster the scatter?
    """
    imgs = _mol_images(exps, mol_size, mol_fontsize)  # , True)
    if figure_kwargs is None:
        figure_kwargs = {"figsize": (12, 8)}
    base_color = "gray"
    if ax is None:
        ax = plt.figure(**figure_kwargs).gca()
    if highlight_clusters:
        colors = [e.cluster for e in examples]

        def normalizer(x):
            return x

        cmap = "Accent"

    else:
        colors = cast(Any, [e.yhat for e in examples])
        normalizer = plt.Normalize(min(colors), max(colors))
        cmap = "viridis"
    space_x = [e.position[0] for e in examples]
    space_y = [e.position[1] for e in examples]
    if cartoon:
        # plot shading, lines, front
        ax.scatter(space_x, space_y, 50, "0.0", lw=2, rasterized=rasterized)
        ax.scatter(space_x, space_y, 50, "1.0", lw=0, rasterized=rasterized)
        ax.scatter(
            space_x,
            space_y,
            40,
            c=normalizer(colors),
            cmap=cmap,
            lw=2,
            alpha=0.1,
            rasterized=rasterized,
        )
    else:
        ax.scatter(
            space_x,
            space_y,
            c=normalizer(colors),
            cmap=cmap,
            alpha=0.5,
            edgecolors="none",
            rasterized=rasterized,
        )
    # now plot cfs/annotated points
    ax.scatter(
        [e.position[0] for e in exps],
        [e.position[1] for e in exps],
        c=normalizer([e.cluster if highlight_clusters else e.yhat for e in exps]),
        cmap=cmap,
        edgecolors="black",
    )

    x = [e.position[0] for e in exps]
    y = [e.position[1] for e in exps]
    titles = []
    colors = []
    for e in exps:
        if not e.is_origin:
            titles.append(f"Similarity = {e.similarity:.2f}\n{e.label}")
            colors.append(cast(Any, base_color))
        else:
            titles.append("Base")
            colors.append(cast(Any, base_color))
    _image_scatter(x, y, imgs, titles, colors, ax, offset=offset)
    ax.axis("off")
    ax.set_aspect("auto")


def plot_cf(
    exps: List[Example],
    fig: Any = None,
    figure_kwargs: Dict = None,
    mol_size: Tuple[int, int] = (200, 200),
    mol_fontsize: int = 10,
    nrows: int = None,
    ncols: int = None,
):
    """Draw the given set of Examples in a grid

    :param exps: Small list of :obj:`Example` which will be drawn
    :param fig: Figure to plot onto
    :param figure_kwargs: kwargs to pass to :func:`plt.figure<matplotlib.pyplot.figure>`
    :param mol_size: size of rdkit molecule rendering, in pixles
    :param mol_fontsize: minimum font size passed to rdkit
    :param nrows: number of rows to draw in grid
    :param ncols: number of columns to draw in grid
    """
    imgs = _mol_images(exps, mol_size, mol_fontsize)
    if nrows is not None:
        R = nrows
    else:
        R = math.ceil(math.sqrt(len(imgs)))
    if ncols is not None:
        C = ncols
    else:
        C = math.ceil(len(imgs) / R)
    if fig is None:
        if figure_kwargs is None:
            figure_kwargs = {"figsize": (12, 8)}
        fig, axs = plt.subplots(R, C, **figure_kwargs)
    else:
        axs = fig.subplots(R, C)
    axs = axs.flatten()
    for i, (img, e) in enumerate(zip(imgs, exps)):
        title = "Base" if e.is_origin else f"Similarity = {e.similarity:.2f}\n{e.label}"
        title += f"\nf(x) = {e.yhat:.3f}"
        axs[i].set_title(title)
        axs[i].imshow(np.asarray(img), gid=f"rdkit-img-{i}")
        axs[i].axis("off")
    for j in range(i, C * R):
        axs[j].axis("off")
        axs[j].set_facecolor("white")
    plt.tight_layout()


def plot_descriptors(
    space: List[Example],
    descriptor_type: str,
    fig: Any = None,
    figure_kwargs: Dict = None,
    output_file: str = None,
):
    """Plot descriptor attributions from given set of Examples are space_tstats

    :param exps: Output from :func:`sample_space`
    :param space_tstats: t-statistics output from :func:`lime_explain`
    :param descriptor_type: Descriptor type to plot, either 'Classic' or 'MACCS'
    :param fig: Figure to plot on to
    :param figure_kwargs: kwargs to pass to :func:`plt.figure<matplotlib.pyplot.figure>`
    :param output_file: Output file name to save the plot
    """
    from importlib_resources import files
    import exmol.lime_data
    import pickle  # type: ignore

    space_tstats = list(space[0].descriptors.tstats)
    if fig is None:
        if figure_kwargs is None:
            figure_kwargs = (
                {"figsize": (5, 5)}
                if descriptor_type == "Classic"
                else {"figsize": (8, 5)}
            )
        fig, ax = plt.subplots(nrows=1, ncols=1, dpi=180, **figure_kwargs)

    # find important descriptors
    d_importance = {
        a: [b, i]
        for i, a, b in zip(
            np.arange(len(space[0].descriptors.descriptors)),
            space[0].descriptors.descriptor_names,
            space_tstats,
        )
        if not np.isnan(b)
    }
    d_importance = dict(
        sorted(d_importance.items(), key=lambda item: abs(item[1][0]), reverse=True)
    )
    t = [a[0] for a in list(d_importance.values())][:5]
    key_ids = [a[1] for a in list(d_importance.values())][:5]
    keys = [a for a in list(d_importance.keys())]

    # set colors
    colors = []
    for ti in t:
        if ti < 0:
            colors.append("#F06060")
        if ti > 0:
            colors.append("#1BBC9B")
    # plot the bars
    bar1 = ax.barh(range(len(t)), t, color=colors, height=0.75)
    new_patches = []
    for patch in reversed(ax.patches):
        bb = patch.get_bbox()
        color = patch.get_facecolor()
        p_bbox = FancyBboxPatch(
            (bb.xmin, bb.ymin),
            abs(bb.width),
            abs(bb.height),
            boxstyle="round,pad=-0.040,rounding_size=0.015",
            ec="none",
            fc=color,
            mutation_aspect=4,
        )
        patch.remove()
        new_patches.append(p_bbox)
    for patch in new_patches:
        ax.add_patch(patch)

    count = 0
    sk_dict, svgs = {}, {}
    if descriptor_type == "MACCS":
        # Load svg images
        mk = files(exmol.lime_data).joinpath("keys.pb")
        with open(str(mk), "rb") as f:
            svgs = pickle.load(f)
    if descriptor_type == "ECFP":
        # get reference for ECFP
        bi = {}  # type: Dict[Any, Any]
        m = smi2mol(space[0].smiles)
        fp = AllChem.GetMorganFingerprint(m, 3, bitInfo=bi)

    for rect, ti, k, ki in zip(bar1, t, keys, key_ids):
        # annotate patches with text desciption
        y = rect.get_y() + rect.get_height() / 2.0
        k = textwrap.fill(str(k), 25)
        if ti < 0:
            x = 0.25
            skx = (
                np.max(np.absolute(t)) + 2
                if descriptor_type == "MACCS"
                else np.max(np.absolute(t))
            )
            box_x = 0.98
            ax.text(
                x,
                y,
                " " if descriptor_type == "ECFP" else k,
                ha="left",
                va="center",
                wrap=True,
                fontsize=12,
            )
        else:
            x = -0.25
            skx = (
                -np.max(np.absolute(t)) - 2
                if descriptor_type == "MACCS"
                else np.max(np.absolute(t))
            )
            box_x = 0.02
            ax.text(
                x,
                y,
                " " if descriptor_type == "ECFP" else k,
                ha="right",
                va="center",
                wrap=True,
                fontsize=12,
            )
        # add SMARTS annotation where applicable
        if descriptor_type == "MACCS" or descriptor_type == "ECFP":
            box = skunk.Box(130, 50, f"sk{count}")
            ab = AnnotationBbox(
                box,
                xy=(skx, count),
                xybox=(box_x, (5 - count) * 0.2 - 0.1),  # Invert axis
                xycoords="data",
                boxcoords="axes fraction",
                bboxprops=dict(lw=0.5),
            )

            ax.add_artist(ab)
            if descriptor_type == "MACCS":
                sk_dict[f"sk{count}"] = svgs[ki]
            if descriptor_type == "ECFP":
                svg = DrawMorganBit(
                    m,
                    int(k),
                    bi,
                    molSize=(300, 200),
                    centerColor=None,
                    aromaticColor=None,
                    ringColor=None,
                    extraColor=(0.8, 0.8, 0.8),
                    useSVG=True,
                )
                sk_dict[f"sk{count}"] = svg.data
        count += 1
    ax.axvline(x=0, color="grey", linewidth=0.5)
    # calculate significant T
    w = np.array([1 / (1 + (1 / (e.similarity + 0.000001) - 1) ** 5) for e in space])
    effective_n = np.sum(w) ** 2 / np.sum(w**2)
    T = ss.t.ppf(0.975, df=effective_n)
    # plot T
    ax.axvline(x=T, color="#f5ad4c", linewidth=0.75, linestyle="--", zorder=0)
    ax.axvline(x=-T, color="#f5ad4c", linewidth=0.75, linestyle="--", zorder=0)
    # set axis
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlabel("Descriptor t-statistics", fontsize=12)
    ax.set_title(f"{descriptor_type} descriptors", fontsize=12)
    # inset SMARTS svg images for MACCS descriptors
    if descriptor_type == "MACCS" or descriptor_type == "ECFP":
        if descriptor_type == "MACCS":
            print(
                "SMARTS annotations for MACCS descriptors were created using SMARTSviewer (smartsview.zbh.uni-hamburg.de, Copyright: ZBH, Center for Bioinformatics Hamburg) developed by K. Schomburg et. al. (J. Chem. Inf. Model. 2010, 50, 9, 1529–1535)"
            )
        xlim = np.max(np.absolute(t)) + 5
        ax.set_xlim(-xlim, xlim)
        svg = skunk.insert(sk_dict)
        plt.tight_layout()
        if output_file is None:
            output_file = f"{descriptor_type}.svg"
        with open(output_file, "w") as f:
            f.write(svg)
        return svg
    elif descriptor_type == "Classic":
        xlim = max(np.max(np.absolute(t)), T + 1)
        ax.set_xlim(-xlim, xlim)
        plt.tight_layout()
        if output_file is None:
            output_file = f"{descriptor_type}.svg"
        plt.savefig(output_file, dpi=180, bbox_inches="tight")
