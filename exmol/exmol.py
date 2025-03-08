from functools import reduce, lru_cache
import inspect
from typing import *
import io
import math
import requests  # type: ignore
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.patches import Rectangle, FancyBboxPatch  # type: ignore
from matplotlib.offsetbox import AnnotationBbox  # type: ignore
import matplotlib as mpl  # type: ignore
import re
import selfies as sf  # type: ignore
import tqdm  # type: ignore
import textwrap  # type: ignore
import skunk  # type: ignore
import synspace  # type: ignore

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
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity, TanimotoSimilarity  # type: ignore

import openai
from . import stoned
from .plot_utils import _mol_images, _image_scatter, _bit2atoms
from .data import *


def _fp_dist_matrix(smiles, fp_type, _pbar):
    mols = [(smi2mol(s), _pbar.update(0.5))[0] for s in smiles]
    # Sorry about the one-line. Just sneaky insertion of progressbar update
    fp = [(stoned.get_fingerprint(m, fp_type), _pbar.update(0.5))[0] for m in mols]
    M = np.array([BulkTanimotoSimilarity(f, fp) for f in fp])
    # 1 - similarity because we want distance
    return 1 - M


def _check_multiple_bases(examples):
    return sum([e.is_origin for e in examples]) > 1


def _ecfp_names(examples, joint_bits):
    # add names for given descriptor indices
    multiple_bases = _check_multiple_bases(examples)
    # need to get base molecule(s) for naming
    bitInfo = {}  # Type Dict[Any, Any]
    base_mol = [smi2mol(e.smiles) for e in examples if e.is_origin == True]
    if multiple_bases:
        multiBitInfo = {}  # type: Dict[int, Tuple[Any, int, int]]
        for b in base_mol:
            bitInfo = {}
            AllChem.GetMorganFingerprint(b, 3, bitInfo=bitInfo)
            for bit in bitInfo:
                if bit not in multiBitInfo:
                    multiBitInfo[bit] = (b, bit, {bit: bitInfo[bit]})
    else:
        base_mol = smi2mol(examples[0].smiles)
        bitInfo = {}  # type: Dict[Any, Any]
        AllChem.GetMorganFingerprint(base_mol, 3, bitInfo=bitInfo)
    result = []  # type: List[str]
    for i in range(len(joint_bits)):
        k = joint_bits[i]
        if multiple_bases:
            m = multiBitInfo[k][0]
            b = multiBitInfo[k][2]
            name = name_morgan_bit(m, b, k)
        else:
            name = name_morgan_bit(base_mol, bitInfo, k)
        result.append(name)
    return tuple(result)


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


def _get_joint_ecfp_descriptors(examples):
    """Create a union of ECFP bits from all base molecules"""
    # get reference
    bases = [smi2mol(e.smiles) for e in examples if e.is_origin]
    ecfp_joint = set()
    for m in bases:
        # Get bitinfo and create a union
        b = {}  # type: Dict[Any, Any]
        temp_fp = AllChem.GetMorganFingerprint(m, 3, bitInfo=b)
        # add if radius greater than 0
        ecfp_joint |= set([(k, v[0][1]) for k, v in b.items() if v[0][1] > 0])
    # want to go in order of radius so when
    # we drop non-unique names, we keep smaller fragments
    ecfp_joint = list(ecfp_joint)
    ecfp_joint.sort(key=lambda x: x[1])
    ecfp_joint = [x[0] for x in ecfp_joint]
    names = _ecfp_names(examples, ecfp_joint)
    # downselect to only unique names
    unique_names = set(names)
    output_ecfp = []
    output_names = []
    for b, n in zip(ecfp_joint, names):
        if n in unique_names and n is not None:
            unique_names.remove(n)
            output_ecfp.append(b)
            output_names.append(n)
    return output_ecfp, output_names


_SMARTS = None


def _load_smarts(path, rank_cutoff=500):
    # we have a rank cut for SMARTS that match too often
    smarts = {}
    with open(path) as f:
        for line in f.readlines():
            if line[0] == "#":
                continue
            i1 = line.find(":")
            i2 = line.find(":", i1 + 1)
            m = MolFromSmarts(line[i2 + 1 :].strip())
            rank = int(line[i1 + 1 : i2])
            if rank > rank_cutoff:
                continue
            name = line[:i1]
            if m is None:
                print(f"Could not parse SMARTS: {line}")
                print(line[i2:].strip())
            smarts[name] = (m, rank)
    return smarts


def name_morgan_bit(m: Any, bitInfo: Dict[Any, Any], key: int) -> str:
    """Get the name of a Morgan bit using a SMARTS dictionary

    :param m: RDKit molecule
    :param bitInfo: bitInfo dictionary from rdkit.Chem.AllChem.GetMorganFingerprint
    :param key: bit key corresponding to the fingerprint you want to have named
    """
    global _SMARTS
    if _SMARTS is None:
        from importlib_resources import files  # type: ignore
        import exmol.lime_data  # type: ignore

        sp = files(exmol.lime_data).joinpath("smarts.txt")
        _SMARTS = _load_smarts(sp)
    morgan_atoms = _bit2atoms(m, bitInfo, key)
    heteroatoms = set()
    for a in morgan_atoms:
        if m.GetAtomWithIdx(a).GetAtomicNum() > 6:
            heteroatoms.add(a)
    names = []
    for name, (sm, r) in _SMARTS.items():
        matches = m.GetSubstructMatches(sm)
        for match in matches:
            # check if match is in morgan bit
            match = set(match)
            if match.issubset(morgan_atoms):
                names.append((r, name, match))
    names.sort(key=lambda x: x[0])
    if len(names) == 0:
        return None
    umatch = names[0][2]
    name = names[0][1][0].lower() + names[0][1][1:].replace("_", " ")
    unique_names = set([names[0][1]])
    for _, n, m in names:
        if len(m.intersection(umatch)) == 0:
            if n not in unique_names:
                name += "/" + n[0].lower() + n[1:].replace("_", " ")
                umatch |= m
                unique_names.add(n)
    if "/" in name and "fragment" not in name.split("/")[-1]:
        name = name + " group"
    # if we failed to match all heteroatoms, fail
    if len(heteroatoms.difference(umatch)) > 0:
        return None
    return name


def get_functional_groups(mol: Any, cutoff: int = 300) -> List[str]:
    """Get a list of functional groups present in a molecule, sorted by priority, avoiding overlaps.

    :param mol: RDKit molecule
    :param cutoff: Maximum rank of functional groups to consider based on popularity
    :return: List of unique functional group names present in the molecule, sorted by priority
    """
    global _SMARTS
    if _SMARTS is None:
        from importlib_resources import files  # type: ignore
        import exmol.lime_data  # type: ignore

        sp = files(exmol.lime_data).joinpath("smarts.txt")
        _SMARTS = _load_smarts(sp)

    if isinstance(mol, str):
        mol = smi2mol(mol)
    if mol is None:
        return []

    matched_atoms = set()
    result = []

    sorted_smarts = sorted(_SMARTS.items(), key=lambda x: x[1][1])

    for name, (sm, rank) in sorted_smarts:
        if rank > cutoff:
            continue
        for match in mol.GetSubstructMatches(sm):
            match_set = set(match)
            if not match_set.intersection(matched_atoms):
                formatted_name = name[0].lower() + name[1:].replace("_", " ")
                result.append(formatted_name)
                matched_atoms.update(match_set)
                break  # Only add group once per molecule

    return result


def clear_descriptors(
    examples: List[Example],
) -> List[Example]:
    """Clears all descriptors from examples

    :param examples: list of examples
    :param descriptor_type: type of descriptor to clear, if None, all descriptors are cleared
    """
    for e in examples:
        e.descriptors = None  # type: ignore
    return examples


def add_descriptors(
    examples: List[Example],
    descriptor_type: str = "MACCS",
    mols: List[Any] = None,
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
    if descriptor_type.lower() == "classic":
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
                plotting_names=descriptor_names,
            )
        return examples
    elif descriptor_type.lower() == "maccs":
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
                plotting_names=descriptor_names,
            )
        return examples
    elif descriptor_type.lower() == "ecfp":
        descriptor_bits, plotting_names = _get_joint_ecfp_descriptors(examples)
        for e, m in zip(examples, mols):
            bitInfo = {}  # type: Dict[Any, Any]
            AllChem.GetMorganFingerprint(m, 3, bitInfo=bitInfo)
            descriptors = tuple(
                [1 if x in bitInfo.keys() else 0 for x in descriptor_bits]
            )
            e.descriptors = Descriptors(
                descriptor_type=descriptor_type,
                descriptors=descriptors,
                descriptor_names=descriptor_bits,
                plotting_names=plotting_names,
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
    to_remove.extend(["[P]", "[#P]", "[=P]", "[B]", "[#B]", "[=B]"])

    a -= set(to_remove)
    a.add("[O-1]")
    return a


# The code below checks for accidental addition of symbols outside of the alphabet.
# the only way this can happen is if the character indicating a ring
# length is mutated to appear eleswhere. The ring length symbol
# is always a plain uncharged element.


def _alphabet_to_elements(alphabet: List[str]) -> Set[str]:
    """Converts SELFIES alphabet to element symbols"""
    symbols = []
    for s in alphabet:
        s = s.replace("[", "").replace("]", "")
        if s.isalpha():
            symbols.append(s)
    return set(symbols)


def _check_alphabet_consistency(
    smiles: str, alphabet_symbols: Set[str], check=False
) -> True:
    """Checks if SMILES only contains tokens from alphabet"""

    alphabet_symbols = _alphabet_to_elements(set(alphabet_symbols))
    # find all elements in smiles (Upper alpha or upper alpha followed by lower alpha)
    smiles_symbols = set(re.findall(r"[A-Z][a-z]?", smiles))

    if check and not smiles_symbols.issubset(alphabet_symbols):
        # show which symbols are not in alphabet
        raise ValueError(
            "symbols not in alphabet" + str(smiles_symbols.difference(alphabet_symbols))
        )
    return smiles_symbols.issubset(alphabet_symbols)


def run_stoned(
    start_smiles: str,
    fp_type: str = "ECFP4",
    num_samples: int = 2000,
    max_mutations: int = 2,
    min_mutations: int = 1,
    alphabet: Union[List[str], Set[str]] = None,
    return_selfies: bool = False,
    _pbar: Any = None,
) -> Union[Tuple[List[str], List[float]], Tuple[List[str], List[str], List[float]]]:
    """Run ths STONED SELFIES algorithm. Typically not used, call :func:`sample_space` instead.

    :param start_smiles: SMILES string to start from
    :param fp_type: Fingerprint type
    :param num_samples: Number of total molecules to generate
    :param max_mutations: Maximum number of mutations
    :param min_mutations: Minimum number of mutations
    :param alphabet: Alphabet to use for mutations, typically from :func:`get_basic_alphabet()`
    :param return_selfies: If SELFIES should be returned as well
    :return: SELFIES, SMILES, and SCORES generated or SMILES and SCORES generated
    """
    if alphabet is None:
        alphabet = get_basic_alphabet()
    if type(alphabet) == set:
        alphabet = list(alphabet)
    alphabet_symbols = _alphabet_to_elements(alphabet)
    # make sure starting smiles is consistent with alphabet
    _ = _check_alphabet_consistency(start_smiles, alphabet_symbols, check=True)
    num_mutation_ls = list(range(min_mutations, max_mutations + 1))

    start_mol = smi2mol(start_smiles)
    if start_mol == None:
        raise Exception("Invalid starting structure encountered")

    # want it so after sampling have num_samples
    randomized_smile_orderings = [
        stoned.randomize_smiles(smi2mol(start_smiles))
        for _ in range(num_samples // len(num_mutation_ls))
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
        # check if smiles are consistent with alphabet and downslect
        selfies_mut, smiles_back = zip(
            *[
                (s, sm)
                for s, sm in zip(selfies_mut, smiles_back)
                if _check_alphabet_consistency(sm, alphabet_symbols)
            ]
        )
        selfies_mut, smiles_back = list(selfies_mut), list(smiles_back)

        all_smiles_collect = all_smiles_collect + smiles_back
        all_selfies_collect = all_selfies_collect + selfies_mut
        if _pbar:
            _pbar.update(len(smiles_back))

    if _pbar:
        _pbar.set_description(f"🥌STONED🥌 Filtering")

    # filter out duplicates
    all_mols = [smi2mol(s) for s in all_smiles_collect]
    all_canon = [
        stoned.largest_mol(mol2smi(m, canonical=True)) if m else None for m in all_mols
    ]
    seen = set()
    to_keep = [False for _ in all_canon]
    for i in range(len(all_canon)):
        if all_canon[i] and all_canon[i] not in seen:
            to_keep[i] = True
            seen.add(all_canon[i])

    # now do filter
    filter_mols = [m for i, m in enumerate(all_mols) if to_keep[i]]
    filter_selfies = [s for i, s in enumerate(all_selfies_collect) if to_keep[i]]
    filter_smiles = [s for i, s in enumerate(all_smiles_collect) if to_keep[i]]

    # compute similarity scores
    base_fp = stoned.get_fingerprint(start_mol, fp_type=fp_type)
    fps = [stoned.get_fingerprint(m, fp_type) for m in filter_mols]
    scores = BulkTanimotoSimilarity(base_fp, fps)  # type: List[float]

    if _pbar:
        _pbar.set_description(f"🥌STONED🥌 Done")

    if return_selfies:
        return filter_selfies, filter_smiles, scores
    else:
        return filter_smiles, scores


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
    all_can = [
        stoned.largest_mol(mol2smi(m, canonical=True)) if m else None for m in mols
    ]
    seen = set()
    to_keep = [False for _ in all_can]
    for i in range(len(all_can)):
        if all_can[i] and all_can[i] not in seen:
            to_keep[i] = True
            seen.add(all_can[i])
    smiles = [s for i, s in enumerate(smiles) if to_keep[i]]
    mols = [m for i, m in enumerate(mols) if to_keep[i]]
    fp0 = stoned.get_fingerprint(mol0, fp_type)
    scores = []
    # drop Nones
    smiles = [s for s, m in zip(smiles, mols) if m is not None]
    for m in mols:
        if m is None:
            continue
        fp = stoned.get_fingerprint(m, fp_type)
        scores.append(TanimotoSimilarity(fp0, fp))
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
        scores.append(TanimotoSimilarity(fp0, fp))
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
    sanitize_smiles: bool = True,
) -> List[Example]:
    """Sample chemical space around given SMILES

    This will evaluate the given function and run the :func:`run_stoned` function over chemical space around molecule. ``num_samples`` will be
    set to 3,000 by default if using STONED and 150 if using ``chemed``. If using ``custom`` then ``num_samples`` will be set to the length of
    of the ``data`` list. If using ``synspace`` then ``num_samples`` will be set to 1,000. See :func:`run_stoned` and :func:`run_chemed` for more details.
    ``synspace`` comes from the package `synspace <https://github.com/whitead/synspace>`. It generates synthetically feasible
    molecules from a given SMILES.

    :param origin_smiles: starting SMILES
    :param f: A function which takes in SMILES or SELFIES and returns predicted value. Assumed to work with lists of SMILES/SELFIES unless `batched = False`
    :param batched: If `f` is batched
    :param preset: Can be `"wide"`, `"medium"`, `"narrow"`, `"chemed"`, `"custom"`, or `"synspace`". Determines how far across chemical space is sampled. Try `"chemed"` preset to only sample pubchem compounds.
    :param data: If not None and preset is `"custom"` will use this data instead of generating new ones.
    :param method_kwargs: More control over STONED, CHEMED and CUSTOM can be set here. See :func:`run_stoned`, :func:`run_chemed` and  :func:`run_custom`
    :param num_samples: Number of desired samples. Can be set in `method_kwargs` (overrides) or here. `None` means default for preset
    :param stoned_kwargs: Backwards compatible alias for `methods_kwargs`
    :param quiet: If True, will not print progress bar
    :param use_selfies: If True, will use SELFIES instead of SMILES for `f`
    :param sanitize_smiles: If True, will sanitize all SMILES
    :return: List of generated :obj:`Example`
    """

    wrapped_f = f

    # if f only takes in 1 arg, wrap it in a function that takes in 2
    # count args with no default value. Looks fancy because of possible objects/partials
    argcount = len(
        [
            i
            for i in inspect.signature(f).parameters.values()
            if i.default == inspect.Parameter.empty
        ]
    )
    if argcount == 1:
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

    if sanitize_smiles:
        origin_smiles = stoned.sanitize_smiles(origin_smiles)[1]
    elif "." in origin_smiles:
        raise ValueError(
            "Given SMILES contains '.', which indicates it is not a single molecule. "
            "Please sanitize it first or set sanitize_smiles=True"
        )
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
        elif preset == "synspace":
            method_kwargs["num_samples"] = 1000 if num_samples is None else num_samples
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
        selfies = [sf.encoder(s) for s in smiles]
    elif preset == "custom":
        smiles, scores = run_custom(
            origin_smiles, data=cast(Any, data), _pbar=pbar, **method_kwargs
        )
        selfies = [sf.encoder(s) for s in smiles]
    elif preset == "synspace":
        mols, _ = synspace.chemical_space(origin_smiles, _pbar=pbar, **method_kwargs)
        if len(mols) < 5:
            raise ValueError(
                "Synspace did not return enough molecules. Try adjusting method_kwargs for synspace"
            )
        data = [mol2smi(mol).replace("~", "") for mol in mols]
        smiles, scores = run_custom(
            origin_smiles, data=cast(Any, data), _pbar=pbar, **method_kwargs
        )
        selfies = [sf.encoder(s) for s in smiles]
    else:
        result = run_stoned(
            origin_smiles, _pbar=pbar, return_selfies=True, **method_kwargs
        )
        selfies, smiles, scores = cast(Tuple[List[str], List[str], List[float]], result)

    pbar.set_description("😀Calling your model function😀")
    if sanitize_smiles:
        smiles = [stoned.sanitize_smiles(s)[1] for s in smiles]
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

    # do clustering everywhere (maybe do counter/same separately?)
    # clustering = AgglomerativeClustering(
    #    n_clusters=max_k, affinity='precomputed', linkage='complete').fit(full_dmat)
    # Just do it on projected so it looks prettier.
    clustering = DBSCAN(eps=0.15, min_samples=5).fit(proj_dmat)

    for i, e in enumerate(exps):  # type: ignore
        e.cluster = clustering.labels_[i]  # type: ignore

    pbar.set_description("🤘Done🤘")
    pbar.close()
    return exps


def _select_examples(cond, examples, nmols, do_filter=False):
    result = []
    if do_filter or do_filter is None:
        from synspace.reos import REOS

        reos = REOS()
        # if do_filter is None, check if 0th smiles passes filter
        if do_filter is None:
            do_filter = reos.process_mol(smi2mol(examples[0].smiles)) == ("ok", "ok")

    # similarity filtered by if cluster/counter
    def cluster_score(e, i):
        score = (e.cluster == i) * cond(e) * e.similarity
        return score

    clusters = set([e.cluster for e in examples])
    for i in clusters:
        close_counter = max(examples, key=lambda e, i=i: cluster_score(e, i))
        # check if actually is (since call could have been zero)
        if cluster_score(close_counter, i):
            result.append(close_counter)

    # sort by similarity
    result = sorted(result, key=lambda v: v.similarity * cond(v), reverse=True)
    # back fill
    result.extend(sorted(examples, key=lambda v: v.similarity * cond(v), reverse=True))
    final_result = []
    if do_filter:
        while len(final_result) < nmols:
            e = result.pop(0)
            if reos.process_mol(smi2mol(e.smiles)) == ("ok", "ok"):
                final_result.append(e)
    else:
        final_result = result[:nmols]
    return list(filter(cond, final_result))


def lime_explain(
    examples: List[Example],
    descriptor_type: str = "MACCS",
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
    # Set tstats for bases, to be used later
    # TODO: Used to put them on examples[0] only,
    # but now copy them to all examples
    for e in examples:
        e.descriptors.tstats = tstat
    # Return beta (feature weights) which are the fits if asked for
    if return_beta:
        return beta
    else:
        return None


def cf_explain(
    examples: List[Example], nmols: int = 3, filter_nondrug: Optional[bool] = None
) -> List[Example]:
    """From given :obj:`Examples<Example>`, find closest counterfactuals (see :doc:`index`)

    :param examples: Output from :func:`sample_space`
    :param nmols: Desired number of molecules
    :param filter_nondrug: Whether or not to filter out non-drug molecules. Default is True if input passes filter
    """

    def is_counter(e):
        return e.yhat != examples[0].yhat

    result = _select_examples(is_counter, examples[1:], nmols, filter_nondrug)
    for i, r in enumerate(result):
        r.label = f"Counterfactual {i+1}"

    return examples[:1] + result


def rcf_explain(
    examples: List[Example],
    delta: Union[Any, Tuple[float, float]] = (-1, 1),
    nmols: int = 4,
    filter_nondrug: Optional[bool] = None,
) -> List[Example]:
    """From given :obj:`Examples<Example>`, find closest counterfactuals (see :doc:`index`)
    This version works with regression, so that a counterfactual is if the given example is higher or
    lower than base.

    :param examples: Output from :func:`sample_space`
    :param delta: float or tuple of hi/lo indicating margin for what is counterfactual
    :param nmols: Desired number of molecules
    :param filter_nondrug: Whether or not to filter out non-drug molecules. Default is True if input passes filter
    """
    if type(delta) is float:
        delta = (-delta, delta)

    def is_high(e):
        return e.yhat + delta[0] >= examples[0].yhat

    def is_low(e):
        return e.yhat + delta[1] <= examples[0].yhat

    hresult = (
        []
        if delta[0] is None
        else _select_examples(is_high, examples[1:], nmols // 2, filter_nondrug)
    )
    for i, h in enumerate(hresult):
        h.label = f"Increase ({i+1})"
    lresult = (
        []
        if delta[1] is None
        else _select_examples(is_low, examples[1:], nmols // 2, filter_nondrug)
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
    if type(axs) != np.ndarray:  # Happens if nrows=ncols=1
        axs = np.array([[axs]])
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
    examples: List[Example],
    output_file: str = None,
    fig: Any = None,
    figure_kwargs: Dict = None,
    title: str = None,
    return_svg: bool = False,
):
    """Plot descriptor attributions from given set of Examples.

    :param examples: Output from :func:`sample_space`
    :param output_file: Output file name to save the plot - optional except for ECFP
    :param fig: Figure to plot on to
    :param figure_kwargs: kwargs to pass to :func:`plt.figure<matplotlib.pyplot.figure>`
    :param title: Title for the plot
    :param return_svg: Whether to return svg for plot
    """

    from importlib_resources import files
    import exmol.lime_data
    import pickle  # type: ignore

    # infer descriptor_type from examples
    descriptor_type = examples[0].descriptors.descriptor_type.lower()

    multiple_bases = _check_multiple_bases(examples)

    if output_file is None and descriptor_type == "ecfp" and not return_svg:
        raise ValueError("No filename provided to save the plot")

    space_tstats = list(examples[0].descriptors.tstats)
    if fig is None:
        if figure_kwargs is None:
            figure_kwargs = (
                {"figsize": (5, 5)}
                if descriptor_type.lower() == "classic"
                else {"figsize": (8, 5)}
            )
        fig, ax = plt.subplots(nrows=1, ncols=1, dpi=180, **figure_kwargs)

    # find important descriptors
    d_importance = {
        a: [b, i, n]
        for i, (a, b, n) in enumerate(
            zip(
                examples[0].descriptors.descriptor_names,
                space_tstats,
                examples[0].descriptors.plotting_names,
            )
        )
        if not np.isnan(b)
    }
    d_importance = dict(
        sorted(d_importance.items(), key=lambda item: abs(item[1][0]), reverse=True)
    )
    t = [a[0] for a in list(d_importance.values())][:5]
    key_ids = [a[1] for a in list(d_importance.values())][:5]
    keys = [a for a in list(d_importance.keys())]
    names = [a[2] for a in list(d_importance.values())][:5]

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
    sk_dict, key_imgs = {}, {}
    if descriptor_type == "maccs":
        # Load svg/png images
        mk = files(exmol.lime_data).joinpath("keys.pb")
        with open(str(mk), "rb") as f:
            key_imgs = pickle.load(f)
    if descriptor_type == "ecfp":
        # get reference for ECFP
        if multiple_bases:
            bases = [smi2mol(e.smiles) for e in examples if e.is_origin == True]
            bi = {}  # type: Dict[Any, Any]
            for b in bases:
                bit_info = {}  # type: Dict[Any, Any]
                fp = AllChem.GetMorganFingerprint(b, 3, bitInfo=bit_info)
                for bit in bit_info:
                    if bit not in bi:
                        bi[bit] = (b, bit, bit_info)
        else:
            bi = {}
            m = smi2mol(examples[0].smiles)
            fp = AllChem.GetMorganFingerprint(m, 3, bitInfo=bi)
    for rect, ti, k, ki, n in zip(bar1, t, keys, key_ids, names):
        # account for Nones
        if n is None:
            n = ""
        # annotate patches with text desciption
        y = rect.get_y() + rect.get_height() / 2.0
        n = textwrap.fill(str(n), 20)
        if ti < 0:
            x = 0.25
            skx = (
                np.max(np.absolute(t)) + 2
                if descriptor_type == "maccs"
                else np.max(np.absolute(t))
            )
            box_x = 0.98
            ax.text(
                x,
                y,
                n,
                ha="left",
                va="center",
                wrap=True,
                fontsize=12,
            )
        else:
            x = -0.25
            skx = (
                -np.max(np.absolute(t)) - 2
                if descriptor_type == "maccs"
                else np.max(np.absolute(t))
            )
            box_x = 0.02
            ax.text(
                x,
                y,
                n,
                ha="right",
                va="center",
                wrap=True,
                fontsize=12,
            )
        # add SMARTS annotation where applicable
        if descriptor_type == "maccs" or descriptor_type == "ecfp":
            if descriptor_type == "maccs":
                key_img = plt.imread(io.BytesIO(key_imgs[ki]["png"]))
                box = skunk.ImageBox(f"sk{count}", key_img, zoom=1)
            else:
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
            if descriptor_type == "maccs":
                sk_dict[f"sk{count}"] = key_imgs[ki]["svg"]
            if descriptor_type == "ecfp":
                if multiple_bases:
                    m = bi[int(k)][0]
                    b = bi[int(k)][2]
                else:
                    b = bi
                svg = DrawMorganBit(
                    m,
                    int(k),
                    b,
                    molSize=(300, 200),
                    centerColor=None,
                    aromaticColor=None,
                    ringColor=None,
                    extraColor=(0.8, 0.8, 0.8),
                    useSVG=True,
                )
                # TODO: Why?
                try:
                    svgdata = svg.data
                except AttributeError:
                    svgdata = svg
                sk_dict[f"sk{count}"] = svgdata
        count += 1
    ax.axvline(x=0, color="grey", linewidth=0.5)
    # calculate significant T
    w = np.array([1 / (1 + (1 / (e.similarity + 0.000001) - 1) ** 5) for e in examples])
    effective_n = np.sum(w) ** 2 / np.sum(w**2)
    T = ss.t.ppf(0.975, df=effective_n)
    # plot T
    ax.axvline(x=T, color="#f5ad4c", linewidth=0.75, linestyle="--", zorder=0)
    ax.axvline(x=-T, color="#f5ad4c", linewidth=0.75, linestyle="--", zorder=0)
    # set axis
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlabel("Descriptor t-statistics", fontsize=12)
    if title is None:
        ax.set_title(f"{descriptor_type} descriptors", fontsize=12)
    else:
        ax.set_title(f"{title}", fontsize=12)
    # inset SMARTS svg images for MACCS descriptors
    if descriptor_type == "maccs" or descriptor_type == "ecfp":
        if descriptor_type == "maccs":
            print(
                "SMARTS annotations for MACCS descriptors were created using SMARTSviewer (smartsview.zbh.uni-hamburg.de, Copyright: ZBH, Center for Bioinformatics Hamburg) developed by K. Schomburg et. al. (J. Chem. Inf. Model. 2010, 50, 9, 1529–1535)"
            )
        xlim = np.max(np.absolute(t)) + 6
        ax.set_xlim(-xlim, xlim)
        svg = skunk.insert(sk_dict)
        if output_file is not None:
            plt.tight_layout()
            with open(output_file, "w") as f:  # type: ignore
                f.write(svg)
        if return_svg:
            plt.close()
            return svg
    elif descriptor_type == "classic":
        xlim = max(np.max(np.absolute(t)), T + 1)
        ax.set_xlim(-xlim, xlim)
        if output_file is not None:
            plt.tight_layout()
            plt.savefig(output_file, dpi=180, bbox_inches="tight")


def check_multiple_aromatic_rings(mol):
    ri = mol.GetRingInfo()
    count = 0
    for bondRing in ri.BondRings():
        flag = True
        for id in bondRing:
            if not mol.GetBondWithIdx(id).GetIsAromatic():
                flag = False
                continue
        if flag:
            count += 1
    return True if count > 1 else False


def merge_text_explains(
    *args: List[Tuple[str, float]], filter: Optional[float] = None
) -> List[Tuple[str, float]]:
    """Merge multiple text explanations into one and sort."""
    # sort them by T value, putting negative examples at the end
    joint = reduce(lambda x, y: x + y, args)
    if len(joint) == 0:
        return []
    # get the highest (hopefully) positive
    m = max([x[1] for x in joint if x[1] > 0])
    pos = [x for x in joint if x[1] == m]
    joint = [x for x in joint if x[1] != m]
    joint = sorted(joint, key=lambda x: np.absolute(x[1]), reverse=True)
    return pos + joint


_multi_prompt = (
    "The following is information about molecules that connect their structures "
    'to the property called "{property}." '
    "The information is attributes of molecules expressed as questions with answers and "
    "relative importance. "
    "Using all aspects of this information, propose an explanation (50-150 words) "
    'for the molecular property "{property}." '
    "Only use the information below. Answer in a scientific "
    'tone and make use of counterfactuals (e.g., "If X were present, {property} would be negatively...").'
    "\n\n"
    "{text}\n\n"
    "Explanation:"
)

_single_prompt = (
    "The following is information about a specific molecule that connects its structure "
    'to the property "{property}." '
    "The information is structural attributes expressed as questions with answers and "
    "relative importance. "
    "Using all aspects of this information, propose an explanation (50-150 words) "
    'for this molecule\'s property "{property}." '
    "Only use the information below. Answer in a scientific "
    'tone and make use of counterfactuals (e.g., "If X were present, its {property} would be negatively...").'
    "\n\n"
    "{text}\n\n"
    "Explanation:"
)


def text_explain_generate(
    text_explanations: List[Tuple[str, float]],
    property_name: str,
    llm_model: str = "gpt-4o",
    single: bool = True,
) -> str:
    """Insert text explanations into template, and generate explanation.

    Args:
        text_explanations: List of text explanations.
        property_name: Name of property.
        llm: Language model to use.
        single: Whether to use a prompt about a single molecule or multiple molecules.
    """
    # want to have negative examples at the end
    text_explanations.sort(key=lambda x: x[1], reverse=True)
    text = "\n".join(
        [
            # f"{x[0][:-1]} {'Positive' if x[1] > 0 else 'Negative'} correlation."
            f"{x[0][:-1]}."
            for x in text_explanations
        ]
    )

    prompt_template = _single_prompt if single else _multi_prompt
    prompt = prompt_template.format(property=property_name, text=text)

    messages = [
        {
            "role": "system",
            "content": "Your goal is to explain which molecular features are important to its properties based on the given text.",
        },
        {"role": "user", "content": prompt},
    ]
    response = openai.chat.completions.create(
        model=llm_model,
        messages=messages,
        temperature=0.05,
    )

    return response.choices[0].message.content


def text_explain(
    examples: List[Example],
    descriptor_type: str = "maccs",
    count: int = 5,
    presence_thresh: float = 0.2,
    include_weak: Optional[bool] = None,
) -> List[Tuple[str, float]]:
    """Take an example and convert t-statistics into text explanations

    :param examples: Output from :func:`sample_space`
    :param descriptor_type: Type of descriptor, either "maccs", or "ecfp".
    :param count: Number of text explanations to return
    :param presence_thresh: Threshold for presence of descriptor in examples
    :param include_weak: Include weak descriptors. If not set, the function
    will be first have this set to False, and if no descriptors are found,
    will be set to True and function will be re-run
    """
    descriptor_type = descriptor_type.lower()
    # populate lime explanation
    if examples[-1].descriptors is None:
        lime_explain(examples, descriptor_type=descriptor_type)
    nbases = sum([1 for e in examples if e.is_origin])

    # Take t-statistics, rank them
    d_importance = [
        (n, t, i)  # name, t-stat, index
        for i, (n, t) in enumerate(
            zip(
                examples[0].descriptors.plotting_names,
                examples[0].descriptors.tstats,
            )
        )
        # don't want NANs and want match (if not multiple bases)
        if not np.isnan(t)
    ]

    d_importance = sorted(d_importance, key=lambda x: abs(x[1]), reverse=True)
    # get significance value - if >significance, then important else weakly important?
    w = np.array([1 / (1 + (1 / (e.similarity + 0.000001) - 1) ** 5) for e in examples])
    effective_n = np.sum(w) ** 2 / np.sum(w**2)
    if np.isnan(effective_n):
        effective_n = len(examples)
    T = ss.t.ppf(0.975, df=effective_n)

    pos_count = 0
    neg_count = 0
    result = []
    existing_names = set()
    for k, v, i in d_importance:
        if pos_count + neg_count == count:
            break
        name = k
        if name is None or name in existing_names:
            continue
        existing_names.add(name)
        if abs(v) > 4:
            imp = "This is very important for the property\n"
        elif abs(v) >= T:
            imp = "This is important for the property\n"
        elif include_weak:
            imp = "This could be relevant for the property\n"
        else:
            continue
        # check if it's present in majority of base molecules

        present = sum(
            [1 for e in examples if e.descriptors.descriptors[i] != 0 and e.is_origin]
        )
        if present / nbases < (1 - presence_thresh) and v < 0:
            if neg_count == count - 2:
                # don't want to have only negative examples
                continue
            kind = "No and it would be negatively correlated with property (counterfactual)."
            neg_count += 1
        elif present / nbases > presence_thresh and v > 0:
            kind = "Yes and this is positively correlated with property."
            pos_count += 1
        else:
            continue
        # adjust name to be question
        if name[-1] != "?":
            name = "Is there " + name + "?"
        s = f"{name} {kind} {imp}"
        result.append((s, v))
    if len(result) == 0 or pos_count == 0 and include_weak is None:
        return text_explain(
            examples,
            descriptor_type=descriptor_type,
            count=count,
            presence_thresh=presence_thresh,
            include_weak=True,
        )
    return result
