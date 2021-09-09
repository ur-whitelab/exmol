from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class Descriptors:
    """Molecular descriptors"""

    #: Descriptor names
    descriptors: tuple
    # Descriptor value
    descriptor_names: tuple
    # t_stats for each molecule
    tstats: tuple = ()


@dataclass
class Example:
    """Example of a molecule"""

    #: SMILES string for molecule
    smiles: str
    #: SELFIES for molecule, as output from :func:`selfies.encoder`
    selfies: str
    #: Tanimoto similarity relative to base
    similarity: float
    #: Output of model function
    yhat: float
    #: Index relative to other examples
    index: int
    #: PCA projected position from similarity
    position: np.ndarray = None
    #: True if base
    is_origin: bool = False
    #: Index of cluster, can be -1 for no cluster
    cluster: int = 0
    #: Label for this example
    label: str = None
    #: Descriptors for this example
    descriptors: Descriptors = None

    # to make it look nicer
    def __str__(self):
        return str(asdict(self))
