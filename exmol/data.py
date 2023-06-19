from dataclasses import dataclass, asdict, field
from typing import Optional
import numpy as np  # type: ignore


@dataclass
class Descriptors:
    """Molecular descriptors"""

    #: Descriptor type
    descriptor_type: str
    #: Descriptor values
    descriptors: tuple
    # Descriptor name
    descriptor_names: tuple
    # plotting name
    plotting_names: tuple = ()
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
    position: np.ndarray = field(default_factory=lambda: np.array(None))
    #: True if base
    is_origin: bool = False
    #: Index of cluster, can be -1 for no cluster
    cluster: int = 0
    #: Label for this example
    label: str = None  # type: ignore
    #: Descriptors for this example
    descriptors: Descriptors = None  # type: ignore

    # to make it look nicer
    def __str__(self):
        return str(asdict(self))
