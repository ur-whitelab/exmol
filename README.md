# Explaining why that molecule

[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ur-whitelab/exmol)
[![tests](https://github.com/ur-whitelab/exmol/actions/workflows/tests.yml/badge.svg)](https://github.com/ur-whitelab/exmol) [![paper](https://github.com/ur-whitelab/exmol/actions/workflows/paper.yml/badge.svg)](https://github.com/ur-whitelab/exmol) [![docs](https://github.com/ur-whitelab/exmol/actions/workflows/docs.yml/badge.svg)](https://ur-whitelab.github.io/exmol/)
[![PyPI version](https://badge.fury.io/py/exmol.svg)](https://badge.fury.io/py/exmol)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

`exmol` is a package to explain black-box predictions of molecules. The package uses model agnostic explanations to help users understand why a molecule is predicted to have a property.

## Install

```sh
pip install exmol
```

## Counterfactual Generation

Our package implements the Model Agnostic Counterfactual Compounds with STONED (MACCS) to generate counterfactuals.
A counterfactual can explain a prediction by showing what would have to change in the molecule to change its predicted class. Here is an eample of a counterfactual:

> This package is not popular. If the package had a logo, it would be popular.

In addition to having a changed prediction, a molecular counterfactual must be similar to its base molecule as much as possible. Here is an example of a molecular counterfactual:

<img alt="counterfactual demo" src="paper/counterfactual.svg" width="400">

The counterfactual shows that if the carboxylic acid were an ester, the molecule would be active. It is up to the user to translate this set of structures into a meaningful sentence.

## Usage

Let's assume you have a deep learning model `my_model(s)` that takes in one SMILES string and outputs a predicted binary class.
To generate counterfactuals, we need to wrap our function so that it can take both SMILES and SELFIES, but
it only needs to use one.

We first expand chemical space around the prediction of interest

```py
import exmol

# mol of interest
base = 'CCCO'

samples = exmol.sample_space(base, lambda smi, sel: my_model(smi), batched=False)
```

Here we use a `lambda` to wrap our function and indicate our function can only take one SMILES string, not a list of them with `batched=False`.
Now we select counterfactuals from that space and plot them.

```py
cfs = exmol.cf_explain(samples)
exmol.plot_cf(cfs)
```

<img alt="set of counterfactuals" src="paper/rf-simple.svg" width="400">

We can also plot the space around the counterfactual. This is computed via PCA of the affinity matrix -- the similarity with the base molecule.
Due to how similarity is calculated, the base is going to be the farthest from all other molecules. Thus your base should fall on the left (or right) extreme of your plot.

```py
cfs = exmol.cf_explain(samples)
exmol.plot_space(samples, cfs)
```
<img alt="chemical space" src="paper/rf-space.svg" width="600">

Each counterfactual is a Python `dataclass` with information allowing it to be used in your own analysis:

```py
print(cfs[0])
```
```
Examples(
  smiles='CCOC(=O)c1ccc(N=CN(Cl)c2ccccc2)cc1',
  selfies='[C][C][O][C][Branch1_2][C][=O][C][=C][C][=C][Branch1_1][#C][N][=C][N][Branch1_1][C][Cl][C][=C][C][=C][C][=C][Ring1][Branch1_2][C][=C][Ring1][S]',
  similarity=0.8181818181818182,
  yhat=-5.459493637084961,
  index=1807,
  position=array([-6.11371691,  1.24629293]),
  is_origin=False,
  cluster=26,
  label='Counterfactual')
```

## Chemical Space

When calling `exmol.sample_space` you can pass `preset=<preset>`, which can be
one of the following:

* `'narrow'`: Only one change to molecular structure, reduced set of possible bonds/elements
* `'medium'`: Default. One or two changes to molecular structure, reduced set of possible bonds/elements
* `'wide'`: One through five changes to molecular structure, large set of possible bonds/elements
* `'chemed'`: A restrictive set where only pubchem molecules are considered. *Experimental*

You can also pass `num_samples` as a "request" for number of samples. You will typically end up with less due to
degenerate molecules. See API for complete description.

## API and Docs

[Read API here](https://ur-whitelab.github.io/exmol/api.html). You should also read the paper (see below) for a more exact
description of the methods and implementation.

## Citation

Please cite [Wellawatte et al.](https://chemrxiv.org/engage/chemrxiv/article-details/6115baf04cb4797dc42df605)

```bibtex
 @article{wellawatte_seshadri_white_2021,
 place={Cambridge},
 title={Model agnostic generation of counterfactual explanations for molecules},
 DOI={10.33774/chemrxiv-2021-4qkg8},
 journal={ChemRxiv},
 publisher={Cambridge Open Engage},
 author={Wellawatte, Geemi P and Seshadri, Aditi and White, Andrew D},
 year={2021}}
```
This content is a preprint and has not been peer-reviewed.


