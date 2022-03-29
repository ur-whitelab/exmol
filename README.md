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

Our package implements the Model Agnostic Counterfactual Compounds with STONED to generate counterfactuals.
A counterfactual can explain a prediction by showing what would have to change in the molecule to change its predicted class. Here is an eample of a counterfactual:

> This package is not popular. If the package had a logo, it would be popular.

In addition to having a changed prediction, a molecular counterfactual must be similar to its base molecule as much as possible. Here is an example of a molecular counterfactual:

<img alt="counterfactual demo" src="https://raw.githubusercontent.com/ur-whitelab/exmol/main/paper/svg_figs/counterfactual.png" width="400">

The counterfactual shows that if the carboxylic acid were an ester, the molecule would be active. It is up to the user to translate this set of structures into a meaningful sentence.

## Usage

Let's assume you have a deep learning model `my_model(s)` that takes in one SMILES string and outputs a predicted binary class.
To generate counterfactuals, we need to wrap our function so that it can take both SMILES and SELFIES, but
it only needs to use one.

We first expand chemical space around the prediction of interest

```py
import exmol

# mol of interest
base = 'Cc1onc(-c2ccccc2Cl)c1C(=O)NC1C(=O)N2C1SC(C)(C)C2C(=O)O'

samples = exmol.sample_space(base, lambda smi, sel: my_model(smi), batched=False)
```

Here we use a `lambda` to wrap our function and indicate our function can only take one SMILES string, not a list of them with `batched=False`.
Now we select counterfactuals from that space and plot them.

```py
cfs = exmol.cf_explain(samples)
exmol.plot_cf(cfs)
```

<img alt="set of counterfactuals" src="https://raw.githubusercontent.com/ur-whitelab/exmol/main/paper/svg_figs/rf-simple.png" width="500">

We can also plot the space around the counterfactual. This is computed via PCA of the affinity matrix -- the similarity with the base molecule.
Due to how similarity is calculated, the base is going to be the farthest from all other molecules. Thus your base should fall on the left (or right) extreme of your plot.

```py
cfs = exmol.cf_explain(samples)
exmol.plot_space(samples, cfs)
```
<img alt="chemical space" src="https://raw.githubusercontent.com/ur-whitelab/exmol/main/paper/svg_figs/rf-space.png" width="600">

Each counterfactual is a Python `dataclass` with information allowing it to be used in your own analysis:

```py
print(cfs[1])
```
```
{
'smiles': 'Cc1onc(-c2ccccc2Cl)c1C(=O)NC1C(=O)N2C1SC(C)(C)C2C',
'selfies': '[C][C][O][N][=C][Branch1_1][Branch2_3][C][=C][C][=C][C][=C][Ring1][Branch1_2][Cl][C]
            [Expl=Ring1][N][C][Branch1_2][C][=O][N][C][C][Branch1_2][C][=O][N][C][Ring1][Branch1_1][S][C]
            [Branch1_1][C][C][Branch1_1][C][C][C][Ring1][Branch1_3][C]',
'similarity': 0.8,
'yhat': 1,
'index': 1813,
'position': array([-7.8032394 ,  0.51781263]),
'is_origin': False,
'cluster': -1,
'label': 'Counterfactual 1'
}
```

## Chemical Space

When calling `exmol.sample_space` you can pass `preset=<preset>`, which can be
one of the following:

* `'narrow'`: Only one change to molecular structure, reduced set of possible bonds/elements
* `'medium'`: Default. One or two changes to molecular structure, reduced set of possible bonds/elements
* `'wide'`: One through five changes to molecular structure, large set of possible bonds/elements
* `'chemed'`: A restrictive set where only pubchem molecules are considered.
* `'custom'`: A restrictive set where only molecules provided by the "data" key are considered.

You can also pass `num_samples` as a "request" for number of samples. You will typically end up with less due to
degenerate molecules. See API for complete description.

## SVG

Molecules are by default drawn as PNGs. If you would like to have them drawn as SVGs, call `insert_svg` after calling
`plot_space` or `plot_cf`

```py
import skunk
exmol.plot_cf(exps)
svg = exmol.insert_svg(exps, mol_fontsize=16)

# for Jupyter Notebook
skunk.display(svg)

# To save to file
with open('myplot.svg', 'w') as f:
    f.write(svg)
```

This is done with the [skunk🦨 library](https://github.com/whitead/skunk).

## Quiet Progress Bars

If `exmol` is being too loud, add `quiet = True` to `sample_space` arguments.

## API and Docs

[Read API here](https://ur-whitelab.github.io/exmol/api.html). You should also read the paper (see below) for a more exact
description of the methods and implementation.

## Developing

This repo uses pre-commit, so after cloning run `pip install -r requirements.txt` and `pre-commit install` prior to committing.

## Citation

Please cite [Wellawatte et al.](https://pubs.rsc.org/en/content/articlelanding/2022/sc/d1sc05259d#!divAbstract)

```bibtex
@Article{wellawatte_seshadri_white_2021,
author ="Wellawatte, Geemi P. and Seshadri, Aditi and White, Andrew D.",
title  ="Model agnostic generation of counterfactual explanations for molecules",
journal  ="Chem. Sci.",
year  ="2022",
pages  ="-",
publisher  ="The Royal Society of Chemistry",
doi  ="10.1039/D1SC05259D",
url  ="http://dx.doi.org/10.1039/D1SC05259D",
}
```
