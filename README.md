# Explaining why that molecule

![tests](https://github.com/ur-whitelab/exmol/actions/workflows/tests.yml/badge.svg) ![paper](https://github.com/ur-whitelab/exmol/actions/workflows/paper.yml/badge.svg) [![docs](https://github.com/ur-whitelab/exmol/actions/workflows/docs.yml/badge.svg)](https://ur-whitelab.github.io/exmol/)

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

<img alt="counterfactual demo" src="https://raw.githubusercontent.com/ur-whitelab/exmol/main/paper/counterfactual.png" width="600">

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

<img alt="set of counterfactuals" src="https://raw.githubusercontent.com/ur-whitelab/exmol/main/paper/rf-simple.png" width="600">

We can also plot the space around the counterfactual:

```py
cfs = exmol.cf_explain(samples)
exmol.plot_space(samples, cfs)
```
<img alt="chemical space" src="https://raw.githubusercontent.com/ur-whitelab/exmol/main/paper/rf-space.png" width="600">

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
## API and Docs

[Read docs here](https://ur-whitelab.github.io/exmol/)

## Citation

Please cite [Wellawatte et al.](https://arxiv.org/abs/)

```bibtex
@article{wellawatte2021model,
  title={Model agnostic generation of counterfactual explanations for molecules},
  author={Wellawatte, Geemi P and Seshadri, Aditi and White, Andrew D},
  journal={arXiv preprint arXiv:????},
  year={2021}
}
```
