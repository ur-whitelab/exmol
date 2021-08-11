# exmol

`exmol` is a package to explain black-box predictions of molecules. The package uses model agnostic explanations to help users understand why a molecule is predicted to have a property.

## Install

```sh
pip install exmol
```

## Counterfactual Generation

Our package implements the Model Agnostic Counterfactual Compounds with STONED (MACCS) to generate counterfactuals.
A counterfactual is method of explaining a prediction by showing what would have to change in the molecule to change its predicted class.

Here is an example of a counterfactual:

The counterfactual shows that if the carboxylic acid were an ester, the molecule would be active. It is up to the user to translate this into a meaningful sentence and examine the multiple counterfactuals.

## Usage

Let's assume you have a model `my_model(s)` that takes in one SMILES string and outputs a predicted binary class.
To generate counterfactuals, we need to wrap our function so that it can take both SMILES and SELFIES, but
it only needs to use one.

We first expand chemical space around the prediction of interest

```py
import exmol

# mol of interest
base = 'CCCO'

samples = exmol.sample_space(base, lambda smi, sel: my_model(smi), batched=False)
```

Now we select counterfactuals from that space and plot them

```py
cfs = exmol.cf_explain(samples)
exmol.plot_cf(cfs)
```

We can also plot the space around the counterfactual:

```py
cfs = exmol.cf_explain(samples)
exmol.plot_space(samples, cfs)
```

