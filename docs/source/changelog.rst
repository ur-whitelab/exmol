Change Log
==========

v3.2.1 (2025-03-08)
-------------------
* Added flag to `get_functional_groups` to get even more functional groups


v3.2.0 (2025-03-07)
-------------------
* Fixed paper github actions
* Added `get_functional_groups` method to get functional groups from SMILES

v3.1.0 (2024-11-21)
-------------------
* Removed langchain and switched to use openai API directly

v3.0.3 (2023-06-19)
-------------------
* Now compatible with python 3.11


v3.0.2 (2023-02-23)
-------------------
* Now try to find largest component with multiple compounds separated via `.` in SMILES
* Added langchain for text explanations


v3.0.1 (2023-02-02)
-------------------
* Removed `~` any bonds from SMILES strings coming from synspace


v3.0.0 (2023-01-24)
-------------------
* Added `synspace` method to generate synthetically feasible space
* Added drug-like filter by default for counterfactuals


v2.2.1 (2022-12-7)
-------------------
* Fixed bug in sorting for text explanations
* Fixed empty plot names saying `None`
* Added priority for naming and removed invalid names
* Added more names (metyhl, ethyl, butyl, etc)
* Fixed `sample_space` to accept partials or objects
* Added openai prompting
* Added `name_morgan_bit` as external facing

v2.2.0 (2022-11-3)
-------------------
* Added natural language explanation method

v2.1.1 (2022-6-03)
------------
* Fixed plotting bug in `plot_descriptors`

v2.1.0 (2022-6-03)
------------
* `plot_descriptors` will no longer save to file without filename

v2.0.1 (2022-5-31)
------------
* Made default `run_stoned` argument use basic instead of semantically robust alphabet, as
    claimed in documentation

v2.0.0 (2022-5-31)
------------
* Added surrogate model explanation method
* Added support for attributing ECFP, MACCS fingerprints, rdkit descriptors and plotting them
* Example notebooks for new method
* Fixed chirality stripping in sanitize
* Made it possible to use multiple base molecules for ECFP descriptors


v1.1.0 (2022-5-2)
-------------------
* Made it possible to have model function that only takes one argument instead of SMILES and SELFIES

v1.0.2 (2022-5-2)
-------------------
* Switched to bulk Tanimoto to improve speed
* Tightened chemed api limit


v1.0.1 (2022-4-4)
-------------------
* Added quiet mode

v1.0.0 (2022-1-21)
-------------------
* Switched to SELFIES v2.0 release, which has a different token list
* Removed "experimental" tag from Chemed and Custom methods
* Type annotations now pass ``mypy``
* Paper models now generate SVGs correctly and fixed token issues

v0.6.0 (2022-1-17)
-------------------
* Changed behavior of num_samples so that it is not affected by mutation count in STONED

v0.5.2 (2022-1-4)
-------------------
* Fixed SMILES escaping in URL in chemed

v0.5.1 (2021-11-16)
-------------------
* Fixed similarity float vs int in chemed

v0.5.0 (2021-10-26)
-------------------
* Added custom lists for counterfactual source (contributed by @maclandrol)

v0.4.1 (2021-09-27)
-------------------
* Fixed randint usage in stoned.py

v0.4.0 (2021-09-17)
-------------------
* Refactored code into files
* Added SVG rewrite so mol structures are SVGs
* SVGs are handled with skunks
* Added cartoon style for scatter plot in plot_space


v0.3.2 (2021-09-02)
-------------------
* Added SMILES sanitization before generating sample space

v0.3.1 (2021-08-26)
-------------------
* Fixed num_samples not passed correctly
* Fixed unparsable simles coming from pubchem

v0.3.0 (2021-08-25)
--------------------

Added CHEMED method and progress bar

v0.2.0 (2021-08-14)
--------------------

Initial package release
