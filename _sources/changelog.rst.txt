Change Log
==========
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
