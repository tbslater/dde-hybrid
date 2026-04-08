# Implementing pipeline delays within a hybrid system dynamics and agent-based model in Python

[![Python 3.14.2](https://img.shields.io/badge/-Python_3.14.2-a8902b?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) ![Code licence](https://img.shields.io/badge/🛡_Licence-GPL--3.0-8a00c2?style=for-the-badge&labelColor=gray) [![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.19471486-blue?style=for-the-badge&labelColor=gray)](https://doi.org/10.5281/zenodo.19471486)

## Abstract

Implementation of hybrid simulation models in open-source software is rare, yet encouraging its use has the potential to increase the development and credibility of future studies. Python offers promise as an alternative to commercial software. However, whilst it is well-established for simulation approaches in isolation, guidance on its use for hybrid simulation remains in its early stages. In this paper we provide an approach to integrating agent-based and system dynamics models in Python with openly available code for model reuse. Specifically, we discuss two possible methods for implementing pipeline delays within a hybrid simulation model, and reflect on the challenges for both approaches. We present a motivating example of infectious disease modeling alongside our methodology.

## Authors

[![ORCID](https://img.shields.io/badge/ORCID_Thomas_Slater-0009--0007--0838--7499-A6CE39?style=for-the-badge&logo=orcid&logoColor=white)](https://orcid.org/0009-0007-0838-7499) [![ORCID](https://img.shields.io/badge/ORCID_Thomas_Monks-0000--0003--2631--4481-A6CE39?style=for-the-badge&logo=orcid&logoColor=white)](https://orcid.org/0000-0003-2631-4481) [![ORCID](https://img.shields.io/badge/ORCID_Mark_Kelson-0000--0001--7744--3780-A6CE39?style=for-the-badge&logo=orcid&logoColor=white)](https://orcid.org/0000-0001-7744-3780)

-   **Thomas Slater**, PhD Student. Department of Mathematics and Statistics, University of Exeter, UK. Website: <https://experts.exeter.ac.uk/44419-tom-slater>.

-   **Thomas Monks**, Associate Professor in Health Data Science. University of Exeter Medical School, UK. Website: <https://experts.exeter.ac.uk/19244-thomas-monks>.

-   **Mark Kelson**, Professor of Statistics for Health. Department of Mathematics and Statistics, University of Exeter, UK. Website: <https://experts.exeter.ac.uk/26360-mark-kelson>.

## Installation

**Step 1**: clone the repository.

```         
git clone https://github.com/tbslater/dde-hybrid.git
cd dde-hybrid
```

**Step 2**: set up the virtual environment.

```         
conda env create --file environment.yaml
conda activate
```

## Run

From the command line, execute the following:

```         
cd code
python run.py
```

## Structure

The repository is structured as follows:

-   **code**: all python code is stored in this folder.

    -   **/hybrid**: code for the hybrid model is contained within this folder.

        -   **/abm.py**: the agent-based model.

        -   **/hybrid.py**: the hybrid model interface.

        -   **/sd.py**: the system dynamics model.

    -   **/sd**: code for the system dynamics model (not part of the hybrid model).

        -   **/model.py**: the system dynamics model.

    -   **/run.py**: code to run replications of all models and produce figures.

-   **figures**: the plots and table produced when run.py is executed is stored in this folder.

    -   **/comp-results.csv**: error and run times for Erlang approximations.

    -   **/peak-infections-plt.png**: plot associated with the hybrid model.

    -   **/pipeline-delay-plt.png:** plot associated with the system dynamics model.

-   **parameters:** parameters for both models are stored within this folder.

    -   **/parameters.json**: parameters stored as a json file.

## Specifications

Simulations were run on a HP EliteBook 640 G10 with a 1.3GHz Intel Core i5 processor and 16GB of memory under Windows 11. Total model run time was approximately 1.9 hours.

## Citation

If you used the code in this repository, please cite us!

> Slater, T., Monks, T., & Kelson, M. (2026). Implementing pipeline delays within a hybrid system dynamics and agent-based model in Python: source code. GitHub. <https://github.com/tbslater/dde-hybrid>.

Or alternatively, you can cite the archived version on Zenodo.

> Slater, T., Monks, T., & Kelson, M. (2026). Implementing pipeline delays within a hybrid system dynamics and agent-based model in Python: source code. (v1.0.1). Zenodo. <https://doi.org/10.5281/zenodo.19471486>.

## Licence

All code is freely available under the copyleft GNU General Public License (GPL) 3.0. See `LICENSE` for details.

## Funding

This research was funded by the EPSRC DTP from October 2024 to March 2028.

## Acknowledgements

The following sources are acknowledged for their help in developing the model code.

| Reference                                                                                                                                                                                                                                                                          | Used for?                                                                                                    |
|-------------------------------------------|-----------------------------|
| Palmer, G., & Tian, Y. (2021). Source code for Ciw hybrid simulations. (2021-03-12). Zenodo. <https://doi.org/10.5281/zenodo.4601529>                                                                                                                                              | Structure for specifying both the system dynamics model and the hybrid interface were inspired by this code. |
| J. Archbold, S. Clohessy, D. Herath, N. Griffiths and O. Oyebode, An Agent-Based Model of the Spread of Behavioural Risk-Factors for Cardiovascular Disease in City-Scale Populations, PLoS ONE, 19(5): e0303051, 2024. <https://github.com/nathangriffiths/CVD-Agent-Based-Model> | Code associated with this paper was used to build the agent-based component of our model.                    |
| Monks, T. (2025). sim-tools (v0.8.0a). Zenodo. <https://doi.org/10.5281/zenodo.15041282>                                                                                                                                                                                           | Package used for random sampling and seed generation.                                                        |
| Heather, A., Monks, T., Mustafee, N., Harper, A., Alidoost, F., Challen, R., & Slater, T. (2026). DES RAP Book: Reproducible Discrete-Event Simulation in Python and R (Version 0.5.0) [Computer software]. <https://github.com/pythonhealthdatascience/des_rap_book>              | Code from this book was used for importing parameters, parallel processing and building the `README` file.   |
