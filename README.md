Code for "Evolution of error correction through a need for speed"

### Citing
```
@article{
  author       = {R. Ravasio, K. Husain, C. G. Evans, R. Phillips, M. Ribezzi, J. W. Szostak, A. Murugan},
  title        = {Evolution of error correction through a need for speed},
  year         = 2024,
  doi          = {doi.org/10.48550/arXiv.2405.10911},
}
```
### Kinetic proofreading

The codes used for the model of kinetic proofreading and the in-silico evolution for speed of a random Markov network. General functions used across the Python notebooks are defined modularly in the Python codes and loaded in each notebook. The data used to make the paper figures can be downloaded from [] and loaded in the notebooks. 

### Self-assembly

Code used for self-assembly simulations.  Python files `sim-*` are used to generate simulation data, while Jupyter notebooks `plot-*` generate the plot panels for the main text and SI.  To use data generated for the paper, download data from [Zenodo 10.5281/zenodo.17478191](https://zenodo.org/uploads/17478191) and extract the zip file in the self-assembly directory.  For both simulation and plotting, creating a virtual environment with the requirements.txt file is recommended.

### Plotting the data from the Supplementary Table of Ravikumar et al. 2018

The notebook used for plotting and looking at the data reported in Table 2 of the Supplementary Material of Ravikumar et al. _Cell_ **175** 2018, as well as a reformatted version of Table 2. The reformatting is necessary to be able to load the data with pandas.
