# Deep-GIST
====

"Deep GIST" is a deep-learning model for rapidly estimating the distribution of hydration free energy around a protein. It's trained on data from grid inhomogeneous solvation theory (GIST). Computation is completed within a few minutes using a single CPU and approximately one minute with a single GPU.
Model weights (415 MB) are available at [Zenodo](https://zenodo.org/record/XXXXXX)

====
## Requirement
Python 3.7~, tensorflow  

## License
“Deep GIST” is available under the GNU General Public License.

## Citing this work
If you use “Deep GIST”, please cite:

```
Deep GIST: A Deep-Learning Model for Predicting the Distribution of Hydration Thermodynamics around Proteins
Yusaku Fukushima and Takashi Yoshidome
XXX (2025).
```
## Contact
If you have any questions, please contact Takashi Yoshidome at takashi.yoshidome.b1@tohoku.ac.jp.

## Usage
* Add hydrogens to the protein in advance.
* Run the Jupyter Lab cells of "prediction.ipynb" in order.
