# Deep GIST

"Deep GIST" is a deep-learning model for rapidly estimating the distribution of hydration free energy around a protein. It's trained on data from grid inhomogeneous solvation theory (GIST). Computation is completed within a few minutes using CPUs and approximately one minute with a single GPU.
Model weights (415 MB) will be available at [Zenodo](https://zenodo.org/record/XXXXXX) after the acceptance of the manuscript.

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
* If you want to compute ∆G<sub>W,replace</sub>, the following computation is conducted:
  1. Water molecules are placed on to the protein surface using [gr Predictor](https://github.com/YoshidomeGroup-Hydration/gr-predictor) and [Placevent](https://github.com/dansind/Placevent/tree/master) programs. Other programs for the placement of water molecules can also be exploited, while we do not check whether the same results are obtained.
  2. Prepare PDB files each of which consists of a protein and one water molecule. Note that "TER" must be required between the protein and a water molecule. See sample.pdb as an example.
  3. Prepare "list.txt" in which the PDB names are listed.
  4. Revise the lines of "protein = ", "path_dx_pred = ", "listfile = ", and "path_pdb = " in comput_SumGISTMap.py.
  5. Run comput_SumGISTMap.py.
