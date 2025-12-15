
## Input data
The analysis relies on the following ROOT files (not stored in the repository):

- `data/FP.root`  
  Contains reference L1 collections (`L1nnPuppiTaus`, `L1PFJets`) built from
  non-extended L1 PF candidates, as well as `GenVisTaus` used for matching.

- `data/SC.root`  
  Contains extended L1 PF candidate information used as input for the
  seeded-cone clustering.  
  The file also includes the seeded-cone clusters reconstructed by the
  reference scouting implementation, used for comparison with the Python
  algorithm.

## Grid search and clustering (`gridsearch.ipynb`)
The notebook performs a grid search to optimise the parameters of the
seeded-cone clustering algorithm starting from L1 PF candidates.

Steps:
1. Load `FP.root` and `SC.root`.
2. Run a grid search to determine the optimal clustering parameters.
3. Apply the optimised algorithm to all events.
4. Produce a new ROOT file (`data/SC_python.root`) containing:
   - extended L1 PF information from `SC.root`,
   - two additional branches:
     - seed flag for each PF candidate,
     - cluster index assigned to each PF candidate,
   - reference L1 algorithm outputs (`L1nnPuppiTaus`, `L1PFJets`) and
     `GenVisTaus` information from `FP.root`.

The resulting file contains all the information required for efficiency studies.

## Efficiency and debugging studies (`efficiencies_debugging.ipynb`)
This notebook loads `data/SC_python.root` and performs:

- efficiency studies in the barrel and endcap regions,
- comparisons between the seeded-cone algorithm and AK4 jets,
- interactive per-event and per-object visualisation using Plotly.

Interactive plots are saved as HTML files for detailed inspection and debugging.

## Notes
- ROOT input files are excluded from the repository due to size.
