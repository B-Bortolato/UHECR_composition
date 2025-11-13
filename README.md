# Learning the Composition of Ultra-High-Energy Cosmic Rays  
(https://arxiv.org/abs/2212.04760) 

[Status: 
- a simulated dataset with 26 primaries is included as a test set
- script to include detector effects is added
Comming soon:
- composition_inference.py script: computing moments, likelihood, sampling from posterior, most probable composition, confidence levels
- jupyter notebook with example how to use
- utilities: plots, confidence levels of moments, distance between primaries
- bootstrapping with different energy spectra
- machine learning model with Bayesian statistics to predict primary type,
- classification of events based on Xmax
- inclusion of ground data to existing method to infer the composition,
- ...]

This repository implements the method proposed in [this paper](https://arxiv.org/abs/2212.04760) to **infer the composition of ultra-high-energy cosmic rays (UHECRs)**. The method estimates the fraction of different primaries based on the longitudinal profile observables Xmax and its uncertainty dXmax.  

The final output is the **posterior distribution of possible compositions**:

`w = (w_p, w_He, w_Li, ...) ~ P(w | measured data)`

---

## Principle

The procedure to compute the composition can be summarized as follows:

1. **Prepare simulated data**  

   Simulated data is stored as a list of NumPy arrays:  

   ```python
   data = [primary1, primary2, ...]
   ```

   Each primary array has shape `(n_events, 3)`:

   ```python
   primary1 = np.array([
       [energy1, Xmax1, dXmax1],
       [energy2, Xmax2, dXmax2],
       ...
   ])
   ```
   - Energy is in EeV  
   - Xmax and dXmax are in g/cm²

2. **Prepare measured data**  

   Measured events should be in the same format:

   ```python
   data_measured = np.array([
       [energy1, Xmax1, dXmax1],
       [energy2, Xmax2, dXmax2],
       ...
   ])
   ```

3. **Add detector effects** to the simulated data.

4. **Compute moments of Xmax**  

   For each primary, compute the first three non-central moments: `z = (z1, z2, z3)` using the bootstrap method to include statistical and systematic uncertainties.  

   Similarly, compute the same moments for the measured data in the chosen energy interval.

5. **Formulate the likelihood**  

   The model solves:

   ```
   P_sim(z | w) = sum_over_Z [ P(z | primary Z) * w_Z ]
   ```

   The posterior is obtained via Bayes’ theorem:

   ```
   P(w | measured data) ∝ L(measured data | w) * Prior(w)
   ```

   where the likelihood is defined as:

   ```
   L(measured data | w) = integral [ log P_sim(z | w) * P(z | measured data) dz ]
   ```

   For full mathematical details, see the [paper](https://arxiv.org/abs/2212.04760).

6. **Sample compositions**  

   Compositions `w` are sampled using **Nested Sampling** via the [UltraNest](https://johannesbuchner.github.io/UltraNest/) package:

   ```
   w ~ P(w | measured data)
   ```

---

## Installation

```bash
# Clone the repository
git clone <repo_url>
cd <repo_directory>

# Create environment (optional)
conda create -n uhecr python=3.9
conda activate uhecr

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

```python
import numpy as np
from detector_effects import DetectorEffects

# Load raw data
data_raw = np.load('dataset_test/EPOS_xmax_Ebin2.npy', allow_pickle=True)

# Initialize detector effects
logE_start = 18
logE_end = 18.3
deteff = DetectorEffects(data=data_raw, logE_start=logE_start, logE_end=logE_end)

# Include detector effects and get processed data
data_processed = deteff.include(data_unit='EeV')

# Preview first 5 events of the first primary
print(data_processed[0][:5])
```

---

## Citation

If you use this code, please cite:  
**[Learning the Composition of Ultra-High-Energy Cosmic Rays, arXiv:2212.04760](https://arxiv.org/abs/2212.04760)**










