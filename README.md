# Learning the Composition of Ultra-High-Energy Cosmic Rays 

This repository contains scripts and instructions for calculating the fractions of H, He … Fe nuclei present in Ultra-High-Energy Cosmic Rays (UHECR) by comparing the moments of the Xmax distribution between measured and simulated data. The method is based on the paper **Learning the Composition of Ultra-High-Energy Cosmic Rays** (https://arxiv.org/abs/2212.04760) and focuses primarily on inferring the UHECR composition from publicly available Pierre Auger Observatory data.



**Strong points**:
+ Likelihood-free approach (it uses a universal likelihood derived by comparing moments of distributions between measured and simulated data).
+ Inclusion of all uncertainties, including **statistical uncertainties of simulated data**, via the bootstrap method.
+ Requires at least 50 measured data points to apply this method.
+ Output: samples from the posterior distribution of possible compositions (high-dimensional) obtained with **Nested Sampling**.
+ Output: log-likelihood values as a function of the confidence interval, making it easy to calculate confidence intervals.




For testing purposes, we added simulated data consisting of 1000 events per primary particle,
with 26 primary particles in total (for each atomic number Z from 1 (proton) to 26 (Fe nuclei), we selected the most stable nucleus).
The simulations were produced with CORSIKA using the EPOS hadronic model.

Before using the simulated data—for example, to infer the composition of UHECR from the publicly available dataset of the Pierre Auger Observatory—**detector effects must be applied to the simulated data** (see the paper or the Jupyter notebook example for details on how this is done).






---
## Method to compute the composition of UHECR

The method uses parameters from the longitudinal profile of UHECR events (additional parameters can be included easily):

- **Xmax** [g/cm²] (slant depth)
- **dXmax** [g/cm²] (uncertainty of the slant depth)
- **E** [EeV] (estimated energy of the primary particle)

The goal is to compute the fractions of primary particles such as H, He, … Fe nuclei.
For short, we write the composition as: `w = (w_p, w_He, w_Li, …)`.

Due to statistical uncertainties, systematic effects, and degeneracies, the inferred composition is not unique; many different compositions may be compatible with the data.
The output of the method consists of samples from the **posterior distribution of possible compositions**:

`w = (w_p, w_He, w_Li, ...) ~ P(w | measured data)`

The method makes use of the **Nested Sampling** algorithm implemented in the
[UltraNest](https://johannesbuchner.github.io/UltraNest/) Python package
to sample from the posterior. The posterior is obtained via Bayes’ theorem as  
**posterior ∝ likelihood × prior**, where the prior can be chosen as a flat Dirichlet
(all compositions are equally probable before looking at the data).


We use the **bootstrap method** to calculate the distribution of moments of Xmax for both measured and simulated data.
For simulated data, we employ a trick: we reuse bootstrapped samples to obtain the distribution of moments for any composition `w`
(an array of fractions):

- `P(<Xmax>, <Xmax²>, <Xmax^3> | w, simulated data),`
- `P(<Xmax>, <Xmax²>, <Xmax^3> | measured data).`

Both distributions of moments tend to be approximately normal due to the **Central Limit Theorem**.
Based on this, the log-likelihood takes the form of a cross-entropy:

- `logL(w) = integral [ log P(<Xmax>, <Xmax²>, <Xmax^3> | w, simulated data) * P(<Xmax>, <Xmax²>, <Xmax^3> | measured data) d<Xmax> d<Xmax²> d<Xmax^3> ]`


The integrals are analytical; see the paper for the explicit expression and details.
The implementation in this repository uses **central moments** instead of **non-central moments**.  
This does not affect the results — the same information is encoded in both.





## Notes

- The energy of the particle is required to correctly include detector effects.  
- Composition should be calculated in narrow log-energy intervals because the bootstrapped methods included in the scripts
  do not account for the relation Xmax ∝ log(E). For larger intervals, assuming a constant composition within the interval,
  please include energy in the bootstrapping method (this will be added later).


---

## Installation & Quick start

```bash
# Clone the repository
git clone https://github.com/B-Bortolato/UHECR_composition.git


# Create environment (optional)
conda create -n uhecr python=3.9
conda activate uhecr

# Install dependencies
pip install -r requirements.txt

# Open Jupyter notebook - containts examples of use with comments
```

---

## Citation

If you find this code useful, please cite:  
**[Learning the Composition of Ultra-High-Energy Cosmic Rays, arXiv:2212.04760](https://arxiv.org/abs/2212.04760)**










