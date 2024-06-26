
# Gourd: Analysis of Cortical Surface fMRI

#### Dependencies
 - [Expat](https://libexpat.github.io) - (Likely already be on your
   system) 
 - [zlib](https://www.zlib.net/) - (Likely already be on your system)
 
 
Additionally, `gourd` will  make use of mathematical functions from
[boost](https://www.boost.org/) if it is available. `gourd` includes a
copy of [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page)
internally. 

We require a `C`/`C++` compiler compatible with the `C++17`
standard (e.g. `gcc` >= `8.3.0` should suffice).


 
#### Installation
Using cmake with dependencies installed:
```properties
mkdir gourd/build && cd gourd/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```



### Estimation of mean process covariance parameters
Estimation of the spatial mean process covariance parameters can be
accomplished using the `gourd_covest` program. This function requires,
at minimum, a list of data files stored in the CIFTI/NIFTI-2 file
format, and a GIFTI shape file. Paths to multiple data files can be
typed one after the other, or wildcard completions can be used for
convenience.

Basic syntax might look like the following:
```properties
./gourd_covest path/to/data*.nii --surface path/to/surf.gii \
  --radius 6.0 \    # Nearest Neighbor Gaussian Process radius
  --tol 1e-8   \   # Optimization tolerance (Default = 1e-6)
  --radial-basis
```


Implemented covariance function options include:
```console
Covariance Functions:
  --radial-basis       (Default) 
  --rational-quadratic 
  --matern 
```
All of these are treated as three parameter covariance functions, with
the parameters corresponding roughly to mean process (i) marginal
variance, (ii) correlation bandwidth, and (iii) smoothness. The
`--matern` option is implemented directly using modified cylindrical
Bessel functions, and may be somewhat slow.

The distance metric options include:
```console
Distance Metrics:
  --great-circle       (Default) 
  --euclidean 
```



### Bayesian estimation of spatially varying coefficient (SVC) regression models

`Gourd` contains several different programs that can be used to
estimate group-level cortical surface SVC regression models. The
`gourd_gplm` function, for example, fits the working regression model
and is suitable for moderate to very large data sets. As above, we
require input in the form of CIFTI/NIFTI-2 outcome images and a GIFTI
shape file.

Basic syntax might look like the following:
```properties
./gourd_gplm path/to/data*.nii --surface path/to/surf.gii \
  --covariates path/to/x.csv  # Mean model design matrix \
  --radial-basis              # GP Covariance function selection \
  --theta 1 0.08 1            # GP Covariance function hyperparams \
  --neighborhood 8            # NNGP radius (for SVCs) \
  --subset path/to/subs.csv   # Filename tokens to subset data*.nii \
  -o path/to/output/prefix    # Output file location and prefix \
  --burnin 4000               # MCMC burnin iterations \
  --samples 1000              # MCMC samples to save \
  --thin 5                    # MCMC post-burnin thinning rate \
  --steps 12                  # HMC numerical integration steps \
  --neighborhood-mass 2       # HMC mass matrix radius \
  --seed 48109                # URNG seed
```

Related model fitting programs include:
- Above is example syntax for a fast "working" model (see [References](#References)).
- `gourd_gplmme` for a "marginal" model variant. This is generally
  more accurate than the above working model, but is also slower and
  more resource intensive.
- `gourd_gplmce` for a "conditional" model variant. We do not
  recommend this version for general practice.
- `gourd_sglm` for analysis with the standard vertex-wise general
  linear model. Our treatment here uses a Bayesian implementation
  with independent Jeffreys priors and Gibbs updates to facilitate
  comparisons with the alternatives above.



#### To-Do
 - [] Enhance help pages



#### References

Whiteman, Johnson, and Kang (2024+) "Bayesian inference for
group-level cortical surface image-on-scalar-regression with Gaussian
process priors." [ArXiv](https://arxiv.org/abs/2306.03663).
