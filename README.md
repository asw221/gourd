
# Gourd: Analysis of Cortical Surface fMRI

#### Dependencies
 - [boost](https://www.boost.org/)
 - [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page)
 - [Expat](https://libexpat.github.io) - (Likely already be on your
   system) 
 - [OpenMP](https://www.openmp.org/)
 - [zlib](https://www.zlib.net/) - (Likely already be on your system)

 
We also require a `C`/`C++` compiler compatable with the `C++17`
standard (e.g. `gcc` >= `8.3.0` should suffice).


#### Notes
 - Cannot include both `gifti_io.h` and `nifti2_io.h` in the same
   code/file. Interference/redifinition of types
   - Resolved.

#### To-Do

 - []  Figure out specific contrast image storage format and duplicate
 - []  Profile sampling schemes for random effects version
 - []  Weighted regression version(s)

 
#### Installation
Using cmake with dependencies installed:
```
mkdir gourd/build && cd gourd/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```



 - [itk](https://itk.org)
