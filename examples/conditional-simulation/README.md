
# Summary of notebook 

∙ Streamline the CMBrings WF+sim. 
∙ Add lensing?
∙ Compare to healpix fullsky and fasttransforms


# Environment for this directory 

```julia
using Pkg
pkg"activate ."
pkg"add https://github.com/EthanAnderes/XFields.jl"
pkg"add https://github.com/EthanAnderes/Spectra.jl"
pkg"add https://github.com/EthanAnderes/FFTransforms.jl"
pkg"add https://github.com/EthanAnderes/HealpixTransforms.jl"
pkg"add https://github.com/EthanAnderes/SphereTransforms.jl"
pkg"add https://github.com/EthanAnderes/LBblocks.jl"
# note: you need to make sure all the non-registered deps on CMBrings are loaded first
pkg"add https://github.com/EthanAnderes/CMBrings.jl"

pkg"add Distributed"
pkg"add LinearAlgebra"
pkg"add FFTW"
pkg"add DelimitedFiles"
pkg"add Dierckx"
pkg"add PyCall"
pkg"add PyPlot"
pkg"add BenchmarkTools"
pkg"add JLD2"
pkg"add Literate"
```

Doing package update will grab current versions of all of the above and 


Now you can run the code in the directory using the saved Manifest/Project by doing this 

```
# open julia in the project directory
julia> using Pkg
julia> activate .
julia> instantiate
julia> include("make.jl")
```




