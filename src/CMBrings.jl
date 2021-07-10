module CMBrings

using XFields
using FFTransforms
using FieldLensing
using SphereTransforms  
# using Spectra # needed for complex_circ_rings

using LinearAlgebra
using Statistics 
using SharedArrays
using Distributed
using JLD2
using FFTW
using ProgressMeter
using PyPlot
using PyCall
using NLopt

const module_dir  = joinpath(@__DIR__, "..") |> normpath


## TODO: add method for checking the FFTransforms transform is the right one for CMBrings

# Extras on SphereTransforms like simulation, getindex etc. 
include("transformations.jl")

include("circ_op.jl")
export CircOp, field2▪, ▪2field

include("lensing.jl")

include("likelihoods.jl")

include("methods.jl")

include("plot.jl")

end
