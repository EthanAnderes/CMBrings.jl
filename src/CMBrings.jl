module CMBrings


using XFields
using FFTransforms
using HealpixTransforms

using LinearAlgebra
using Statistics 
using SharedArrays
using Distributed
using FFTW: plan_rfft 

const module_dir  = joinpath(@__DIR__, "..") |> normpath

include("cov_sheets.jl")

#include("fft_ring_transforms.jl")
#export RingSpinTransform, RingS2Transform

#include("xfield_extras.jl")

# include("grid_extras.jl")
# export	Δpix, Δfreq, nyq, Ωx, Ωk,
# 		inv_scale, unitary_scale, ordinary_scale,
# 		pix, freq, fullpix, fullfreq, wavenum

# include("methods.jl")
# export ωη




end
