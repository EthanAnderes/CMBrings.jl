module CMBrings


using XFields
using FFTransforms
using SphereTransforms
using HealpixTransforms

import LinearAlgebra: dot
using LinearAlgebra

using Statistics 
using SharedArrays
using Distributed
using JLD2
using FFTW
using ProgressMeter
using PyPlot


const module_dir  = joinpath(@__DIR__, "..") |> normpath

include("cov_sheets.jl")

include("methods.jl")

include("plot.jl")


#include("fft_ring_transforms.jl")
#export RingSpinTransform, RingS2Transform

#include("xfield_extras.jl")

# include("grid_extras.jl")
# export	Δpix, Δfreq, nyq, Ωx, Ωk,
# 		inv_scale, unitary_scale, ordinary_scale,
# 		pix, freq, fullpix, fullfreq, wavenum





end
