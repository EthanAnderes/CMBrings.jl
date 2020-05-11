module CMBrings

using Reexport
@reexport using XFields
@reexport using FFTransforms
using FFTransforms: FFTR
using LinearAlgebra
using Statistics 
using HealpixHelper
const HH = HealpixHelper
const module_dir  = joinpath(@__DIR__, "..") |> normpath

include("spin_transforms.jl")
export RingSpinTransform, RingS2Transform

include("xfield_extras.jl")

include("grid_extras.jl")
export	Δpix, Δfreq, nyq, Ωx, Ωk,
		inv_scale, unitary_scale, ordinary_scale,
		pix, freq, fullpix, fullfreq, wavenum

include("methods.jl")
export ωη




end
