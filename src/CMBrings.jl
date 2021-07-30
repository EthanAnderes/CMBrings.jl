module CMBrings


using XFields
using FFTransforms
using FieldLensing
using SphereTransforms  

using LinearAlgebra
using FFTW
using Statistics 
using SparseArrays
using Distributed
using JLD2
using ProgressMeter
using PyPlot
using PyCall
using NLopt

const module_dir  = joinpath(@__DIR__, "..") |> normpath

# This is temporary and tries to circumvent the issues with Real Symmetric * Complex vec
# @inline function LinearAlgebra.mul!(
#         C::Vector{Complex{T}},
#         A::Symmetric{T,Matrix{T}}, 
#         B::Vector{Complex{T}}
#     ) where T <: LinearAlgebra.BlasReal
#     @inbounds C .= complex.(A*real(B), A*imag(B)) # .* α .+ C .* β
#     return C
# end


function LinearAlgebra.:*(A::Symmetric{T,Matrix{T}}, x::AbstractVector{Complex{T}}) where T <: LinearAlgebra.BlasReal
    complex.(A*real(x), A*imag(x))
end

# TODO: you may also want to do something similar for triangular matrices:
# trmv, Triangular matrix-vector multiplication

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
