module CMBrings

using XFields
using FFTransforms
using FieldLensing
using SphereTransforms  # specify what we need from this
using HealpixTransforms # do we need this?

using LinearAlgebra
using Statistics 
using SharedArrays
using Distributed
using JLD2
using FFTW
using ProgressMeter
using PyPlot

const module_dir  = joinpath(@__DIR__, "..") |> normpath

include("az_blocks.jl")

include("az_cov.jl")

include("methods.jl")

include("plot.jl")

LinearAlgebra.dot(f::Xfield{FT},g::Xfield{FT}) where FT<:𝕎 = dot(f[:], g[:])

LinearAlgebra.adjoint(C::DiagOp) = C  

# Lensing 
# ======================================


struct Nabla!{Tθ,Tφ}
    ∂θ::Tθ
    ∂φᵀ::Tφ
end

function (∇!::Nabla!{Tθ,Tφ})(∇y::NTuple{2,A}, y::NTuple{2,A}) where {Tθ,Tφ,Tf,A<:Array{Tf,2}}
    mul!(∇y[1], ∇!.∂θ, y[1])
    mul!(∇y[2], y[2], ∇!.∂φᵀ)
    ∇y
end

function (∇!::Nabla!{Tθ,Tφ})(∇y::NTuple{2,A}, y::A) where {Tθ,Tφ,Tf,A<:Array{Tf,2}}
    ∇!(∇y, (y,y))
end

function (∇!::Nabla!{Tθ,Tφ})(y::A) where {Tθ,Tφ,Tf,A<:Array{Tf,2}}
    ∇y = (similar(y), similar(y))
    ∇!(∇y, (y,y))
    ∇y
end



end
