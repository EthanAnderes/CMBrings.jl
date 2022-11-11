module CMBrings

using EAZTransforms
using EAZTransforms: pix, freq, nyq, Ωpix 
import EAZTransforms as EZ 
# should we export EZ and pix, freq, nyq, Ωpix ?

using FFTransforms: 𝕀, ⊗, 𝕎
import FFTransforms as FT

import HealpixTransforms as HT

using XFields
using FieldLensing
import CirculantCov as CC
import VecchiaFactorization as VF
using BlockArrays: Block, BlockArray, PseudoBlockArray, 
                    blocks, undef_blocks, blocksizes,
                    blockaxes, blockcolsupport
using BlockBandedMatrices: BlockBandedMatrix, Zeros, Ones

# using Distributed
using LinearAlgebra
using FFTW
using Statistics 
using SparseArrays
using JLD2
using ProgressMeter
using PyCall
using PyPlot 
using NLopt
using DelimitedFiles: readdlm
import ImageFiltering as IF
import LowRankCholesky as LRC

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

include("transformations.jl")

include("circ_op.jl")
export CircOp, field2▪, ▪2field

include("block_diag_construct.jl")

include("beam_filters.jl")
# beam▫

include("masking.jl")
# pix_point_src_mask

include("pixel_wf_decimation.jl")
# healpix_pwf▫
# TODO: add decimation

include("poly_hp_filters.jl")
# TODO: add the local methods from examples/transfer-fun-est

include("lensing.jl")

include("likelihoods.jl")

include("wf_pcg.jl")

include("alm_from_eaz.jl")

include("misc_methods.jl")

include("plot.jl")

end
