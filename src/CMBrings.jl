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

include("az_blocks.jl")

include("az_cov.jl")

include("methods.jl")

include("plot.jl")


LinearAlgebra.dot(f::Xfield{FT},g::Xfield{FT}) where FT<:𝕎 = dot(f[:], g[:])

LinearAlgebra.adjoint(C::DiagOp) = C  



# Move these to CMBsphere and CMBhealpix ...  healpix and fasttransforms
# =====================================


function LinearAlgebra.dot(f::Xmap{FT},g::Xmap{FT}) where FT<:𝕊
    trn  = fieldtransform(f)
    sqrtΩ = sqrt.(SphereTransforms.Ωpix(trn))
    return  dot(f[:].*sqrtΩ, g[:].*sqrtΩ)
end

function LinearAlgebra.dot(f::Xfield{FT},g::Xfield{FT}) where FT<:𝕊
    dot(f[!], g[!])
end

function LinearAlgebra.dot(f::Xfourier{FT},g::Xfourier{FT}) where FT<:ℍ0
    trn = fieldtransform(f)
    return HealpixTransforms.Ωpix(trn) * dot(f[:],g[:])
end

function LinearAlgebra.dot(f::Xfield{FT},g::Xfield{FT}) where FT<:ℍ0
    trn = fieldtransform(f)
    l,m   = HealpixTransforms.lm(trn)
    fl = f[!]
    gl = g[!]
    flm = fl[.!(m .== 0)]
    glm = gl[.!(m .== 0)]    
    fl0 = fl[m .== 0]
    gl0 = gl[m .== 0]
    rtn  = dot(real.(flm),real.(glm)) * 2  
    rtn += dot(imag.(flm),imag.(glm)) * 2
    rtn += dot(real.(fl0),real.(gl0)) 
    return rtn
end


function ℍlm2𝕊lm(hlm, h0, s0)
    ls0′, ms0′ = SphereTransforms.lm(s0)
    lh0,  mh0  = HealpixTransforms.lm(h0)
    bew =  findall(ls0′ .<= maximum(lh0))
    ls0 = ls0′[bew]
    ms0 = ms0′[bew]

    idx_into_h0 = HealpixTransforms.lm2index(ls0, abs.(ms0), h0)
    mp = findall(ms0 .>= 0)
    mn = findall(ms0 .< 0)
    mp_bew = findall((ms0′ .>= 0) .& (ls0′ .<= maximum(lh0)))
    mn_bew = findall((ms0′ .< 0) .& (ls0′ .<= maximum(lh0)))


    slm = fill(0.0, size_out(s0))
    slm[mp_bew] = (-1).^(ms0[mp]) .* real.(hlm[idx_into_h0[mp]])
    slm[mn_bew] = (-1).^(ms0[mn].+1) .* imag.(hlm[idx_into_h0[mn]])
    slm 
end

function whitefourier(trn::𝕊)
    wlm  = SphereTransforms.white_fourier(trn)
    return Xfourier(trn, wlm)
end

function whitemap(trn::𝕊)
    wx   = SphereTransforms.white_map(trn) 
    return Xmap(trn, wx)
end

function whitefourier(trn::ℍ0)
    wlm    = randn(eltype_out(trn),size_out(trn))
    m      = HealpixTransforms.lm(trn)[2]
    m0Bool = findall(m .== 0)
    wlm[m0Bool] = randn(real(eltype_out(trn)), size(m0Bool))
    Xfourier(trn, wlm)
end

function whitemap(trn::ℍ0)
    wx  = randn(eltype_in(trn),size_in(trn)) ./ sqrt(HealpixTransforms.Ωpix(trn))
    Xmap(trn, wx)
end

flatnoisemap(μK′n::Number, trn)     = (μK′n * π / 60 / 180) * whitemap(trn)

flatnoisefourier(μK′n::Number, trn) = (μK′n * π / 60 / 180) * whitefourier(trn)

simfourier(Cl::DiagOp{<:Xfourier}) = √Cl * whitefourier(fieldtransform(Cl.f))






end
