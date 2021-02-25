## Spin 2 lensing example which uses CMBsphere transform to handle the QU cov operator


# Modules
# ==============================
# using FFTW
# FFTW.FFTW.set_num_threads(8)

using XFields
using CMBrings
using CMBsphere # we will use CMBsphere to do the EBcovariance operator
using CMBflat: PrQr # Eventually remove this CMBflat.PrQr dependence ...

import FFTransforms as FT
import SphereTransforms as ST

using Spectra
using FieldLensing 

using  LinearAlgebra
using  SparseArrays
import Dierckx 
import NLopt

using DelimitedFiles
using LBblocks: @sblock
using PyPlot
using BenchmarkTools
using ProgressMeter

#- 

if isdefined(Main,:PlutoRunner)
    import PlutoUI
    hide_plots = false
elseif isdefined(Main, :IJulia) && Main.IJulia.inited
    hide_plots = false
else 
    hide_plots = true
end

# Template out a wrapper for tmS2 
# ==========================================

# The fourier transform is just the broadcased unitary 𝕀(nθ) ⊗ 𝕎(Tf, nφ, 2π)

struct Az𝕊0{Tf<:Real, C<:CartesianIndices} <: XFields.Transform{Tf,2} 
    tmAz::FT.𝕎{Tf, 2, Tf, Tf}
    tm𝕊::ST.𝕊0
    ringidx::C     
    function Az𝕊0(tmAz::FT.𝕎{Tf, 2, Tp, Tf}, tm𝕊::ST.𝕊0, ringidx::C) where {Tf, Tp, C}
        nθAz, nφAz = size_in(tmAz)
        nθ𝕊, nφ𝕊   = size_in(tm𝕊)
        @assert nθAz <= nθ𝕊
        @assert nφAz == nφ𝕊
        @assert isodd(nφ𝕊)
        @assert size(ringidx) == (nθAz, nφAz)
        ## ensure the transformation is unitary
        tmAz′ = FT.unscale(tmAz) |> tm -> FT.unitary_scale(tm)*tm 
        new{Tf,C}(tmAz′, tm𝕊, ringidx)
    end 
end 

@inline XFields.size_in(tm::Az𝕊0)   = XFields.size_in(tm.tmAz)
@inline XFields.size_out(tm::Az𝕊0)  = XFields.size_out(tm.tmAz)
@inline XFields.eltype_in(tm::Az𝕊0{Tf})  where {Tf}       = Tf
@inline XFields.eltype_out(tm::Az𝕊0{Tf}) where {Tf<:Real} = Complex{Tf}
@inline XFields.plan(tm::Az𝕊0) = XFields.plan(tm.tmAz) 

struct Az𝕊2{Tf<:Real, C<:CartesianIndices} <: XFields.Transform{Tf,3} 
    tmAz::FT.𝕎{Tf, 3, Tf, Tf}
    tm𝕊::ST.𝕊2
    ringidx::C     
    function Az𝕊2(tmAz::FT.𝕎{Tf, 3, Tp, Tf}, tm𝕊::ST.𝕊2, ringidx::C) where {Tf, Tp, C}
        nθAz, nφAz, = size_in(tmAz)
        nθ𝕊, nφ𝕊,   = size_in(tm𝕊)
        @assert nθAz <= nθ𝕊
        @assert nφAz == nφ𝕊
        @assert isodd(nφ𝕊)
        @assert size(ringidx) == (nθAz, nφAz)
        ## ensure the transformation is unitary
        tmAz′ = FT.unscale(tmAz) |> tm -> FT.unitary_scale(tm)*tm 
        new{Tf,C}(tmAz′, tm𝕊, ringidx)
    end 
end 

@inline XFields.size_in(tm::Az𝕊2)   = XFields.size_in(tm.tmAz)
@inline XFields.size_out(tm::Az𝕊2)  = XFields.size_out(tm.tmAz)
@inline XFields.eltype_in(tm::Az𝕊2{Tf})  where {Tf}       = Tf
@inline XFields.eltype_out(tm::Az𝕊2{Tf}) where {Tf<:Real} = Complex{Tf}
@inline XFields.plan(tm::Az𝕊2) = XFields.plan(tm.tmAz) 

# struct QU𝕊2ring{Tf<:Number, C<:CartesianIndices} <: XFields.Transform{Tf,3} 
#     sz::NTuple{2,Int}
#     tm𝕊2::ST.𝕊2
#     index𝕊2ring::C     
#     function QU𝕊2ring{Tf}(sz, tm𝕊2::ST.𝕊2, index𝕊2ring::C) where {Tf,C}
#         nθ𝕊, nφ𝕊 = size_in(tm𝕊2)
#         @assert sz[1] <= nθ𝕊
#         @assert sz[2] == nφ𝕊
#         @assert isodd(nφ𝕊)
#         @assert size(index𝕊2ring) == sz
#         new{Tf,C}(sz, tm𝕊2, index𝕊2ring)
#     end 
# end 

# @inline XFields.size_in(tm::QU𝕊2ring)             = (tm.sz[1], tm.sz[2], 2)
# @inline XFields.size_out(tm::QU𝕊2ring{<:Complex}) = (tm.sz[1], tm.sz[2], 2)
# @inline XFields.size_out(tm::QU𝕊2ring{<:Real})    = (tm.sz[1], tm.sz[2]÷2+1, 2)
# @inline XFields.eltype_in(tm::QU𝕊2ring{Tf})  where {Tf} = Tf
# @inline XFields.eltype_out(tm::QU𝕊2ring{Tf}) where {Tf<:Real}    = Complex{Tf}
# @inline XFields.eltype_out(tm::QU𝕊2ring{Tf}) where {Tf<:Complex} = Tf
# @inline XFields.plan(tm::QU𝕊2ring{Tf}) where {Tf} = XFields.plan(𝕌(st)) 


# function FT.𝕎(tm::QU𝕊2ring{Tf}) where {Tf}
#     nθ, nφ = tm.sz[1], tm.sz[2]
#     return FT.:⊗(FT.𝕀(nθ) , FT.𝕎(Tf, nφ, 2π) , FT.𝕀(2) ) 
# end
# function 𝕌(tm::QU𝕊2ring{Tf}) where {Tf}
#     tmW2  = FT.𝕎(tm)
#     return FT.unitary_scale(tmW2) * tmW2
# end
# ST.𝕊2(tm::QU𝕊2ring) = tm.tm𝕊2


ST.Ωpix(tm::Union{Az𝕊0,Az𝕊2}) = ST.Ωpix(tm.tm𝕊)[tm.ringidx[:,1]]

function ST.pix(tm::Union{Az𝕊0,Az𝕊2})
    θ, φ = ST.pix(tm.tm𝕊)
    return θ[tm.ringidx[:,1]], φ
end


# extras ========

function XFields.Xmap(tm::Az𝕊2{Tf}, x1, x2) where {Tf}
    mat = zeros(Tf, size_in(tm))
    mat[:,:,1] .= x1
    mat[:,:,2] .= x2
    return Xmap(tm, mat)
end

XFields.Xmap(tm::Az𝕊2, x::AbstractMatrix) = Xmap(tm, x, x)

function XFields.Xfourier(tm::Az𝕊2{Tf}, x1, x2) where {Tf}
    mat = zeros(Complex{Tf},size_out(tm))
    mat[:,:,1] .= x1
    mat[:,:,2] .= x2
    return Xfourier(tm, mat)
end

XFields.Xfourier(tm::Az𝕊2, x::AbstractMatrix) = Xfourier(tm, x, x)

function Base.getindex(f::Xfield{<:Az𝕊2}, sym::Symbol)
    (sym == :Qx) ? fielddata(MapField(f))[:,:,1] :
    (sym == :Ux) ? fielddata(MapField(f))[:,:,2] :
    (sym == :Qk) ? fielddata(FourierField(f))[:,:,1] :
    (sym == :Uk) ? fielddata(FourierField(f))[:,:,2] :
    error("index is not defined")
end

function LinearAlgebra.dot(f::Xfield{TM}, g::Xfield{TM}) where TM<:Union{Az𝕊0,Az𝕊2}
    FT.sum_kbn(f[:].*g[:])
end


# need to teach AzBlocks how to multiply and divide Xfield{<:Az𝕊} ========



# need to teach DiagOp{XField{<:𝕊}} how to multiply and divide Xfield{<:Az𝕊} ========


# Simulation ======


# function simmap(Cl::DiagOp{Fi}) where {Fi<:Xfourier} 
#     tm  = fieldtransform(Cl.f)
#     √Cl * Xmap(tm, FT.randn_in(tm))
# end

# function simfourier(Cl::DiagOp{Fi}) where {Fi<:Xfourier} 
#     tm  = fieldtransform(Cl.f)
#     #√Cl * Xfourier(tm, FT.randn_out(tm))
#     # We need the following instead since we don't have randn for fft yet
#     Xfourier(simmap(Cl))
# end 
 
# function flatnoisemap(μK′n::Number, tm::Union{𝕎,QU𝕊2ring}) 
#     (μK′n * π / 60 / 180) * Xmap(tm, FT.randn_in(tm))
# end 

# function flatnoisefourier(μK′n::Number, tm::Union{𝕎,QU𝕊2ring}) 
#     # We need the following instead since we don't have randn for fft yet
#     Xfourier(flatnoisemap(μK′n, tm)) 
# end



# Set ring transforms
# ==============================

tmAzS0, tmAzS2 = @sblock let 

    ## size of the embedding full sphere
    𝕊nθ, 𝕊nφ = (2560, 2560-1)
    ## 𝕊nθ, 𝕊nφ = (3584, 2048-1)

    ## Spin ±2 transform
    tmS2 = ST.𝕊2(𝕊nθ, 𝕊nφ)
    tmS0 = ST.𝕊0(𝕊nθ, 𝕊nφ)

    ## grid coords on full sphere
    θ𝕊, φ𝕊 = ST.pix(tmS0) 

    ## north and southern boundaries and the corresponding indices
    θnorth∂ = 2.2 # 2.12
    θsouth∂ = 2.85
    θrng    = findall(θnorth∂ .<= θ𝕊 .<= θsouth∂)
    ringidx = CartesianIndices((θrng[1]:θrng[end], 1:length(φ𝕊)))
    nθ, nφ  = size(ringidx)

    ## Spin 0 ring transform is just inherited from FFTransforms
    Tf = Float64
    tmW0  = FT.:⊗(FT.𝕀(nθ), FT.𝕎(Tf, nφ, 2π)) # 𝕀(nθ) ⊗ 𝕎(Tf, nφ, 2π)
    tmW2  = FT.:⊗(FT.𝕀(nθ), FT.𝕎(Tf, nφ, 2π), FT.𝕀(2)) 

    ## Spin 2 transform includes the ring embedding ...
    tmAzS0 = Az𝕊0(tmW0, tmS0, ringidx)
    tmAzS2 = Az𝕊2(tmW2, tmS2, ringidx)

    return tmAzS0, tmAzS2
end



# Mask and CMBring observation region
# ==============================


data_mask_init, Ω, θ, φ = @sblock let tmAzS0, tmAzS2, QP_bdry=1e-5, fwhm′=150

    pr_mat_init  = readdlm(joinpath(CMBrings.module_dir,"examples/artifacts/FastTransform_mask_nθ3072_nφ4095.csv"), ',', Bool)
    
    full_sky_tm𝕊0 = ST.𝕊0(size(pr_mat_init)...)
    θ_mat_init, φ_mat_init = ST.pix(full_sky_tm𝕊0)
    spline_mask = Dierckx.Spline2D(θ_mat_init, φ_mat_init, pr_mat_init, kx=1, ky=1, s=0.0)

    nθ, nφ,  = size_in(tmAzS2)
    θ, φ  = ST.pix(tmAzS2)
    Ω     = ST.Ωpix(tmAzS2)
    ## θ = θnorth∂ .+ ((θsouth∂ - θnorth∂) / nθ) .* (0:nθ-1)
    ## φ = (2π / nφ) .* (0:nφ-1)
    ## Ω = ST.Ωpix.(θ, θ[2] - θ[1], φ[2] .- φ[1])

    data_mask_init = spline_mask.(θ, φ') .> 0
    data_mask_init[1:30,:] .= 0
    data_mask_init[end - 30 + 1:end,:] .= 0

    return data_mask_init, Ω, θ, φ

end;

#- 

Pr, Qr = @sblock let tmAzS0, tmAzS2, data_mask_init, QP_bdry=1e-5, fwhm′=150

    θ, φ  = ST.pix(tmAzS2)
    tmFlat = FT.𝕎(Float64, size(data_mask_init), ((θ[2] - θ[1])*length(θ), 2π))
    pr0x, qr0x = PrQr(tmFlat, data_mask_init, fwhm′, fwhm′, QP_bdry)
    pr0 = Xmap(tmAzS2, pr0x, pr0x)
    qr0 = Xmap(tmAzS2, qr0x, qr0x)

    DiagOp(pr0), DiagOp(qr0)
end;

# Localize lensing vector field to data mask.

Mϕ = @sblock let tmAzS0, tmAzS2, data_mask_init, QP_bdry=1e-5, fwhm′=75

    θ, φ  = ST.pix(tmAzS2)
    tmFlat = FT.𝕎(Float64, size(data_mask_init), ((θ[2] - θ[1])*length(θ), 2π))
    pr0x, qr0x = PrQr(tmFlat, data_mask_init, fwhm′, fwhm′, QP_bdry)

    ## mϕx = pr0x .+ qr0x
    mϕx = pr0x 

    ## make sure it hits zero and 1
    mϕx .-= minimum(mϕx)
    mϕx ./= maximum(mϕx)
    Mϕ    = DiagOp(Xmap(tmAzS0, mϕx))

    Mϕ
end;

# Azimuthal ring mask

@sblock let ma=Pr[:Qx], φ, θ, hide_plots
    hide_plots && return
    imgs = Dict(1=>ma)
    txt  = Dict(1=>"Mask")
    ctxt = Dict(1=>"w")
    ## fig, ax = CMBrings.brickplot(imgs; txt=txt, ctxt=ctxt, fφ=1)
    fig, ax = CMBrings.diskplot(imgs, φ', π.-θ; txt=txt, nrows=1, fontsize=14)
    return fig
end

# Plot √Ωpix over ring θ's 

@sblock let θ, φ, Ω, hide_plots
    hide_plots && return
    fig,ax = subplots(1)
    ax.plot(θ, rad2deg.(sqrt.(Ω)).*60, label="sqrt pixel area (arcmin)")
    ax.plot(θ, zero(θ) .+ rad2deg.(θ[2] - θ[1]).*60, label="Δθ (arcmin)")
    ## ax.plot(θ, zero(θ) .+ rad2deg.(φ[2] - φ[1]).*60, label="Δφ (arcmin)")
    ax.set_xlabel(L"polar coordinate $\theta$")
    ax.legend()
    return fig
end


# Spectral densities
# ==============================

# ϕϕ, EB spectra

eel, bbl, ẽel, b̃bl, ϕϕl = @sblock let
    
    r  = 0.01

    lmax = 11000
    l = 0:lmax
    cld = Spectra.camb_cls(;lmax=lmax, r)
    
    eesl = cld[:unlen_scalar] |> x->(x[:Cee] ./ x[:factor_on_cl_cmb])
    eetl = cld[:unlen_tensor] |> x->(x[:Cee] ./ x[:factor_on_cl_cmb])
    eel  = eesl .+ eetl
    eel[1] = 0

    bbsl = cld[:unlen_scalar] |> x->(x[:Cbb] ./ x[:factor_on_cl_cmb])
    bbtl = cld[:unlen_tensor] |> x->(x[:Cbb] ./ x[:factor_on_cl_cmb])
    ## note: bbsl == 0 
    bbl    = bbsl .+ bbtl
    bbl[1] = 0

    ẽesl   = cld[:len_scalar] |> x->(x[:Cee] ./ x[:factor_on_cl_cmb])
    ẽel    = ẽesl .+ eetl # we only have lensed spectra for scalar
    ẽel[1] = 0

    b̃bsl   = cld[:len_scalar] |> x->(x[:Cbb] ./ x[:factor_on_cl_cmb])
    b̃bl    = b̃bsl .+ eetl # we only have lensed spectra for scalar
    b̃bl[1] = 0

    ϕϕl    = cld[:phi] |> x->(x[:Cϕϕ] ./ x[:factor_on_cl_phi])
    ϕϕl[1] =  0

    return eel, bbl, ẽel, b̃bl, ϕϕl

end;

# beam/transfer

bl = @sblock let 

    beamfwhm  = 4.0 |> arcmin -> deg2rad(arcmin/60)

    lmax = 11000 
    l = 0:lmax
    σ² = beamfwhm^2 / 8 / log(2)
    bl = @. exp( - σ²*l*(l+1) / 2)
    return bl

end;

# noise

## nnl, wnl, snl = @sblock let 
## 
##     μK′n      = 2.5 
##     ellknee   = 0   
##     alphaknee = 3
## 
##     lmax = 11000
##     l = 0:lmax
##     whitenoisel    = fill(μK′n^2 * (π/60/180)^2, size(l))
##     smoothnoisel   = @. μK′n^2 * (π/60/180)^2 * Spectra.knee(l; ell=ellknee, alpha=alphaknee) 
##     smoothnoisel .-= μK′n^2 * (π/60/180)^2 
##     smoothnoisel[smoothnoisel .< 0] .= 0    
##     noisel = smoothnoisel .+ whitenoisel
##     return noisel, whitenoisel, smoothnoisel
## 
## end;

#-

## @sblock let hide_plots, nnl, eel, bbl, ϕϕl, bl # , lmax=5000
##     hide_plots && return
## 
##     l = 0:length(nnl)-1
##     rng = 2:5000
## 
##     fig,ax = subplots(1)
##     ax.plot(l[rng], l[rng].^2 .* eel[rng], label="ee")
##     ax.plot(l[rng], l[rng].^2 .* bbl[rng], label="bb")
##     ax.plot(l[rng], l[rng].^2 .* nnl[rng], ":", label="noise")
##     ax.plot(l[rng], l[rng].^2 .* nnl[rng]./bl[rng], ":", label="noise/beam")
##     ## ax.axvline(x=lmax, color="r", label="data lmax")
##     ax.set_xscale("log")
##     ax.set_yscale("log")
##     ax.set_xlabel(L"\ell")
##     ax.set_ylabel(L"\ell^2 C_\ell")
##     ax.legend()
##     return fig
## end




# AzBlock Beam and Noise operators (only white noise for now)
# ==============================


Naz = @sblock let tmAzS0, Ω, μK′n = 2.5
    μKᵒn = μK′n / 60
    σ²   = deg2rad(μKᵒn)^2
    Vector_M = [Diagonal(σ²./Ω) for k in 1:size_out(tmAzS0)[2]]
    CMBrings.AzBlock(Vector_M)
end

# quick test

#=

ei = Xmap(tmAzS0)
ei.fd[end - 50,100] = 1
Nei = Naz * ei
Nei[:][end - 50,100]
deg2rad(2.5 / 60)^2 / Ω[end - 50]

=#


Baz = @sblock let tmAzS0,  bl, θ, φ, Ω

	tmW=FT.unscale(tmAzS0.tmAz)
    
    dmax = 1.2maximum(CMBrings.geoθ1θ2Δφcol(θ[1], θ[1], φ .- φ[1]))
    θgrid = range(0, dmax^(1/2), length=100_000).^2
    
    covf  = Dierckx.Spline1D(
        θgrid, 
        Spectra.spec2spherecov(bl, θgrid), 
        k=3
    )
    
    covf_θ1θ2Δφℝ = (θ1, θ2, Δφ) -> covf(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφ)) 

    Baz  = CMBrings.AzBlock(covf_θ1θ2Δφℝ, θ, φ, tmW) do Σ, k
        real.(Σ) * LinearAlgebra.Diagonal(Ω)
    end

    return Baz
end;




#=
eiS0 = Xmap(tmAzS0)
eiS0.fd[end - 50,100] = 1
eiS2 = Xmap(tmAzS2)
eiS2.fd[end - 50,100,1] = 1

@time XFields._lmult(Baz, eiS0)
@time XFields._lmult(Baz, eiS2)
=#

# Simulation
# ==============================


EBcov, Lcut, Φcov = @sblock let tmAzS0, tmAzS2, eel, bbl, ϕϕl, lcut = 2000

    n𝕊θ, n𝕊φ, = size_in(tmAzS2.tm𝕊)
    l2,m2,a2 = ST.lma(-2, n𝕊θ, n𝕊φ)
    l0,m0,a0 = ST.lma(0, n𝕊θ, n𝕊φ)
    
    ECL  = @. getindex((eel,), l2 + 1)
    BCL  = @. getindex((bbl,), l2 + 1)
    ΦCL  = @. getindex((ϕϕl,), l0 + 1)
    LCL  =  (0 .< l2 .<= lcut)
    ECL[.!a2] .= 0
    BCL[.!a2] .= 0
    ΦCL[.!a0] .= 0

    EBcov = DiagOp(Xfourier(tmAzS2.tm𝕊, ECL, BCL))
    Lcut  = DiagOp(Xfourier(tmAzS2.tm𝕊, LCL, LCL))
    Φcov  = DiagOp(Xfourier(tmAzS0.tm𝕊, ΦCL))

    return EBcov, Lcut, Φcov

end


## We need to teach EBcov, Lcut and Φcov how to multiply against 


ei  = Xmap(tmAzS2)
ei.fd[50,100,1] = 1
ei𝕊 = Xmap(tmAzS2.tm𝕊)
ei𝕊.fd[tmAzS2.ringidx,:] .= ei.fd
Σei𝕊 = EBcov * ei𝕊
Σei  = Xmap(tmAzS2, Σei𝕊.fd[tmAzS2.ringidx,:])
Σei[:Qx]