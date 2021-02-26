## Spin 2 lensing example which uses CMBsphere transform to handle the QU cov operator


# Modules
# ==============================
# using FFTW
# FFTW.FFTW.set_num_threads(8)

using XFields
using CMBrings
using CMBsphere     # we will use CMBsphere to do the EBcovariance operator
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
    ringidxS0 = CartesianIndices((θrng[1]:θrng[end], 1:length(φ𝕊)))
    ringidxS2 = CartesianIndices((θrng[1]:θrng[end], 1:length(φ𝕊), 1:2))
    nθ, nφ  = size(ringidxS0)

    ## Spin 0 ring transform is just inherited from FFTransforms
    Tf = Float64
    tmW0  = FT.:⊗(FT.𝕀(nθ), FT.𝕎(Tf, nφ, 2π)) # 𝕀(nθ) ⊗ 𝕎(Tf, nφ, 2π)
    tmW2  = FT.:⊗(FT.𝕀(nθ), FT.𝕎(Tf, nφ, 2π), FT.𝕀(2)) 

    ## Spin 2 transform includes the ring embedding ...
    tmAzS0 = CMBrings.Az𝕊0(tmW0, tmS0, ringidxS0)
    tmAzS2 = CMBrings.Az𝕊2(tmW2, tmS2, ringidxS2)

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

@time ei′ = EBcov * ei
@time ei′ = Lcut * ei

ei′[:Qx] |> matshow
ei′[:Ux] |> matshow

## ========

ei  = Xmap(tmAzS2)
ei.fd[50,100,1] = 1

@time ei′ = EBcov * ei
@time ei′ = Lcut * ei

ei′[:Qx] |> matshow
ei′[:Ux] |> matshow



ϕ_sim = Xmap(tmAzS0, CMBsphere.simmap(Φcov)[:][tmAzS0.ringidx])
p_sim = Xmap(tmAzS2, CMBsphere.simmap(EBcov)[:][tmAzS2.ringidx])
