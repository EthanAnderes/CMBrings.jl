# estimation utalizing parameters -> Cℓ -> EAZ fourier quasi-bandpower

# TODO
# ==============================
#=
• given Cℓ, plot obs quasi-bandpowers vrs expected quasi-bandpowers
• Try ClassicOrthogonalPolynomials instead of AssociatedLegendrePolynomials
=#



# Modules
# ==============================

using LinearAlgebra
using FFTW
FFTW.set_num_threads(BLAS.get_num_threads())

using CMBrings

using  XFields
using  EAZTransforms
using  EAZTransforms: pix, freq, nyq, Ωpix # these work for FFTransforms too
import EAZTransforms as EZ

import FFTransforms as FT
import HealpixTransforms as HT

import CirculantCov as CC
using  FieldLensing
using  Spectra: camb_cls
using  VecchiaFactorization
import VecchiaFactorization as VF

using LBblocks: @sblock

using AssociatedLegendrePolynomials
# using SparseArrays
# using BenchmarkTools
# using ProgressMeter
# using BlockArrays
# using Dierckx: Spline1D
# using ImageFiltering
import JLD2
using PyPlot
# import PyCall as PC
# HP = PC.pyimport("healpy")

include(joinpath(CMBrings.module_dir,"examples/quasi-bandpwr2parm/LocalMethods.jl"))
import .LocalMethods as LM



# EAZ pixel grid
# ========================================

tm0, tm2, grid_type = @sblock let 

    ## set φ grid parameters: φspan and nφ
    φspan = deg2rad.((-60,60)) # deg2rad.((-45, 45))
    nφ    = 1575  # 18000, 18000÷4, 768, 1536, 1575, 2048, 1024, 972,  1280

    ## set θ grid parameters: θ, θ∂
    ## ---- option
    # type  = :healpix
    # Nside = 2048 # 8192
    # ri_offset_from_SP = round(Int, sqrt(3*Nside^2*(1+cos(2.8))))
    # ri = (3*Nside+1):1:(4*Nside-1 - ri_offset_from_SP)
    # θ  = CC.θ_healpix(Nside)[ri]
    # θ∂ = CC.θ_healpix(Nside)[ri.start:ri.step:ri.stop+ri.step]
    ## ---- option
    type = :equiθ # :equicosθ 
    nθ     = 600 # 805
    θspan  = π/2 .- deg2rad.((-51,-69)) # π/2 .- deg2rad.((-41.78,-70.43))
    θ, θ∂  = CC.θ_grid(; θspan, N=nθ, type)

    tm0 = EAZ0{Float64}(θ, φspan, nφ; θ∂)
    tm2 = EAZ2{Float64}(θ, φspan, nφ; θ∂)

    return tm0, tm2, type
end


# Plot Grid statistics

@sblock let tm0, hide_plots=false
    hide_plots && return
    fig,ax = subplots(1, dpi=147)
    ax.plot(tm0.θ, rad2deg.(.√(EZ.Ωpix(tm0)).*60), label="sqrt pixel area")
    ax.plot(tm0.θ, rad2deg.(EZ.Δθ(tm0).*60), label="Δθ")
    ax.plot(tm0.θ, rad2deg.(sin.(tm0.θ).*EZ.Δφ(tm0).*60), label="pix φ side arclen")
    ax.plot(tm0.θ, EZ.pix_diag_arcmin(tm0), label="pix diag arclen")
    ax.set_xlabel(L"polar coordinate $\theta$")
    ax.set_ylabel("arcmin")
    ax.legend()
    return nothing
end

@show (tm0.nθ, tm0.nφ)
@show extrema(rad2deg.(.√(EZ.Ωpix(tm0)).*60))
@show extrema(rad2deg.(EZ.Δθ(tm0).*60))
@show extrema(rad2deg.(sin.(tm0.θ) .* EZ.Δφ(tm0) .* 60))
@show extrema(EZ.pix_diag_arcmin(tm0));


# Demonstraight modes loaded by alm's
# ================================

# a_lm impulse response
let 
    ls_max     = 500
    ms_absmax  = 300
    θ_vector = range(1.8, 3.0, length=1000) # or θ
    spn = 0 # other possibilities are -2, 2
    l   = 200

    λlm_cache  = λlm(0:ls_max, 0:ms_absmax, cos.(θ_vector))

    fig, ax = subplots(1)
    ms = Int.(freq(tm0)[2])[1:5:end] |> m->m[0 .< m .< l]
    Δms = ms[2] - ms[1]

    for i = 1:length(ms)
        X₀  = CMBrings.CMBrings.index_λlm(l, ms[i], spn; λlm_cache) # 0 for spin 0
        ax.plot((Δms/2) .* X₀ ./ maximum(abs.(X₀)) .+ ms[i], θ_vector)
    end 
    ax.set_ylim(θ_vector[end], θ_vector[1])
    ax.set_xlabel(L"azimuthal frequency $m$")
    ax.set_ylabel(L"polar $\theta$ [rad]")
    ax.set_title(L"Impulse response of $a_{\ell = %$(l), m}$ on $[I(\theta,\cdot)]_m$ for different values of $m$.")
    ax.plot(ms, π .- asin.(ms ./ l), ":k", label=L"$\pi - \sin^{-1}(m/\ell)$ where $\ell = %$(l)$")
    ax.legend(loc="lower right")
end 




# Construct and plot the Cl conv kernel
# ================================

ls_max     = 1000
ms_absmax  = 1000
θ_vector = range(2.2, 2.85, length=1000) # or θ
s = 0    # other possibilities are -2, 2
λlm_cache  = λlm(0:ls_max, 0:ms_absmax, cos.(θ_vector));

# The Chi weights are computed via this method
# CMBrings. Ξ(l₀, l₁, ls_max, θ_vector, s, λlm_cache) 
# These are the weights that dot with C_l and which give the expected value of quasi-bandpower total power in bin (l₀, l₁)

# %%

figure()
Δl = 50
plot(CMBrings.Ξ(50, 50+Δl, ls_max, θ_vector, s, λlm_cache))
plot(CMBrings.Ξ(150, 150+Δl, ls_max, θ_vector, s, λlm_cache))
plot(CMBrings.Ξ(250, 250+Δl, ls_max, θ_vector, s, λlm_cache))
plot(CMBrings.Ξ(350, 350+Δl, ls_max, θ_vector, s, λlm_cache))
plot(CMBrings.Ξ(800, 800+Δl, ls_max, θ_vector, s, λlm_cache))

# %%

[
    mean(CMBrings.Ξ(50, 50+Δl, ls_max, θ_vector, s, λlm_cache))
    mean(CMBrings.Ξ(150, 150+Δl, ls_max, θ_vector, s, λlm_cache))
    mean(CMBrings.Ξ(250, 250+Δl, ls_max, θ_vector, s, λlm_cache))
    mean(CMBrings.Ξ(350, 350+Δl, ls_max, θ_vector, s, λlm_cache))
    mean(CMBrings.Ξ(800, 800+Δl, ls_max, θ_vector, s, λlm_cache))
]

# %%

figure()
Δl = 20
plot(CMBrings.Ξ(200,     200+Δl, ls_max, θ_vector, s, λlm_cache))
plot(CMBrings.Ξ(200+Δl,  200+2Δl, ls_max, θ_vector, s, λlm_cache))
plot(CMBrings.Ξ(200+2Δl, 200+3Δl, ls_max, θ_vector, s, λlm_cache))
plot(CMBrings.Ξ(200+3Δl, 200+4Δl, ls_max, θ_vector, s, λlm_cache))
plot(CMBrings.Ξ(200+4Δl, 200+5Δl, ls_max, θ_vector, s, λlm_cache))


# %%





figure()
plot(CMBrings.Ξ(50 , 600, ls_max, θ_vector, s, λlm_cache))
plot(CMBrings.Ξ(150, 600, ls_max, θ_vector, s, λlm_cache))
plot(CMBrings.Ξ(250, 600, ls_max, θ_vector, s, λlm_cache))
plot(CMBrings.Ξ(350, 600, ls_max, θ_vector, s, λlm_cache))




# %%

Δl  = 20
lbd = [Δl*i for i in 20:30]
𝛘s = [CMBrings.Ξ(lbd[i], lbd[i+1], ls_max, θ_vector, s, λlm_cache) for i=1:length(lbd)-1];


figure()
for i in 1:length(𝛘s)   
    plot(𝛘s[i])
end



figure()
for i in 1:length(𝛘s)-1   
    plot(𝛘s[i] .- 𝛘s[i+1])
end



