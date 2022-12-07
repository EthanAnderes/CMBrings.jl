using LinearAlgebra
using FFTW
FFTW.set_num_threads(6)

using CMBrings

using  XFields
using  EAZTransforms
using  EAZTransforms: pix, freq, nyq, Ωpix 
import EAZTransforms as EZ

import FFTransforms as FT
import HealpixTransforms as HT

import CirculantCov as CC
using  FieldLensing
using  Spectra: camb_cls
using  VecchiaFactorization
import VecchiaFactorization as VF
using LowRankCholesky: pdeigen

using LBblocks: @sblock

using PyPlot
import PyCall as PC
hp = PC.pyimport("healpy");

# EAZ pixel grid
# ========================================

eaz0, eaz2, grid_type = @sblock let 

    ## set φ grid parameters: φspan and nφ
    # φspan = deg2rad.((-60,60)) 
    φspan = deg2rad.((-45, 45))
    nφ    = 18000÷4 # 2048 # 3072  # 1575 # 18000, 18000÷4, 768, 1536, 1575, 2048, 1024, 972,  1280
    # nφ    = 1575

    ## set θ grid parameters: θ, θ∂
    ## ---- option
    # type  = :healpix
    # Nside = 2048 # 8192
    # ri_offset_from_SP = round(Int, sqrt(3*Nside^2*(1+cos(2.8))))
    # ri = (3*Nside+1):1:(4*Nside-1 - ri_offset_from_SP)
    # θ  = CC.θ_healpix(Nside)[ri]
    # θ∂ = CC.θ_healpix(Nside)[ri.start:ri.step:ri.stop+ri.step]
    ## ---- option
    type = :equicosθ # :equiθ # 
    nθ     = 600 # 500 # 800
    # nθ    = 400
    θspan  = π/2 .+ deg2rad.((51,67))
    # θspan  = π/2 .+ deg2rad.((41.78,70.43)) 
    θ, θ∂  = CC.θ_grid(; θspan, N=nθ, type)

    ## Good smallish run settings
    # φspan = deg2rad.((-60,60))
    # nφ    = 1575
    # type  = :equiθ
    # nθ    = 400
    # θspan = π/2 .+ deg2rad.((51,69))


    eaz0 = EAZ0{Float64}(θ, φspan, nφ; θ∂)
    eaz2 = EAZ2{Float64}(θ, φspan, nφ; θ∂)

    return eaz0, eaz2, type
end;

@sblock let eaz0, hide_plots=false
    hide_plots && return
    fig,ax = subplots(1, dpi=147)
    ax.plot(eaz0.θ, rad2deg.(.√(EZ.Ωpix(eaz0)).*60), label="sqrt pixel area")
    ax.plot(eaz0.θ, rad2deg.(EZ.Δθ(eaz0).*60), label="Δθ")
    ax.plot(eaz0.θ, rad2deg.(sin.(eaz0.θ).*EZ.Δφ(eaz0).*60), label="pix φ side arclen")
    ax.plot(eaz0.θ, EZ.pix_diag_arcmin(eaz0), label="pix diag arclen")
    ax.set_xlabel(L"polar coordinate $\theta$")
    ax.set_ylabel("arcmin")
    ax.legend()
    return nothing
end;

# Spectral densities
# ==============================

φ_approx_nyq = eaz0.φfreq_mult * eaz0.nφ / minimum(sin.(eaz0.θ)) / 2
θ_approx_nyq = π / minimum(EZ.Δθ(eaz0)) 
@show approx_lmax = ceil(Int, sqrt(φ_approx_nyq^2 + θ_approx_nyq^2))

approx_lmax += ceil(Int, approx_lmax * 0.05) # for good measure:)
## override ...
## approx_lmax = 25_000

ℓ, ttℓ, eeℓ, bbℓ = @sblock let lmax=approx_lmax, r=0.001, T=Float64
    
    l = 0:lmax
    cld = camb_cls(;lmax=lmax, r,
        lSampleBoost   = 4.0,
        lAccuracyBoost = 4.0,
        KmaxBoost = 6.0,
        )
    
    ttl = cld[:unlen_scalar] |> x->(x[:Ctt] ./ x[:factor_on_cl_cmb])
    ttl[1] = ttl[3] /100
    ttl[2] = ttl[3] /10

    eesl = cld[:unlen_scalar] |> x->(x[:Cee] ./ x[:factor_on_cl_cmb])
    eetl = cld[:unlen_tensor] |> x->(x[:Cee] ./ x[:factor_on_cl_cmb])
    eel  = eesl .+ eetl
    eel[1] = eel[2] = 0

    bbsl = cld[:unlen_scalar] |> x->(x[:Cbb] ./ x[:factor_on_cl_cmb])
    bbtl = cld[:unlen_tensor] |> x->(x[:Cbb] ./ x[:factor_on_cl_cmb])
    ## note: bbsl == 0 
    bbl    = bbsl .+ bbtl
    bbl[1] = bbl[2] = 0

    return l, T.(ttl), T.(eel), T.(bbl)
end;

# Wide Gaussian Beam
# ----------------------------------------
#=
fwhm_arcmin = 1.5 * maximum(EZ.pix_diag_arcmin(eaz0)) # optional settings....
σ² = CMBrings.arcmin2rad(fwhm_arcmin)^2 / 8 / log(2)
beamℓ_pre =  @. exp( - σ²*ℓ*(ℓ+1) / 2);
=# 

# Subpixel Gaussian Beam
# ----------------------------------------
#
# fwhm_arcmin = 0.9 * minimum(EZ.pix_diag_arcmin(eaz0)) # optional settings....
# σ² = CMBrings.arcmin2rad(fwhm_arcmin)^2 / 8 / log(2)
# beamℓ_pre =  @. exp( - σ²*ℓ*(ℓ+1) / 2);


# Healpix pixel window function .... option
# ----------------------------------------
pwf0ℓ, pwf2ℓ = hp.pixwin(8192, pol=true, lmax=maximum(ℓ))
beamℓ_pre = pwf0ℓ;

# Now we taper so we don't get aliasing
# ----------------------------------------
# note we are setting the taper at the ℓ_nyq for the top edge
φ_approx_ℓ_nyq = eaz0.φfreq_mult * eaz0.nφ / sin.(minimum(eaz0.θ)) / 2


#####
# srt_ramp  = 0.9 * φ_approx_ℓ_nyq           # optional settings....
# end_ramp  = 1.0 * φ_approx_ℓ_nyq            # optional settings...
# ℓ_taper   = CMBrings.pixweight.(Float64.(ℓ); ▮l=0, ▯l=0, ▮r=end_ramp, ▯r=srt_ramp)
# ℓ_taper .+= 0.001                         # optional settings...
# ℓ_taper ./= maximum(ℓ_taper)              # optional settings...
#####
# ℓ_taper = @. exp(-(ℓ/end_ramp)^6)           # seems to work well
#####
srt_ramp  = 0.75 * φ_approx_ℓ_nyq           # optional settings....
end_ramp  = 1.0 * φ_approx_ℓ_nyq            # optional settings...
ℓ_taper = map(ℓ) do l
    if l < srt_ramp
        return 1 
    else
        lpost = l-srt_ramp
        σ     =  (end_ramp - srt_ramp)/2 
        return exp(-(lpost/σ)^2)
    end
end
#####

beamℓ = beamℓ_pre .* ℓ_taper; 


# Plot the tapered beam
m_max_top = round(Int, eaz0.φfreq_mult * eaz0.nφ / sin.(minimum(eaz0.θ)) / 2)
m_max_btm = round(Int, eaz0.φfreq_mult * eaz0.nφ / sin.(maximum(eaz0.θ)) / 2)
fig,ax = subplots(2, dpi=147)
ax[1].plot(0:m_max_btm, beamℓ_pre[1:m_max_btm+1]);
ax[1].plot(0:m_max_btm, beamℓ[1:m_max_btm+1]);
ax[1].axvline(x=m_max_top, color="black", linestyle="--")
ax[1].axvline(x=m_max_btm, color="black", linestyle="--")
ax[2].plot(0:m_max_btm, beamℓ[1:m_max_btm+1]./beamℓ_pre[1:m_max_btm+1]);
ax[2].set_ylim([0.90, 1.10])
ax[1].axvline(x=m_max_btm, color="black", linestyle="--")


# Sqrt of the cov matrix and beamed cov matrix of the fields we will test on
# ------------------------------------------

T▪½  = let 
    T▪ = CMBrings.eaz_cov(eaz0, ℓ, ttℓ) |> CircOp
    map(x->sqrt(pdeigen(Symmetric(x))), T▪) |> CircOp
end

BT▪½  = let
    # Note that the beam response on ttℓ is beamℓ.^2 .* ttℓ !!!
    BT▪   = CMBrings.eaz_cov(eaz0, ℓ, beamℓ.^2 .* ttℓ) |> CircOp;
    map(x->sqrt(pdeigen(Symmetric(x))), BT▪) |> CircOp
end;

# Beam operators
# ----------------------------------------

# Full eaz beam operator
Beam1▪  = let fℓ=beamℓ
    DΩ = Diagonal(EZ.Ωpix(eaz0))
    B_pre▫  = CMBrings.eaz_cov(eaz0, ℓ, fℓ)
    CircOp(B_pre▫) * DiagOp(Xfourier(eaz0, EZ.Ωpix(eaz0) .+ falses(size_out(eaz0))))
end;

# Vecchia eaz beam operator
using Distributed
block_sizesθ=VF.block_split(eaz0.nθ, 40) # optional settings .....
Beam2▪  = let ℓ=ℓ, fℓ=beamℓ
    
    # B_pre▫  = CMBrings.eaz_cov_vecchia(eaz0, ℓ, fℓ; block_sizesθ) |> CircOp;
    # ---------- alternative that doesn't require postive definite
    Γ  = CC.Γθ₁θ₂φ₁φ⃗_Iso(ℓ, fℓ)
    B_pre▫ = CMBrings.eaz_cov_btridiag(eaz0, Γ; block_sizesθ)
    B▫     = pmap(B_pre▫) do B
        VF.vecchia_general(B, block_sizesθ)
    end

    # DΩ = Diagonal(EZ.Ωpix(eaz0))
    # B▫ = map(B->B*DΩ, B_pre▫)
    # CircOp(B▫)

    CircOp(B▫) * DiagOp(Xfourier(eaz0, EZ.Ωpix(eaz0) .+ falses(size_out(eaz0))))
end;
# w0    = Xmap(eaz0,randn(eltype_in(eaz0), size_in(eaz0)))
# CMBrings.map_plot(Beam2▪ * Beam2▪ * w0, title1="eaz vecchia iterative beamed white noise")
# CMBrings.fourier_power(
#     Beam2▪ * Beam2▪ * w0, 
#     ℓs = [round(Int,srt_ramp), round(Int,end_ramp)], 
#     imag_fun=CMBrings.imag_logabs2clip,
#     xaxis_units = :m # :Hz
# );


# Diag m multiplier
Beam3▪  = let ℓ=ℓ, fℓ=beamℓ
    Bmθ = ones(real(eltype_out(eaz0)), size_out(eaz0))
    θs = EZ.θ(eaz0)
    φs = EZ.φ(eaz0)
    Γ  = CC.Γθ₁θ₂φ₁φ⃗_Iso(ℓ, fℓ) 
    for i in axes(Bmθ,1)
        θᵢ = θs[i]
        # TODO: possibly change Γ that changes the band limit on each ring using
        # φ_approx_ℓ_nyq = eaz0.φfreq_mult * eaz0.nφ / sin(θᵢ) / 2
        Bᵢ_pre▫     = CMBrings.eaz_cov(eaz0, Γ; θ=θᵢ, φ=φs)
        Bᵢ_in_col   = map(x->x[1], Bᵢ_pre▫)
        Bᵢ_in_col ./= Bᵢ_in_col[1]
        Bmθ[i,:] = Bᵢ_in_col
    end
    DiagOp(Xfourier(eaz0, Bmθ))
end;

# Simulate the pre-beamed and beamed field. The beamed field here is considered the ground truth.
# ==========================================================

w0    = Xmap(eaz0,randn(eltype_in(eaz0), size_in(eaz0)))
t     = T▪½   * w0     # pre-beamed field
bt    = BT▪½  * w0     # ground truth beamed field
b1t   = Beam1▪ * t     # full eaz beam
b2t   = Beam2▪ * t;    # vecchia eaz beam

# Plots
# -------------------

# Map plots.....

CMBrings.map_plot(t, title1=L"T(\theta,\varphi)")
CMBrings.map_plot(bt, title1=L"ground truth beamed: $BT(\theta,\varphi)$")
CMBrings.map_plot(b1t, title1=L"eaz beamed: $B_1T(\theta,\varphi)$")
CMBrings.map_plot(b2t, title1=L"eaz vecchia beamed: $B_2T(\theta,\varphi)$")
CMBrings.map_plot(Beam1▪ * Beam1▪ * Beam1▪ * Beam1▪ * w0, title1="eaz iterative beamed white noise")
CMBrings.map_plot(Beam2▪ * Beam2▪ * Beam2▪ * Beam2▪ * w0, title1="eaz vecchia iterative beamed white noise")

# Fourier plots....

CMBrings.fourier_power(
    t,  
    ℓs = [round(Int,srt_ramp), round(Int,end_ramp)], 
    imag_fun=CMBrings.imag_logabs2clip,
    title1=L"log|T(\theta,m)|^2",
    xaxis_units = :m # :Hz
);
CMBrings.fourier_power(
    bt, 
    ℓs = [round(Int,srt_ramp), round(Int,end_ramp)], 
    imag_fun=CMBrings.imag_logabs2clip,
    title1=L"log|BT(\theta,m)|^2",
    xaxis_units = :m # :Hz
);
CMBrings.fourier_power(
    b1t, 
    ℓs = [round(Int,srt_ramp), round(Int,end_ramp)], 
    imag_fun=CMBrings.imag_logabs2clip,
    title1=L"log|B_1T(\theta,m)|^2",
    xaxis_units = :m # :Hz
);
CMBrings.fourier_power(
    b2t, 
    ℓs = [round(Int,srt_ramp), round(Int,end_ramp)], 
    imag_fun=CMBrings.imag_logabs2clip,
    title1=L"log|B_2T(\theta,m)|^2",
    xaxis_units = :m # :Hz
);

## Power ratio .........

r1bt  = real(b1t[!] .* conj.(bt[!])) |> x->CMBrings.imag_blur(x;blur=2) 
r2bt  = real(b2t[!] .* conj.(bt[!])) |> x->CMBrings.imag_blur(x;blur=2) 
rbtbt = abs2.(bt[!])              |> x->CMBrings.imag_blur(x;blur=2) 

CMBrings.fourier_power(
    Xfourier(eaz0, r1bt ./ rbtbt); 
    title1=L"b1t(\theta,m) bt^*(\theta,m)/|bt(\theta,m)|^2", # imag_fun=CMBrings.imag_logabs2clip,
    vmin=0.95, vmax=1.05, # for t
    ℓs = [round(Int,srt_ramp), round(Int,end_ramp)], 
    xaxis_units = :m # :Hz
);

CMBrings.fourier_power(
    Xfourier(eaz0, r2bt ./ rbtbt); 
    title1=L"b2t(\theta,m) bt^*(\theta,m)/|bt(\theta,m)|^2", # imag_fun=CMBrings.imag_logabs2clip,
    vmin=0.95, vmax=1.05, # for t
    ℓs = [round(Int,srt_ramp), round(Int,end_ramp)], 
    xaxis_units = :m # :Hz
);

## EAZ quasi-bandpowers .........

b1t_kpwr, b2t_kpwr, bt_kpwr, t_kpwr, ℓbn = @sblock let b1t, b2t, bt, t
    ℓbn, b1t_kpwr = CMBrings.quasi_bandpowers(b1t; Δℓsph_bin = 10)
    ℓbn, b2t_kpwr = CMBrings.quasi_bandpowers(b2t; Δℓsph_bin = 10)
    ℓbn, bt_kpwr = CMBrings.quasi_bandpowers(bt; Δℓsph_bin = 10)
    ℓbn, t_kpwr = CMBrings.quasi_bandpowers(t; Δℓsph_bin = 10)
    b1t_kpwr, b2t_kpwr, bt_kpwr, t_kpwr, ℓbn
end

fig,ax = subplots(2, dpi=147)
ul = findfirst(ℓbn .> srt_ramp) |> x->(isnothing(x) ? length(ℓbn) : x[1])
ll = findfirst(0 .< ℓbn)    |> x->(isnothing(x) ? length(ℓbn) : x[1])

ax[1].plot(ℓbn[ll:ul], bt_kpwr[ll:ul] ./ t_kpwr[ll:ul], label="power ratio:  BT / T")
ax[1].plot(ℓbn[ll:ul], b1t_kpwr[ll:ul] ./ t_kpwr[ll:ul], label="power ratio: B1T / T")
ax[1].plot(ℓbn[ll:ul], b2t_kpwr[ll:ul] ./ t_kpwr[ll:ul], label="power ratio: B2T / T")
ax[1].axhline(y=1, color="black", linestyle="--")
ax[1].legend()

ax[2].plot(ℓbn[ll:ul], b1t_kpwr[ll:ul] ./ bt_kpwr[ll:ul], label="power ratio: B1T / BT")
ax[2].plot(ℓbn[ll:ul], b2t_kpwr[ll:ul] ./ bt_kpwr[ll:ul], label="power ratio: B2T / BT")
ax[2].axhline(y=1, color="black", linestyle="--")
ax[2].legend()

