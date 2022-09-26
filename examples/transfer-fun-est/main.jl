#####################
# TODO items
"""
✓ post-process QU pre-filtered maps so they are in the same format
  as the filtered QU 
• remove the loading dependance on lb1, rb1, Δl1, Δr1 = -50, 50, 10, 10 ...
  use uniform mask Mu, or M_hard instead ...
• make some basic plots of the 2D and quasi-bandpower 1D transfer functions
• make all the hpix2... methods return Xfield(<:EAZ)
• Figure out how to get rid of ring_idx_rng
• 
""" 
#####################


# adapted from 2022/m04-d21-hpix3G/script4
# and 2022/m07_d21_point_src_mask


# Modules
# =========================================

using LinearAlgebra
using FFTW
FFTW.set_num_threads(BLAS.get_num_threads())

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

using LBblocks: @sblock

using PyPlot
import PyCall as PC
HP = PC.pyimport("healpy")

# include("LocalMethods.jl")
# import .LocalMethods as LM




# Set files and load healpix files
# =========================================
cmb_file_root = "/Users/ethananderes/Downloads/3gmaps/sims"
noise_file_root = "/Users/ethananderes/Downloads/3gmaps/data"

TF_cmb_file_, preTF_cmb_file_ = @sblock let cmb_file_root, noise_file_root

    preTF_cmb_file_ = joinpath(cmb_file_root, "lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed1_lmax17000_nside8192_interp1.6_method1_pol_1_lensedmap.fits")
    
    # return TF_cmb_file_ =  joinpath(cmb_file_root, "Coadd_allfields_lencmbonly_spt3g90ghz.hpix")
    TF_cmb_file_ =  joinpath(cmb_file_root, "Coadd_allfields_lencmbonly_spt3g150ghz.hpix")
    # return TF_cmb_file_ =  joinpath(cmb_file_root, "Coadd_allfields_lencmbonly_spt3g220ghz.hpix")
    
    # return cmb_file_ =  joinpath(noise_file_root,"signflip_001_bundle_000.g3.gz_90.hpix")
    # return cmb_file_ = joinpath(noise_file_root,"signflip_001_bundle_000.g3.gz_150.hpix")
    # return cmb_file_ = joinpath(noise_file_root,"signflip_001_bundle_000.g3.gz_220.hpix")
    # return cmb_file_ = joinpath(noise_file_root,"wei_signflip/signflip_000_bundle_000_150Ghz.hpx")

    return TF_cmb_file_, preTF_cmb_file_
end


# Set point source file
# =========================================

point_src_file_ = "/Users/ethananderes/Downloads/3gmaps/resources/spt3g_1500d_mask_list_eete+lensing-19-20_S150=6mJycut_v3.txt"


# Set Healpix grid
# =========================================
Nside  = 2048*4
lmax   = Int(2.5*Nside) #  3*Nside-1

tmℍ2 = HT.ℍ2{Float64}(Nside; lmax)
tmℍ0 = HT.ℍ0{Float64}(Nside; lmax)

l, m  = HT.lm(lmax);


# Set EAZ grid
# ========================================

eaz0, eaz2, ring_idx_rng = @sblock let Nside

    nφ    = 4 * (Nside-2) ÷ 4 # note 4(Nside-2) == 2^3 * 3^2 * 5 * 7
    φspan = (-π/3, π/3) # deg2rad.((-60,60))

    ri_offset_from_SP = round(Int, sqrt(3*Nside^2*(1+cos(2.8))))
    # ri = (3*Nside+1):1:(4*Nside-1 - ri_offset_from_SP)
    ri = (3*Nside+1):2:(4*Nside-1 - ri_offset_from_SP)
    θ  = CC.θ_healpix(Nside)[ri]
    θ∂ = CC.θ_healpix(Nside)[ri.start:ri.step:ri.stop+ri.step]

    eaz0 = EAZ0{Float64}(θ, φspan, nφ; θ∂)
    eaz2 = EAZ2{Float64}(θ, φspan, nφ; θ∂)

    return eaz0, eaz2, ri 
end;


@sblock let eaz0, hide_plots=false
    hide_plots && return
    fig,ax = subplots(1, dpi=147)
    ax.plot(eaz0.θ, rad2deg.(.√(EZ.Ωpix(eaz0)).*60), label="sqrt pixel area (arcmin)")
    ax.plot(eaz0.θ, rad2deg.(EZ.Δθ(eaz0).*60), label="Δθ (arcmin)")
    ax.plot(eaz0.θ, rad2deg.(sin.(eaz0.θ).*EZ.Δφ(eaz0).*60), label="pix φ side arclen (arcmin)")
    ax.plot(eaz0.θ, EZ.pix_diag_arcmin(eaz0), label="pix diag arclen (arcmin)")
    ax.set_xlabel(L"polar coordinate $\theta$")
    ax.legend()
    return nothing
end

@show (eaz0.nθ, eaz0.nφ)
@show extrema(rad2deg.(.√(EZ.Ωpix(eaz0)).*60))
@show extrema(rad2deg.(EZ.Δθ(eaz0).*60))
@show extrema(rad2deg.(sin.(eaz0.θ) .* EZ.Δφ(eaz0) .* 60))
@show extrema(EZ.pix_diag_arcmin(eaz0));


# Healpix pwf 
# =============================
PWF2▪  = CMBrings.healpix_pwf▫(eaz2; Nside=2048) |> CircOp 

PWF0▪  = CMBrings.healpix_pwf▫(eaz0; Nside=2048) |> CircOp 

# @time nhpx_θ = CMBrings.healpix_count_θ(eaz0, 2048)
# plot(nhpx_θ)

PWF▪[1] |> matshow


# Map space masks: Mp (point source) and Mu (uniform region), M = Mp * Mu
# =======================================================================

# Mp (point source mask)
Mp0 = CMBrings.pix_point_src_mask(eaz0, point_src_file_; smooth_border_Δ′= 10); 
Mp2 = DiagOp(Xmap(eaz2, Mp0[:]))

# Mu (uniform scan region pixel mask)
Mu0 = @sblock let eaz0
    ## parameters ...
    lb1, rb1, Δl1, Δr1 = -48, 48, 6, 6
    # lb1, rb1, Δl1, Δr1 = -50, 50, 10, 10
    # lb1, rb1, Δl1, Δr1 = -40, 40, 7, 7
    
    φ = EZ.φ(eaz0)
    mask   = zeros(eltype_in(eaz0),size_in(eaz0))
    mask .+= CMBrings.cosφ°Mask.(rad2deg.(φ'); lb=lb1, rb=rb1, Δl=Δl1, Δr=Δr1)
    DiagOp(Xmap(eaz0, mask))
end
Mu2 = DiagOp(Xmap(eaz2, Mu0[:]))


# M (combined mask) 
M0 = Mu0 * Mp0
M2 = Mu2 * Mp2

# M_hard (Hard-cut mask, i.e. all observed pixels) 
M_hard0 = DiagOp(Xmap(eaz0, M0[:].>0))
M_hard2 = DiagOp(Xmap(eaz2, M0[:].>0))

# Map plot
CMBrings.map_plot(
    # Mp0.f, title1="point source pixel mask",
    # Mu0.f, title1="uniform scan region pixel mask",
    M0.f, title1="full pixel mask",
);


# Load pre filtered eaz maps
# ========================================

@time qu_eaz, t_eaz = @sblock let  g3_adjust=1, cmb_file_=preTF_cmb_file_, HP, tmℍ0, tmℍ2, eaz0, eaz2, ring_idx_rng

    φ, φ_full = EZ.φ(eaz0), EZ.φ_full(eaz0)
    hpix_map_IQU  = g3_adjust .* HP.read_map(cmb_file_, field=(0,1,2))

    t_hpx   = Xmap(tmℍ0, hpix_map_IQU[1,:])
    # qu_hpx  = Xmap(tmℍ2, hcat(hpix_map_IQU[2,:], hpix_map_IQU[3,:]) )
    # ↓↓↓ here is the adjustment to put into healpix convention ...
    qu_hpx  = Xmap(tmℍ2, hcat(.- hpix_map_IQU[2,:], .- hpix_map_IQU[3,:]) )

    # -- default
    # lb1, rb1, Δl1, Δr1 = -50, 50, 10, 10
    # lb2, rb2, Δl2, Δr2 = -50, 50, 10, 10
    # ---
    # lb1, rb1, Δl1, Δr1 = -59, 59, 10, 10
    # lb2, rb2, Δl2, Δr2 = -59, 59, 10, 10
    #  ---- 
    lb1, rb1, Δl1, Δr1  = -180, 180, 0, 0
    lb2, rb2, Δl2, Δr2  = -180, 180, 0, 0

    qu_eaz  = CMBrings.hpix2equirect_patch(
        qu_hpx;
        ring_idx_rng, φ, φ_full, 
        lb=lb1, rb=rb1, Δl=Δl1, Δr=Δr1,
    ) # |> x->Xmap(eaz2, x.*CMBrings.cosφ°Mask.(rad2deg.(φ'); lb=lb2, rb=rb2, Δl=Δl2, Δr=Δr2))

    t_eaz  = CMBrings.hpix2equirect_patch(
        t_hpx;
        ring_idx_rng, φ, φ_full, 
        lb=lb1, rb=rb1, Δl=Δl1, Δr=Δr1,
    ) # |> x->Xmap(eaz0, x.*CMBrings.cosφ°Mask.(rad2deg.(φ'); lb=lb2, rb=rb2, Δl=Δl2, Δr=Δr2))

    return Xmap(eaz2, qu_eaz), Xmap(eaz0, t_eaz)
end;



# Load filtered eaz maps
# ========================================

@time TF_qu_eaz, TF_t_eaz = @sblock let  g3_adjust=1, cmb_file_=TF_cmb_file_, HP, tmℍ0, tmℍ2, eaz0, eaz2, ring_idx_rng

    φ, φ_full = EZ.φ(eaz0), EZ.φ_full(eaz0)
    hpix_map_IQU  = g3_adjust .* HP.read_map(cmb_file_, field=(0,1,2))

    qu_hpx  = Xmap(tmℍ2, hcat(hpix_map_IQU[2,:], hpix_map_IQU[3,:]) )
    t_hpx   = Xmap(tmℍ0, hpix_map_IQU[1,:])

    # -- default
    # lb1, rb1, Δl1, Δr1 = -50, 50, 10, 10
    # lb2, rb2, Δl2, Δr2 = -50, 50, 10, 10
    # ---
    # lb1, rb1, Δl1, Δr1 = -59, 59, 10, 10
    # lb2, rb2, Δl2, Δr2 = -59, 59, 10, 10
    #  ---- 
    lb1, rb1, Δl1, Δr1  = -180, 180, 0, 0
    lb2, rb2, Δl2, Δr2  = -180, 180, 0, 0

    qu_eaz  = CMBrings.hpix2equirect_patch(
        qu_hpx;
        ring_idx_rng, φ, φ_full, 
        lb=lb1, rb=rb1, Δl=Δl1, Δr=Δr1,
    ) # |> x->Xmap(eaz2, x.*CMBrings.cosφ°Mask.(rad2deg.(φ'); lb=lb2, rb=rb2, Δl=Δl2, Δr=Δr2))

    t_eaz  = CMBrings.hpix2equirect_patch(
        t_hpx;
        ring_idx_rng, φ, φ_full, 
        lb=lb1, rb=rb1, Δl=Δl1, Δr=Δr1,
    ) # |> x->Xmap(eaz0, x.*CMBrings.cosφ°Mask.(rad2deg.(φ'); lb=lb2, rb=rb2, Δl=Δl2, Δr=Δr2))

    return Xmap(eaz2, qu_eaz), Xmap(eaz0, t_eaz)
end;

# approx 2D Tf multiplier
# =============================

# initialize
Tf0 = DiagOp(Xfourier(eaz0,1))
Tf2 = DiagOp(Xfourier(eaz2,1))

# add high pass
ℓ_Hp  = 275
Tf0  *= DiagOp(Xfourier(eaz0, abs.(EZ.ell(eaz0)) .> ℓ_Hp))
Tf2  *= DiagOp(Xfourier(eaz2, abs.(EZ.ell(eaz2)) .> ℓ_Hp))

# add low pass 
ℓ_Lp = 13_000
Tf0  *= DiagOp(Xfourier(eaz0, exp.(.- (abs.(EZ.ell(eaz0))./ℓ_Lp).^6) ))
Tf2  *= DiagOp(Xfourier(eaz2, exp.(.- (abs.(EZ.ell(eaz2))./ℓ_Lp).^6) ))

# add beam 
fwhm′  = 1.5 # 1.7
B0, B2 = @sblock let eaz0, eaz2, fwhm′
    fwhmrad = CMBrings.arcmin2rad(fwhm′)
    σ²      = CMBrings.fwhmrad2σ²(fwhmrad)
    ℓ0  = abs.(EZ.ell(eaz0))
    ℓ2  = abs.(EZ.ell(eaz2))
    bℓ0 = @. exp( - ℓ0 * (ℓ0+1) * σ² / 2)
    bℓ2 = @. exp( - ℓ2 * (ℓ2+1) * σ² / 2)
    DiagOp(Xfourier(eaz0, bℓ0)), DiagOp(Xfourier(eaz2, bℓ2))
end
Tf0  *= B0
Tf2  *= B2





# some plots
# =============================

CMBrings.map_plot(
    # M0 * TF_t_eaz; title1=L"$Tf * T$", # imag_fun=x->CMBrings.imag_blur(x;blur=25),
    # M0 * Tf0 * t_eaz; title1=L"approximated $Tf * T$", # imag_fun=x->CMBrings.imag_blur(x;blur=25),
    #
    M2 * TF_qu_eaz; title1=L"$Tf * Q$ mock-sim", title2=L"$Tf * U$ mock-sim",  # imag_fun=x->CMBrings.imag_blur(x;blur=25),
    # M2 * Tf2 * qu_eaz; title1=L"$Tf * Q$ mock-sim", title2=L"$Tf * U$ mock-sim", #  imag_fun=x->CMBrings.imag_blur(x;blur=25),
);

# %%

CMBrings.fourier_power(
    # M0 * TF_t_eaz; title1=L"log EAZ-fourier power: $Tf * T$ mock-sim", imag_fun=CMBrings.imag_logabs2clip,
    M0 * Tf0 * t_eaz; title1=L"log EAZ-fourier power: $T$ mock-sim", imag_fun=CMBrings.imag_logabs2clip,
    #
    # Mu2 * TF_qu_eaz; title1=L"log EAZ-fourier power: $Tf * P$ mock-sim", imag_fun=CMBrings.imag_logabs2clip,
    # Mu2 * Tf2 * qu_eaz; title1=L"log EAZ-fourier power: $P$ mock-sim", imag_fun=CMBrings.imag_logabs2clip,
    ℓs = [275, 10_000, 13_000, Int(2048*2.5-1)], 
    xaxis_units = :m # :Hz
);





# EAZ quasi bandpowers
# =============================


f1_kpwr, f2_kpwr, ℓbn = @sblock let f1 = M0 * TF_t_eaz, # ... or Mu0
                                    f2 = M0 * Tf0 * t_eaz
#f1_kpwr, f2_kpwr, ℓbn = @sblock let f1 = M2 * TF_qu_eaz, # ... or Mu2
#                                     f2 = M2 * Tf2 * qu_eaz                                 
    ℓbn, f1_kpwr = CMBrings.quasi_bandpowers(f1; Δℓsph_bin = 15)
    ℓbn, f2_kpwr = CMBrings.quasi_bandpowers(f2; Δℓsph_bin = 15)
    f1_kpwr, f2_kpwr, ℓbn
end

fig,ax = subplots(2, dpi=147)
ul = findfirst(ℓbn .> 10_000) |> x->(isnothing(x) ? length(ℓbn) : x[1])
ll = findfirst(ℓ_Hp .< ℓbn) |> x->(isnothing(x) ? length(ℓbn) : x[1])
ax[1].semilogy(ℓbn[ll:ul], f1_kpwr[ll:ul], label="spt filtered sim sky")
ax[1].plot(ℓbn[ll:ul], f2_kpwr[ll:ul], label="2d filtered sim sky")
ax[2].plot(ℓbn[ll:ul], f1_kpwr[ll:ul] ./ f2_kpwr[ll:ul], label="power ratio: spt filt / 2d filt")
# ↑↑↑ Is the missing factor the beam ?? or a pixel window function ??
ax[2].axhline(y=1, color="black", linestyle="--")
ax[1].legend()
ax[2].legend()



# mode-by-mode eaz fourier ratio 
# =============================
# This allows us to check if the missing filter has ell based contours

CMBrings.fourier_power(
    Xfourier(eaz0, (M0 * TF_t_eaz)[!] ./ (M0 * Tf0 * t_eaz)[!]),
    vmin = 0.5, vmax=2.0,
    ℓs = [275, 10_000, 13_000, Int(2048*2.5-1)], 
);


"""
From this picture it does appear that the missing filter is isotropic. 
Could this be the beam?  
"""








# Old ....
# =============================

# perhaps this should go into the directory examples/signal-noise-sims








# Modeling the signal part (spectra, EAZ blocks, poly trough, beam, mask)
# • The model is M * B▪ * PT▪ * Ł(ϕ) * qu
# • With a beam of 2.15 arcmins (ell cut of 10_000) our EAZ model power is too small after 3000 or so.
# • Perhaps this missing power above 3000 is point sources, a pixel window effect ?? Interpolation effect?? 
# TODO: model point sources, beam
# ==============================

# Spectra
# --------

φ_approx_nyq = freq_mult * nφ / minimum(sin.(θ)) / 2
θ_approx_nyq = π / minimum(Δθ) 
@show approx_lmax = ceil(Int, sqrt(φ_approx_nyq^2 + θ_approx_nyq^2))

approx_lmax += ceil(Int, approx_lmax * 0.1) # for good measure:)
## override ...
## approx_lmax = 25_000

ℓ, ttℓ, eeℓ, bbℓ,  ϕϕℓ, ẽẽℓ, b̃b̃ℓ = @sblock let lmax=approx_lmax, r=0.01, T=Float64
    
    l = 0:lmax
    cld = camb_cls(;lmax=lmax, r,
        lSampleBoost   = 4.0,
        lAccuracyBoost = 4.0,
        KmaxBoost = 4.0,
    )
    
    ttsl = cld[:unlen_scalar] |> x->(x[:Ctt] ./ x[:factor_on_cl_cmb])
    tttl = cld[:unlen_tensor] |> x->(x[:Ctt] ./ x[:factor_on_cl_cmb])
    ttl  = ttsl .+ tttl
    ttl[1] = 0

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
    b̃bl    = b̃bsl .+ bbtl # we only have lensed spectra for scalar
    b̃bl[1] = 0

    ϕϕl    = cld[:phi] |> x->(x[:Cϕϕ] ./ x[:factor_on_cl_phi])
    ϕϕl[1] =  ϕϕl[2] ### trying to fix a rank degeneracy here ...

    return l,T.(ttl), T.(eel), T.(bbl), T.(ϕϕl), T.(ẽel), T.(b̃bl) 
end;

fig,ax = subplots(1)
ax.loglog( ℓ.^2 .* ttℓ)
ax.loglog( ℓ.^2 .* eeℓ)
ax.loglog( ℓ.^2 .* bbℓ)
ax.loglog( ℓ.^2 .* ẽẽℓ)
ax.loglog( ℓ.^2 .* b̃b̃ℓ)

#=
EB▫_θ = CMBrings.az_cov_blks(ℓ, eeℓ, bbℓ; θ=θ[end-500:end], φ, ℓrange=[nφ÷2-5,nφ÷2+1], ngrid=100_000);
EB▫_θ = CMBrings.az_cov_blks(ℓ, eeℓ, bbℓ; θ=θ[1:500], φ, ℓrange=[nφ÷2-5,nφ÷2+1], ngrid=100_000);
EB▫_θ = CMBrings.az_cov_blks(ℓ, ℓ.*(ℓ .+ 1).*eeℓ, ℓ.*(ℓ .+ 1).*bbℓ; θ=θ[end-500:end], φ, ℓrange=[nφ÷2-5,nφ÷2+1], ngrid=100_000);
EB▫_θ[1]   |> Hermitian |> eigen |> x->x.values
EB▫_θ[end] |> Hermitian |> eigen |> x->x.values
=#



# Coordinate pivot, blocks and queries for Vecchia
# ------------------------------------------------

permθ, block_sizesθ = @sblock let nθ, bsd_nθ=bsd_nθ 
    block_sizesθ = VF.block_split(nθ, bsd_nθ)
    permθ=1:nθ
    permθ, block_sizesθ
end

# Spin 2 signal
# ------------

@time EB▪½ = CMBrings.spin2_az_cov½_vecchia_blks(ℓ, eeℓ, bbℓ, block_sizesθ, permθ; θ, φ) |> CircOp;
## sum(Base.summarysize, EB▪½) / 1e9 # 7.41 GB, 3.55min construction, high res
## EB▪⁻½ = map(inv, EB▪½) |> CircOp;
EB▪⁻½ = map(VF.posdef_inv, EB▪½) |> CircOp;

# Spin 0 signals
# ------------

# TT▪½ = CMBrings.spin0_az_cov½_vecchia_blks(ℓ, ttℓ, block_sizesθ, permθ; θ, φ) |> CircOp;
# TT▪⁻½ = map(VF.posdef_inv, TT▪½) |> CircOp;

Phi▪½ = CMBrings.spin0_az_cov½_vecchia_blks(ℓ, ϕϕℓ, block_sizesθ, permθ; θ, φ) |> CircOp;
Phi▪⁻½ = map(VF.posdef_inv, Phi▪½) |> CircOp;

# simulation
# ----------

# t = TT▪½ * Xmap(eaz0,randn(Float64,nθ,nφ));
# TODO: add non-Vecchia version ...

ϕ = Phi▪½ * Xmap(eaz0,randn(Float64,nθ,nφ));
# ------ alt: full non-Vecchia approximate simulation
# @time ϕ = @sblock let ℓ, ϕϕℓ, blksiz=nφ÷5, θ, φ, w=Xmap(eaz0,randn(Float64,nθ,nφ)) 
#     nθ, nφ = length(θ), length(φ)
#     wθ▪    = CMBrings.field2▪(w)
#     fθ▪    = map(similar, wθ▪)
#     ℓfull  = 1:nφ÷2+1
#     ℓblks  = blocks(PseudoBlockArray(ℓfull, VF.block_split(length(ℓfull), blksiz)))
#     for ℓblk in ℓblks
#         Σ▪_ℓblk = CMBrings.az_cov_blks(ℓ, ϕϕℓ; θ, φ, ℓrange=ℓblk)
#         for (i,ℓi) in enumerate(ℓblk)
#             ## L = cholesky(Symmetric(Σ▪_ℓblk[i])).L
#             ## lmul!(L, fθ▪[ℓi])
#             M = sqrt(Symmetric(Σ▪_ℓblk[i]))
#             mul!(fθ▪[ℓi], M, wθ▪[ℓi])
#         end
#     end
#     return CMBrings.▪2field(fieldtransform(w), fθ▪)
# end;

#-

qu = EB▪½ * Xmap(eaz2,randn(ComplexF64,nθ,nφ));
# ------ alt: full non-Vecchia approximate simulation
# qu = @sblock let ℓ, eeℓ, bbℓ, blksiz=nφ÷10, θ, φ, w=Xmap(eaz2,randn(ComplexF64,nθ,nφ)) 
#     nθ, nφ = length(θ), length(φ)
#     wθ▪    = CMBrings.field2▪(w)
#     fθ▪    = map(similar, wθ▪)
#     ℓfull  = 1:nφ÷2+1
#     ℓblks  = blocks(PseudoBlockArray(ℓfull, VF.block_split(length(ℓfull), blksiz)))
#     for ℓblk in ℓblks
#         Σ▪_ℓblk = CMBrings.az_cov_blks(ℓ, eeℓ, bbℓ; θ, φ, ℓrange=ℓblk)
#         for (i,ℓi) in enumerate(ℓblk)
#             ## L = cholesky(Hermitian(Σ▪_ℓblk[i])).L
#             ## lmul!(L, fθ▪[ℓi]) ## This leads to striations in U for some reason
#             M = sqrt(Hermitian(Σ▪_ℓblk[i]))
#             mul!(fθ▪[ℓi], M, wθ▪[ℓi])
#         end
#     end
#     return CMBrings.▪2field(fieldtransform(w), fθ▪)
# end;


# Mask 
# ----

prφ    = CMBrings.cosφ°Mask.(rad2deg.(φ); lb=lb1, rb=rb1, Δl=Δl1, Δr=Δr1)
prφ  .*= CMBrings.cosφ°Mask.(rad2deg.(φ); lb=lb2, rb=rb2, Δl=Δl2, Δr=Δr2)
prθ     = CMBrings.cosφ°Mask.(rad2deg.(θ); lb=132, rb=159, Δl=1/4, Δr=1/4)
M_prθ   = DiagOp(Xmap(eaz2, prθ  .+ falses(size_in(eaz2)) ));
M_prφ   = DiagOp(Xmap(eaz2, prφ' .+ falses(size_in(eaz2)) ));
M       = DiagOp(Xmap(eaz2, prθ .* prφ' ));

ln_prθ  = CMBrings.cosφ°Mask.(rad2deg.(θ); lb=132, rb=159, Δl=1/5, Δr=1/5)
Mϕ      = DiagOp(Xmap(eaz0, ln_prθ .+ falses(size_in(eaz0)) ))

## Mϕ[:] .|> real |> matshow; colorbar()
## prθ .* prφ' .|> real |> matshow; colorbar()

# Lensing operators
# -----------------

∇!,  ∇!_ϕ = CMBrings.generate_∇!∇!ϕ(θ, φ; uniformΔθ = (grid_type == :equiθ) ? true : false); 

Ł, ϕ2v!, ϕ2vᴴ!, ∇! = CMBrings.generate_lense(;
    θ, mv1x=Mϕ[:], mv2x=Mϕ[:], ∇!,  ∇!_ϕ, 
    nsteps_lensing=14
);

# PT▪ == poly trough
# ------------------
PT▪ = @sblock let eaz2, θ

    arcl_filt_width = deg2rad(1.31) # corresponds to l==275
    
    ks = FT.freq(eaz2)[2]' .+ falses(size_out(eaz2))
    @assert size(ks,1) == length(θ)
    for (θi, rowks) in zip(θ, eachrow(ks))
        kmin_cut = 2 * π * sin(θi) / arcl_filt_width
        rowks .= (abs.(rowks) .>= kmin_cut)
    end 

    return DiagOp(Xfourier(eaz2, ks))
end




# B▪ == beam
# ----------

B▪ = @sblock let eaz2, θ

    # beamfwhm_arcmin =  0 
    beamfwhm_arcmin =  2.15 # 2π / 10_000 |> rad2deg |> x->x*60
    # beamfwhm_arcmin =  0.25 
       
    if beamfwhm_arcmin == 0
        return Xfourier(eaz2, 1) |> DiagOp
    else 
        beamfwhm_rad    =  deg2rad(beamfwhm_arcmin / 60)
        beamσ² = beamfwhm_rad^2 / 8 / log(2)
        arclength_k = FT.freq(eaz2)[2]' ./ sin.(θ)
        beamℓ  = @. exp( - abs2(arclength_k)*beamσ² / 2)
        return DiagOp(Xfourier(eaz2, beamℓ))
    end
end


# B▪ = @sblock let eaz2, θ, φ, θ∂, Ω, block_sizesθ, permθ

#     pix_diag_rad   = CC.geoβ.(θ∂[2:end], θ∂[1:end-1], φ[1], φ[2]) # arclength of the pixel diagonals
#     beamfwhm_rad_θ = pix_diag_rad # * 0.95
#     σ²θ            = @. CMBrings.fwhmrad2σ²(beamfwhm_rad_θ)

#     Γbeam_θ₁θ₂φ₁φ⃗ = let σ²θ_spl = Spline1D(θ,σ²θ,k=2)
#         function (θ₁, θ₂, φ₁, φ⃗)
#             complex.(CMBrings.B̃eam1.(θ₁, θ₂, σ²θ_spl(θ₁), σ²θ_spl(θ₂), φ₁ .- φ⃗))
#         end
#     end;


#     nθ, nφ = length(θ), length(φ)
#     DΩΩ  = Diagonal(vcat(Ω, Ω))
    
#     Bspin0▪ = CMBrings.spin0_az_cov_vecchia_blks(
#         Γbeam_θ₁θ₂φ₁φ⃗, block_sizesθ,  permθ; θ, φ
#     ) |> CircOp;

#     B▪ = map(Bspin0▪) do B
#         ## B = Bspin0▪[2]
#         P = B[1]'
#         R = inv(B[2])
#         Mpre = B[3] ## B[3]*B[3]'
#         M = VF.Midiagonal(Mpre.data) # What is the speed effect here??

#         a1 = 1:2nθ |> x->reshape(x,nθ,2)
#         P2 = VF.Piv(a1[P.perm,:][:])
#         M2 = vcat(M.data, M.data) |> VF.Midiagonal
#         invR2 = vcat(
#             R.data, 
#             [zeros(eltype(M.data[1]), size(M.data[1],1), size(M.data[end],2))], 
#             R.data
#         ) |> VF.Ridiagonal |> inv

#         P2' * invR2 * M2 * invR2' * P2 * DΩΩ
#     end |> CircOp

#     return B▪
# end;





# Compare Signal model to mock-sim
# ==========================

lnqu_signal_eaz  = M_prθ * qu_signal_eaz
lnqu_signal_eaz′ = M * B▪ * PT▪ * Ł(ϕ) * qu

# %%

CMBrings.map_plot_QU(
    lnqu_signal_eaz ;  title1=L"$Q$ mock-sim", title2=L"$U$ mock-sim",  # vmin=-4, vmax=4,
    # lnqu_signal_eaz′ ; title1=L"$Q$ EAZ model", title2=L"$U$ EAZ model",  # vmin=-4, vmax=4,
    # lnqu_signal_eaz ;  title1=L"$Q$ mock-sim w/blur", title2=L"$U$ mock-sim w/blur", imag_fun=x->CMBrings.imag_blur(x;blur=20), # vmin=-4, vmax=4,
    # lnqu_signal_eaz′ ; title1=L"$Q$ EAZ model w/blur", title2=L"$U$ EAZ model w/blur", imag_fun=x->CMBrings.imag_blur(x;blur=20), # vmin=-4, vmax=4,
    θ, φ, 
);


# %%

CMBrings.fourier_power(
    #lnqu_signal_eaz; title1=L"log EAZ-fourier power: $P$ mock-sim",  imag_fun=CMBrings.imag_logabs2clip,vmin=-25, vmax = 8, 
    # lnqu_signal_eaz′; title1=L"log EAZ-fourier power: $P$ EAZ model", imag_fun=CMBrings.imag_logabs2clip, vmin=-25, vmax = 8, 
    # lnqu_signal_eaz; title1=L"EAZ-fourier power w/blur: $P$ mock-sim",  imag_fun=x->CMBrings.imag_blur(abs2.(x);blur=5), vmax = 250, 
    # lnqu_signal_eaz′; title1=L"EAZ-fourier power w/blur: $P$ EAZ model", imag_fun=x->CMBrings.imag_blur(abs2.(x);blur=5), vmax = 250, 
    # just for comparison 
    qu_noise_eaz; title1=L"EAZ-fourier power w/blur: $P$ sign-flip noise", imag_fun=x->CMBrings.imag_blur(abs2.(x);blur=5), # vmax=0.00075,
    θ, φ, ℓs = [275, 3000, Int(2048*2.5-1)], 
);


# %%

ℓsph_bin, spt_power = CMBrings.quasi_bandpowers(lnqu_signal_eaz;  θ, Δℓsph_bin = 15)
ℓsph_bin, sim_power = CMBrings.quasi_bandpowers(lnqu_signal_eaz′; θ, Δℓsph_bin = 15)

fig,ax = subplots(2)

ul = 400
ax[1].semilogy(ℓsph_bin[1:ul], spt_power[1:ul], label="mock-sim quasi-power")
ax[1].semilogy(ℓsph_bin[1:ul], sim_power[1:ul], label="EAZ model quasi-power")
ax[2].plot(ℓsph_bin[1:ul], (sim_power./spt_power)[1:ul], label="(EAZ model power)/(mock-sim power)")
ax[1].set_xlabel(L"\ell")
ax[2].set_xlabel(L"\ell")
ax[1].legend()
ax[2].legend()




## The invBeam excess in the spt mock-sims seems to kick in at an arclength of 3.76 arcmin. 
## 
## Lets check the arclength of the az pixel Δφ for healpix
## 
## 
## θ₀, θ¹ = extrema(θ)
## 
## Nside′ = 1024
## r1θ, r1φ, r1idx, r1Δφ, r1nφ = HT.θ_φ_idx_4_rings(Nside′)
## (@. rad2deg(r1Δφ * sin(r1θ))*60)[θ₀ .<= r1θ .<= θ¹] |> plot
## 
## 
## 2π / deg2rad(3.95 / 60)





# Modeling the noise part (TODO)
# ============================


## N▪ = @sblock let μK_arcmin = 1.0, Ω, nφ 
##     σ²   = deg2rad(μK_arcmin/60)^2 # Cⁿℓ == μK_arcmin |> arcmin2radians |> abs2
##     σ²_Ω = σ² ./ Ω
##     Nmat = Diagonal(vcat(σ²_Ω,σ²_Ω))
##     N▫   = [Nmat for ℓ = 1:nφ÷2+1]
##     CircOp(N▫)
## end; 

# This one fixes the noise to match healpix
N▪ = @sblock let μK_arcmin = 6.6, eaz2, Ω, nφ, nθ, θ
    σ²   = deg2rad(μK_arcmin/60)^2 # Cⁿℓ == μK_arcmin |> arcmin2radians |> abs2

    Nside′ = 1024*8
    r1θ, r1φ, r1idx, r1Δφ, r1nφ = HT.θ_φ_idx_4_rings(Nside′)
    θ₀, θ¹ = extrema(θ)
    Ω′ = (r1Δφ[2:end] .* diff(.- cos.(r1θ)))[θ₀ .<= r1θ[2:end] .<= θ¹] |> mean
    σ²_Ω =  fill(σ² ./ Ω′, nθ)

    
    ## Nmat = Diagonal(vcat(σ²_Ω,σ²_Ω))
    ## N▫   = [Nmat for ℓ = 1:nφ÷2+1]
    ## CircOp(N▫)
    DiagOp(Xfourier(eaz2, σ²_Ω .+ falses(nθ, nφ)))
end; 


# # ≈ 1/f noise

CiF▪ = @sblock let eaz2, θ

    c           = 1.0
    arclength_k = FT.freq(eaz2)[2]' ./ sin.(θ)

    return DiagOp(Xfourier(eaz2, @.  c * pinv(abs(arclength_k))))
end;



#-

## no_wht = map(N▪, Xmap(eaz2,randn(ComplexF64,nθ,nφ))) do Σ,v
##     sqrt(Σ)*v
## end 

no_wht  = sqrt(N▪) * Xmap(eaz2,randn(ComplexF64,nθ,nφ)) 
no_wht′ = sqrt(N▪) * Xmap(eaz2,randn(ComplexF64,nθ,nφ)) 
no_invf = sqrt(CiF▪) * Xmap(eaz2,randn(ComplexF64,nθ,nφ)); 



# Modeling the full datan (TODO)
# ========


## d = M * (T▪ * Ł(ϕ) * qu + no) |> Xfourier;


## d[:] |> real |> matshow; colorbar()
## d[:] |> imag |> matshow; colorbar()
## qu[:] |> real |> matshow; colorbar()
## qu[:] |> imag |> matshow; colorbar()
## ϕ[:] |> matshow; colorbar()
## (Ł(ϕ)*qu - qu)[:] |> real |> matshow; colorbar()
## qu[:] |> imag |> matshow; colorbar()
## (B▪ * B▪ * B▪ * B▪ * B▪ * no)[:] |> real |> matshow; colorbar()
## (B▪ * B▪ * B▪ * B▪ * B▪ * no)[:] |> imag |> matshow; colorbar()

#  comparison: sim and mock-sim or noise flip
# ---------

# * what is Tcal ???
# * get the pixel weights for the noise (as a function of θ if possible)
# * Is the amplification of power in the signal at high freq & small sinθ due to pixel window function?
# * Point source mask for the signal sims (only visible with large blur)

# Noise comparison
# ---------



### outside the poly trough variance est
row_weights = real.(PT▪[!])
row_weights = row_weights ./ sum.(eachrow(row_weights))
σ²_otPT = sum.(eachrow(row_weights .* abs2.((PT▪* QUnoise)[!])))
X1_otPT = [1 .+ 0θ;; θ;; θ.^2] .* (θ .< 2.6)
X2_otPT = (1 .+ 0θ)            .* (2.6 .<= θ .< 2.67)
X3_otPT = [1 .+ 0θ;; θ]        .* (2.67 .<= θ .< 2.69)
X4_otPT = [1 .+ 0θ;; θ]        .* (2.69 .<= θ)
X_otPT  = [X1_otPT ;; X2_otPT ;; X3_otPT ;; X4_otPT]
β_otPT  = X_otPT \ σ²_otPT
W▪_otPT  = DiagOp(Xfourier(eaz2, X_otPT * β_otPT .+ falses(nθ, nφ)))


### inside the poly trough variance est
row_weights = real.(1 .- PT▪[!])
row_weights = row_weights ./ sum.(eachrow(row_weights))
σ²_inPT = sum.(eachrow(row_weights .* abs2.((QUnoise - PT▪* QUnoise)[!]))) 
X1_inPT = [1 .+ 0θ;; θ;; θ.^2] .* (θ .< 2.61)
X2_inPT = (1 .+ 0θ)            .* (2.61 .<= θ .< 2.66)
X3_inPT = [1 .+ 0θ;; θ]        .* (2.66 .<= θ .< 2.68)
X4_inPT = [1 .+ 0θ;; θ]        .* (2.68 .<= θ)
X_inPT  = [X1_inPT ;; X2_inPT ;; X3_inPT ;; X4_inPT]
β_inPT  = X_inPT \ σ²_inPT
W▪_inPT  = DiagOp(Xfourier(eaz2, X_inPT * β_inPT .+ falses(nθ, nφ)))


## plot(σ²_inPT)
## plot(σ²_otPT)
## plot(X_otPT * β_otPT)
## plot(X_inPT * β_inPT)


no_otPT = sqrt(W▪_otPT) * Xfourier(eaz2,randn(ComplexF64,nθ,nφ)) 
no_inPT = sqrt(W▪_inPT) * Xfourier(eaz2,randn(ComplexF64,nθ,nφ)) 
no_sim   = M_prφ * (PT▪ * no_otPT + (no_inPT - PT▪ * no_inPT) + PT▪ * no_invf / 10)


no_spt   = QUnoise
## no_spt   = QUnoise - PT▪* QUnoise



# ## fourier level plots ...

CMBrings.fourier_plot_QU(eaz2, no_spt, θ, φ; 
    blur = 2, 
    logs = false, # vmin = 0, vmax=15, 
    ## logs = true, vmin = -13, vmax=-10, 
    title=L"$|P\,(\theta,\ell_\varphi)|^2$ where $P=Q+iU$ is sign-flip noise",
    save_fig,
    save_fig_filename = "fourier_plot_QU_noise_spt",
)

#-

CMBrings.fourier_plot_QU(eaz2, no_sim, θ, φ; 
    blur = 2, 
    logs = false,  # vmin = 0, vmax=15, 
    ## logs = true, vmin = -11, vmax=-9, 
    title=L"$|P\,(\theta,\ell_\varphi)|$^2 where $P=Q+iU$ is CMBrings-sim noise",
    save_fig,
    save_fig_filename = "fourier_plot_QU_noise_CMBrings",
)


# ## map level plots ...

CMBrings.map_plot_QU(
    no_spt[:] .|> real, 
    no_spt[:] .|> imag,  
    θ, φ; 
    blur = 3, 
    title1=L"$Q(\theta,\varphi)$ spt noise, w/small Gaussian blur", 
    title2=L"$U(\theta,\varphi)$ spt noise, w/small Gaussian blur",
    save_fig,
    save_fig_filename = "map_plot_QU_noiseBlur_spt"
)

#-

CMBrings.map_plot_QU(
    no_sim[:] .|> real, 
    no_sim[:] .|> imag,  
    θ, φ; 
    blur = 3, 
    title1=L"$Q(\theta,\varphi)$ CMBrings noise, w/small Gaussian blur", 
    title2=L"$U(\theta,\varphi)$ CMBrings noise, w/small Gaussian blur",
    save_fig,
    save_fig_filename = "map_plot_QU_noiseBlur_sim"
)

# %% 

ℓsph_bin, no_spt_power = CMBrings.quasi_bandpowers(no_spt; θ, Δℓsph_bin = 15)
ℓsph_bin, no_sim_power = CMBrings.quasi_bandpowers(no_sim; θ, Δℓsph_bin = 15)


fig,ax = subplots(2)

ul = 400

ax[1].semilogy(ℓsph_bin[1:ul], (no_sim_power./no_spt_power)[1:ul])
ax[2].semilogy(ℓsph_bin[1:ul], no_spt_power[1:ul])
ax[2].semilogy(ℓsph_bin[1:ul], no_sim_power[1:ul])






