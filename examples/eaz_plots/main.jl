#####################
# TODO items
"""
•
• 
""" 
#####################

# Modules
# =========================================

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

using LBblocks: @sblock

using PyPlot
import PyCall as PC
HP = PC.pyimport("healpy")

# include(joinpath(CMBrings.module_dir,"examples/eaz_plots/LocalMethods.jl"))
# import .LocalMethods as LM


# Set files and load healpix files
# =========================================
cmb_file_root = "/Users/ethananderes/Downloads/3gmaps/sims"
data_file_root = "/Users/ethananderes/Downloads/3gmaps/data"

cmb_file_, ghz = @sblock let cmb_file_root, data_file_root

    # cmb_file_, ghz = joinpath(cmb_file_root, "lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed1_lmax17000_nside8192_interp1.6_method1_pol_1_lensedmap.fits")

    # cmb_file_, ghz =  joinpath(cmb_file_root, "mockobs_v2/Coadd_allfields_90ghz.hpix"), 90
    # cmb_file_, ghz =  joinpath(cmb_file_root, "mockobs_v2/Coadd_allfields_150ghz.hpix"), 150
    # cmb_file_, ghz =  joinpath(cmb_file_root, "mockobs_v2/Coadd_allfields_220ghz.hpix"), 220

    # cmb_file_, ghz =  joinpath(cmb_file_root, "mockobs_v2/tqu1_cambphiG1_fg_mdpl2v0.7_90ghz_seed1_3gpatch.fits"), 90
    # cmb_file_, ghz =  joinpath(cmb_file_root, "mockobs_v2/tqu1_cambphiG1_fg_mdpl2v0.7_150ghz_seed1_3gpatch.fits"), 150
    # cmb_file_, ghz =  joinpath(cmb_file_root, "mockobs_v2/tqu1_cambphiG1_fg_mdpl2v0.7_220ghz_seed1_3gpatch.fits"), 220
    
    # cmb_file_, ghz =  joinpath(cmb_file_root, "Coadd_allfields_lencmbonly_spt3g90ghz.hpix"), 90
    # cmb_file_, ghz =  joinpath(cmb_file_root, "Coadd_allfields_lencmbonly_spt3g150ghz.hpix"), 150
    # cmb_file_, ghz =  joinpath(cmb_file_root, "Coadd_allfields_lencmbonly_spt3g220ghz.hpix"), 220
    
    # cmb_file_, ghz = joinpath(data_file_root,"signflip_001_bundle_000.g3.gz_90.hpix"), 90
    # cmb_file_, ghz = joinpath(data_file_root,"signflip_001_bundle_000.g3.gz_150.hpix"), 150
    # cmb_file_, ghz = joinpath(data_file_root,"signflip_001_bundle_000.g3.gz_220.hpix"), 220
    # cmb_file_, ghz = joinpath(data_file_root,"wei_signflip/signflip_000_bundle_000_150Ghz.hpx"), 150

    # scanfir null tests ...

    # cmb_file_, ghz =  joinpath(data_file_root, "bump_600_test_maps_left.fits"), 90
    # cmb_file_, ghz =  joinpath(data_file_root, "bump_600_test_maps_right.fits"), 90
    # cmb_file_, ghz =  joinpath(data_file_root, "bump_600_test_maps_left_minus_right_divided_by_two.fits"), 90
    # cmb_file_, ghz =  joinpath(data_file_root, "bump_600_test_maps_left_plus_right_divided_by_two.fits"), 90
    # cmb_file_, ghz =  joinpath(data_file_root, "bump_600_test_maps_noise.fits"), 90

    
    # filter on/off tests ...

    # filter on
    # cmb_file_, ghz =  joinpath(data_file_root, "filter_test_on_left.fits" ), 90
    # cmb_file_, ghz =  joinpath(data_file_root, "filter_test_on_right.fits"), 90
    # scmb_file_, ghz =  joinpath(data_file_root, "filter_test_on_lmrd2.fits"), 90
    # filter off
    # cmb_file_, ghz =  joinpath(data_file_root, "filter_test_off_left.fits" ), 90
    cmb_file_, ghz =  joinpath(data_file_root, "filter_test_off_right.fits"), 90
    # cmb_file_, ghz =  joinpath(data_file_root, "filter_test_off_lmrd2.fits"), 90

    return cmb_file_, ghz
end

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

    nφ    = 4 * (Nside-2) ÷ 3 # Default.  note 4(Nside-2) == 2^3 * 3^2 * 5 * 7
    # nφ    = 4 * (Nside-2) ÷ 6 #  for testing ...
    φspan = (-π/3, π/3) # deg2rad.((-60,60))

    ri_offset_from_SP = round(Int, sqrt(3*Nside^2*(1+cos(2.8))))
    ri = (3*Nside+1):1:(4*Nside-1 - ri_offset_from_SP) # Default.
    # ri = (3*Nside+1):2:(4*Nside-1 - ri_offset_from_SP)
    # ri = (3*Nside+1):3:(4*Nside-1 - ri_offset_from_SP) # for testing ...
    
    θ  = CC.θ_healpix(Nside)[ri]
    θ∂ = CC.θ_healpix(Nside)[ri.start:ri.step:ri.stop+ri.step]

    eaz0 = EAZ0{Float64}(θ, φspan, nφ; θ∂)
    eaz2 = EAZ2{Float64}(θ, φspan, nφ; θ∂)

    return eaz0, eaz2, ri 
end;

@sblock let eaz0, hide_plots=true
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



# Load cmb_file_
# ========================================

@time qu_eaz, t_eaz = @sblock let  g3_adjust=1, cmb_file_, HP, tmℍ0, tmℍ2, eaz0, eaz2, ring_idx_rng

    φ, φ_full = EZ.φ(eaz0), EZ.φ_full(eaz0)
    hpix_map_IQU  = g3_adjust .* HP.read_map(cmb_file_, field=(0,1,2))
    hpix_map_IQU[hpix_map_IQU .== HT.UNSEEN] .= 0

    t_hpx   = Xmap(tmℍ0, hpix_map_IQU[1,:])
    # qu_hpx  = Xmap(tmℍ2, hcat(hpix_map_IQU[2,:], hpix_map_IQU[3,:]) )
    # ↓↓↓ here is the adjustment to put into healpix convention ...
    qu_hpx  = Xmap(tmℍ2, hcat(hpix_map_IQU[2,:], .- hpix_map_IQU[3,:]) )

    # -- default
    # lb1, rb1, Δl1, Δr1 = -50, 50, 5, 5
    # lb2, rb2, Δl2, Δr2 = -50, 50, 5, 5
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

# qu_eaz_on_diff, t_eaz_on_diff = qu_eaz, t_eaz

# qu_eaz_off_left, t_eaz_off_left = qu_eaz, t_eaz
# qu_eaz_off_right, t_eaz_off_right = qu_eaz, t_eaz
# qu_eaz_off_diff, t_eaz_off_diff = qu_eaz, t_eaz

# Map space masks: Mp (point source) and Mu (uniform region), M = Mp * Mu
# =======================================================================

# Mp (point source mask)
point_src_file_ = "/Users/ethananderes/Downloads/3gmaps/resources/spt3g_1500d_mask_list_eete+lensing-19-20_S150=6mJycut.txt"
Mp0 = CMBrings.pix_point_src_mask(eaz0, point_src_file_; radius_in=:deg, smooth_border_Δ′= 2, skipstart=22); 
# ------
# point_src_file_ = "/Users/ethananderes/Downloads/3gmaps/resources/spt3g_1500d_mask_list_eete+lensing-19-20_S150=6mJycut_v3.txt"
# Mp0 = CMBrings.pix_point_src_mask(eaz0, point_src_file_; radius_in=:arcmin, smooth_border_Δ′= 10, skipstart=22); 

Mp2 = DiagOp(Xmap(eaz2, Mp0[:]))

# Mu (uniform scan region pixel mask)
# ------------- option 1
Mu0 = @sblock let eaz0
    ## parameters ...
    # lb1, rb1, Δl1, Δr1 = -50, 50, 3, 3 
    lb1, rb1, Δl1, Δr1 = -45, 45, 3, 3 # default    
    # lb1, rb1, Δl1, Δr1 = -45, 45, 20, 20 # default    

    φ = EZ.φ(eaz0)
    mask   = zeros(eltype_in(eaz0),size_in(eaz0))
    mask .+= CMBrings.cosφ°Mask.(rad2deg.(φ'); lb=lb1, rb=rb1, Δl=Δl1, Δr=Δr1)
    DiagOp(Xmap(eaz0, mask))
end
Mu2 = DiagOp(Xmap(eaz2, Mu0[:]))
# ------------- option 2
# Mu0 = DiagOp(Xmap(eaz0, (abs2.(t_eaz[:]) .+ abs2.(qu_eaz[:]).*320 .> 50)))
# Mu2 = DiagOp(Xmap(eaz2, Mu0[:]))

# M (combined mask) 
M0 = Mu0 * Mp0
M2 = Mu2 * Mp2

# M_hard (Hard-cut mask, i.e. all observed pixels) 
# Mu0_hard = @sblock let eaz0
#     # lb1, rb1, Δl1, Δr1 = -53, 53, 1, 1    
#     lb1, rb1, Δl1, Δr1 = -50, 50, 1, 1    
#     φ = EZ.φ(eaz0)
#     mask   = zeros(eltype_in(eaz0),size_in(eaz0))
#     mask .+= CMBrings.cosφ°Mask.(rad2deg.(φ'); lb=lb1, rb=rb1, Δl=Δl1, Δr=Δr1)
#     DiagOp(Xmap(eaz0, mask.>0))
# end
# Mu2_hard = DiagOp(Xmap(eaz2, Mu0_hard[:]))
# ------------- option 2
Mu0_hard = DiagOp(Xmap(eaz0, Mu0[:].>0))
Mu2_hard = DiagOp(Xmap(eaz2, Mu0[:].>0))
# ------------- option 3
# Mu0_hard = DiagOp(Xmap(eaz0, (abs2.(t_eaz[:]) .+ abs2.(qu_eaz[:]).*320 .> 50)))
# Mu2_hard = DiagOp(Xmap(eaz2, Mu0[:]))

Mp0_hard = DiagOp(Xmap(eaz0, Mp0[:].>0))
Mp2_hard = DiagOp(Xmap(eaz2, Mp0[:].>0))

M0_hard = Mu0_hard * Mp0_hard
M2_hard = Mu2_hard * Mp2_hard


# Map plot
#=
CMBrings.map_plot(
    # Mp0.f, title1="point source pixel mask",
    # Mu0.f, title1="uniform scan region pixel mask",
    # M0.f, title1="full pixel mask",
    M0_hard.f, title1="full pixel mask",
);
=#
 
# Low pass / High pass / Poly filters
# ===============================================
ℓ_Lp = 12_000 # 13_000 
ℓ_Hp  = 268   # 300
LP0  = DiagOp(Xfourier(eaz0, exp.(.- (abs.(EZ.ell(eaz0))./ℓ_Lp).^6) ))
LP2  = DiagOp(Xfourier(eaz2, exp.(.- (abs.(EZ.ell(eaz2))./ℓ_Lp).^6) ))
HP0 = DiagOp(Xfourier(eaz0, abs.(EZ.ell(eaz0)) .> ℓ_Hp))
HP2 = DiagOp(Xfourier(eaz2, abs.(EZ.ell(eaz2)) .> ℓ_Hp))

using ClassicalOrthogonalPolynomials
include(joinpath(CMBrings.module_dir,"examples/transfer-fun-est/LocalMethods.jl"))
import .LocalMethods as LM
Po_order  = 8
t_pre = range(-1, 1; length=sum(Mu0[:][1,:].>0))
t = zeros(eaz0.nφ)
t[Mu0[:][1,:].>0] .= t_pre
Pfilter   = Legendre() # Normalized(Legendre()) # Normalized(ChebyshevT()) #  
X         = Pfilter[t, 1:(Po_order+1)]
Poly = LM.RingDeprojector(X, M0_hard[:]);



# Extra filter 
# =========

# F0  = DiagOp(Xfourier(eaz0, 550 .< abs.(EZ.ell(eaz0)) .< 610))
# F2  = DiagOp(Xfourier(eaz2, 550 .< abs.(EZ.ell(eaz2)) .< 610))

## @ 320 ell
# F0  = DiagOp(Xfourier(eaz0, exp.(.- 0.5 .* abs2.((EZ.ell(eaz0) .- 320) ./ 20) )))
# F2  = DiagOp(Xfourier(eaz2, exp.(.- 0.5 .* abs2.((EZ.ell(eaz2) .- 320) ./ 20) )))
## @ 580 ell
# F0  = DiagOp(Xfourier(eaz0, exp.(.- 0.5 .* abs2.((EZ.ell(eaz0) .- 580) ./ 30) )))
# F2  = DiagOp(Xfourier(eaz2, exp.(.- 0.5 .* abs2.((EZ.ell(eaz2) .- 580) ./ 30) )))
## @ 1.1Hz both lines
# Hz0 = map(m -> deg2rad(1) / (2π / m), EZ.freq(eaz0)[2])' .+ zeros(size_out(eaz0))
# Hz2 = map(m -> deg2rad(1) / (2π / m), EZ.freq(eaz2)[2])' .+ zeros(size_out(eaz2))
# F0  = DiagOp(Xfourier(eaz0, exp.(.- 0.5 .* abs2.((Hz0 .- 1.1) ./ 0.05) )))
# F2  = DiagOp(Xfourier(eaz2, exp.(.- 0.5 .* abs2.((abs.(Hz2) .- 1.1) ./ 0.05) )))
## @ 1.3Hz right line
# Hz0 = map(m -> deg2rad(1) / (2π / m), EZ.freq(eaz0)[2])' .+ zeros(size_out(eaz0))
# Hz2 = map(m -> deg2rad(1) / (2π / m), EZ.freq(eaz2)[2])' .+ zeros(size_out(eaz2))
# F0  = DiagOp(Xfourier(eaz0, exp.(.- 0.5 .* ((Hz0 .- 1.13) ./ 0.01).^4 )))
# F2  = DiagOp(Xfourier(eaz2, exp.(.- 0.5 .* ((abs.(Hz2) .- 1.13) ./ 0.01).^4 )))
## @ 1.09Hz left line
# Hz0 = map(m -> deg2rad(1) / (2π / m), EZ.freq(eaz0)[2])' .+ zeros(size_out(eaz0))
# Hz2 = map(m -> deg2rad(1) / (2π / m), EZ.freq(eaz2)[2])' .+ zeros(size_out(eaz2))
# F0  = DiagOp(Xfourier(eaz0, exp.(.- 0.5 .* ((Hz0 .- 1.09) ./ 0.01).^4 )))
# F2  = DiagOp(Xfourier(eaz2, exp.(.- 0.5 .* ((abs.(Hz2) .- 1.09) ./ 0.01).^4 )))

## @ 5.5 Hz
# Hz0 = map(m -> deg2rad(1) / (2π / m), EZ.freq(eaz0)[2])' .+ zeros(size_out(eaz0))
# Hz2 = map(m -> deg2rad(1) / (2π / m), EZ.freq(eaz2)[2])' .+ zeros(size_out(eaz2))
# F0  = DiagOp(Xfourier(eaz0, exp.(.- 0.5 .* abs2.((Hz0 .- 5.5) ./ 0.5) )))
# F2  = DiagOp(Xfourier(eaz2, exp.(.- 0.5 .* abs2.((abs.(Hz2) .- 5.5) ./ 0.5) )))
## @ 13.5 Hz
# Hz0 = map(m -> deg2rad(1) / (2π / m), EZ.freq(eaz0)[2])' .+ zeros(size_out(eaz0))
# Hz2 = map(m -> deg2rad(1) / (2π / m), EZ.freq(eaz2)[2])' .+ zeros(size_out(eaz2))
# F0  = DiagOp(Xfourier(eaz0, exp.(.- 0.5 .* abs2.((Hz0 .- 13.5) ./ 0.5) )))
# F2  = DiagOp(Xfourier(eaz2, exp.(.- 0.5 .* abs2.((abs.(Hz2) .- 13.5) ./ 0.5) )))
## @ 16.5 Hz
# Hz0 = map(m -> deg2rad(1) / (2π / m), EZ.freq(eaz0)[2])' .+ zeros(size_out(eaz0))
# Hz2 = map(m -> deg2rad(1) / (2π / m), EZ.freq(eaz2)[2])' .+ zeros(size_out(eaz2))
# F0  = DiagOp(Xfourier(eaz0, exp.(.- 0.5 .* abs2.((Hz0 .- 16.5) ./ 0.5) )))
# F2  = DiagOp(Xfourier(eaz2, exp.(.- 0.5 .* abs2.((abs.(Hz2) .- 16.5) ./ 0.5) )))


## LowHigh/HighPass/Poly filter
F0 = Poly # HP0 * Poly
F2 = Poly # HP2 * Poly

# turn off extra filter 
# F0 = F2 = 1


# (θ,φ) plots
# =============================

CMBrings.map_plot(
    M0 * t_eaz; title1=L"$T(\theta,\varphi)$ (%$ghz Ghz)",
    # M2 * qu_eaz; title1=L"$Q(\theta,\varphi)$ (%$ghz Ghz)", title2=L"$U(\theta,\varphi)$ (%$ghz Ghz)", 

    # Mu0 * t_eaz; title1=L"band pass (left-right)/2 $T(\theta,\varphi)$",
    # F0 * Mu0 * t_eaz; title1=L"band pass (left-right)/2 $T(\theta,\varphi)$",
    # M0 * Xmap(eaz0, real(qu_eaz_off_diff[:])); title1=L"left scan: $Q(\theta,\varphi)$",
    # M0 * Xmap(eaz0, imag(qu_eaz_off_diff[:])); title1=L"(left-right)/2: $U(\theta,\varphi)$,
    #
    imag_fun=x->CMBrings.imag_blur(x;blur=5),
    #
    # vmin = ...,
    # vmax = ...,
);

# (θ,m) plots
# =============================

CMBrings.fourier_power(
    # M0 * t_eaz; title1=L"$T(\theta,m)$ (%$ghz Ghz)",
    M2 * qu_eaz; title1=L"$[Q+iU](\theta,\varphi)$ (%$ghz Ghz)",    
    
    # M0 * t_eaz_off_diff; title1=L"filter_off_lmrd2 : $T(\theta,m)$",
    # M0 * F0 * M0 *  t_eaz_off_left; title1=L"filter_off_left : EAZ Lp * Hp * $T(\theta,\varphi)$",
    # M0 * F0 * M0 *  t_eaz_on_left; title1=L"filter_on_left : EAZ Lp * Hp * $T(\theta,\varphi)$",
    # M0 * Xmap(eaz0, real(qu_eaz_off_diff[:])); title1=L"log power of (left-right)/2: $Q(\theta,m)$",
    # M0 * Xmap(eaz0, imag(qu_eaz_off_diff[:])); title1=L"log power of (left-right)/2: $U(\theta,m)$",
    # 
    # imag_fun=x->abs.(x),
    # imag_fun=CMBrings.imag_logabs2clip,
    imag_fun=x->CMBrings.imag_blur(CMBrings.imag_logabs2clip(x);blur=2),
    # imag_fun=x->CMBrings.imag_blur(CMBrings.angle.(x);blur=2),
    # vmax = 0.001,
    # vmin = -11,
    vmin = -9,
    # vmin = -2, vmax = 2,
    # vmax = -2,
    # ℓs = [550, 580,610, 13_000, 16_000, Int(Nside*2.5-1)], 
    ℓs = [13_000, 16_000, Int(Nside*2.5-1)], 
    # xaxis_units = :m,
    xaxis_units = :Hz
);


# EAZ quasi bandpowers
# =============================

ℓbn, t_kpwr  = CMBrings.quasi_bandpowers(M0 * t_eaz;   Δℓsph_bin = 15)
ℓbn, qu_kpwr = CMBrings.quasi_bandpowers(M2 * qu_eaz; Δℓsph_bin = 15)

fig,ax = subplots(2, dpi=147)
ul = findfirst(ℓbn .> 5_000) |> x->(isnothing(x) ? length(ℓbn) : x[1])
ll = findfirst(50 .< ℓbn) |> x->(isnothing(x) ? length(ℓbn) : x[1])
ax[1].semilogy(ℓbn[ll:ul], ℓbn[ll:ul].^0 .* t_kpwr[ll:ul], label=L"quasi-bandpowers from $T(\theta,m)$")
ax[2].semilogy(ℓbn[ll:ul], ℓbn[ll:ul].^0 .* qu_kpwr[ll:ul], label=L"quasi-bandpowers from $[Q+iU](\theta,m)$")
ax[1].legend()
ax[2].legend()



# Load into CMBLensing.EquiRect
# ==========================================

import CMBLensing as CMBL

t_eqR, qu_eqR = @sblock let eaz0, T=Float64, t_eaz, qu_eaz, M0, M2, F0, F2
	θspan = eaz0.θ∂ |> extrema
	φspan = eaz0.φspan .|> CC.in_negπ_π |> extrema
	Ny    = eaz0.nθ
	Nx    = eaz0.nφ
	proj = CMBL.ProjEquiRect(;Ny, Nx, T, θspan, φspan)
	t_eqR = CMBL.EquiRectMap((F0 * M0 * t_eaz)[:], proj)
	qu_eqR = CMBL.EquiRectQUMap(
		real((F2 * M2 * qu_eaz)[:]), 
		imag((F2 * M2 * qu_eaz)[:]), 
		proj,
	)
	t_eqR, qu_eqR
end;

# plot(t_eqR)
# plot(qu_eqR)

# Project from CMBL.EquiRect back to healpix

Nside′ = 2048*4

qu_eqR_2_q = function (f)
	q = CMBL.Map(f).arr[:,:,1]
	CMBL.EquiRectMap(q, f.proj)
end 

qu_eqR_2_u = function (f)
	u = CMBL.Map(f).arr[:,:,2]
	## notice the sign change on u for healpix convention
	CMBL.EquiRectMap(.- u, f.proj)
end 

t_hpx  = CMBL.project(t_eqR => CMBL.ProjHealpix(Nside′))
q_hpx  = CMBL.project(qu_eqR_2_q(qu_eqR) => CMBL.ProjHealpix(Nside′))
u_hpx  = CMBL.project(qu_eqR_2_u(qu_eqR)  => CMBL.ProjHealpix(Nside′))

# Bandpowers

lmax_bp = 8_000

# auto
t_t_ℓ, e_e_ℓ, b_b_ℓ = HP.sphtfunc.anafast(
	map(x->x.arr, (t_hpx, q_hpx, u_hpx)), 
	lmax=lmax_bp, pol=true, alm=false,
) |> x->(x[1,:], x[2,:], x[3,:]);

# # cross
# t′_t_ℓ, e′_e_ℓ, b′_b_ℓ = HP.sphtfunc.anafast(
# 	map(x->x.arr, (t_hpx,  q_hpx,  u_hpx)), 
# 	map(x->x.arr, (t′_hpx, q′_hpx, u′_hpx)), 
# 	lmax=lmax_bp, pol=true, alm=false,
# ) |> x->(x[1,:], x[2,:], x[3,:]);

# cross corr
# t′_t′_ℓ, e′_e′_ℓ, b′_b′_ℓ = HP.sphtfunc.anafast(
# 	map(x->x.arr, (t′_hpx, q′_hpx, u′_hpx)), 
# 	lmax=lmax_bp, pol=true, alm=false,
# ) |> x->(x[1,:], x[2,:], x[3,:]);
# 
# corr_t′_t_ℓ     = t′_t_ℓ ./ .√(t_t_ℓ .* t′_t′_ℓ)
# corr_e′_e_ℓ     = e′_e_ℓ ./ .√(e_e_ℓ .* e′_e′_ℓ)
# corr_b′_b_ℓ     = b′_b_ℓ ./ .√(b_b_ℓ .* b′_b′_ℓ)

# Band powers

C = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

fig,ax = subplots(1, dpi=147)
ℓx = 0:lmax_bp
ℓr = 10:lmax_bp-10
ax.semilogy(ℓx[ℓr], ℓx[ℓr].^2 .* t_t_ℓ[ℓr], color=C[1] , label=L"$E$ bandpowers")
ax.semilogy(ℓx[ℓr], ℓx[ℓr].^2 .* e_e_ℓ[ℓr], color=C[2] , label=L"$B$ bandpowers")
ax.semilogy(ℓx[ℓr], ℓx[ℓr].^2 .* b_b_ℓ[ℓr], color=C[4] , label=L"$T$ bandpowers")
ax.set_xlabel(L"\ell")
ax.set_ylabel(L"\ell^2 C_\ell")
ax.legend()





