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

# include(joinpath(CMBrings.module_dir,"examples/eaz_plots/LocalMethods.jl"))
# import .LocalMethods as LM


# Set files and load healpix files
# =========================================
cmb_file_root = "/Users/ethananderes/Downloads/3gmaps/sims"
noise_file_root = "/Users/ethananderes/Downloads/3gmaps/data"

cmb_file_, ghz = @sblock let cmb_file_root, noise_file_root

    # cmb_file_, ghz = joinpath(cmb_file_root, "lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed1_lmax17000_nside8192_interp1.6_method1_pol_1_lensedmap.fits")

    # cmb_file_, ghz =  joinpath(cmb_file_root, "mockobs_v2/Coadd_allfields_90ghz.hpix"), 90
    cmb_file_, ghz =  joinpath(cmb_file_root, "mockobs_v2/Coadd_allfields_150ghz.hpix"), 150
    # cmb_file_, ghz =  joinpath(cmb_file_root, "mockobs_v2/Coadd_allfields_220ghz.hpix"), 220

    # cmb_file_, ghz =  joinpath(cmb_file_root, "mockobs_v2/tqu1_cambphiG1_fg_mdpl2v0.7_90ghz_seed1_3gpatch.fits"), 90
    # cmb_file_, ghz =  joinpath(cmb_file_root, "mockobs_v2/tqu1_cambphiG1_fg_mdpl2v0.7_150ghz_seed1_3gpatch.fits"), 150
    # cmb_file_, ghz =  joinpath(cmb_file_root, "mockobs_v2/tqu1_cambphiG1_fg_mdpl2v0.7_220ghz_seed1_3gpatch.fits"), 220
    
    # cmb_file_, ghz =  joinpath(cmb_file_root, "Coadd_allfields_lencmbonly_spt3g90ghz.hpix"), 90
    # cmb_file_, ghz =  joinpath(cmb_file_root, "Coadd_allfields_lencmbonly_spt3g150ghz.hpix"), 150
    # cmb_file_, ghz =  joinpath(cmb_file_root, "Coadd_allfields_lencmbonly_spt3g220ghz.hpix"), 220
    
    # cmb_file_, ghz = joinpath(noise_file_root,"signflip_001_bundle_000.g3.gz_90.hpix"), 90
    # cmb_file_, ghz = joinpath(noise_file_root,"signflip_001_bundle_000.g3.gz_150.hpix"), 150
    # cmb_file_, ghz = joinpath(noise_file_root,"signflip_001_bundle_000.g3.gz_220.hpix"), 220
    # cmb_file_, ghz = joinpath(noise_file_root,"wei_signflip/signflip_000_bundle_000_150Ghz.hpx")


    return cmb_file_, ghz
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



# Load cmb_file_
# ========================================

@time qu_eaz, t_eaz = @sblock let  g3_adjust=1, cmb_file_, HP, tmℍ0, tmℍ2, eaz0, eaz2, ring_idx_rng

    φ, φ_full = EZ.φ(eaz0), EZ.φ_full(eaz0)
    hpix_map_IQU  = g3_adjust .* HP.read_map(cmb_file_, field=(0,1,2))

    # qu_hpx  = Xmap(tmℍ2, hcat(hpix_map_IQU[2,:], hpix_map_IQU[3,:]) )
    # !!!!! note the signs here
    qu_hpx  = Xmap(tmℍ2, hcat(.- hpix_map_IQU[2,:], .- hpix_map_IQU[3,:]) )
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



# Map space masks: Mp (point source) and Mu (uniform region), M = Mp * Mu
# =======================================================================

# Mp (point source mask)
Mp0 = CMBrings.pix_point_src_mask(eaz0, point_src_file_; smooth_border_Δ′= 10); 
Mp2 = DiagOp(Xmap(eaz2, Mp0[:]))

# Mu (uniform scan region pixel mask)
Mu0 = @sblock let eaz0
    ## parameters ...
    lb1, rb1, Δl1, Δr1 = -48, 48, 6, 6 # default
    # lb1, rb1, Δl1, Δr1 = -50, 50, 10, 10
    # lb1, rb1, Δl1, Δr1 = -40, 40, 7, 7
    # lb1, rb1, Δl1, Δr1 = -58, 58, 2, 2
    
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
#=
CMBrings.map_plot(
    # Mp0.f, title1="point source pixel mask",
    # Mu0.f, title1="uniform scan region pixel mask",
    M0.f, title1="full pixel mask",
);
=#
 

# Extra filter 
# =========

# F0  = DiagOp(Xfourier(eaz0, abs.(EZ.ell(eaz0)) .< 260))
# F2  = DiagOp(Xfourier(eaz2, abs.(EZ.ell(eaz2)) .< 260))
# turn off extra filter 
F0 = F2 = 1



# (θ,φ) plots
# =============================

CMBrings.map_plot(
    # F0 * M0 * t_eaz; title1=L"$T(\theta,\varphi)$ mockobs_v2 (%$ghz Ghz)", 
    # F0 * M0 * t_eaz; title1=L"$T(\theta,\varphi)$  mockobs_v2 (%$ghz Ghz) with Gaussian blur", imag_fun=x->CMBrings.imag_blur(x;blur=5),
    #
    F2 * M2 * qu_eaz; title1=L"$Q(\theta,\varphi)$ mockobs_v2 (%$ghz Ghz)", title2=L"$U(\theta,\varphi)$ mockobs_v2 (%$ghz Ghz)", 
    # F2 * M2 *  qu_eaz; title1=L"$Q(\theta,\varphi)$ mockobs_v2 (%$ghz Ghz) with Gaussian blur", title2=L"$U(\theta,\varphi)$  mockobs_v2 (%$ghz Ghz) with Gaussian blur", imag_fun=x->CMBrings.imag_blur(x;blur=25),
);

# (θ,m) plots
# =============================

CMBrings.fourier_power(
    M0 * t_eaz; title1=L"log ECP-fourier power: $T(\theta,m)$ mockobs_v2 (%$ghz Ghz)", vmin = -8, imag_fun=CMBrings.imag_logabs2clip,
    #
    # M2 * qu_eaz; title1=L"log ECP-fourier power: $[Q+iU](\theta,m)$ mockobs_v2 (%$ghz Ghz)", vmin = -8, imag_fun=CMBrings.imag_logabs2clip,
    # 
    ℓs = [300, 4_000, 13_000, 16_000, Int(Nside*2.5-1)], 
    xaxis_units = :m # :Hz
);



# EAZ quasi bandpowers
# =============================

ℓbn, t_kpwr  = CMBrings.quasi_bandpowers(M0 * t_eaz;   Δℓsph_bin = 20)
ℓbn, qu_kpwr = CMBrings.quasi_bandpowers(M2 * qu_eaz; Δℓsph_bin = 20)

fig,ax = subplots(2, dpi=147)
ul = findfirst(ℓbn .> 15_000) |> x->(isnothing(x) ? length(ℓbn) : x[1])
ll = findfirst(50 .< ℓbn) |> x->(isnothing(x) ? length(ℓbn) : x[1])
ax[1].semilogy(ℓbn[ll:ul], ℓbn[ll:ul].^2 .* t_kpwr[ll:ul], label=L"quasi-bandpowers from $T(\theta,m)$")
ax[2].semilogy(ℓbn[ll:ul], ℓbn[ll:ul].^2 .* qu_kpwr[ll:ul], label=L"quasi-bandpowers from $[Q+iU](\theta,m)$")
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





