# Testing out using EAZTransforms

#####################
# TODO items
"""
• make Hpℓ like Hpm
• remove redunant Modules
• add polarization noise sim 
• make some of the filters EAZ transform agnositic and move them to CMBrings
• Make binned versions of the filters to compare bin(filter*field) vrs filter*bin(field)
• Healpix pixel binner and put in CMBrings
• pixel weights and put in CMBrings
• make a master regression projector and put in CMBrings
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

using ClassicalOrthogonalPolynomials
using KahanSummation: sum_kbn
# using SparseArrays
# using BenchmarkTools
# using ProgressMeter
# using BlockArrays
# using Dierckx: Spline1D
# using Measurements
# using ImageFiltering
# import JLD2
using PyPlot
# import PyCall as PC
# HP = PC.pyimport("healpy")

include("LocalMethods.jl")
import .LocalMethods as LM




# Set point source file
# =========================================

point_src_file_ = "/Users/ethananderes/Downloads/3gmaps/resources/spt3g_1500d_mask_list_eete+lensing-19-20_S150=6mJycut_v3.txt"

# Set EAZ grid
# ========================================


tm0, tm2 = @sblock let nφ=18000÷4, φspan=deg2rad.((-60,60)) # default
# tm0, tm2 = @sblock let nφ=18000÷4, φspan=deg2rad.((-45,45)) 
# tm0, tm2 = @sblock let nφ=18000, φspan=deg2rad.((-60,60)) 

    Nside = 2048 # 8192
    ri_offset_from_SP = round(Int, sqrt(3*Nside^2*(1+cos(2.8))))
    ri = (3*Nside+1):1:(4*Nside-1 - ri_offset_from_SP)
    θ  = CC.θ_healpix(Nside)[ri]
    θ∂ = CC.θ_healpix(Nside)[ri.start:ri.step:ri.stop+ri.step]

    tm0 = EAZ0{Float64}(θ, φspan, nφ; θ∂)
    tm2 = EAZ2{Float32}(θ, φspan, nφ; θ∂)

    return tm0, tm2 
end;


# Plot Grid statistics
# ========================================
@sblock let tm0, hide_plots=false
    hide_plots && return
    fig,ax = subplots(1, dpi=147)
    ax.plot(tm0.θ, rad2deg.(.√(EZ.Ωpix(tm0)).*60), label="sqrt pixel area (arcmin)")
    ax.plot(tm0.θ, rad2deg.(EZ.Δθ(tm0).*60), label="Δθ (arcmin)")
    ax.plot(tm0.θ, rad2deg.(sin.(tm0.θ).*EZ.Δφ(tm0).*60), label="pix φ side arclen (arcmin)")
    ax.plot(tm0.θ, EZ.pix_diag_arcmin(tm0), label="pix diag arclen (arcmin)")
    ax.set_xlabel(L"polar coordinate $\theta$")
    ax.legend()
    return nothing
end

@show (tm0.nθ, tm0.nφ)
@show extrema(rad2deg.(.√(EZ.Ωpix(tm0)).*60))
@show extrema(rad2deg.(EZ.Δθ(tm0).*60))
@show extrema(rad2deg.(sin.(tm0.θ) .* EZ.Δφ(tm0) .* 60))
@show extrema(EZ.pix_diag_arcmin(tm0));



# Map space masks: Mp (point source) and Mu (uniform region), M = Mp * Mu
# =======================================================================

# Mp (point source mask)
Mp = CMBrings.pix_point_src_mask(tm0, point_src_file_); 

# Mu (uniform scan region pixel mask)
Mu = @sblock let tm0
    φ = EZ.φ(tm0)
    # lb1, rb1, Δl1, Δr1 = -50, 50, 10, 10
    lb1, rb1, Δl1, Δr1 = -40, 40, 7, 7
    mask   = zeros(eltype_in(tm0),size_in(tm0))
    mask .+= CMBrings.cosφ°Mask.(rad2deg.(φ'); lb=lb1, rb=rb1, Δl=Δl1, Δr=Δr1)
    DiagOp(Xmap(tm0, mask))
end

# M (combined mask) 
M = Mu * Mp

# M_hard (Hard-cut mask, i.e. all observed pixels) 
M_hard = DiagOp(Xmap(tm0, M[:].>0))

# Map plot
CMBrings.map_plot(
    Mp.f, title1="point source pixel mask",
    # Mu.f, title1="uniform scan region pixel mask",
    # M.f, title1="full pixel mask",
);



# Masked mode deprojections
# ======================================

modeMask = M # default
# modeMask = Mu
# modeMask = Mp
# modeMask = M_hard
# modeMask = DiagOp(Xmap(tm0,1))

function ring_regressor(X, maskOp=modeMask)
    maskᵀ = Matrix(transpose(maskOp[:]))
    return function (f::Xfield{<:EZ.EAZ})
        fmapᵀ = Matrix(transpose(f[:]))
        for (v,m) in zip(eachcol(fmapᵀ), eachcol(maskᵀ))
            Xm   = X .* m
            v  .-= Xm * (Xm \ v) 
        end
        Xmap(fieldtransform(f), Matrix(transpose(fmapᵀ)))
    end
end 

# stepwise deprojection ... with KahanSummation
function ring_stepwise_regressor(X, maskOp=modeMask)
    maskᵀ = Matrix(transpose(maskOp[:]))
    return function (f::Xfield{<:EZ.EAZ})
        fmapᵀ = Matrix(transpose(f[:]))
        for (v,m) in zip(eachcol(fmapᵀ), eachcol(maskᵀ))
            Xm   = X .* m
            for x in eachcol(Xm)
                v  .-= x .* (sum_kbn(x .* v) / sum_kbn(abs2.(x)))
            end
        end
        Xmap(fieldtransform(f), Matrix(transpose(fmapᵀ)))
    end
end 



# Filter: Poly (Po),  m based Hp (Hpm), combined (Hpm_Po)
# ==================================================

# -------
# Pfilter = Normalized(ChebyshevT())
Pfilter= Normalized(Legendre())
Po_order = 15
Po, Xpoly = @sblock let tm0, Pfilter,
                        poly_orders = 0:Po_order # default
    x = range(-1, 1; length=tm0.nφ)
    Xpoly = Pfilter[x, poly_orders .+ 1]
    ring_regressor(Xpoly), Xpoly
    # ring_stepwise_regressor(Xpoly), Xpoly
end

# -------
m_Hp_cut = 150
Hpm, Xcos, Xsin = @sblock let   tm0, 
                                mcos=collect(0:m_Hp_cut),  # default
                                msin=collect(1:m_Hp_cut),  # default
    φ = EZ.pix(tm0)[2]
    Xcos  = cos.(mcos' .* φ)
    Xsin  = sin.(msin' .* φ)
    Xhpm = hcat(Xcos, Xsin)
    ring_regressor(Xhpm), Xcos, Xsin
    # ring_stepwise_regressor(Xhpm), Xcos, Xsin
end

# -------
Hpm_Po = @sblock let  X = hcat(Xcos, Xsin, Xpoly[:,2:end]) # default
# Hpm_Po = @sblock let X = hcat(Xsin, Xpoly) # still has bars
# Hpm_Po = @sblock let X = hcat(Xcos[:,2], Xpoly) # still has bars
# Hpm_Po = @sblock let X = hcat(Xcos[:,6], Xpoly) # still has bars
# Hpm_Po = @sblock let X = hcat(Xcos[:,6], Xpoly[:, 1:4], Xpoly[:, 6:end]) # better!!
# Hpm_Po = @sblock let maskOp=Mp, X = mean(eachrow(Mu[:])).*hcat(Xpoly, Xcos[:,2:end], Xsin) |> x->Matrix(qr(x).Q)
    ring_stepwise_regressor(X)
    # ring_regressor(X)
end

#=
X = mean(eachrow(Mu[:])).* hcat(Xcos[:,Po_order+4:50], Xsin[:,Po_order+3:50], Xpoly)
Cov = X'*X
sqrt_inv_Dov = Diagonal(inv.(sqrt.(diag(Cov)))) 
Corr = sqrt_inv_Dov * Cov * sqrt_inv_Dov
Corr |> matshow; colorbar()
=# 
# ======================================

ℓ_Hp_cut = 150 # 50 # 250


# fit version
Hp = @sblock let tm0, ℓ_Hp_cut, Hp_add_poly_order, Pfilter, maskOp=modeMask
    θ, φ = EZ.pix(tm0)

    # k = FT.freq(tm0)[2]
    k = collect(1:1_000)
    # k = collect(0:5:27_000)
    # all the k's that we need for the HP
    k_all  = k[k .<= maximum(ℓ_Hp_cut .* sin.(θ))]

    # the unmasked full column set of modes needed
    Xcos_all  = cos.(k_all' .* φ)
    Xsin_all  = sin.(k_all' .* φ)

    # Poly terms too 
    x = range(-1, 1; length=tm0.nφ)
    Xpoly = Pfilter[x, 1:(Hp_add_poly_order+1)] 

    # uses k, maskᵀ, θ, φ and ℓ_Hp_cut
    maskᵀ  = Matrix(transpose(maskOp[:]))
    get_Xm = function (i4θ)
        msk      = maskᵀ[:,i4θ]
        cols_all = 0 .< k_all .<= ℓ_Hp_cut*sin(θ[i4θ])
        Xm  = msk .* hcat(
            Xpoly,
            Xcos_all[:, cols_all],
            Xsin_all[:, cols_all],
        )
        return Xm 
    end

    # cache the permute in QR
    # TODO: fix the type hardcoding ... 
    pm_vec = Vector{Int64}[]
    for i4θ in eachindex(θ)
        Xm   = get_Xm(i4θ)
        qrXm = qr(Xm, ColumnNorm())
        pm   = qrXm.p[1:rank(Xm)] # default 
        ## pm = 1:size(Xm,2) # testing !!!!!
        push!(pm_vec, pm)
    end

    function (f::Xfield)
        fmapᵀ = Matrix(transpose(f[:]))
        for (v,i4θ) in zip(eachcol(fmapᵀ), eachindex(θ))
            pm     = pm_vec[i4θ]
            Xm     = get_Xm(i4θ)[:,pm] # works nicely ....
            v  .-= Xm * (Xm \ v)
        end
        Xmap(fieldtransform(f), Matrix(transpose(fmapᵀ)))
    end
end


# # Fourier version
# Hp_fft = @sblock let tm0, ℓ_Hp_cut
#     ℓ_from_m = EZ.ell(tm0)
#     DiagOp(Xfourier(tm0, ℓ_from_m .> ℓ_Hp_cut))
# end


# Filter: Nc (notch)
# ==================================
Nc = @sblock let tm0
    X = [(-1)^j for j in 1:tm0.nφ] 
    ring_regressor(X)
end 

# fourier version
# Nc_fft = @sblock let tm0, cutoff=1.0
#     k_cut = EZ.nyq(tm0)[2]
#     k     = EZ.freq(tm0)[2]' .+ falses(tm0.nθ)
#     DiagOp(Xfourier(tm0, k .< k_cut))
# end



# Filter: Lp (low pass)
# ==================================
ℓ_Lp_cut = 13_000

Lp_fft = @sblock let tm0, ℓ_Lp_cut
    # ℓ_from_m = EZ.freq(tm0)[2]' ./ sin.(tm0.θ)
    ℓ_from_m = EZ.ell(tm0)
    DiagOp(Xfourier(tm0, @. exp( - (ℓ_from_m / ℓ_Lp_cut)^6 )))
end


# Pixel bin conv: Pbin
# ==================================
nφ_bin_width = 4

Pbin = @sblock let tm0, kernel_width = nφ_bin_width
    kernel = centered(ones(1,kernel_width) / kernel_width)
    function (f::Xfield)
        Xmap(fieldtransform(f), imfilter(f[:], kernel))
    end
end 


# Decimate pixels: Dci
# ==================================
# dci_factor = nφ_bin_width # default
dci_factor = 2

# Dci, tm0_bin, tmUS2_bin, φ_bin, nφ_bin = @sblock let dci_factor, tm0, φspan
Dci = @sblock let dci_factor

    Dci = function (f::Xfield{EZ.EAZ0{T}}) where T
        tm = fieldtransform(f)
        tm1 = EZ.EAZ0{T}(
            tm.θ, 
            tm.φspan, 
            tm.nφ ÷ dci_factor; 
            θ∂=tm.θ∂
        )
        Xmap(tm1, f[:][:,1:dci_factor:end])
    end

    Dci
end 


# time_sinc filter: Snc
# ==================================
snc_kernel_size = 16 * 2^ceil(Int,log(dci_factor)/log(2)) + 1

Snc = @sblock let snc_kernel_size, dci_factor, tm0
    Δφ = EZ.Δφ(tm0)    # CC.counterclock_Δφ(φ∂[1], φ∂[2])
    Δ  = Δφ*dci_factor
    rb = Δφ*((snc_kernel_size - 1)÷2)
    lb = - rb
    L  = (rb - lb)
    x  = range(lb, rb, snc_kernel_size)
    
    # Hann window
    wx     = @. cospi(x / L)^2 / L

    # kernel
    kernel = centered(
        Array(transpose(wx .* sinc.(x ./ Δ) ./ Δ))
    )

    function (f::Xfield)
        Xmap(fieldtransform(f), imfilter(f[:], kernel))
    end
end 



# Sim iid noise (before filtering and stuff)
# ==================================


μK′ₒ = 5 
Ωₒ   = 2.7e-8 # Ω[end÷2] # setting it this way will change the noise levels depending on the ring spacing
# CMBrings.μKarcmin.(CMBrings.σpix(μK′ₒ, Ωₒ), Ω) # gives the μKarcmin for all other rings.

ℓkₒ  = 1500 # 1500 # 1000 # angular scale that  1/f^α noise crosses the noise
αₒ   = 1 # 3    # power in the 1/f^α noise

μK′pnt_srcₒ = 400 # noise corruption on point sources

poly_order_noiseₒ = Po_order - 1 # TOD polynomial corruption 
poly_σμK_noiseₒ   = 1e1 
poly_stitchₒ    = false # if true simulate discontinuous polynomal
# poly_stitchₒ     = true #<--- switching this on causes some bands that could alias to zig-zags

Pₒ= Pfilter # Normalized(Legendre())

w_eaz, f_eaz, pts_eaz, poly_eaz = @sblock let tm0, μK′ₒ, Ωₒ, αₒ, ℓkₒ, Pₒ, Mp, μK′pnt_srcₒ, poly_order_noiseₒ, poly_σμK_noiseₒ, poly_stitchₒ

    # White noise
    σₒ   = CMBrings.σpix(μK′ₒ, Ωₒ) # pixel noise sd on all rings (=> this isn't white noise)
    w_eaz = σₒ * Xmap(tm0, randn(eltype_in(tm0),size_in(tm0))) # |> Xfourier

    # 1/f noise
    ## kₒ parameterizes where it crosses σₒ. α is the power.
    ## Note that either kₒ or σₒ should be held fix as a parameter since they
    ## are degenerate. Typical use case is to set σₒ to the base white noise level.
    ## Then treat kₒ as the location of the cross-over where 1/f noise dominates (to the left for kₒ)
    ## f⁻¹spec(k; σₒ, kₒ, αₒ=1) = σₒ*(kₒ/k)^αₒ
    # k  = EZ.freq(tm0)[2]
    # ℓk = k' ./ sin.(θ)
    ℓk = EZ.ell(tm0)
    cf = @. XFields.nan2zero(σₒ*(ℓkₒ/ℓk)^αₒ)
    Cf = DiagOp(Xfourier(tm0, cf))
    f_eaz = Cf * Xmap(tm0, randn(eltype_in(tm0),size_in(tm0))) # |> Xfourier

    # Add noise to masked pixels
    pnt_srcs = DiagOp(Xmap(tm0, (Mp[:] .== 0)))
    pts_eaz  = CMBrings.σpix(μK′pnt_srcₒ, Ωₒ) * pnt_srcs * Xmap(tm0, randn(eltype_in(tm0),size_in(tm0)))

    # TOD polynomial corruption 
    # pords = collect(0:poly_order_noiseₒ)'
    xφ  = range(-1, 1; length=tm0.nφ)
    X   = Pₒ[xφ, 1:(poly_order_noiseₒ+1)] 
    poly_eazᵀ = Array(transpose(zeros(eltype_in(tm0),size_in(tm0))))
    for x in eachcol(poly_eazᵀ)
        # μ  = poly_stitchₒ ? rand(φ) : rand(CC.in_negπ_π.(φ))
        # X  = poly_stitchₒ ? (φ .- μ).^pords : (CC.in_negπ_π.(φ) .- μ).^pords
        x .= poly_σμK_noiseₒ * X * randn(size(X,2))
    end
    poly_eaz  = Xmap(tm0, Array(transpose(poly_eazᵀ)))

    w_eaz, f_eaz, pts_eaz, poly_eaz
end 

n_eaz = w_eaz + f_eaz + pts_eaz + poly_eaz


# Filter the noise and plot it
# =============================

# default
# @time filt_n_eaz =  Hp(Po(M*(w_eaz + f_eaz + pts_eaz + poly_eaz))); 
# @time filt_n_eaz =  Hpm_Po(M * n_eaz); 
# @time filt_n_eaz =  Hpm(Po(M * n_eaz)); 
@time filt_n_eaz =  Po(Hpm(M * n_eaz)); 


# @time filt_n_eaz =  Po(M * w_eaz); # ✓
# @time filt_n_eaz =  Hpm(M * w_eaz); # ✓
# @time filt_n_eaz =  Po(Hpm(M * w_eaz)); # ✓
# @time filt_n_eaz =  Hpm(Po(M * w_eaz)); # ✓
# @time filt_n_eaz =  Hpm_Po(M * w_eaz); # x declination bands
# filt_n_eaz = filt_n_eaz - M * w_eaz


# impulse
# δ_eaz_mat = 0*n_eaz[:] 
# δ_eaz_mat[:,end÷2] .= 1
# δ_eaz = Xmap(tm0, δ_eaz_mat)
# @time filt_n_eaz =  Hp(δ_eaz); 


# Fourier plot
# ---------------------

CMBrings.fourier_power(
    filt_n_eaz;
    imag_fun=x->CMBrings.imag_blur(abs2.(x);blur=1),  vmax = 400,
    # imag_fun=CMBrings.imag_logabs2clip, # vmin = -5, # vmin = -20,
    title1="pure time-stream noise on each row",  
    # ℓs = [ℓ_Hp_cut, 4000, 10000, ℓ_Lp_cut, Int(2.5*8192 - 1), Int(3*8192 - 1)], 
    ℓs = [300, 4000, 10000, 13000, Int(2.5*8192 - 1), Int(3*8192 - 1)], 
);

# Map plot
# ---------------------

CMBrings.map_plot(
    filt_n_eaz,
    imag_fun=x->CMBrings.imag_blur(x;blur=0), # vmin = -25, vmax = 25,
    #imag_fun=x->CMBrings.imag_blur(x;blur=3), # vmin = -15, vmax = 15,
    title1="noise sim",
);


# EAZ quasi bandpowers
# ---------------------

# setting `f1 = w_eaz, f2 = Pbin(w_eaz)` probes the pixel window function
# actually not, since in this case with an eaz -> eaz pixel bin, the 
# output will still be white. 

f1_kpwr, f2_kpwr, ℓbn = @sblock let f1 = w_eaz+f_eaz, f2 = filt_n_eaz
# f1_kpwr, f2_kpwr, ℓbn = @sblock let f1 = w_eaz, f2 = Pbin(w_eaz)
# f1_kpwr, f2_kpwr, ℓbn = @sblock let f1 = f_eaz, f2 = Pbin(f_eaz)
    ℓbn, f1_kpwr = CMBrings.quasi_bandpowers(f1; Δℓsph_bin = 25)
    ℓbn, f2_kpwr = CMBrings.quasi_bandpowers(f2; Δℓsph_bin = 25)
    f1_kpwr, f2_kpwr, ℓbn
end




fig,ax = subplots(1) # , dpi=147)
ul = findfirst(ℓbn .> 4_000) |> x->(isnothing(x) ? length(ℓbn) : x[1])
ax.plot(ℓbn[2:ul], f1_kpwr[2:ul])
ax.plot(ℓbn[2:ul], f2_kpwr[2:ul])
# ax.plot(ℓbn[2:ul], f2_kpwr[2:ul]./f1_kpwr[2:ul])

X = [ones(ul-1) ;; ℓbn[2:ul] ;; ℓbn[2:ul].^2 ;; ℓbn[2:ul].^3 ;; ℓbn[2:ul].^4]
# β = X \ (log.(f2_kpwr[2:ul]) .- log.(f1_kpwr[2:ul]))
β = [
-1.9823545418378444e-10
 -3.308283032146229e-7
 -2.978442041888445e-9
 -1.2334147276440859e-14
 -3.4261603311566984e-19
]
ax.semilogy(ℓbn[2:ul], exp.(X*β))



plot(ℓbn[2:ul], f2_kpwr[2:ul]./f1_kpwr[2:ul] .- exp.(X*β))


ax.grid(true)
ax.set_title("quasi-bandpowers");

