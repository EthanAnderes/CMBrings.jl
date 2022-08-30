# Use CircOp projection matrices to de-project the modes on EAZ/EquiRect fourier seeded by alm for l<=lmin


# TODO
# ==============================
#=

• Using EAZTransforms for all fields

=#



# Modules
# ==============================

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

using AssociatedLegendrePolynomials
# using SparseArrays
# using BenchmarkTools
# using ProgressMeter
# using BlockArrays
# using Dierckx: Spline1D
# using ImageFiltering
import JLD2
using PyPlot
import PyCall as PC
HP = PC.pyimport("healpy")





# Set Healpix grid
# ================================================

Nside  = 2048 # 2048*4
lmax   = Int(2.5*Nside) #  3*Nside-1

tmℍ2 = HT.ℍ2{Float64}(Nside; lmax)
tmℍ0 = HT.ℍ0{Float64}(Nside; lmax)

l, m  = HT.lm(lmax)



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

# Generate spectral density
# ========================

ℓ, eeℓ, bbℓ, ϕϕℓ, ttℓ = @sblock let lmax=max(lmax, 10_000), r=0.01, rT=eltype_in(tmℍ0)

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
    eel[2] = 0

    bbsl = cld[:unlen_scalar] |> x->(x[:Cbb] ./ x[:factor_on_cl_cmb])
    bbtl = cld[:unlen_tensor] |> x->(x[:Cbb] ./ x[:factor_on_cl_cmb])
    # note: bbsl == 0
    bbl    = bbsl .+ bbtl
    bbl[1] = 0
    bbl[2] = 0

    bbl[bbl .<= 0] .= minimum(bbl[3:end][bbl[3:end] .> 0])
    eel[eel .<= 0] .= minimum(eel[3:end][eel[3:end] .> 0])
    bbl[1] = bbl[2] = 0
    eel[1] = eel[2] = 0

    ϕϕl    = cld[:phi] |> x->(x[:Cϕϕ] ./ x[:factor_on_cl_phi])
    ϕϕl[1] = 0

    return l, rT.(eel), rT.(bbl), rT.(ϕϕl), rT.(ttl)
end

figure()
loglog( ℓ.^2 .* eeℓ)
loglog( ℓ.^2 .* bbℓ)
loglog( ℓ.^2 .* ttℓ)
axvline(x=lmax, color="black", label="lmax")



# Simulate on Healpix
lmax_cut = lmax 
low_pass_cut = 200
# Note: HealpixTransforms.jl only works for Float64 (TODO, fix this restriction)
# ===================

# Simulate Healpix $a_{\ell m}$'s 

eblm, tlm, ϕlm = let rT=eltype_in(tmℍ0)
    lmax_ = l .<= lmax_cut
    full_elm = lmax_ .* sqrt.(eeℓ[l.+1]) .* randn(complex(rT), length(l))
    full_blm = lmax_ .* sqrt.(bbℓ[l.+1]) .* randn(complex(rT), length(l))
    full_tlm = lmax_ .* sqrt.(ttℓ[l.+1]) .* randn(complex(rT), length(l))
    full_ϕlm = lmax_ .* sqrt.(ϕϕℓ[l.+1]) .* randn(complex(rT), length(l))
    hcat(full_elm, full_blm), full_tlm, full_ϕlm
end 

low_pass_eblm, low_pass_tlm, low_pass_ϕlm = let 
    low_pass = l .<= low_pass_cut
    eblm.*low_pass, tlm.*low_pass, ϕlm.*low_pass
end

# Convert $a_{\ell m}$'s to healpix Q,U

quθφ           = tmℍ2 \ Xfourier(tmℍ2, eblm)
low_pass_quθφ  = tmℍ2 \ Xfourier(tmℍ2, low_pass_eblm)

tθφ           = tmℍ0 \  Xfourier(tmℍ0, tlm)
low_pass_tθφ  = tmℍ0 \  Xfourier(tmℍ0, low_pass_tlm)

ϕθφ           = tmℍ0 \  Xfourier(tmℍ0, ϕlm)
low_pass_ϕθφ  = tmℍ0 \  Xfourier(tmℍ0, low_pass_ϕlm)




# Healpix -> EAZ grid projection
# ---- options ---------
# For partial sky
lb1, rb1, Δl1, Δr1 = -59, 59, 10, 10
lb2, rb2, Δl2, Δr2 = -59, 59, 10, 10
#  For full sky
## lb1, rb1, Δl1, Δr1  = -180, 180, 0, 0
## lb2, rb2, Δl2, Δr2  = -180, 180, 0, 0
# ---- end options -----
# ==============================

qu_rings  = CMBrings.hpix2equirect_patch(
    quθφ;
    ring_idx_rng, φ, φ_full, 
    lb=lb1, rb=rb1, Δl=Δl1, Δr=Δr1,
) |> x->Xmap(tm2, x.*CMBrings.cosφ°Mask.(rad2deg.(φ'); lb=lb2, rb=rb2, Δl=Δl2, Δr=Δr2))

t_rings  = CMBrings.hpix2equirect_patch(
    tθφ;
    ring_idx_rng, φ, φ_full, 
    lb=lb1, rb=rb1, Δl=Δl1, Δr=Δr1,
) |> x->Xmap(tm0, x.*CMBrings.cosφ°Mask.(rad2deg.(φ'); lb=lb2, rb=rb2, Δl=Δl2, Δr=Δr2))

ϕ_rings  = CMBrings.hpix2equirect_patch(
    ϕθφ;
    ring_idx_rng, φ, φ_full, 
    lb=lb1, rb=rb1, Δl=Δl1, Δr=Δr1,
) |> x->Xmap(tm0, x.*CMBrings.cosφ°Mask.(rad2deg.(φ'); lb=lb2, rb=rb2, Δl=Δl2, Δr=Δr2))

low_pass_qu_rings  = CMBrings.hpix2equirect_patch(
    low_pass_quθφ;
    ring_idx_rng, φ, φ_full, 
    lb=lb1, rb=rb1, Δl=Δl1, Δr=Δr1,
) |> x->Xmap(tm2, x.*CMBrings.cosφ°Mask.(rad2deg.(φ'); lb=lb2, rb=rb2, Δl=Δl2, Δr=Δr2))

low_pass_t_rings  = CMBrings.hpix2equirect_patch(
    low_pass_tθφ;
    ring_idx_rng, φ, φ_full, 
    lb=lb1, rb=rb1, Δl=Δl1, Δr=Δr1,
) |> x->Xmap(tm0, x.*CMBrings.cosφ°Mask.(rad2deg.(φ'); lb=lb2, rb=rb2, Δl=Δl2, Δr=Δr2))

low_pass_ϕ_rings  = CMBrings.hpix2equirect_patch(
    low_pass_ϕθφ;
    ring_idx_rng, φ, φ_full, 
    lb=lb1, rb=rb1, Δl=Δl1, Δr=Δr1,
) |> x->Xmap(tm0, x.*CMBrings.cosφ°Mask.(rad2deg.(φ'); lb=lb2, rb=rb2, Δl=Δl2, Δr=Δr2))



# Spin 0 EAZ low-pass projection with CircOp
# ===================================

ms          = Int.(FT.freq(tm0)[2])  # all the m's
ms_idx      = findall(ms .<= low_pass_cut) # subset of m's we will operate on
ls_given_m  = m -> max(1, m):low_pass_cut                # for each m, what modes to we project to

LPI▪, LPXI▪ = @sblock let θ, ms, ms_idx, ls_given_m, low_pass_cut
    nθ = length(θ)
    λlm_cache   = λlm(0:low_pass_cut, 0:low_pass_cut, cos.(θ))
    I▫  = AbstractMatrix[ false*I(nθ) for i in eachindex(ms)]
    XI▫ = AbstractMatrix[ false*I(nθ) for i in eachindex(ms)]

    for i in ms_idx
        msi = ms[i] 
        ls  = ls_given_m(msi)
        X₀  = CMBrings.index_λlm(ls, msi, 0; λlm_cache) 
        
        # R₀  = qr(X₀).R
        # I▫[i] = X₀ * ((R₀'*R₀) \ X₀') 
        # --- or 
        I▫[i] = X₀ / X₀

        XI▫[i]  = X₀ 
    end

    return CircOp(I▫), CircOp(XI▫) 
end

CMBrings.map_plot(
    low_pass_t_rings; title1=L"Healpix low-pass $T$", vmin=-250, vmax=250,
    # low_pass_ϕ_rings; title1=L"Healpix low-pass $\phi$",
    imag_fun=x->CMBrings.imag_blur(x;blur=0),
);
CMBrings.map_plot(
    # LPI▪*t_rings; title1=L"EAZ EB-low-pass $T$",
    map((X,v)-> (X isa Diagonal) ? 0*v : X*(X\v), LPXI▪, t_rings);  title1=L"EAZ low-pass $T$",  vmin=-250, vmax=250,
    # map((X,v)-> (X isa Diagonal) ? 0*v : X*(X\v), LPXI▪, ϕ_rings);  title1=L"EAZ low-pass $\phi$", 
    imag_fun=x->CMBrings.imag_blur(x;blur=0),
);


CMBrings.fourier_power(
    low_pass_t_rings; title1=L"Healpix low-pass $T$ power",
    # low_pass_ϕ_rings; title1=L"Healpix low-pass $\phi$ power",
    θ, φ, ℓs = [low_pass_cut, lmax_cut], 
    imag_fun=CMBrings.imag_logabs2clip,
);
CMBrings.fourier_power(
    map((X,v)-> (X isa Diagonal) ? 0*v : X*(X\v), LPXI▪, t_rings);  title1=L"EAZ low-pass $T$ power",
    # map((X,v)-> (X isa Diagonal) ? 0*v : X*(X\v), LPXI▪, ϕ_rings);  title1=L"EAZ low-pass $\phi$ power",
    θ, φ, ℓs = [low_pass_cut, lmax_cut], 
    imag_fun=CMBrings.imag_logabs2clip,
    # vmin = -39,
);



# Polarization EAZ low-pass projection with CircOp
# ===================================

ms          = Int.(FT.freq(tm0)[2])  # all the m's
ms_idx      = findall(ms .<= low_pass_cut) # subset of m's we will operate on
ls_given_m  = m -> max(1, m):low_pass_cut                # for each m, what modes to we project to

LPEB▪, LPB▪, LPXEB▪, LPXB▪, LPXE▪ = @sblock let θ, ms, ms_idx, ls_given_m, low_pass_cut
    
    nθ = length(θ)
    λlm_cache   = λlm(0:low_pass_cut, 0:low_pass_cut, cos.(θ))

    B▫    = AbstractMatrix[ false*I(2nθ) for i in eachindex(ms)]
    EB▫   = AbstractMatrix[ false*I(2nθ) for i in eachindex(ms)]
    XE▫   = AbstractMatrix[ false*I(2nθ) for i in eachindex(ms)]
    XB▫   = AbstractMatrix[ false*I(2nθ) for i in eachindex(ms)]
    XEB▫  = AbstractMatrix[ false*I(2nθ) for i in eachindex(ms)]

    for i in ms_idx
        msi  = ms[i] 
        ls   = ls_given_m(msi)
        X₋₂  = CMBrings.index_λlm(ls, msi, -2; λlm_cache) 
        X₊₂  = CMBrings.index_λlm(ls, msi,  2; λlm_cache) 
        eX   = vcat(X₋₂, X₊₂)
        bX   = vcat(- X₋₂, X₊₂)
        ebX  = hcat(eX, bX)

        # --- 
        # bR  = qr(bX).R
        # ebR = qr(ebX).R
        # B▫[i] = bX * ((bR'*bR) \ bX') 
        # EB▫[i] = ebX * ((ebR'*ebR) \ ebX') 
        # --- Not sure which one of these options is the best
        # bR  = qr(bX).R
        # ebR = qr(ebX).R
        # B▫[i] = bX * (pinv(bR'*bR) * bX') 
        # EB▫[i] = ebX * (pinv(ebR'*ebR) * ebX') 
        
        B▫[i] = bX / bX
        EB▫[i] = ebX / ebX

        XE▫[i]  = eX 
        XB▫[i]  = bX 
        XEB▫[i] = ebX
    end

    return CircOp(EB▫), CircOp(B▫), CircOp(XEB▫), CircOp(XB▫), CircOp(XE▫) 
end


CMBrings.map_plot_QU(
    low_pass_qu_rings; title1=L"Healpix EB-low-pass $Q$", title2=L"Healpix EB-low-pass $U$",  vmin = -4, vmax = 4, 
    θ, φ, imag_fun=x->CMBrings.imag_blur(x;blur=0),
);
CMBrings.map_plot_QU(
    # LPEB▪*qu_rings; title1=L"EAZ EB-low-pass $Q$", title2=L"EAZ EB-low-pass $U$",
    map((X,v)-> (X isa Diagonal) ? 0*v : X*(X\v), LPXEB▪, qu_rings);  title1=L"EAZ EB-low-pass $Q$", title2=L"EAZ EB-low-pass $U$",  vmin = -4, vmax = 4, 
    # map((X,v)-> (X isa Diagonal) ? 0*v : X*(X\v), LPXB▪, qu_rings);  title1=L"EAZ B-low-pass $Q$", title2=L"EAZ B-low-pass $U$",
    θ, φ, imag_fun=x->CMBrings.imag_blur(x;blur=0),
);


CMBrings.fourier_power(
    low_pass_qu_rings; title1="Healpix EB-low-pass power",
    θ, φ, ℓs = [low_pass_cut, lmax_cut], 
    imag_fun=CMBrings.imag_logabs2clip,
);
CMBrings.fourier_power(
    map((X,v)-> (X isa Diagonal) ? 0*v : X*(X\v), LPXEB▪, qu_rings);  title1="EAZ EB-low-pass power",
    # map((X,v)-> (X isa Diagonal) ? 0*v : X*(X\v), LPXB▪, qu_rings);  title1="EAZ B-low-pass power",
    θ, φ, ℓs = [low_pass_cut, lmax_cut], 
    imag_fun=CMBrings.imag_logabs2clip,
);





# pseudo-scalar B & E mode reconstruction on EAZ
# ===================================

# Healpix pseudo-scalar B & E
bθφ           = tmℍ0 \ Xfourier(tmℍ0, eblm[:,2])
eθφ           = tmℍ0 \ Xfourier(tmℍ0, eblm[:,1])
low_pass_eθφ  = tmℍ0 \ Xfourier(tmℍ0, low_pass_eblm[:,1])
low_pass_bθφ  = tmℍ0 \ Xfourier(tmℍ0, low_pass_eblm[:,2])

# Projected to EAZ
e_rings  = CMBrings.hpix2equirect_patch(
    eθφ;
    ring_idx_rng, φ, φ_full, 
    lb=lb1, rb=rb1, Δl=Δl1, Δr=Δr1,
) |> x->Xmap(tm0, x.*CMBrings.cosφ°Mask.(rad2deg.(φ'); lb=lb2, rb=rb2, Δl=Δl2, Δr=Δr2))

b_rings  = CMBrings.hpix2equirect_patch(
    bθφ;
    ring_idx_rng, φ, φ_full, 
    lb=lb1, rb=rb1, Δl=Δl1, Δr=Δr1,
) |> x->Xmap(tm0, x.*CMBrings.cosφ°Mask.(rad2deg.(φ'); lb=lb2, rb=rb2, Δl=Δl2, Δr=Δr2))

low_pass_e_rings  = CMBrings.hpix2equirect_patch(
    low_pass_eθφ;
    ring_idx_rng, φ, φ_full, 
    lb=lb1, rb=rb1, Δl=Δl1, Δr=Δr1,
) |> x->Xmap(tm0, x.*CMBrings.cosφ°Mask.(rad2deg.(φ'); lb=lb2, rb=rb2, Δl=Δl2, Δr=Δr2))

low_pass_b_rings  = CMBrings.hpix2equirect_patch(
    low_pass_bθφ;
    ring_idx_rng, φ, φ_full, 
    lb=lb1, rb=rb1, Δl=Δl1, Δr=Δr1,
) |> x->Xmap(tm0, x.*CMBrings.cosφ°Mask.(rad2deg.(φ'); lb=lb2, rb=rb2, Δl=Δl2, Δr=Δr2))



hp_bbℓ = deepcopy(bbℓ)
hp_bbℓ[ℓ .<= maximum(ms[ms_idx])] .= 0

hp_eeℓ = deepcopy(eeℓ)
hp_eeℓ[ℓ .<= maximum(ms[ms_idx])] .= 0

# N_E_hpB▪  = CMBrings.az_cov_blks(ℓ, eeℓ, hp_bbℓ; θ, φ, ℓrange=ms_idx, ngrid=100_000);
# N_hpE_B▪  = CMBrings.az_cov_blks(ℓ, hp_eeℓ, bbℓ; θ, φ, ℓrange=ms_idx, ngrid=100_000);

block_sizesθ = VF.block_split(nθ, bsd_nθ)
N_E_hpB▪  = CMBrings.spin2_az_cov_vecchia_blks(ℓ, eeℓ, hp_bbℓ, block_sizesθ; θ, φ, ℓrange=ms_idx);
N_hpE_B▪  = CMBrings.spin2_az_cov_vecchia_blks(ℓ, hp_eeℓ, bbℓ, block_sizesθ; θ, φ, ℓrange=ms_idx);



# construct an estimate of iblm's and treat it as pseudo-scalar modes 
est_low_pass_b_rings = let fk▪ = CC.ℂfθk2▪(qu_rings[!])
    T       = eltype(fk▪[1])
    rtn_fk▪ = Vector{T}[ zeros(T,nθ) for i in ms]
    for (i4idx, i4ms) in enumerate(ms_idx)
        X2 = LPXB▪[i4ms]
        X0 = LPXI▪[i4ms] 
        
        N   = N_E_hpB▪[i4idx] # the index doesn't map to ms but to ms_idx
        # if using Vecchia !!!!!!!
        # N⁻¹ = VF.posdef_inv(N) # if using Vecchia ...
        N⁻¹ = VF.inv(N) # if using Vecchia ...
        # !!!!!!!


        ℓs       = ls_given_m(ms[i4ms])
        inv_bbℓs = pinv.(bbℓ[ℓs .+ 1])
        inv_bbℓs[ℓs .<= 2] .= Inf
        Σ⁻¹      = Diagonal(inv_bbℓs) 

        # iblm_est = (X2'* (N \ X2) + Σ⁻¹) \ (X2' * (N \ fk▪[i4ms])) # default
        # if using Vecchia !!!!!!!
        iblm_est = (X2'* mapslices(x->N⁻¹*x, X2, dims=1) + Σ⁻¹) \ (X2' * (N⁻¹ * fk▪[i4ms])) 
        # !!!!!!!
        iblm_est[ℓs .<= 2] .= 0 # just to make sure these are set to zero

        rtn_fk▪[i4ms] =  - im * X0 * iblm_est
    end
    Xfourier(tm0, CC.▪2ℝfθk(rtn_fk▪))
end 


# construct an estimate of elm's and treat it as pseudo-scalar modes 
est_low_pass_e_rings = let fk▪ = CC.ℂfθk2▪(qu_rings[!])
    T       = eltype(fk▪[1])
    rtn_fk▪ = Vector{T}[ zeros(T,nθ) for i in ms]
    for (i4idx, i4ms) in enumerate(ms_idx)
        X2 = LPXE▪[i4ms]
        X0 = LPXI▪[i4ms] 
        
        N  = N_hpE_B▪[i4idx] # the index doesn't map to ms but to ms_idx
        # if using Vecchia !!!!!!!
        # N⁻¹ = VF.posdef_inv(N) # if using Vecchia ...
        N⁻¹ = inv(N) # if using Vecchia ...
        # Note: you could also try to instanciate the inverse ...
        # !!!!!!!

        ℓs       = ls_given_m(ms[i4ms])
        inv_eeℓs = pinv.(eeℓ[ℓs .+ 1])
        inv_eeℓs[ℓs .<= 2] .= Inf
        Σ⁻¹      = Diagonal(inv_eeℓs) 
        
        # elm_est = (X2'* (N \ X2) + Σ⁻¹) \ (X2' * (N \ fk▪[i4ms]))
        # if using Vecchia !!!!!!!
        elm_est = (X2'* mapslices(x->N⁻¹*x, X2, dims=1) + Σ⁻¹) \ (X2' * (N⁻¹ * fk▪[i4ms])) 
        # !!!!!!!
        elm_est[ℓs .<= 2] .= 0 # just to make sure these are set to zero

        rtn_fk▪[i4ms] =  X0 * elm_est
    end
    Xfourier(tm0, CC.▪2ℝfθk(rtn_fk▪))
end 


## different sign for e and b....this is noted in healpix doc 
CMBrings.map_plot(
    # - est_low_pass_e_rings;  title1=L"EAZ generated pseudo-scalar low pass $E$", vmin = -4, vmax = 4,
    - est_low_pass_b_rings;  title1=L"EAZ generated pseudo-scalar low pass $B$", vmin = -0.15, vmax = 0.15,
    θ, φ, imag_fun=x->CMBrings.imag_blur(x;blur=0),
);
CMBrings.map_plot(
    # e_rings;  title1=L"Healpix generated pseudo-scalar $E$", 
    # b_rings;  title1=L"Healpix generated pseudo-scalar $B$", 
    # low_pass_e_rings;  title1=L"Healpix generated pseudo-scalar low pass $E$", vmin = -4, vmax = 4,
    low_pass_b_rings;  title1=L"Healpix generated pseudo-scalar low pass $B$", vmin = -0.15, vmax = 0.15,
    θ, φ, imag_fun=x->CMBrings.imag_blur(x;blur=0),
);




CMBrings.fourier_power(
    est_low_pass_e_rings;  title1=L"EAZ generated pseudo-scalar low pass $E$", vmin=-80, vmax=10,
    # est_low_pass_b_rings;  title1=L"EAZ generated pseudo-scalar low pass $B$", 
    θ, φ, ℓs = [low_pass_cut, lmax_cut], 
    imag_fun=CMBrings.imag_logabs2clip,
);
CMBrings.fourier_power(
    # e_rings;  title1=L"Healpix generated pseudo-scalar $E$", 
    # b_rings;  title1=L"Healpix generated pseudo-scalar $B$", 
    low_pass_e_rings;  title1=L"Healpix generated pseudo-scalar low pass $E$", vmin=-80, vmax=10,
    # low_pass_b_rings;  title1=L"Healpix generated pseudo-scalar low pass $B$", 
    θ, φ, ℓs = [low_pass_cut, lmax_cut], 
    imag_fun=CMBrings.imag_logabs2clip,
);




# subset the full maps in Az so they are more clear

sub_factor = 4
tmUS0_sub = 𝕀(nθ) ⊗ 𝕌(eltype_in(tm0), nφ÷sub_factor, 2π/freq_mult)
est_low_pass_b_rings_sub = Xmap(tmUS0_sub, est_low_pass_b_rings[:][:,1:sub_factor:end])
est_low_pass_e_rings_sub = Xmap(tmUS0_sub, est_low_pass_e_rings[:][:,1:sub_factor:end])
low_pass_b_rings_sub = Xmap(tmUS0_sub, low_pass_b_rings[:][:,1:sub_factor:end])
low_pass_e_rings_sub = Xmap(tmUS0_sub, low_pass_e_rings[:][:,1:sub_factor:end])
b_rings_sub = Xmap(tmUS0_sub, b_rings[:][:,1:sub_factor:end])
e_rings_sub = Xmap(tmUS0_sub, e_rings[:][:,1:sub_factor:end])

#

CMBrings.map_plot(
    b_rings_sub;  title1=L"Healpix generated pseudo-scalar $B$", 
    θ, φ=φ[1:sub_factor:end], imag_fun=x->CMBrings.imag_blur(x;blur=0),
    vmin = -0.15, vmax = 0.15,
);
CMBrings.map_plot(
    low_pass_b_rings_sub;  title1=L"Healpix generated pseudo-scalar low pass $B$", 
    θ, φ=φ[1:sub_factor:end], imag_fun=x->CMBrings.imag_blur(x;blur=0),
    vmin = -0.15, vmax = 0.15,
);
CMBrings.map_plot(
    - est_low_pass_b_rings_sub;  title1=L"EAZ generated pseudo-scalar low pass $B$", 
    θ, φ=φ[1:sub_factor:end], imag_fun=x->CMBrings.imag_blur(x;blur=0),
    vmin = -0.15, vmax = 0.15,
);
CMBrings.map_plot(
    - est_low_pass_b_rings_sub - low_pass_b_rings_sub;  title1=L"Error (EAZ-Healpix) generated pseudo-scalar low pass $B$", 
    θ, φ=φ[1:sub_factor:end], imag_fun=x->CMBrings.imag_blur(x;blur=0),
    vmin = -0.15, vmax = 0.15,
);

#

CMBrings.map_plot(
    e_rings_sub;  title1=L"Healpix generated pseudo-scalar $E$", 
    θ, φ=φ[1:sub_factor:end], imag_fun=x->CMBrings.imag_blur(x;blur=0),
    # vmin = -6, vmax = 6,
);
CMBrings.map_plot(
    low_pass_e_rings_sub;  title1=L"Healpix generated pseudo-scalar low pass $E$", 
    θ, φ=φ[1:sub_factor:end], imag_fun=x->CMBrings.imag_blur(x;blur=0),
    vmin = -6, vmax = 6,
);
CMBrings.map_plot(
    - est_low_pass_e_rings_sub;  title1=L"EAZ generated pseudo-scalar low pass $E$", 
    θ, φ=φ[1:sub_factor:end], imag_fun=x->CMBrings.imag_blur(x;blur=0),
    vmin = -6, vmax = 6,
);
CMBrings.map_plot(
    - est_low_pass_e_rings_sub - low_pass_e_rings_sub;  title1=L"Error (EAZ-Healpix) generated pseudo-scalar low pass $E$", 
    θ, φ=φ[1:sub_factor:end], imag_fun=x->CMBrings.imag_blur(x;blur=0),
    vmin = -6, vmax = 6,
);
