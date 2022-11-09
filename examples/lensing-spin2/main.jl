

# TODO
# ==============================
#=

• Convert to using EAZTransforms for all fields
• Upgrade the mask to include point sources
• Upgrade the noise to include 1/f noise (for spin0) and filtering effects

=#


# Modules
# ==============================

using LinearAlgebra
using FFTW
FFTW.set_num_threads(BLAS.get_num_threads())

using  CMBrings
using  XFields
using  EAZTransforms
using  EAZTransforms: pix, freq, nyq, Ωpix # these work for FFTransforms too
import EAZTransforms as EZ 
import FFTransforms as FT
import HealpixTransforms as HT

import CirculantCov as CC
using FieldLensing 
using Spectra: camb_cls
using VecchiaFactorization
import VecchiaFactorization as VF
import LowRankCholesky as LRC
using LBblocks: @sblock

using SparseArrays
using PyPlot
using BenchmarkTools
using ProgressMeter
using BlockArrays
using Dierckx: Spline1D 

include(joinpath(CMBrings.module_dir,"examples/lensing-spin2/LocalMethods.jl"))
import .LocalMethods as LM

## import Random
## Random.seed!(1234)

save_jld2    = false # !!!!!!
save_figures = false 
hide_plots   = true
polar_plots  = false
# if isdefined(Main, :IJulia) && Main.IJulia.inited
#     hide_plots = false
# else 
#     hide_plots = true
# end



# EAZ pixel grid
# ========================================

eaz0, eaz2, grid_type = @sblock let 

    ## set φ grid parameters: φspan and nφ
    φspan = deg2rad.((-60,60)) # deg2rad.((-45, 45))
    #nφ    = 2048 # 3072  # 1575 # 18000, 18000÷4, 768, 1536, 1575, 2048, 1024, 972,  1280
    nφ    = 1575

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
    # nθ     = 500 #  600 # 800
    nθ    = 400
    θspan  = π/2 .+ deg2rad.((51,69)) # π/2 .+ deg2rad.((41.78,70.43))
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
end


# Plot Grid statistics

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
end

@show (eaz0.nθ, eaz0.nφ)
@show extrema(rad2deg.(.√(EZ.Ωpix(eaz0)).*60))
@show extrema(rad2deg.(EZ.Δθ(eaz0).*60))
@show extrema(rad2deg.(sin.(eaz0.θ) .* EZ.Δφ(eaz0) .* 60))
@show extrema(EZ.pix_diag_arcmin(eaz0));


# Coordinate pivot, blocks and queries for Vecchia
# ==============================
## using Primes; factor(length(eaz0.θ)) # ; @assert nθ÷bks == nθ/bks

bsd_nθ       = 100 # 50 # 100 #  150 # 161
block_sizesθ = VF.block_split(eaz0.nθ, bsd_nθ) # |> sort
permθ        = 1:eaz0.nθ

# Spectral densities
# ==============================

φ_approx_nyq = eaz0.φfreq_mult * eaz0.nφ / minimum(sin.(eaz0.θ)) / 2
θ_approx_nyq = π / minimum(EZ.Δθ(eaz0)) 
@show approx_lmax = ceil(Int, sqrt(φ_approx_nyq^2 + θ_approx_nyq^2))

approx_lmax += ceil(Int, approx_lmax * 0.05) # for good measure:)
## override ...
## approx_lmax = 25_000

ℓ, ϕϕℓ, eeℓ, bbℓ, ẽẽℓ, b̃b̃ℓ = @sblock let lmax=approx_lmax, r=0.01, T=Float64
    
    l = 0:lmax
    cld = camb_cls(;lmax=lmax, r,
        lSampleBoost   = 4.0,
        lAccuracyBoost = 4.0,
        KmaxBoost = 6.0,
        )
    
    eesl = cld[:unlen_scalar] |> x->(x[:Cee] ./ x[:factor_on_cl_cmb])
    eetl = cld[:unlen_tensor] |> x->(x[:Cee] ./ x[:factor_on_cl_cmb])
    eel  = eesl .+ eetl
    eel[1] = eel[2] = 0

    bbsl = cld[:unlen_scalar] |> x->(x[:Cbb] ./ x[:factor_on_cl_cmb])
    bbtl = cld[:unlen_tensor] |> x->(x[:Cbb] ./ x[:factor_on_cl_cmb])
    ## note: bbsl == 0 
    bbl    = bbsl .+ bbtl
    bbl[1] = bbl[2] = 0

    ẽesl   = cld[:len_scalar] |> x->(x[:Cee] ./ x[:factor_on_cl_cmb])
    ẽel    = ẽesl .+ eetl # we only have lensed spectra for scalar
    ẽel[1] = ẽel[2] = 0

    b̃bsl   = cld[:len_scalar] |> x->(x[:Cbb] ./ x[:factor_on_cl_cmb])
    b̃bl    = b̃bsl .+ bbtl # we only have lensed spectra for scalar
    b̃bl[1] = b̃bl[2] = 0

    ϕϕl    = cld[:phi] |> x->(x[:Cϕϕ] ./ x[:factor_on_cl_phi])
    ϕϕl[1] =  ϕϕl[2] = 0 

    return l, T.(ϕϕl), T.(eel), T.(bbl), T.(ẽel), T.(b̃bl) 
end;


#=
loglog( ℓ.^2 .* eeℓ)
loglog( ℓ.^2 .* bbℓ)
loglog( ℓ.^2 .* ẽẽℓ)
loglog( ℓ.^2 .* b̃b̃ℓ)
=#

# this is a hack ...
#=
bbℓ[bbℓ .<= 0] .= 1e-18 # minimum(bbℓ[3:end][bbℓ[3:end] .> 0])
eeℓ[eeℓ .<= 0] .= 1e-18 # minimum(eeℓ[3:end][eeℓ[3:end] .> 0])
b̃b̃ℓ[b̃b̃ℓ .<= 0] .= 1e-18 # minimum(bbℓ[3:end][bbℓ[3:end] .> 0])
ẽẽℓ[ẽẽℓ .<= 0] .= 1e-18 # minimum(eeℓ[3:end][eeℓ[3:end] .> 0])
b̃b̃ℓ[1] = b̃b̃ℓ[2] = 0
ẽẽℓ[1] = ẽẽℓ[2] = 0
bbℓ[1] = bbℓ[2] = 0
eeℓ[1] = eeℓ[2] = 0
=#




# Check the block cov matrices for problems with pos def 
# =========================================
#=
EB▫_test = CMBrings.az_cov_blks(
    ℓ, eeℓ, bbℓ; 
    θ=eaz0.θ[end-2*bsd_nθ + 1:end], # eaz0.θ[1:2*bsd_nθ], 
    φ=EZ.φ(eaz0), 
    ℓrange=[eaz0.nφ÷2-5,eaz0.nφ÷2+1], 
    ngrid=100_000
);

Σ = EB▫_test[1]
M = LRC.low_rank_cov(
    Hermitian(Σ);
    tol=0
)

VF.vecchia(EB▫_test[1], [200,200]; atol=1e-15)

EB▫_test[1] |> Hermitian |> eigen |> x->x.values
EB▫_test[end] |> Hermitian |> eigen |> x->x.values

EB▫_test[end] |> Hermitian |> eigen |> x->x.vectors[:,end] |> plot
EB▫_test[end] |> Hermitian |> eigen |> x->x.vectors[:,end-1] |> plot
EB▫_test[end] |> Hermitian |> eigen |> x->x.vectors[:,end÷2] |> plot
EB▫_test[end] |> Hermitian |> eigen |> x->x.vectors[:,2] |> plot
=#


#=
# TODO: try out ClassicalOrthogonalPolynomials
nℓ = @. (2ℓ+1)/(4π)
j0⁺0tℓ = @. ϕϕℓ * nℓ
f0⁺0t = ((a,b,jℓ)=(0,0,j0⁺0tℓ); CC.Fun(CC.Jacobi(b,a),jℓ))
f0⁺0t_F64 = ((a,b,jℓ)=(0,0,Float64.(j0⁺0tℓ)); CC.Fun(CC.Jacobi(b,a),jℓ))
covtt = x-> f0⁺0t(cos(x))
covtt_F64 = x-> f0⁺0t_F64(cos(x))

@benchmark f0⁺0t($(BigFloat(0.1))) # 43 ms
@benchmark f0⁺0t_F64(0.1)          # 50 μs

@benchmark cos($(BigFloat(0.1))) # 1.050 μs
@benchmark cos(0.1)              # 0.875 ns
=#


# Mask 
# =========================================

# kron product mask
prθ, prφ  =  @sblock let eaz0

    rT=real(eltype_in(eaz0))

    ## θ part of the mask
    # ▮lθ, ▯lθ = 20, 60 
    ▮lθ, ▯lθ = 15, 50 
    ▮rθ, ▯rθ = eaz0.nθ-▮lθ+1, eaz0.nθ-▯lθ+1 
    prθ    = CMBrings.pixweight.(rT.(1:eaz0.nθ); ▮l=▮lθ, ▯l=▯lθ, ▯r=▯rθ, ▮r=▮rθ)
    
    ## φ part of the mask
    # ▮lφ, ▯lφ = 30, 60 
    # ▮rφ, ▯rφ = eaz0.nφ-▮lφ+1, eaz0.nφ-▯lφ+1 
    # prφ    = CMBrings.pixweight.(rT.(1:eaz0.nφ); ▮l=▮lφ, ▯l=▯lφ, ▯r=▯rφ, ▮r=▮rφ)
    # ----- option ----- ↓↓ No azmuthal mask ↓↓
    prφ = ones(rT,eaz0.nφ)

    prθ, prφ
end;


# Lensing mask (to keep the lense from transporting off the polar cut)
Mϕ = @sblock let eaz0, prθφ = prθ.*prφ'
    
    rT=real(eltype_in(eaz0))
    nθ, nφ = eaz0.nθ, eaz0.nφ

    ## Set mϕx
    ## ... option: ...
    # ▮lθ, ▯lθ = 1, 10 
    # ▮rθ, ▯rθ = nθ-1+1, nθ-10+1 
    # prθ  = CMBrings.pixweight.(rT.(1:nθ); ▮l=▮lθ,    ▯l=▯lθ, ▯r=▯rθ, ▮r=▮rθ)
    # mϕx = prθ * ones(rT,nφ)'
    ## ... option: ...
    sqz = 4
    sft = 0.4
    mϕx = prθφ .|> x-> clamp((atan(sqz*(x-sft)) + π/2)/π, .05, .95)

    ## Scale mϕx so it hits zero and 1
    mϕx .-= minimum(mϕx)
    mϕx ./= maximum(mϕx)
    Mϕ    = DiagOp(Xmap(eaz0, mϕx))
    Mϕ
end;

# Mask plot
# ========================


#= Old ... slated for removal
@sblock let prθ, prφ, Mϕ, φ, θ, hide_plots, save_figures
    hide_plots && return
    prθφ = prθ .* prφ'
    dma = prθφ .> 0
    ma  = prθφ
    ## imgs = Dict(1=>dma, 2=>ma)
    ## txt  = Dict(1=>"pre-smoothed mask", 2=>"mask")
    imgs = Dict(1=>ma, 2=>Mϕ[:])
    txt  = Dict(1=>"data mask", 2=>"lensing mask")

    fig, ax = CMBrings.diskplot(
        imgs, CC.in_negπ_π.(φ)', π.-θ; 
        txt=txt, 
        figsize=(10,8), nrows=1, fontsize=14
    )
    save_figures && savefig("figure$(fig.number).png", dpi=250, bbox_inches="tight")
    return nothing
end
=# 


## Mϕ[:] .|> real |> matshow; colorbar()
## prθ .* prφ' .|> real |> matshow; colorbar()


@sblock let eaz0, Mϕ, prθφ = prθ.*prφ', hide_plots, save_figures
    hide_plots && return
    
    fig1, ax1 = CMBrings.map_plot(
        Mϕ.f,
        title1="Lensing displacement mask",
    );

    fig2, ax2 = CMBrings.map_plot(
        Xmap(eaz0, prθφ),
        title1="Data pixel mask",
    );

    save_figures && savefig("figure$(fig1.number).png", dpi=250, bbox_inches="tight")
    save_figures && savefig("figure$(fig2.number).png", dpi=250, bbox_inches="tight")
    
    return nothing
end


# Spin 2 signal
# =================================================
# TODO: make custom sqrt and LowRankCov/Chol with a clamp...


@time EB▪½ = CMBrings.spin2_az_cov½_vecchia_blks(
    ℓ, eeℓ, bbℓ, block_sizesθ, permθ; 
    θ=EZ.θ(eaz0), φ=EZ.φ(eaz0), 
    # atol      = 1e-10, # 1e-14, # for the low rank Chol
) |> CircOp;

#=
@time qu = EB▪½ * Xmap(eaz2,randn(eltype_in(eaz2), size_in(eaz2)));
CMBrings.map_plot(qu)
CMBrings.fourier_power(qu, ℓs = [1000, 4000], imag_fun=CMBrings.imag_logabs2clip);

@time EB▪½ \ Xmap(eaz2,randn(eltype_in(eaz2), size_in(eaz2)));
CMBrings.fourier_power(EB▪½ \ qu, ℓs = [1000, 4000], imag_fun=CMBrings.imag_logabs2clip);
CMBrings.map_plot(EB▪½ \ qu)

=#

## sum(Base.summarysize, EB▪½) / 1e9 # 7.41 GB, 3.55min construction, high res
## EB▪½[end-5][3].data[2]
## EB▪⁻½[end-5][2].data[2]

# EB▪⁻½ = map(VF.posdef_inv, EB▪½) |> CircOp;

# @time EB▪ = CMBrings.spin2_az_cov_vecchia_blks(
#     ℓ, eeℓ, bbℓ, block_sizesθ, permθ; θ=EZ.θ(eaz0), φ=EZ.φ(eaz0), 
#     atol = 0, 
#     ) |> CircOp;

#=
qu = EB▪ * Xmap(eaz2,randn(eltype_in(eaz2), size_in(eaz2)));
qu = EB▪½ * EB▪½' * Xmap(eaz2,randn(eltype_in(eaz2), size_in(eaz2)));
CMBrings.fourier_power(EB▪ \ qu, ℓs = [1000, 4000], imag_fun=CMBrings.imag_logabs2clip);

CMBrings.fourier_power(EB▪⁻½ * qu, ℓs = [1000, 4000], imag_fun=CMBrings.imag_logabs2clip);
CMBrings.map_plot(EB▪⁻½ * qu)
=#

# EB▪½[1][3].data[3].info

    # Γ, C   = CC.ΓCθ₁θ₂φ₁φ⃗_CMBpol(ℓ, eeℓ, bbℓ; ngrid=100_000)
    # Σ_pre▫, P = CMBrings.spin2_az_bidiagΣ▫_P(Γ, C,  block_sizesθ, permθ[:]; θ=EZ.θ(eaz0), φ=EZ.φ(eaz0))
    # blk_sizes′ = VF.blocksizes(Σ_pre▫[1],1) # for spin2 block sizes get doubled ...
    # Σ▫ = map(Σ_pre▫) do Σ
    #     R, preM, = VF.R_M_P(Σ, blk_sizes′; atol)
    #     # M½ = VF.Midiagonal(map(sqrt, preM.data))
    #     M½ = VF.Midiagonal(map(x->LRC.low_rank_cov(sqrt(VF.Sym_or_Hrm(Matrix(x)))), preM.data)) # testing
    #     P' * inv(R) * M½ * P 
    # end



#=
@time EB▪ = CMBrings.spin2_az_cov_vecchia_blks(
    ℓ, eeℓ, bbℓ, block_sizesθ, permθ; 
    θ=EZ.θ(eaz0), φ=EZ.φ(eaz0), 
    atol = 0, 
    atol = 1e-14, # for the low rank Chol
) |> CircOp;

Γ, C   = CC.ΓCθ₁θ₂φ₁φ⃗_CMBpol(ℓ, eeℓ, bbℓ; ngrid=100_000)
Σ_pre▫, P = spin2_az_bidiagΣ▫_P(Γ, C, blk_sizes, perm; θ, φ, ℓrange)
blk_sizes′ = VF.blocksizes(Σ_pre▫[1],1) # for spin2 block sizes get doubled ...
Σ▫ = map(Σ_pre▫) do Σ
    R, preM, = VF.R_M_P(Σ, blk_sizes′; atol)
    M½ = VF.Midiagonal(map(x->sqrt(x;tol), preM.data))
    P' * inv(R) * M½ * P 
end

=#


## @time EB = let 
##     EB▫  = CMBrings.az_cov_blks(ℓ, eeℓ, bbℓ ; θ,  φ)
##     map(EB▫) do M 
##         Array(sqrt(Hermitian(M)))
##     end |> CircOp
## end
## EB▪⁻½ = map(inv, EB▪½) |> CircOp;
## -------




# Spin 0 signal
# =================================================

@time Phi▪½ = CMBrings.spin0_az_cov½_vecchia_blks(
    ℓ, ϕϕℓ, block_sizesθ, permθ; θ=EZ.θ(eaz0), φ=EZ.φ(eaz0)
) |> CircOp;


#=
@time ϕ = Phi▪½ * Xmap(eaz0,randn(eltype_in(eaz0), size_in(eaz0)));
CMBrings.map_plot(ϕ)
CMBrings.fourier_power(ϕ, ℓs = [1000, 4000], imag_fun=CMBrings.imag_logabs2clip);

@time Phi▪½ \ Xmap(eaz0,randn(eltype_in(eaz0), size_in(eaz0)));
CMBrings.fourier_power(Phi▪½ \ϕ, ℓs = [1000, 4000], imag_fun=CMBrings.imag_logabs2clip);
CMBrings.map_plot(Phi▪½ \ϕ)
=#
## sum(Base.summarysize, Phi▪½) / 1e9 # 1.4 GB, 2.5min construction, high res

# Phi▪⁻½ = map(VF.posdef_inv, Phi▪½) |> CircOp;
##  Phi▪½[20][3].data[end].info

# Noise
# ============================

# μK_arcmin  = 5.0 # default 
μK_arcmin  = 1.0 # testing !!!

N▪ = @sblock let μK_arcmin, eaz0
    Ω, nφ = EZ.Ωpix(eaz0), eaz0.nφ
    σ²   = deg2rad(μK_arcmin/60)^2 # Cⁿℓ == μK_arcmin |> arcmin2radians |> abs2
    σ²_Ω = σ² ./ Ω
    Nmat = Diagonal(vcat(σ²_Ω,σ²_Ω))
    N▫   = [Nmat for ℓ = 1:nφ÷2+1]
    CircOp(N▫)
end; 

N▪⁻¹ = map(Nℓ->Diagonal(1 ./ diag(Nℓ)), N▪.Σ) |> CircOp;

# Now add pure BB noise * large factor bb_noise_factor

## N▪ = let bb_noise_factor = 100 
##     zeroEB▪  = CMBrings.az_cov_blks(ℓ, 0 .* eeℓ, bbℓ ; θ=EZ.θ(eaz0), φ=EZ.φ(eaz0), ngrid=100_000) |> CircOp
##     map(N▪, zeroEB▪) do A, B
##         A + bb_noise_factor * B
##     end |> CircOp
## end 
## 
## ## N▪⁻¹ = map(Nℓ->Diagonal(1 ./ diag(Nℓ)), N▪.Σ) |> CircOp;
## N▪⁻¹ = map(inv, N▪) |> CircOp;


# Mask
# ============================

M = DiagOp(Xmap(eaz2, prθ .* prφ' ));

# Beam
# ============================


fwhmθ_rad = EZ.pix_diag_rad(eaz0) # pix_diag_rad # * 0.95
## -- option --
# fwhm′ = 2.0 
# fwhmθ_rad = fill(CMBrings.arcmin2rad(fwhm′), eaz0.nθ)

normalizeθ = :row_ave
B▪ = CMBrings.beam▫(eaz2; fwhmθ_rad, block_sizesθ, normalizeθ) |> CircOp;

# Lensing operators
# ============================

∇!,  ∇!_ϕ = CMBrings.generate_∇!∇!ϕ(EZ.θ(eaz0), EZ.φ(eaz0); uniformΔθ = (grid_type == :equiθ) ? true : false); 

Ł, ϕ2v!, ϕ2vᴴ!, ∇! = CMBrings.generate_lense(;
    θ=EZ.θ(eaz0), mv1x=Mϕ[:], mv2x=Mϕ[:], ∇!,  ∇!_ϕ, 
    nsteps_lensing=14
);

# simulation
# ==============================

ϕ = Phi▪½ * Xmap(eaz0,randn(eltype_in(eaz0), size_in(eaz0)));
## ------ alt: full non-Vecchia approximate simulation
# @time ϕ = @sblock let ℓ, ϕϕℓ, blksiz=eaz0.nφ÷5, eaz0
#     θ, φ   = EZ.θ(eaz0), EZ.φ(eaz0)
#     nθ, nφ = length(θ), length(φ)
#     w      = Xmap(eaz0,randn(eltype_in(eaz0), size_in(eaz0))) 
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

qu = EB▪½ * Xmap(eaz2,randn(eltype_in(eaz2), size_in(eaz2)));
## ------ alt: full non-Vecchia approximate simulation
# qu = @sblock let ℓ, eeℓ, bbℓ, blksiz=eaz2.nφ÷5, eaz2
#     θ, φ   = EZ.θ(eaz0), EZ.φ(eaz0)
#     nθ, nφ = length(θ), length(φ)
#     w      = Xmap(eaz2,randn(eltype_in(eaz2), size_in(eaz2)))
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

#-

no = map(N▪, Xmap(eaz2,randn(eltype_in(eaz2), size_in(eaz2)))) do Σ,v
    sqrt(Σ)*v
end 

#-

d = M * (B▪ * Ł(ϕ) * qu + no) |> Xfourier;

#-

#=

CMBrings.map_plot(
    # d,
    # qu,
    # ϕ,
    Ł(ϕ)*qu - qu,
    # Ł(ϕ)*qu,
    # no, 
    # B▪ * B▪ * B▪ * B▪ * B▪ * no,
    # imag_fun=x->CMBrings.imag_blur(x;blur=0),
);



CMBrings.fourier_power(
    # d,
    # qu,
    # ϕ,
    # Ł(ϕ)*qu - qu,
    # Ł(ϕ)*qu,
    # no, 
    B▪ * B▪ * B▪ * B▪ * B▪ * no,
    ℓs = [400, 1000, 3000, 4000], 
    imag_fun=CMBrings.imag_logabs2clip,
);


=#

# Mixflow operator
# ============================

nnℓ = deg2rad(μK_arcmin/60)^2 # Cⁿℓ == μK_arcmin |> arcmin2radians |> abs2

Ð▪⁻¹ = CMBrings.spin2_az_cov½_vecchia_blks(
   ℓ, (@. eeℓ/(ẽẽℓ+2nnℓ)), (@. bbℓ/(b̃b̃ℓ+2nnℓ)),  
   block_sizesθ,  permθ; θ=EZ.θ(eaz0), φ=EZ.φ(eaz0)
) |> CircOp;


# Initalize opps for ϕ gradient
# ==============================================


import CMBflat

N0ℓ, NΦNℓ = @sblock let pix_side_rad = mean(.√EZ.Ωpix(eaz0)), n_iter=5, ℓ, eeℓ, bbℓ, ϕϕℓ, fwhmθ_rad, nnℓ=fill(nnℓ,length(ℓ)) 
    
    ## not sure which version of σ² is the best here???
    ## σ² = mean(fwhmθ_rad)^2 / 8 / log(2)
    ## σ² = minimum(fwhmθ_rad)^2 / 8 / log(2)    
    ## σ² = maximum(fwhmθ_rad)^2 / 8 / log(2) # default
    σ² = 1.25 * maximum(fwhmθ_rad)^2 / 8 / log(2) # testing ...
    beamℓ = @. exp( - σ²*ℓ*(ℓ+1) / 2)

    T_fld   = Float64
    nθ, nφ  = 512, 512   
    periodθ = T_fld(nθ * pix_side_rad)
    periodφ = T_fld(nφ * pix_side_rad)
    tm      = FT.𝕎(T_fld, (nθ, nφ), (periodθ, periodφ))
    tmΦ     = FT.ordinary_scale(tm) * tm
    tmEB    = CMBflat.QU2EB(T_fld, (nθ, nφ), (periodθ, periodφ))
    Idx     = round.(Int,FT.wavenum(tmΦ)) .+ 1
    ecl     = map(i -> getindex(eeℓ, i), Idx)
    bcl     = map(i -> getindex(bbℓ, i), Idx)
    ϕcl     = map(i -> getindex(ϕϕℓ, i), Idx)
    ncl     = map(i -> getindex(nnℓ, i), Idx)
    bmcl    = map(i -> getindex(beamℓ, i), Idx)
    EBcov   = DiagOp(Xfourier(tmEB, cat(ecl,bcl;dims=3))) 
    Ncov    = DiagOp(Xfourier(tmEB, cat(ncl,ncl;dims=3))) 
    Bm      = DiagOp(Xfourier(tmEB, cat(bmcl,bmcl;dims=3)))
    Φcov    = DiagOp(Xfourier(tmΦ, ϕcl))
    ## lcut_prpn = [0.75, 0.95]    
    ## kf  =  [abs.(FT.fullfreq(FT.𝕎(tmEB))[i]) .<= lcut_prpn[i]*FT.nyq(FT.𝕎(tmEB))[i] for i = 1:2]
    ## Bm *= DiagOp(Xfourier(tmEB, kf[1] ))
    ## Bm *= DiagOp(Xfourier(tmEB, kf[2] ))
    ## ----- 
    Ncov_local = Ncov / Bm^2
    Ncov_local.f.fd[real.(Bm.f.fd) .<= 0] .= Inf
    Ncov_local.f.fd[1,1,1] = Inf
    Ncov_local.f.fd[1,1,2] = Inf
    ## ----- EBcov_local: unlensed signal
    ## Not sure if we want zero B power here??
    ## EBcov_local = Xfourier(tmEB, EBcov[:El], 0) |> DiagOp
    ## -- alternative 
    EBcov_local = deepcopy(EBcov)
    ## ----- Nϕ with tot power == EBcov_local + B̃fromE + Ncov_local
    ## In the iterations B̃fromE will get reduced. 
    B̃fromE  = CMBflat.lnB_matpwr(tmΦ, EBcov_local[:El], Φcov[!]) |> 
                    x-> Xfourier(tmEB, 0, x) |> 
                    DiagOp    
    Nϕ  = CMBflat.N0ℓ_EB(
        tmΦ, 
        EBcov_local, 
        inv(EBcov_local + B̃fromE + Ncov_local), # inv total power: signal + effective noise
    )
    Nϕ.f.fd[real.(Nϕ.f.fd) .<= 0] .= Inf 
    Nϕ.f.fd[1,1] = Inf 
    for cntr = 1:n_iter
        wf_B̃fromE  = CMBflat.lnB_matpwr(
            tmΦ, 
            (EBcov_local^2 * inv(EBcov_local + Ncov_local))[:El], 
            (Φcov^2 * inv(Φcov + Nϕ))[!],
        ) |> x-> Xfourier(tmEB, 0, x) |> DiagOp    
        Nϕ  = CMBflat.N0ℓ_EB(
            tmΦ, 
            EBcov_local, 
            inv(EBcov_local + B̃fromE - wf_B̃fromE + Ncov_local), # inv total power: signal + effective noise
        )
        Nϕ.f.fd[real.(Nϕ.f.fd) .<= 0] .= Inf 
        Nϕ.f.fd[1,1] = Inf 

    end
    k      = FT.wavenum(tmΦ)[:,1]
    k4n0ck = k.^4 .* real.(Nϕ[!][:,1])
    spline_k4n0ck = Spline1D(
        vcat(2,k[3:end]), vcat(k4n0ck[3], k4n0ck[3:end])
        ; k=1, bc="zero",
    )
    N0ℓ = spline_k4n0ck.(ℓ) ./ ℓ.^4
    N0ℓ[real.(N0ℓ) .<= 0] .= Inf 
    N0ℓ[isnan.(N0ℓ)]      .= Inf 
    NΦNℓ = @. inv(inv(N0ℓ) + inv(ϕϕℓ))
    N0ℓ, NΦNℓ
end;


# NΦN▪ = CMBrings.spin0_az_cov½_vecchia_blks(
#     ℓ, NΦNℓ,  
#     block_sizesθ,  permθ; θ=EZ.θ(eaz0), φ=EZ.φ(eaz0)
# ) |> x->map(m->m*m',x) |> CircOp;


NΦN▪ = CMBrings.spin0_az_cov_vecchia_blks(
    ℓ, NΦNℓ,  
    block_sizesθ,  permθ; θ=EZ.θ(eaz0), φ=EZ.φ(eaz0)
) |> CircOp;



# Initalize opps for WF
# ==============================================

## we apparently need this to commute with M ....
## diag(W▪[1])[1:end÷2] == diag(W▪[1])[end÷2+1:end]

mult_nnℓ = 0.95

wwℓ  = mult_nnℓ .*  nnℓ
nn⁺ℓ = nnℓ .- wwℓ

W▪    = map(N▪) do N 
    Diagonal(real(diag(N)) * mult_nnℓ) 
end |> CircOp;

N▪⁺ᵍ  = map(W▪, N▪) do W, N 
    pinv(N - W)
end |> CircOp;

MWMᵀᵍ = @sblock let W▪, M, eaz2
    ## MWMᵀ_pxl = abs2.(prθφM) .* prθW
    prθW = diag(W▪[1])[1:end÷2]
    ## prθM = M[:][:,end÷2]
    ## MWMᵀ_pxl = prθW .* abs2.(prθM) .* ones(1,eaz2.nφ)
    MWMᵀ_pxl = prθW .* abs2.(M[:]) # Testing !!!!!!!!
    DiagOp(Xmap(eaz2, pinv.(MWMᵀ_pxl)))
end;


@time _A₁₁ᵍ▪, _A₂₂_A₂₁A₁₁ᵍA₁₂_ᵍ▪ = @sblock let B▪, ℓ, eeℓ, bbℓ, N▪⁺ᵍ, W▪, M, MWMᵀᵍ, normalizeθ, block_sizesθ, permθ, eaz0
    
    nθ = eaz0.nθ
    
    Mθ     = M[:][:,end÷2] |> x->vcat(x,x)
    ## Mθ     = mean(eachcol(M[:])) |> x->vcat(x,x)

    MWMᵀᵍθ = MWMᵀᵍ[:][:,end÷2] |> x->vcat(x,x)
    
    EB▪ = CMBrings.spin2_az_cov_vecchia_blks(
        ℓ, eeℓ, bbℓ, block_sizesθ, permθ; θ=EZ.θ(eaz0), φ=EZ.φ(eaz0), 
        atol = 0, 
    ) |> CircOp

    _A₁₁ᵍ▪ = map(W▪, N▪⁺ᵍ) do W, iN
        Diagonal(pinv.(Mθ .* MWMᵀᵍθ .* conj.(Mθ) .+ diag(iN)))
    end |> CircOp

    _A₂₂_A₂₁A₁₁ᵍA₁₂_ᵍ▪ = map(_A₁₁ᵍ▪, B▪, N▪⁺ᵍ, EB▪) do iA, Bl, iN, Σ
        # iA, Bl, iN, Σ = _A₁₁ᵍ▪[1], B▪[1], N▪⁺ᵍ[1], EB▪[1]

        PΣ, RΣ, MΣ = Σ[1], inv(Σ[2]), Σ[3]
        invΣ = VF.instantiate_inv(RΣ, MΣ, PΣ)

        if normalizeθ == :Ω
            PB, RB, MB, matΩ = Bl[1], inv(Bl[2]), Bl[3], Bl[6]
            invB = VF.instantiate_inv(RB, MB, PB)
            matB = inv(cholesky(VF.Sym_or_Hrm(invB)))
            matB′ = sqrt(iN - iN*iA*iN) * matB * matΩ
        elseif normalizeθ == :row_ave
            mat_row_ave, RB, MB = Bl[1], inv(Bl[2]), Bl[3]
            invB = VF.instantiate_inv(RB, MB)
            matB = inv(cholesky(VF.Sym_or_Hrm(invB)))
            matB′ = sqrt(iN - iN*iA*iN) * mat_row_ave * matB
        end

        invΣ += matB′'*matB′
        # X = invΣ + matB′'*(iN - iN*iA*iN)*matB′
        invX = inv(cholesky(VF.Sym_or_Hrm(invΣ))) # default

        return VF.vecchia(
                invX, 
                2 .* block_sizesθ,  
                ## VF.block_split(2nθ, 250),
                1:2nθ |> x->(reshape(x,nθ,2)')[:],
                atol = 1e-10, # !!!! testing 
                )
    end |> CircOp

    _A₁₁ᵍ▪, _A₂₂_A₂₁A₁₁ᵍA₁₂_ᵍ▪
end;

# Try some gradient moves
# ==============================================

# Initalize
f_cr = 0*d
g_cr = 0*d
ϕ_cr = 0*ϕ


let M=M, MWMᵀᵍ=MWMᵀᵍ, N▪⁺ᵍ=N▪⁺ᵍ, B▪=B▪, _A₁₁ᵍ▪=_A₁₁ᵍ▪, _A₂₂_A₂₁A₁₁ᵍA₁₂_ᵍ▪=_A₂₂_A₂₁A₁₁ᵍA₁₂_ᵍ▪, eaz2=eaz2, EB▪½=EB▪½

    global function A(g, f, L)
        Afg_g = (M'*MWMᵀᵍ*M*g + N▪⁺ᵍ*g) - (N▪⁺ᵍ*B▪*L*f)
        Afg_f = - (L'*B▪'*N▪⁺ᵍ*g) + (L'*B▪'*N▪⁺ᵍ*B▪*L*f + EB▪½'\(EB▪½\f))
        Afg_g, Afg_f
    end

    global function _Aᵍ(g, f, L)
        f1 = _A₂₂_A₂₁A₁₁ᵍA₁₂_ᵍ▪ * (L'*B▪'*N▪⁺ᵍ*_A₁₁ᵍ▪*g + f)
        _A₁₁ᵍ▪*(g + N▪⁺ᵍ*B▪*L*f1), f1
    end

    global function sim_bg_bf(L)
        γ₁  = sqrt(MWMᵀᵍ) * Xmap(eaz2,randn(eltype_in(eaz2), size_in(eaz2)))
        γ₂  = map((Σ,v)->sqrt(Σ)*v, N▪⁺ᵍ, Xmap(eaz2,randn(eltype_in(eaz2), size_in(eaz2))))
        γ₃  = EB▪½' \ Xmap(eaz2,randn(eltype_in(eaz2), size_in(eaz2)))
        b_g = M'* MWMᵀᵍ * d + Xfourier(M'*γ₁ + γ₂)
        b_f = Xfourier(γ₃ - L'*B▪'*γ₂)
        return  b_g, b_f
    end
      
end;




# WF for conditional expected value
## -----------------------
g_cr, f_cr, reshist = CMBrings.pcg_coupled(;
    nsteps=200, # 50 
    rel_tol=1e-15, 
    _Aᵍ = (g, f) -> _Aᵍ(g, f, DiagOp(Xmap(eaz2,1))), 
    A   = (g, f) ->   A(g, f, DiagOp(Xmap(eaz2,1))),
    b_g = M'* MWMᵀᵍ * d, 
    b_f = 0 * d, 
    x_g = 0 * d, 
    x_f = 0 * d, 
)


## CMBrings.map_plot(f_cr);
## CMBrings.fourier_power(f_cr, ℓs = [400, 1000], imag_fun=CMBrings.imag_logabs2clip)

## semilogy(reshist)
## f_cr[:] |> real |> matshow; colorbar()
## g_cr[:] |> real |> matshow; colorbar()
## f_cr[:] .- g_cr[:] |> real |> matshow; colorbar()
## CMBrings.map_plot(  A(d, qu, DiagOp(Xmap(eaz2,1)))[2] )
## CMBrings.map_plot(_Aᵍ(d, qu, DiagOp(Xmap(eaz2,1)))[2] )
## CMBrings.fourier_power(  A(d, qu, DiagOp(Xmap(eaz2,1)))[2], imag_fun=CMBrings.imag_logabs2clip )
## CMBrings.fourier_power(_Aᵍ(d, qu, DiagOp(Xmap(eaz2,1)))[2], imag_fun=CMBrings.imag_logabs2clip )
## _Aᵍ(A(d, qu, )...)[2][:] .- qu[:] |> real |> matshow; colorbar()
## _Aᵍ(A(d, qu)...)[2][:] .- qu[:]  |> real |> matshow; colorbar()
## (M*(_Aᵍv1(A(d, qu)...)[1] - d))[:] |> real |> matshow; colorbar()
## (M*(_Aᵍv2(A(d, qu)...)[1] - d))[:] |> real |> matshow; colorbar()


## ------ initialize f′_cr
f′_cr = Ł(ϕ_cr) * (Ð▪⁻¹ \ f_cr) 


# Now gradient moves
ϕ_cr, f_cr,  g_cr, f′_cr, reshist = let ϕ_cr=ϕ_cr, f_cr=f_cr,  g_cr=g_cr, f′_cr=f′_cr, reshist=reshist

    for otr = 1:50 # default

        ## ------- update ϕ_cr (inputs are updated f′_cr and f_cr)
        @time gradϕ = CMBrings.∇ll_ϕf′_usingf(
            ϕ_cr, f_cr, Phi▪½, EB▪½; 
            data=d, Ł, Ð⁻¹=Ð▪⁻¹, M=M, B=B▪, N⁻¹=N▪⁻¹, 
            ϕ2v!, ϕ2vᴴ!, ∇!, grad_nsteps=14
        )
        ∇ϕ_cr = NΦN▪ * gradϕ 
        @time β = CMBrings.linesearch_ϕf′(
            ∇ϕ_cr, ϕ_cr, f′_cr,  Phi▪½, EB▪½; 
            data=d, Ł, Ð⁻¹=Ð▪⁻¹, M=M, B=B▪, N⁻¹=N▪⁻¹,
            eval_max=500, 
            startval=0.0001 , # default 0.0001 
            upper_bound = 1,  # default 2
            ftol_abs=10,      # default 100
            solver=:LN_COBYLA,  
        )
        @show β
        ϕ_cr += β * ∇ϕ_cr
        L_cr  = Ł(ϕ_cr)

        ## ------ update f_cr
        b_g_sim, b_f_sim = sim_bg_bf(L_cr)
        @time g_cr, f_cr, reshist = CMBrings.pcg_coupled(;
            nsteps  = 50, 
            rel_tol = 1e-15, 
            _Aᵍ = (g,f) -> _Aᵍ(g,f,L_cr), 
            A   = (g,f) ->   A(g,f,L_cr),
            b_g = M'*MWMᵀᵍ*d, 
            b_f = 0*d, 
            x_g = g_cr, 
            x_f = f_cr, 
            ## b_g = b_g_sim, 
            ## b_f = b_f_sim, 
            ## x_g = 0*g_cr, 
            ## x_f = 0*f_cr, 
        )
        hist_tail = isempty(reshist) ? nothing : reshist[end] 
        @show (hist_tail, length(reshist))

        ## ------ update f′_cr
        f′_cr = L_cr * (Ð▪⁻¹ \ f_cr) 

        ## ------ show stats
        @show CMBrings.ll_ϕf′(
            ϕ_cr, f′_cr, Phi▪½, EB▪½; 
            data=d, Ł, Ð⁻¹=Ð▪⁻¹, M, B=B▪, N⁻¹=N▪⁻¹
        )
        
    end # end for-loop

    ϕ_cr, f_cr, g_cr, f′_cr, reshist
end # end let


kappa = function (ϕ0)
    v   = (deepcopy(ϕ0[:]), deepcopy(ϕ0[:]))
    tmp = deepcopy(ϕ0[:])

    ∇!_ϕ(tmp, ϕ0[:], Val(2))
    ∇!_ϕ(v[2], tmp, Val(2))
    v[2] .*= csc.(eaz0.θ).^2


    ∇!_ϕ(tmp, ϕ0[:], Val(1))
    tmp .*= sin.(eaz0.θ)
    ∇!_ϕ(v[1], tmp, Val(1))
    v[1] ./= sin.(eaz0.θ)
    v[1][1:4,:] .= 0
    v[1][end-3:end,:] .= 0

    κ = v[1] .+ v[2]
    κ
end

## kappa(ϕ_cr) |> matshow


if save_jld2
    include("save_src.jl")
end


# Plots
# ================================

# using ImageFiltering

# log₊(x::T) where T = x > 0 ? log(x) : T(-Inf)

# function log_clip(x)
#     lx = log₊.(x)
#     finite_idx = @. isfinite(lx)
#     if !any(finite_idx)
#         return lx
#     else
#         lx[.!(finite_idx)] .= minimum(lx[finite_idx])
#         return lx
#     end
# end


# imag_logabs2clip(x) = log_clip(abs2.(x))

# function imag_blur(x;blur=0)
#     nθ, nφ = size(x)
#     imfilter(x, Kernel.gaussian(blur.*(1,(nφ÷2)/nθ)), "circular")
# end




#- 





## different sign for e and b....this is noted in healpix doc 
CMBrings.map_plot(
    # ϕ_cr; title1=L"Estimated $\phi$",
    # ϕ; title1=L"True $\phi$",
    Xmap(eaz0, kappa(ϕ_cr));  title1=L"Estimated $\kappa$", # vmin = -0.15, vmax = 0.15,
    # Xmap(eaz0, kappa(ϕ));  title1=L"Simulation truth $\kappa$", # vmin = -0.15, vmax = 0.15,
    # imag_fun=x->CMBrings.imag_blur(x;blur=2),
);



## different sign for e and b....this is noted in healpix doc 
CMBrings.map_plot(
    f_cr;  title1=L"Estimated unlensed $Q$", title2=L"Estimated unlensed $U$", # vmin = -0.15, vmax = 0.15,
    # qu;  title1=L"Truth unlensed $Q$", title2=L"Truth unlensed $U$", # vmin = -0.15, vmax = 0.15,
    # qu - f_cr;  title1=L"Truth - Estimated unlensed $Q$", title2=L"Truth - Estimated unlensed $U$", # vmin = -0.15, vmax = 0.15,
    # M * (Ł(ϕ)*qu - Ł(ϕ_cr)*f_cr);  title1=L"Truth - Estimated lensed $Q$", title2=L"Truth - Estimated lensed $U$", # vmin = -0.15, vmax = 0.15,
    imag_fun=x->CMBrings.imag_blur(x;blur=0),
);




CMBrings.fourier_power(
    Xmap(eaz0, kappa(ϕ_cr));  title1=L"Estimated $\kappa$", vmin = -15, # vmax = 0,
    # Xmap(eaz0, kappa(ϕ));  title1=L"Simulation truth $\kappa$",  vmin = -15, # vmax = 0,
    # Xmap(eaz0, kappa(ϕ_cr - ϕ));  title1=L"truth - est $\kappa$", # vmin = -15, # vmax = 0,
    ℓs = [400, 1000, 3000], 
    imag_fun=CMBrings.imag_logabs2clip,
);

#-

ℓbin, cr_power = CMBrings.quasi_bandpowers(
    Xmap(eaz0, kappa(ϕ_cr)); 
    Δℓsph_bin = 15
)
ℓbin, tu_power    = CMBrings.quasi_bandpowers(
    Xmap(eaz0, kappa(ϕ)); 
    Δℓsph_bin = 15
)
ℓbin, tu_cr_power = CMBrings.quasi_bandpowers(
    Xmap(eaz0, kappa(ϕ_cr)), 
    Xmap(eaz0, kappa(ϕ)); 
    Δℓsph_bin = 15
)

corr_power = tu_cr_power ./ sqrt.(cr_power) ./ sqrt.(tu_power)

fig,ax = subplots(1, dpi=147)
ax.plot(ℓbin, abs2.(real.(corr_power)))
# ax.plot(ℓbin, save_corr_power_sq)


hcat(abs2.(real.(corr_power)), save_corr_power_sq)

save_corr_power_sq = abs2.(real.(corr_power))

# TODO: fixup the following ....







#-

## ϕ[:] |> matshow; colorbar()
## ϕ_cr[:] |> matshow; colorbar()
## f_cr[:] |> real |> matshow; colorbar()
## qu[:] |> real |> matshow; colorbar()
## f_cr[:] |> imag |> matshow; colorbar()
## qu[:] |> imag |> matshow; colorbar()
## f_cr[:] .- qu[:] |> real |> matshow; colorbar()


#-

@sblock let ϕtru = ϕ, ϕest = ϕ_cr, kappa, φ, θ, hide_plots, save_figures, polar_plots
    hide_plots && return

    imgs = Dict(
        1=>kappa(ϕtru), 
        2=>kappa(ϕest)
    )
    txt  = Dict(1=>L"true $\kappa$", 2=>L"est $\kappa$")
    
    vmin, vmax = .7 .* extrema(imgs[1])

    if polar_plots
        fig, ax = CMBrings.diskplot(imgs, CC.in_negπ_π.(φ)', π.-θ, figsize=(6,5))
    else 
        fig,ax = subplots(nrows=1, ncols=2, figsize=(6,2))
        imgs[1] |> imshow(-,fig,ax[1];vmin,vmax)
        imgs[2] |> imshow(-,fig,ax[2];vmin,vmax)
    end
    ax[1].set_title(txt[1])
    ax[2].set_title(txt[2])

    
    ## fig.suptitle(L"true (top) vrs est (bottom) $\nabla_\theta \phi$")

    save_figures && savefig("figure$(fig.number).png", dpi=250, bbox_inches="tight")
    return nothing
end




#-

@sblock let ϕtru = ϕ, ϕest = ϕ_cr, ϕ2v!, φ, θ, hide_plots, save_figures, polar_plots
    hide_plots && return

    viz = function (ϕ0)
        v = (deepcopy(ϕ0[:]), deepcopy(ϕ0[:]))
        ϕ2v!(v, ϕ0[:])
        v 
    end


    imgs = Dict(1=>viz(ϕtru)[1], 2=>viz(ϕest)[1])
    txt  = Dict(1=>L"true $\theta$ displacement", 2=>L"est $\theta$ displacement")
    
    vmin, vmax = extrema(imgs[1])

    if polar_plots
        fig, ax = CMBrings.diskplot(imgs, CC.in_negπ_π.(φ)', π.-θ, figsize=(6,5))
    else 
        fig,ax = subplots(nrows=1, ncols=2, figsize=(6,2))
        imgs[1] |> imshow(-,fig,ax[1];vmin,vmax)
        imgs[2] |> imshow(-,fig,ax[2];vmin,vmax)
    end
    ax[1].set_title(txt[1])
    ax[2].set_title(txt[2])

    
    ## fig.suptitle(L"true (top) vrs est (bottom) $\nabla_\theta \phi$")

    save_figures && savefig("figure$(fig.number).png", dpi=250, bbox_inches="tight")
    return nothing
end




#-

@sblock let ϕtru = ϕ, ϕest = ϕ_cr, ϕ2v!, φ, θ, hide_plots, save_figures, polar_plots
    hide_plots && return
    viz = function (ϕ0)
        v = (deepcopy(ϕ0[:]), deepcopy(ϕ0[:]))
        ϕ2v!(v, ϕ0[:])
        v 
    end
    imgs = Dict(1=>viz(ϕtru)[2], 2=>viz(ϕest)[2])
    txt  = Dict(1=>L"true $\varphi$ displacement", 2=>L"est $\varphi$ displacement")
    
    vmin, vmax = extrema(imgs[1])

    if polar_plots
        fig, ax = CMBrings.diskplot(imgs, CC.in_negπ_π.(φ)', π.-θ, figsize=(6,5))
    else 
        fig,ax = subplots(nrows=1, ncols=2, figsize=(6,2))
        imgs[1] |> imshow(-,fig,ax[1];vmin,vmax)
        imgs[2] |> imshow(-,fig,ax[2];vmin,vmax)
    end
    ax[1].set_title(txt[1])
    ax[2].set_title(txt[2])

    save_figures && savefig("figure$(fig.number).png", dpi=250, bbox_inches="tight")
    return nothing
end


#- 


@sblock let ϕtru = ϕ, ϕest = ϕ_cr, ϕ2v!, φ, θ, hide_plots, save_figures, polar_plots
    hide_plots && return
    viz = function (ϕ0)
        v = (deepcopy(ϕ0[:]), deepcopy(ϕ0[:]))
        ϕ2v!(v, ϕ0[:])
        v 
    end
    imgs = Dict(1=>ϕtru[:] .- mean(ϕtru[:]), 2=>ϕest[:] .- mean(ϕest[:]))
    txt  = Dict(1=>"true lensing potential", 2=>"est lensing potential")
    
    vmin, vmax = extrema(imgs[1])

    if polar_plots
        fig, ax = CMBrings.diskplot(imgs, CC.in_negπ_π.(φ)', π.-θ, figsize=(6,5))
    else 
        fig,ax = subplots(nrows=1, ncols=2, figsize=(6,2))
        imgs[1] |> imshow(-,fig,ax[1];vmin,vmax)
        imgs[2] |> imshow(-,fig,ax[2];vmin,vmax)
    end
    ax[1].set_title(txt[1])
    ax[2].set_title(txt[2])


    save_figures && savefig("figure$(fig.number).png", dpi=250, bbox_inches="tight")

    return nothing
end








#-

@sblock let d, φ, θ, hide_plots, save_figures, polar_plots

    hide_plots && return

    imgs = Dict(1=>real(d[:]), 2=>imag(d[:]))
    txt  = Dict(
        1=>"data Q",     2=>"data U",
    )

    vmin, vmax = extrema(imgs[1])

    if polar_plots
        fig, ax = CMBrings.diskplot(imgs, CC.in_negπ_π.(φ)', π.-θ, figsize=(6,5))
    else 
        fig,ax = subplots(nrows=1, ncols=2, figsize=(6,2))
        imgs[1] |> imshow(-,fig,ax[1])
        imgs[2] |> imshow(-,fig,ax[2])
    end
    ax[1].set_title(txt[1])
    ax[2].set_title(txt[2])

    
    ## fig.suptitle("unlensed Q (top) and U (bottom)")

    save_figures && savefig("figure$(fig.number).png", dpi=250, bbox_inches="tight")

    return nothing

end






#-

@sblock let f_cr, φ, θ, hide_plots, save_figures, polar_plots

    hide_plots && return

    imgs = Dict(1=>real(f_cr[:]), 2=>imag(f_cr[:]))
    txt  = Dict(
        1=>"unlensed Q est",     2=>"unlensed U est",
    )


    if polar_plots
        fig, ax = CMBrings.diskplot(imgs, CC.in_negπ_π.(φ)', π.-θ, figsize=(6,5))
    else 
        fig,ax = subplots(nrows=1, ncols=2, figsize=(6,2))
        imgs[1] |> imshow(-,fig,ax[1])
        imgs[2] |> imshow(-,fig,ax[2])
    end
    ax[1].set_title(txt[1])
    ax[2].set_title(txt[2])


    save_figures && savefig("figure$(fig.number).png", dpi=250, bbox_inches="tight")

    return nothing

end



#-

@sblock let f_cr, qu, φ, θ, hide_plots, save_figures, polar_plots

    hide_plots && return

    imgs = Dict(1=>real(f_cr[:] .- qu[:]), 2=>imag(f_cr[:] .- qu[:]))
    txt  = Dict(
        1=>"unlensed Q (est - tru)",     2=>"unlensed U (est - tru)",
    )

    if polar_plots
        fig, ax = CMBrings.diskplot(imgs, CC.in_negπ_π.(φ)', π.-θ, figsize=(6,5))
    else 
        fig,ax = subplots(nrows=1, ncols=2, figsize=(6,2))
        imgs[1] |> imshow(-,fig,ax[1])
        imgs[2] |> imshow(-,fig,ax[2])
    end
    ax[1].set_title(txt[1])
    ax[2].set_title(txt[2])


    save_figures && savefig("figure$(fig.number).png", dpi=250, bbox_inches="tight")

    return nothing

end



#-

@sblock let f_cr, ϕ_cr, ϕ, qu, Ł, M, φ, θ, hide_plots, save_figures, polar_plots

    hide_plots && return

    L_cr = Ł(ϕ_cr)
    L = Ł(ϕ)
    lnf_cr = M*L_cr*f_cr
    lnf = M*L*qu

    imgs = Dict(1=>real(lnf_cr[:] .- lnf[:]), 2=>imag(lnf_cr[:] .- lnf[:]))
    txt  = Dict(
        1=>"masked lensed Q (est - tru)",     2=>"masked lensed U (est - tru)",
    )

    if polar_plots
        fig, ax = CMBrings.diskplot(imgs, CC.in_negπ_π.(φ)', π.-θ, figsize=(6,5))
    else 
        fig,ax = subplots(nrows=1, ncols=2, figsize=(6,2))
        imgs[1] |> imshow(-,fig,ax[1])
        imgs[2] |> imshow(-,fig,ax[2])
    end
    ax[1].set_title(txt[1])
    ax[2].set_title(txt[2])


    save_figures && savefig("figure$(fig.number).png", dpi=250, bbox_inches="tight")

    return nothing

end





