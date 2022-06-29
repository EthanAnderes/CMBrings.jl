


# Modules
# ==============================

using LinearAlgebra
using FFTW
FFTW.set_num_threads(BLAS.get_num_threads())

using XFields
using  FFTransforms
import FFTransforms as FT
import CirculantCov as CC
using CMBrings
using FieldLensing 
using Spectra: camb_cls
using VecchiaFactorization
import VecchiaFactorization as VF
using LBblocks: @sblock

using SparseArrays
using PyPlot
using BenchmarkTools
using ProgressMeter
using BlockArrays
using Dierckx: Spline1D 

## import Random
## Random.seed!(1234)

#- 

if isdefined(Main, :IJulia) && Main.IJulia.inited
    hide_plots = false
else 
    hide_plots = true
end
save_figures = false 
polar_plots = false
save_jld2 = true # !!!!!!

# Pixel grid
# ==============================

θ, φ, θ∂, φ∂, Ω, Δθ, nθ, nφ, freq_mult, grid_type, bsd_nθ = @sblock let 
    
    # --------- hi-res, equiθ
    # φspan, freq_mult = deg2rad.((-60, 60)), 3
    # φ, φ∂ = CC.φ_grid(;φspan, N=1536)    # N=768 or N=1536, 2048, 1024, 972,  1280
    # type, N, θspan  = :equiθ,  805, π/2 .- deg2rad.((-41.78,-70.43)) 
    # θ, θ∂  = CC.θ_grid(; θspan, N, type)
    # bsd_nθ = 161

    φspan, freq_mult = deg2rad.((-60, 60)), 3
    φ, φ∂ = CC.φ_grid(;φspan, N=1536)    # N=768 or N=1536, 2048, 1024, 972,  1280
    type, N, θspan  = :equicosθ,  805, π/2 .- deg2rad.((-41.78,-70.43)) 
    θ, θ∂  = CC.θ_grid(; θspan, N, type)
    bsd_nθ = 161

    #  -------- med-res
    # φspan, freq_mult = deg2rad.((-45, 45)), 4
    # φ, φ∂ = CC.φ_grid(;φspan, N=1280)    # N=768 or N=1024, 972, 1536, 1280
    # type, N, θspan  = :equiθ,  600, π/2 .- deg2rad.((-51,-69)) 
    # θ, θ∂  = CC.θ_grid(; θspan, N, type)
    # bsd_nθ = 150

    # --------- hi-res, healpix rings
    # Nside = 8192
    # type  = :healpix
    # ri_offset_from_SP = round(Int, sqrt(3*Nside^2*(1+cos(2.805))))
    # ri = (3*Nside+1):6:(4*Nside-1 - ri_offset_from_SP) # upper limit should be 4*Nside-1
    # θ  = CC.θ_healpix(Nside)[ri]
    # θ∂ = CC.θ_healpix(Nside)[ri.start:ri.step:ri.stop+ri.step]
    # # ... Now choose the Az number of grid points
    # # Make sure the portion of azimuth is a factor of nφ_full
    # # 4Nside should be largest value for nφ_full
    # # nφ_full = 3*Nside÷4
    # nφ_full = 1536 * 3
    # # nφ_full = 3*Nside÷4 - 3*512÷4
    # # nφ_full = 3*Nside        
    # # nφ_full = 4*(Nside-1) # 2^3 * 3^2 * 5 * 7 * 13 
    # φ_full = 2 * π * (0:nφ_full-1) / nφ_full
    # φspan, freq_mult = deg2rad.((-60, 60)), 3
    # φspan, freq_mult = deg2rad.((0, 360)), 1
    # # φspan, freq_mult = deg2rad.((-45, 45)), 4
    # φ, φ∂ = CC.φ_grid(;φspan, N=nφ_full÷freq_mult)  
    # bsd_nθ = 161

    
    nθ, nφ = length(θ), length(φ)
    Ω  = CC.counterclock_Δφ(φ∂[1], φ∂[2]) .* diff(.- cos.(θ∂))
    Δθ = diff(θ∂)

    collect(θ), φ, θ∂, φ∂, Ω, Δθ, nθ, nφ, freq_mult, type, bsd_nθ
end 


# Plot √Ωpix over ring θ's 

@sblock let θ, φ, Ω, Δθ, hide_plots=false, save_figures
    hide_plots && return

    pix_diag_rad = CC.geoβ.(θ[2:end], θ[1:end-1], φ[1], φ[2]) # arclength of the pixel diagonals
    pixφside_rad = sin.(θ) .* CC.counterclock_Δφ(φ[1], φ[2])
    pixθside_rad = Δθ


    fig,ax = subplots(1)
    ax.plot(θ, (@. rad2deg(√Ω)*60), label="sqrt pixel area (arcmin)")
    ax.plot(θ, (@. rad2deg(pixθside_rad)*60), label="Δθ (arcmin)")
    ax.plot(θ, (@. rad2deg(pixφside_rad)*60), label="pix φ side arclen (arcmin)")
    ax.plot(θ[1:end-1], (@. rad2deg(pix_diag_rad)*60), label="pix diag arclen (arcmin)")
    ax.set_xlabel(L"polar coordinate $\theta$")
    ax.legend()
    save_figures && savefig("figure$(fig.number).png", dpi=250)
    return nothing
end



pix_diag_arcmin = CC.geoβ.(θ[2:end],θ[1:end-1],φ[1],φ[2]) .|> x->60*rad2deg(x)
@show (nθ, nφ)
@show extrema(@. rad2deg(√Ω)*60) 
@show extrema(@. rad2deg(Δθ)*60) 
@show extrema(pix_diag_arcmin) 

# Transformations
# ==============================

tmUS2, tmUS0, T = @sblock let nθ, nφ, freq_mult
    ## T  = ComplexF32
    T  = ComplexF64
    Tr = real(T)
    tmUS2 = 𝕀(nθ) ⊗ 𝕌(T, nφ, 2π/freq_mult)
    tmUS0 = 𝕀(nθ) ⊗ 𝕌(Tr, nφ, 2π/freq_mult)
    return tmUS2, tmUS0, T
end;


# Spectral densities
# ==============================

φ_approx_nyq = freq_mult * nφ / minimum(sin.(θ)) / 2
θ_approx_nyq = π / minimum(Δθ) 
@show approx_lmax = ceil(Int, sqrt(φ_approx_nyq^2 + θ_approx_nyq^2))

approx_lmax += ceil(Int, approx_lmax * 0.1) # for good measure:)
## override ...
## approx_lmax = 25_000

ℓ, ϕϕℓ, eeℓ, bbℓ, ẽẽℓ, b̃b̃ℓ = @sblock let lmax=approx_lmax, r=0.01, T=Float64
    
    l = 0:lmax
    cld = camb_cls(;lmax=lmax, r,
        lSampleBoost   = 4.0,
        lAccuracyBoost = 4.0,
        KmaxBoost = 4.0,
        )
    
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

    return l, T.(ϕϕl), T.(eel), T.(bbl), T.(ẽel), T.(b̃bl) 
end;


# this is a hack ...
bbℓ[bbℓ .<= 0] .= 1e-18 # minimum(bbℓ[3:end][bbℓ[3:end] .> 0])
eeℓ[eeℓ .<= 0] .= 1e-18 # minimum(eeℓ[3:end][eeℓ[3:end] .> 0])
b̃b̃ℓ[b̃b̃ℓ .<= 0] .= 1e-18 # minimum(bbℓ[3:end][bbℓ[3:end] .> 0])
ẽẽℓ[ẽẽℓ .<= 0] .= 1e-18 # minimum(eeℓ[3:end][eeℓ[3:end] .> 0])
b̃b̃ℓ[1] = b̃b̃ℓ[2] = 0
ẽẽℓ[1] = ẽẽℓ[2] = 0
bbℓ[1] = bbℓ[2] = 0
eeℓ[1] = eeℓ[2] = 0

#=
loglog( ℓ.^2 .* eeℓ)
loglog( ℓ.^2 .* bbℓ)
loglog( ℓ.^2 .* ẽẽℓ)
loglog( ℓ.^2 .* b̃b̃ℓ)
=#


## semilogy(ℓ, eeℓ)
## semilogy(ℓ, bbℓ)
## semilogy(ℓ, b̃b̃ℓ)

## ## testing !!!!
## semilogy(ℓ, bbℓ)
## numerical_floor_i = findall(bbℓ .< 1e-16)[3]
## bbℓi = bbℓ[numerical_floor_i]
## ℓi   = ℓ[numerical_floor_i]
## for_clamp_ℓ      = @. bbℓi * (ℓi/ℓ)^3
## for_clamp_ℓ[1:ℓi] .= -Inf
## semilogy(ℓ, max.(bbℓ, for_clamp_ℓ))


## ???? can you do something like set the numerical floor spectra
## ???? to be a scaled beam 
## For that matter, what happens if you multiply by a beam to attinuate
## near grid scale local variation. ??



#-

#=
EB▫_θhead = CMBrings.az_cov_blks(ℓ, eeℓ, bbℓ; θ=θ[1:2*bsd_nθ], φ, ℓrange=[nφ÷2-5,nφ÷2+1], ngrid=100_000);
EB▫_θhead[1] |> Hermitian |> eigen |> x->x.values
EB▫_θhead[end] |> Hermitian |> eigen |> x->x.values


EB▫_θtail = CMBrings.az_cov_blks(ℓ, eeℓ, bbℓ; θ=θ[end-2*bsd_nθ:end], φ, ℓrange=[nφ÷2-5,nφ÷2+1], ngrid=100_000);
EB▫_θtail[1] |> Hermitian |> eigen |> x->x.values
EB▫_θtail[end] |> Hermitian |> eigen |> x->x.values


EB▫_θtail[end] |> Hermitian |> eigen |> x->x.vectors[:,end] |> plot
EB▫_θtail[end] |> Hermitian |> eigen |> x->x.vectors[:,end-1] |> plot
EB▫_θtail[end] |> Hermitian |> eigen |> x->x.vectors[:,end÷2] |> plot
EB▫_θtail[end] |> Hermitian |> eigen |> x->x.vectors[:,2] |> plot




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
prθ, prφ  =  @sblock let rT=real(T), nθ, nφ, tmUS2

    ## ▮lθ, ▯lθ = 20, 60 
    ## ▮rθ, ▯rθ = nθ-▮lθ+1, nθ-▯lθ+1 
    ▮lθ, ▯lθ = 15, 50 
    ▮rθ, ▯rθ = nθ-▮lθ+1, nθ-▯lθ+1 
    prθ    = CMBrings.pixweight.(rT.(1:nθ); ▮l=▮lθ,    ▯l=▯lθ, ▯r=▯rθ, ▮r=▮rθ)
    
    ## ---------------------------- 
    ## ▮lφ, ▯lφ = 30, 60 
    ## ▮rφ, ▯rφ = nφ-▮lφ+1, nφ-▯lφ+1 
    ## prφ    = CMBrings.pixweight.(rT.(1:nφ); ▮l=▮lφ,    ▯l=▯lφ, ▯r=▯rφ, ▮r=▮rφ)
    ## ----- alt ----- ↓↓ No azmuthal mask ↓↓
    prφ = ones(rT,nφ)
    ## ---------------------------- 

    prθ, prφ
end;

# Lensing mask (to keep the lense from transporting off the polar cut)
Mϕ = @sblock let rT=real(T), nθ, nφ, tmUS0, prθφ = prθ.*prφ'
    
    ## ▮lθ, ▯lθ = 1, 10 
    ## ▮rθ, ▯rθ = nθ-1+1, nθ-10+1 
    ## prθ  = CMBrings.pixweight.(rT.(1:nθ); ▮l=▮lθ,    ▯l=▯lθ, ▯r=▯rθ, ▮r=▮rθ)
    ## mϕx = prθ * ones(rT,nφ)'
    ## ---------- alt -------------
    sqz = 4
    sft = 0.4
    mϕx = prθφ .|> x-> clamp((atan(sqz*(x-sft)) + π/2)/π, .05, .95)

    ## make sure it hits zero and 1
    mϕx .-= minimum(mϕx)
    mϕx ./= maximum(mϕx)
    Mϕ    = DiagOp(Xmap(tmUS0, mϕx))
    Mϕ
end;

## Mϕ[:] .|> real |> matshow; colorbar()
## prθ .* prφ' .|> real |> matshow; colorbar()

# Azimuthal ring mask

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


# Coordinate pivot, blocks and queries for Vecchia
# ==============================
## using Primes; factor(length(θ)) # ; @assert nθ÷bks == nθ/bks

permθ, block_sizesθ = @sblock let prθ, nθ, bsd_nθ=bsd_nθ 
    block_sizesθ = VF.block_split(nθ, bsd_nθ)
    ## block_sizesθ = VF.block_split(nθ, bsd_nθ) |> sort

    permθ=1:nθ
    permθ, block_sizesθ
end


# Spin 2 signal
# =================================================

## @time EB▪½ = let 
##     EB▫  = CMBrings.az_cov_blks(ℓ, eeℓ, bbℓ ; θ,  φ)
##     map(EB▫) do M 
##         Array(sqrt(Hermitian(M)))
##     end |> CircOp
## end
## EB▪⁻½ = map(inv, EB▪½) |> CircOp;
## -------
@time EB▪½ = CMBrings.spin2_az_cov½_vecchia_blks(ℓ, eeℓ, bbℓ, block_sizesθ, permθ; θ, φ) |> CircOp;
EB▪⁻½ = map(VF.posdef_inv, EB▪½) |> CircOp;


## sum(Base.summarysize, EB▪½) / 1e9 # 7.41 GB, 3.55min construction, high res
## EB▪½[end-5][3].data[2]
## EB▪⁻½[end-5][2].data[2]


# Spin 0 signal
# =================================================

## @time Phi▪½ = let 
##     Phi▫  = CMBrings.az_cov_blks(ℓ, ϕϕℓ; θ,  φ)
##     map(Phi▫) do M 
##         Array(sqrt(Symmetric(M))) 
##     end |> CircOp
## end
## Phi▪⁻½ = map(inv, Phi▪½) |> CircOp;
## -------
@time Phi▪½ = CMBrings.spin0_az_cov½_vecchia_blks(ℓ, ϕϕℓ, block_sizesθ, permθ; θ, φ) |> CircOp;
Phi▪⁻½ = map(VF.posdef_inv, Phi▪½) |> CircOp;

## sum(Base.summarysize, Phi▪½) / 1e9 # 1.4 GB, 2.5min construction, high res


# Noise
# ============================

μK_arcmin  = 5.0 # 1.0

N▪ = @sblock let μK_arcmin, Ω, nφ 
    σ²   = deg2rad(μK_arcmin/60)^2 # Cⁿℓ == μK_arcmin |> arcmin2radians |> abs2
    σ²_Ω = σ² ./ Ω
    Nmat = Diagonal(vcat(σ²_Ω,σ²_Ω))
    N▫   = [Nmat for ℓ = 1:nφ÷2+1]
    CircOp(N▫)
end; 

N▪⁻¹ = map(Nℓ->Diagonal(1 ./ diag(Nℓ)), N▪.Σ) |> CircOp;

# Now add pure BB noise * large factor bb_noise_factor

## N▪ = let bb_noise_factor = 100 
##     zeroEB▪  = CMBrings.az_cov_blks(ℓ, 0 .* eeℓ, bbℓ ; θ,  φ, ngrid=100_000) |> CircOp
##     map(N▪, zeroEB▪) do A, B
##         A + bb_noise_factor * B
##     end |> CircOp
## end 
## 
## ## N▪⁻¹ = map(Nℓ->Diagonal(1 ./ diag(Nℓ)), N▪.Σ) |> CircOp;
## N▪⁻¹ = map(inv, N▪) |> CircOp;


# Mask
# ============================

M = DiagOp(Xmap(tmUS2, prθ .* prφ' ));

# Beam
# ============================

pix_diag_rad   = CC.geoβ.(θ∂[2:end], θ∂[1:end-1], φ[1], φ[2]) # arclength of the pixel diagonals
beamfwhm_rad_θ = pix_diag_rad # * 0.95
σ²θ            = @. CMBrings.fwhmrad2σ²(beamfwhm_rad_θ)

Γbeam_θ₁θ₂φ₁φ⃗ = let σ²θ_spl = Spline1D(θ,σ²θ,k=2)
    function (θ₁, θ₂, φ₁, φ⃗)
        complex.(CMBrings.B̃eam1.(θ₁, θ₂, σ²θ_spl(θ₁), σ²θ_spl(θ₂), φ₁ .- φ⃗))
    end
end;

B▪ = @sblock let Γbeam_θ₁θ₂φ₁φ⃗, block_sizesθ, permθ, θ, φ, Ω

    nθ, nφ = length(θ), length(φ)
    DΩΩ  = Diagonal(vcat(Ω, Ω))
    
    Bspin0▪ = CMBrings.spin0_az_cov_vecchia_blks(
        Γbeam_θ₁θ₂φ₁φ⃗, block_sizesθ,  permθ; θ, φ
    ) |> CircOp;

    Bspin2▪ = map(Bspin0▪) do B
        ## B = Bspin0▪[2]
        P = B[1]'
        R = inv(B[2])
        Mpre = B[3] ## B[3]*B[3]'
        M = VF.Midiagonal(Mpre.data) # What is the speed effect here??

        a1 = 1:2nθ |> x->reshape(x,nθ,2)
        P2 = VF.Piv(a1[P.perm,:][:])
        M2 = vcat(M.data, M.data) |> VF.Midiagonal
        invR2 = vcat(
            R.data, 
            [zeros(eltype(M.data[1]), size(M.data[1],1), size(M.data[end],2))], 
            R.data
        ) |> VF.Ridiagonal |> inv

        P2' * invR2 * M2 * invR2' * P2 * DΩΩ
    end |> CircOp

    return Bspin2▪
end;  



# Lensing operators
# ============================

∇!,  ∇!_ϕ = CMBrings.generate_∇!∇!ϕ(θ, φ; uniformΔθ = (grid_type == :equiθ) ? true : false); 

Ł, ϕ2v!, ϕ2vᴴ!, ∇! = CMBrings.generate_lense(;
    θ, mv1x=Mϕ[:], mv2x=Mϕ[:], ∇!,  ∇!_ϕ, 
    nsteps_lensing=14
);

# simulation
# ==============================

## ϕ = Phi▪½ * Xmap(tmUS0,randn(Float64,nθ,nφ));
## ------ alt: full non-Vecchia approximate simulation
@time ϕ = @sblock let ℓ, ϕϕℓ, blksiz=nφ÷5, θ, φ, w=Xmap(tmUS0,randn(Float64,nθ,nφ)) 
    nθ, nφ = length(θ), length(φ)
    wθ▪    = CMBrings.field2▪(w)
    fθ▪    = map(similar, wθ▪)
    ℓfull  = 1:nφ÷2+1
    ℓblks  = blocks(PseudoBlockArray(ℓfull, VF.block_split(length(ℓfull), blksiz)))
    for ℓblk in ℓblks
        Σ▪_ℓblk = CMBrings.az_cov_blks(ℓ, ϕϕℓ; θ, φ, ℓrange=ℓblk)
        for (i,ℓi) in enumerate(ℓblk)
            ## L = cholesky(Symmetric(Σ▪_ℓblk[i])).L
            ## lmul!(L, fθ▪[ℓi])
            M = sqrt(Symmetric(Σ▪_ℓblk[i]))
            mul!(fθ▪[ℓi], M, wθ▪[ℓi])
        end
    end
    return CMBrings.▪2field(fieldtransform(w), fθ▪)
end;

#-

## qu = EB▪½ * Xmap(tmUS2,randn(ComplexF64,nθ,nφ))
## ------ alt: full non-Vecchia approximate simulation
qu = @sblock let ℓ, eeℓ, bbℓ, blksiz=nφ÷5, θ, φ, w=Xmap(tmUS2,randn(ComplexF64,nθ,nφ)) 
    nθ, nφ = length(θ), length(φ)
    wθ▪    = CMBrings.field2▪(w)
    fθ▪    = map(similar, wθ▪)
    ℓfull  = 1:nφ÷2+1
    ℓblks  = blocks(PseudoBlockArray(ℓfull, VF.block_split(length(ℓfull), blksiz)))
    for ℓblk in ℓblks
        Σ▪_ℓblk = CMBrings.az_cov_blks(ℓ, eeℓ, bbℓ; θ, φ, ℓrange=ℓblk)
        for (i,ℓi) in enumerate(ℓblk)
            ## L = cholesky(Hermitian(Σ▪_ℓblk[i])).L
            ## lmul!(L, fθ▪[ℓi]) ## This leads to striations in U for some reason
            M = sqrt(Hermitian(Σ▪_ℓblk[i]))
            mul!(fθ▪[ℓi], M, wθ▪[ℓi])
        end
    end
    return CMBrings.▪2field(fieldtransform(w), fθ▪)
end;

#-

no = map(N▪, Xmap(tmUS2,randn(ComplexF64,nθ,nφ))) do Σ,v
    sqrt(Σ)*v
end 

#-

d = M * (B▪ * Ł(ϕ) * qu + no) |> Xfourier;

#-

## d[:] |> real |> matshow; colorbar()
## d[:] |> imag |> matshow; colorbar()
## qu[:] |> real |> matshow; colorbar()
## qu[:] |> imag |> matshow; colorbar()
## ϕ[:] |> matshow; colorbar()
## (Ł(ϕ)*qu - qu)[:] |> real |> matshow; colorbar()
## qu[:] |> imag |> matshow; colorbar()
## (B▪ * B▪ * B▪ * B▪ * B▪ * no)[:] |> real |> matshow; colorbar()
## (B▪ * B▪ * B▪ * B▪ * B▪ * no)[:] |> imag |> matshow; colorbar()

# Mixflow operator
# ============================

nnℓ = deg2rad(μK_arcmin/60)^2 # Cⁿℓ == μK_arcmin |> arcmin2radians |> abs2

Ð▪⁻¹ = CMBrings.spin2_az_cov½_vecchia_blks(
   ℓ, (@. eeℓ/(ẽẽℓ+2nnℓ)), (@. bbℓ/(b̃b̃ℓ+2nnℓ)),  
   block_sizesθ,  permθ; θ, φ
) |> CircOp;


# Initalize opps for ϕ gradient
# ==============================================


import CMBflat

N0ℓ, NΦNℓ = @sblock let pix_side_rad = mean(@. √Ω), n_iter=5, ℓ, eeℓ, bbℓ, ϕϕℓ, beamfwhm_rad_θ, nnℓ=fill(nnℓ,length(ℓ)) 
    
    ## not sure which version of σ² is the best here???
    ## σ² = mean(beamfwhm_rad_θ)^2 / 8 / log(2)
    ## σ² = minimum(beamfwhm_rad_θ)^2 / 8 / log(2)    
    σ² = maximum(beamfwhm_rad_θ)^2 / 8 / log(2) ## original ...
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

NΦN▪ = CMBrings.spin0_az_cov½_vecchia_blks(
    ℓ, NΦNℓ,  
    block_sizesθ,  permθ; θ, φ
) |> x->map(m->m*m',x) |> CircOp;

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

MWMᵀᵍ = @sblock let W▪, M, nφ, tmUS2
    ## MWMᵀ_pxl = abs2.(prθφM) .* prθW
    prθW = diag(W▪[1])[1:end÷2]
    ## prθM = M[:][:,end÷2]
    ## MWMᵀ_pxl = prθW .* abs2.(prθM) .* ones(1,nφ)
    MWMᵀ_pxl = prθW .* abs2.(M[:]) # Testing !!!!!!!!
    DiagOp(Xmap(tmUS2, pinv.(MWMᵀ_pxl)))
end;


@time _A₁₁ᵍ▪, _A₂₂_A₂₁A₁₁ᵍA₁₂_ᵍ▪ = @sblock let B▪, EB▪½,  N▪⁺ᵍ, W▪, M, MWMᵀᵍ, block_sizesθ, nθ = length(θ)
    Mθ     = M[:][:,end÷2] |> x->vcat(x,x)
    ## Mθ     = mean(eachcol(M[:])) |> x->vcat(x,x)

    MWMᵀᵍθ = MWMᵀᵍ[:][:,end÷2] |> x->vcat(x,x)
    
    _A₁₁ᵍ▪ = map(W▪, N▪⁺ᵍ) do W, iN
        Diagonal(pinv.(Mθ .* MWMᵀᵍθ .* conj.(Mθ) .+ diag(iN)))
    end |> CircOp

    _A₂₂_A₂₁A₁₁ᵍA₁₂_ᵍ▪ = map(_A₁₁ᵍ▪, B▪, N▪⁺ᵍ, EB▪½) do iA, Bl, iN, Σ½
        PΣ, RΣ, M½Σ = Σ½[1], inv(Σ½[2]), Σ½[3]
        invΣ = VF.instantiate_inv(RΣ, M½Σ*M½Σ', PΣ)

        PB, RB, MB, matΩ = Bl[1], inv(Bl[2]), Bl[3], Bl[6]
        invB = VF.instantiate_inv(RB, MB, PB)
        matB = inv(cholesky(Symmetric(invB)))

        iN_iNiAiN½ = sqrt(iN - iN*iA*iN)
        lmul!(iN_iNiAiN½, matB)
        rmul!(matB, matΩ)
        invΣ += matB'*matB  
        ## X = invΣ + matΩ'*(matB'*(iN - iN*iA*iN)*matB)*matΩ
        invX = inv(cholesky(Hermitian(invΣ))) 
        return VF.vecchia(invX, 
                    2 .* block_sizesθ,  
                    ## VF.block_split(2nθ, 250),
                    1:2nθ |> x->(reshape(x,nθ,2)')[:] 
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


let M=M, MWMᵀᵍ=MWMᵀᵍ, N▪⁺ᵍ=N▪⁺ᵍ, B▪=B▪, _A₁₁ᵍ▪=_A₁₁ᵍ▪, _A₂₂_A₂₁A₁₁ᵍA₁₂_ᵍ▪=_A₂₂_A₂₁A₁₁ᵍA₁₂_ᵍ▪, tmUS2=tmUS2, EB▪⁻½=EB▪⁻½

    global function A(g, f, L)
        Afg_g = (M'*MWMᵀᵍ*M*g + N▪⁺ᵍ*g) - (N▪⁺ᵍ*B▪*L*f)
        Afg_f = - (L'*B▪'*N▪⁺ᵍ*g) + (L'*B▪'*N▪⁺ᵍ*B▪*L*f + EB▪⁻½'*EB▪⁻½*f)
        Afg_g, Afg_f
    end

    global function _Aᵍ(g, f, L)
        f1 = _A₂₂_A₂₁A₁₁ᵍA₁₂_ᵍ▪ * (L'*B▪'*N▪⁺ᵍ*_A₁₁ᵍ▪*g + f)
        _A₁₁ᵍ▪*(g + N▪⁺ᵍ*B▪*L*f1), f1
    end

    global function sim_bg_bf(L)
        γ₁  = sqrt(MWMᵀᵍ) * Xmap(tmUS2,randn(ComplexF64,nθ,nφ))
        γ₂  = map((Σ,v)->sqrt(Σ)*v, N▪⁺ᵍ, Xmap(tmUS2,randn(ComplexF64,nθ,nφ)))
        γ₃  = EB▪⁻½' * Xmap(tmUS2,randn(ComplexF64,nθ,nφ))
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
    _Aᵍ = (g, f) -> _Aᵍ(g, f, DiagOp(Xmap(tmUS2,1))), 
    A   = (g, f) ->   A(g, f, DiagOp(Xmap(tmUS2,1))),
    b_g = M'* MWMᵀᵍ * d, 
    b_f = 0 * d, 
    x_g = 0*d, 
    x_f = 0*d, 
)


## semilogy(reshist)
## f_cr[:] |> imag |> matshow; colorbar()
## g_cr[:] |> imag |> matshow; colorbar()
## f_cr[:] .- g_cr[:] |> imag |> matshow; colorbar()
## _Aᵍv1(A(d, qu)...)[2][:] .- qu[:] |> imag |> matshow; colorbar()
## _Aᵍv2(A(d, qu)...)[2][:] .- qu[:]  |> imag |> matshow; colorbar()
## (M*(_Aᵍv1(A(d, qu)...)[1] - d))[:] |> imag |> matshow; colorbar()
## (M*(_Aᵍv2(A(d, qu)...)[1] - d))[:] |> imag |> matshow; colorbar()


## ------ initialize f′_cr
f′_cr = Ł(ϕ_cr) * (Ð▪⁻¹ \ f_cr) 


# Now gradient moves
ϕ_cr, f_cr,  g_cr, f′_cr, reshist = let ϕ_cr=ϕ_cr, f_cr=f_cr,  g_cr=g_cr, f′_cr=f′_cr, reshist=reshist

    for otr = 1:40

        ## ------- update ϕ_cr (inputs are updated f′_cr and f_cr)
        @time gradϕ = CMBrings.∇ll_ϕf′_usingf(
            ϕ_cr, f_cr, Phi▪⁻½, EB▪⁻½; 
            data=d, Ł, Ð⁻¹=Ð▪⁻¹, M=M, B=B▪, N⁻¹=N▪⁻¹, 
            ϕ2v!, ϕ2vᴴ!, ∇!, grad_nsteps=14
        )
        ∇ϕ_cr = NΦN▪ * gradϕ 
        @time β = CMBrings.linesearch_ϕf′(
            ∇ϕ_cr, ϕ_cr, f′_cr,  Phi▪⁻½, EB▪⁻½; 
            data=d, Ł, Ð⁻¹=Ð▪⁻¹, M=M, B=B▪, N⁻¹=N▪⁻¹,
            eval_max=500, startval=0.0001, ftol_abs=100, solver=:LN_COBYLA,  
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
            ϕ_cr, f′_cr, Phi▪⁻½, EB▪⁻½; 
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
    v[2] .*= csc.(θ).^2


    ∇!_ϕ(tmp, ϕ0[:], Val(1))
    tmp .*= sin.(θ)
    ∇!_ϕ(v[1], tmp, Val(1))
    v[1] ./= sin.(θ)
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

log₊(x::T) where T = x > 0 ? log(x) : T(-Inf)

log_clip = function (x)
    lx = log₊.(x)
    finite_idx = @. isfinite(lx)
    lx[.!(finite_idx)] .= minimum(lx[finite_idx])
    lx 
end

#- 


CMBrings.fourier_power(
    f_cr; 
    θ, φ,
    x->log_clip(abs2.(x)) ,
    ## ℓs = [1000, 3000], 
    title1=L"$|P\,(\theta,\ell_\varphi)|^2$ where $P=Q+iU$, lmax_cut",
)

#-

ℓbin, f_cr_power = CMBrings.quasi_bandpowers(f_cr; θ, Δℓsph_bin = 15)
ℓbin, f_power    = CMBrings.quasi_bandpowers(qu; θ, Δℓsph_bin = 15)
figure()
semilogy(ℓbin, ℓbin.^2 .* f_cr_power)
semilogy(ℓbin, ℓbin.^2 .* f_power)


#- 
using ImageFiltering
blur     = 2
imag_fun = x -> imfilter(x, Kernel.gaussian(blur.*(1,(nφ÷2)/nθ)), "circular")
CMBrings.map_plot_QU(
    f_cr;
    θ, φ,
    #imag_fun,
    title1=L"$Q(\theta,\varphi)$ w/small Gaussian blur",
    title2=L"$U(\theta,\varphi)$ w/small Gaussian blur",
)


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





