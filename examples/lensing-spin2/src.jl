
## In progress: Test conditional simulations in gradient flows
## In progress: Test extension of masking mask

## TODO: Add full simulation to compare with Vecchia
## TODO: Try different Vecchia blocks at different ell's
## TODO: Test an Asmuthal component to the mask


# Modules
# ==============================
using LinearAlgebra
## LinearAlgebra.BLAS.set_num_threads(1)
using FFTW 
## FFTW.set_num_threads(6)

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

#- 

if isdefined(Main, :IJulia) && Main.IJulia.inited
    hide_plots = false
else 
    hide_plots = true
end
save_figures = false 


# Pixel grid
# ==============================

θ, φ, θ∂, φ∂, Ω, Δθ, nθ, nφ, freq_mult, grid_type = @sblock let 
    ## --------- hi-res
    ## φspan, freq_mult = deg2rad.((-60, 60)), 3
    ## φ, φ∂ = CC.φ_grid(;φspan, N=2048)    # N=768 or N=1536, 2048, 1024, 972,  1280
    ## type, N, θspan  = :healpix,  2048, π/2 .- deg2rad.((-41,-70)) 
    ## θ, θ∂  = CC.θ_grid(; θspan, N, type)
    ##  -------- med-res
    ## φspan, freq_mult = deg2rad.((-45, 45)), 4
    ## φ, φ∂ = CC.φ_grid(;φspan, N=1536)    # N=768 or N=1024, 972, 1536, 1280
    ## type, N, θspan  = :equiθ,  500, π/2 .- deg2rad.((-50,-65)) 
    ## θ, θ∂  = CC.θ_grid(; θspan, N, type)
    ##  -------- low-res
    φspan, freq_mult = deg2rad.((-45, 45)), 4
    φ, φ∂ = CC.φ_grid(;φspan, N=1024)    # N=768 or N=1024, 972, 1536, 1280
    type, N, θspan  = :equiθ,  300, π/2 .- deg2rad.((-57,-69)) 
    θ, θ∂  = CC.θ_grid(; θspan, N, type)

    
    nθ, nφ = length(θ), length(φ)
    Ω  = CC.counterclock_Δφ(φ∂[1], φ∂[2]) .* diff(.- cos.(θ∂))
    Δθ = diff(θ∂)

    collect(θ), φ, θ∂, φ∂, Ω, Δθ, nθ, nφ, freq_mult, type
end 


pix_diag_arcmin = CC.geoβ.(θ[2:end],θ[1:end-1],φ[1],φ[2]) .|> x->60*rad2deg(x)
@show (nθ, nφ)
@show extrema(@. rad2deg(√Ω)*60) 
@show extrema(@. rad2deg(Δθ)*60) 
@show extrema(pix_diag_arcmin) 

# Plot √Ωpix over ring θ's 

@sblock let θ, φ, Ω, Δθ, hide_plots, save_figures
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

approx_lmax += ceil(Int, approx_lmax * 0.3) # for good measure:)

ℓ, ϕϕℓ, eeℓ, bbℓ, ẽẽℓ, b̃b̃ℓ = @sblock let lmax=approx_lmax, r=0.01
    
    l = 0:lmax
    cld = camb_cls(;lmax=lmax, r)
    
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
    ϕϕl[1] =  ϕϕl[2] ### trying to fix a rank degeneracy here ...

    return l, ϕϕl, eel, bbl, ẽel, b̃bl 
end;

## semilogy(ℓ, eeℓ)
## semilogy(ℓ, bbℓ)

## can use this to check if we are getting pos definite 
## EB▫_tail = CMBrings.az_cov_blks(ℓ, eeℓ, bbℓ; θ=θ[1:150], φ);
## EB▫_tail = CMBrings.az_cov_blks(ℓ, eeℓ, bbℓ; θ=θ[end-250:end], φ, ℓrange=nφ÷2-2:nφ÷2+1);

## EB▫_tail[1] |> Hermitian |> eigen |> x->x.values
## EB▫_tail[end] |> Hermitian |> eigen |> x->x.values

## EB▫_tail[50] |> Hermitian |> eigen |> x->x.vectors[:,end] |> plot
## EB▫_tail[50] |> Hermitian |> eigen |> x->x.vectors[:,end-1] |> plot
## EB▫_tail[50] |> Hermitian |> eigen |> x->x.vectors[:,end-2] |> plot
## EB▫_tail[50] |> Hermitian |> eigen |> x->x.vectors[:,end-3] |> plot




# Mask 
# =========================================

# kron product mask
prθ, prφ  =  @sblock let rT=real(T), nθ, nφ, tmUS2
    ##
    ▮lθ, ▯lθ = 25, 40 
    ▮rθ, ▯rθ = nθ-▮lθ+1, nθ-▯lθ+1 
    ## ▮lθ, ▯lθ = 40, 70 
    ## ▮rθ, ▯rθ = nθ-10+1, nθ-20+1 
    prθ    = CMBrings.pixweight.(rT.(1:nθ); ▮l=▮lθ,    ▯l=▯lθ, ▯r=▯rθ, ▮r=▮rθ)
    ## 
    ## ▮lφ, ▯lφ = 5, 40 
    ## ▮rφ, ▯rφ = nφ-▮lφ+1, nφ-▯lφ+1 
    ## prφ    = CMBrings.pixweight.(rT.(1:nφ); ▮l=▮lφ,    ▯l=▯lφ, ▯r=▯rφ, ▮r=▮rφ)
    ## ----- alt ----- ↓↓ No azmuthal mask ↓↓
    prφ = ones(rT,nφ)
    ##
    
    prθ, prφ
end;

# Lensing mask (to keep the lense from transporting off the polar cut)
Mϕ = @sblock let rT=real(T), nθ, nφ, tmUS0, prθφ = prθ.*prφ'
    
    ▮lθ, ▯lθ = 5, 25 ### Testing !!!!!
    ▮rθ, ▯rθ = nθ-▮lθ+1, nθ-▯lθ+1 
    prθ  = CMBrings.pixweight.(rT.(1:nθ); ▮l=▮lθ,    ▯l=▯lθ, ▯r=▯rθ, ▮r=▮rθ)
    mϕx = prθ * ones(rT,nφ)'
    ## ---------- or 
    ## sqz = 6 # 8
    ## sft = 0.5
    ## mϕx = prθφ .|> x-> clamp((atan(sqz*(x-sft)) + π/2)/π, .05, .95)

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

permθ, block_sizesθ = @sblock let prθ, nθ, bsd_nθ=100 
    block_sizesθ = VF.block_split(nθ, bsd_nθ)
    ## block_sizesθ = VF.block_split(nθ, bsd_nθ) |> sort

    permθ=1:nθ
    permθ, block_sizesθ
end



# Data operators
# ============================

# ## Noise

μK_arcmin       = 1.0

N▪ = @sblock let μK_arcmin, Ω, nφ 
    σ²   = deg2rad(μK_arcmin/60)^2 # Cⁿℓ == μK_arcmin |> arcmin2radians |> abs2
    σ²_Ω = σ² ./ Ω
    Nmat = Diagonal(vcat(σ²_Ω,σ²_Ω))
    N▫   = [Nmat for ℓ = 1:nφ÷2+1]
    CircOp(N▫)
end; 

# ## Mask

M = DiagOp(Xmap(tmUS2, prθ .* prφ' ));

# ## Beam
# Conjecturing here that the arclength of the pixel diagonals 
# is what determines quality of the AzEq beam. 

pix_diag_rad = CC.geoβ.(θ[2:end], θ[1:end-1], φ[1], φ[2]) # arclength of the pixel diagonals
beamfwhm_arcmin = maximum(60 .* rad2deg.(pix_diag_rad))

## pixφside = sin.(θ) .* CC.counterclock_Δφ(φ∂[1], φ∂[2])
## pixθside = Δθ
## beamfwhm_arcmin = 2.0 * maximum(60 .* rad2deg.(vcat(pixθside, pixφside)))
##
## beamfwhm_arcmin = 1.0 * maximum(@. rad2deg(√Ω)*60)


beamℓ = @sblock let ℓ, beamfwhm_arcmin
    beamfwhm_rad = beamfwhm_arcmin |> arcmin -> deg2rad(arcmin/60)
    σ² = beamfwhm_rad^2 / 8 / log(2)
    beamℓ = @. exp( - σ²*ℓ*(ℓ+1) / 2)
end 


B▪ = @sblock let ℓ, beamℓ, block_sizesθ, permθ, θ, φ, Ω

    nθ, nφ = length(θ), length(φ)
    DΩΩ  = Diagonal(vcat(Ω, Ω))

    ## Bspin0▫½ = CMBrings.az_cov½_vecchia_blks(ℓ, beamℓ, block_sizesθ, permθ; θ, φ);
    
    Bspin0▪ = CMBrings.az_cov_vecchia_blks(
        ℓ, beamℓ, 
        block_sizesθ,  permθ; θ, φ
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



# Spin 2 signal
# =================================================

@time EB▪½ = CMBrings.az_cov½_vecchia_blks(ℓ, eeℓ, bbℓ, block_sizesθ, permθ; θ, φ) |> CircOp;
## sum(Base.summarysize, EB▪½) / 1e9 # 7.41 GB, 3.55min construction, high res
## EB▪⁻½ = map(inv, EB▪½) |> CircOp;
EB▪⁻½ = map(VF.posdef_inv, EB▪½) |> CircOp;


# EB▪½[end-5][3].data[2]
# EB▪⁻½[end-5][2].data[2]

## EB▫ = CMBrings.az_cov_blks(
##     ℓ, eeℓ, bbℓ; θ, φ, 
##    ℓrange = [1,2,3,4, nφ÷2-1, nφ÷2, nφ÷2+1]
## );

# Spin 0 signal
# =================================================

Phi▪½ = CMBrings.az_cov½_vecchia_blks(ℓ, ϕϕℓ, block_sizesθ, permθ; θ, φ) |> CircOp;
## sum(Base.summarysize, Phi▪½) / 1e9 # 1.4 GB, 2.5min construction, high res
## Phi▪⁻½ = map(inv, Phi▪½) |> CircOp;
Phi▪⁻½ = map(VF.posdef_inv, Phi▪½) |> CircOp;

# Lensing operators
# ============================

∇!,  ∇!_ϕ = CMBrings.generate_∇!∇!ϕ(θ, φ; uniformΔθ = (grid_type == :equiθ) ? true : false); 

Ł, ϕ2v!, ϕ2vᴴ!, ∇! = CMBrings.generate_lense(;
        θ, mv1x=Mϕ[:], mv2x=Mϕ[:], ∇!,  ∇!_ϕ, 
        nsteps_lensing=14
);

# Mixflow operator
# ============================

nnℓ = deg2rad(μK_arcmin/60)^2 # Cⁿℓ == μK_arcmin |> arcmin2radians |> abs2

Ð▪⁻¹ = CMBrings.az_cov½_vecchia_blks(
   ℓ, (@. eeℓ/(ẽẽℓ+2nnℓ)), (@. bbℓ/(b̃b̃ℓ+2nnℓ)),  
   block_sizesθ,  permθ; θ, φ
) |> CircOp;

# simulation
# ==============================

ϕ = Phi▪½ * Xmap(tmUS0,randn(Float64,nθ,nφ))

qu = EB▪½ * Xmap(tmUS2,randn(ComplexF64,nθ,nφ))

no = map(N▪, Xmap(tmUS2,randn(ComplexF64,nθ,nφ))) do Σ,v
    sqrt(Σ)*v
end 

d = M * (B▪ * Ł(ϕ) * qu + no) |> Xfourier;

#-

## d[:] |> real |> matshow; colorbar()
## d[:] |> imag |> matshow; colorbar()
## ϕ[:] |> matshow; colorbar()
## qu[:] |> real |> matshow; colorbar()
## qu[:] |> imag |> matshow; colorbar()
## (B▪ * B▪ * B▪ * B▪ * B▪ * no)[:] |> real |> matshow; colorbar()
## (B▪ * B▪ * B▪ * B▪ * B▪ * no)[:] |> imag |> matshow; colorbar()


# Initalize opps for ϕ gradient
# ==============================================

N▪⁻¹ = map(Nℓ->Diagonal(1 ./ diag(Nℓ)), N▪.Σ) |> CircOp;

import CMBflat
import Dierckx

N0ℓ, NΦNℓ = @sblock let pix_side_rad = mean(@. √Ω), n_iter=5, ℓ, eeℓ, bbℓ, ϕϕℓ, beamℓ, nnℓ=fill(nnℓ,length(ℓ)) 
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
    spline_k4n0ck = Dierckx.Spline1D(
        vcat(2,k[3:end]), vcat(k4n0ck[3], k4n0ck[3:end])
        ; k=1, bc="zero",
    )
    N0ℓ = spline_k4n0ck.(ℓ) ./ ℓ.^4
    N0ℓ[real.(N0ℓ) .<= 0] .= Inf 
    N0ℓ[isnan.(N0ℓ)]      .= Inf 
    NΦNℓ = @. inv(inv(N0ℓ) + inv(ϕϕℓ))
    N0ℓ, NΦNℓ
end;

NΦN▪ = CMBrings.az_cov½_vecchia_blks(
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
    prθM = M[:][:,end÷2]
    MWMᵀ_pxl = prθW .* abs2.(prθM) .* ones(1,nφ)
    DiagOp(Xmap(tmUS2, pinv.(MWMᵀ_pxl)))
end;


@time _A₁₁ᵍ▪, _A₂₂_A₂₁A₁₁ᵍA₁₂_ᵍ▪ = @sblock let B▪, EB▪½,  N▪⁺ᵍ, W▪, M, MWMᵀᵍ, block_sizesθ, nθ = length(θ)
    Mθ     = M[:][:,end÷2] |> x->vcat(x,x)
    MWMᵀᵍθ = MWMᵀᵍ[:][:,end÷2] |> x->vcat(x,x)
    
    _A₁₁ᵍ▪ = map(W▪, N▪⁺ᵍ) do W, iN
        Diagonal(pinv.(Mθ .* MWMᵀᵍθ .* conj.(Mθ) .+ diag(iN)))
    end |> CircOp

    _A₂₂_A₂₁A₁₁ᵍA₁₂_ᵍ▪ = map(_A₁₁ᵍ▪, B▪, N▪⁺ᵍ, EB▪½) do iA, Bl, iN, Σ½
        # iA,  Bl, iN, Σ½ = _A₁₁ᵍ▪[2], B▪[2], N▪⁺ᵍ[2], EB▪½[2]
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

b_g, b_f, A, _Aᵍ = let L=DiagOp(Xmap(tmUS2,1)), # d, N▪⁺ᵍ, MWMᵀᵍ, EB▪⁻½, B▪, M, _A₁₁ᵍ▪, _A₂₂_A₂₁A₁₁ᵍA₁₂_ᵍ▪
    
    b_g    = M'* MWMᵀᵍ * d 
    b_f    = 0 * d 
    A = function (g, f)
        Afg_g = (M'*MWMᵀᵍ*M*g + N▪⁺ᵍ*g) - (N▪⁺ᵍ*B▪*L*f)
        Afg_f = - (L'*B▪'*N▪⁺ᵍ*g) + (L'*B▪'*N▪⁺ᵍ*B▪*L*f + EB▪⁻½'*EB▪⁻½*f)
        Afg_g, Afg_f
    end
    _Aᵍ = function (g, f)
        f1 = _A₂₂_A₂₁A₁₁ᵍA₁₂_ᵍ▪ * (L'*B▪'*N▪⁺ᵍ*_A₁₁ᵍ▪*g + f)
        _A₁₁ᵍ▪*(g + N▪⁺ᵍ*B▪*L*f1), f1
    end
    #### these are for conditional simulations
    ## γ₁ = sqrt(MWMᵀᵍ) * Xmap(tmUS2,randn(ComplexF64,nθ,nφ))
    ## γ₂ = map((Σ,v)->sqrt(Σ)*v, N▪⁺ᵍ, Xmap(tmUS2,randn(ComplexF64,nθ,nφ)))
    ## γ₃ = EB▪⁻½' * Xmap(tmUS2,randn(ComplexF64,nθ,nφ))
    ## b_g += Xfourier(M'*γ₁ + γ₂)
    ## b_f += Xfourier(γ₃ - Ł(ϕ_cr)'*B▪'*γ₂) 

    b_g, b_f, A, _Aᵍ
end;




# WF for conditional expected value
## -----------------------
g_cr, f_cr, reshist = CMBrings.pcg_coupled(;
    nsteps=50, 
    rel_tol=1e-15, 
    _Aᵍ, A, 
    b_g, b_f, 
    x_g=0*d, x_f=0*d, 
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

    for otr = 1:7

        ## ------- update ϕ_cr (inputs are updated f′_cr and f_cr)
        gradϕ = CMBrings.∇ll_ϕf′_usingf(
            ϕ_cr, f_cr, Phi▪⁻½, EB▪⁻½; 
            data=d, Ł, Ð⁻¹=Ð▪⁻¹, M=M, B=B▪, N⁻¹=N▪⁻¹, 
            ϕ2v!, ϕ2vᴴ!, ∇!, grad_nsteps=14
        )
        ∇ϕ_cr = NΦN▪ * gradϕ 
        @time β = CMBrings.linesearch_ϕf′(
            ∇ϕ_cr, ϕ_cr, f′_cr,  Phi▪⁻½, EB▪⁻½; 
            data=d, Ł, Ð⁻¹=Ð▪⁻¹, M=M, B=B▪, N⁻¹=N▪⁻¹,
            eval_max=500, startval=0.0001, ftol_abs=20, solver=:LN_COBYLA,  
        )
        @show β
        ϕ_cr += β * ∇ϕ_cr

        ## ------ update _Aᵍ, b_g, b_f, A for WF operators and preconditioner
        b_g, b_f, A, _Aᵍ = let L=Ł(ϕ_cr), # d, N▪⁺ᵍ, MWMᵀᵍ, EB▪⁻½, B▪, M, _A₁₁ᵍ▪, _A₂₂_A₂₁A₁₁ᵍA₁₂_ᵍ▪
            
            b_g    = M'* MWMᵀᵍ * d 
            b_f    = 0 * d 
            A = function (g, f)
                Afg_g = (M'*MWMᵀᵍ*M*g + N▪⁺ᵍ*g) - (N▪⁺ᵍ*B▪*L*f)
                Afg_f = - (L'*B▪'*N▪⁺ᵍ*g) + (L'*B▪'*N▪⁺ᵍ*B▪*L*f + EB▪⁻½'*EB▪⁻½*f)
                Afg_g, Afg_f
            end
            _Aᵍ = function (g, f)
                f1 = _A₂₂_A₂₁A₁₁ᵍA₁₂_ᵍ▪ * (L'*B▪'*N▪⁺ᵍ*_A₁₁ᵍ▪*g + f)
                _A₁₁ᵍ▪*(g + N▪⁺ᵍ*B▪*L*f1), f1
            end
            #### these are for conditional simulations
            ## γ₁ = sqrt(MWMᵀᵍ) * Xmap(tmUS2,randn(ComplexF64,nθ,nφ))
            ## γ₂ = map((Σ,v)->sqrt(Σ)*v, N▪⁺ᵍ, Xmap(tmUS2,randn(ComplexF64,nθ,nφ)))
            ## γ₃ = EB▪⁻½' * Xmap(tmUS2,randn(ComplexF64,nθ,nφ))
            ## b_g += Xfourier(M'*γ₁ + γ₂)
            ## b_f += Xfourier(γ₃ - Ł(ϕ_cr)'*B▪'*γ₂) 

            b_g, b_f, A, _Aᵍ
        end;

        # ------ update f_cr
        g_cr, f_cr, reshist = CMBrings.pcg_coupled(;
            nsteps=50, 
            rel_tol=1e-15, 
            _Aᵍ, A, 
            b_g, b_f, 
            x_g=g_cr, x_f=f_cr,  #### Try turning this back on to see if it helps 
            ## x_g=0*g_cr, x_f=0*f_cr, #### Testing!!! 
        )
        @show reshist

        ## ------ update f′_cr
        f′_cr = Ł(ϕ_cr) * (Ð▪⁻¹ \ f_cr) 

        ## ------ show stats
        @show CMBrings.ll_ϕf′(
            ϕ_cr, f′_cr, Phi▪⁻½, EB▪⁻½; 
            data=d, Ł, Ð⁻¹=Ð▪⁻¹, M, B=B▪, N⁻¹=N▪⁻¹
        )
        
    end # end for-loop

    ϕ_cr, f_cr,  g_cr, f′_cr, reshist
end # end let



#-

## ϕ[:] |> matshow; colorbar()
## ϕ_cr[:] |> matshow; colorbar()
## f_cr[:] |> real |> matshow; colorbar()
## qu[:] |> real |> matshow; colorbar()
## f_cr[:] |> imag |> matshow; colorbar()
## qu[:] |> imag |> matshow; colorbar()
## f_cr[:] .- qu[:] |> real |> matshow; colorbar()

#-

@sblock let ϕtru = ϕ, ϕest = ϕ_cr, ϕ2v!, φ, θ, hide_plots, save_figures
    hide_plots && return
    viz = function (ϕ0)
        v = (deepcopy(ϕ0[:]), deepcopy(ϕ0[:]))
        ϕ2v!(v, ϕ0[:])
        v 
    end
    imgs = Dict(1=>viz(ϕtru)[1], 2=>viz(ϕest)[1])
    txt  = Dict(1=>L"true $\nabla_\theta \phi$", 2=>L"est $\nabla_\theta \phi$")
    
    fig,ax = subplots(2, figsize=(9,8))
    imgs[1] |> imshow(-,fig,ax[1])
    imgs[2] |> imshow(-,fig,ax[2])
    fig.suptitle(L"true (top) vrs est (bottom) $\nabla_\theta \phi$")
    ## fig, ax = CMBrings.diskplot(
    ##     imgs, CC.in_negπ_π.(φ)', π.-θ; txt=txt, fontsize=14
    ## )

    save_figures && savefig("figure$(fig.number).png", dpi=250, bbox_inches="tight")
    return nothing
end




#-

@sblock let ϕtru = ϕ, ϕest = ϕ_cr, ϕ2v!, φ, θ, hide_plots, save_figures
    hide_plots && return
    viz = function (ϕ0)
        v = (deepcopy(ϕ0[:]), deepcopy(ϕ0[:]))
        ϕ2v!(v, ϕ0[:])
        v 
    end
    imgs = Dict(1=>viz(ϕtru)[2], 2=>viz(ϕest)[2])
    txt  = Dict(1=>L"true $\nabla_\varphi \phi$", 2=>L"est $\nabla_\varphi \phi$")
    
    fig,ax = subplots(2, figsize=(9,8))
    imgs[1] |> imshow(-,fig,ax[1])
    imgs[2] |> imshow(-,fig,ax[2])
    fig.suptitle(L"true (top) vrs est (bottom) $\nabla_\varphi \phi$")
    ## fig, ax = CMBrings.diskplot(
    ##     imgs, CC.in_negπ_π.(φ)', π.-θ; txt=txt, fontsize=14
    ##)

    save_figures && savefig("figure$(fig.number).png", dpi=250, bbox_inches="tight")
    return nothing
end


#- 


@sblock let ϕtru = ϕ, ϕest = ϕ_cr, ϕ2v!, φ, θ, hide_plots, save_figures
    hide_plots && return
    viz = function (ϕ0)
        v = (deepcopy(ϕ0[:]), deepcopy(ϕ0[:]))
        ϕ2v!(v, ϕ0[:])
        v 
    end
    imgs = Dict(1=>ϕtru[:], 2=>ϕest[:])
    txt  = Dict(1=>L"true $\phi$", 2=>L"est $\phi$")
    
    fig,ax = subplots(2, figsize=(9,8))
    imgs[1] |> imshow(-,fig,ax[1])
    imgs[2] |> imshow(-,fig,ax[2])
    fig.suptitle(L"true (top) vrs est (bottom) $\phi$")
    ## fig, ax = CMBrings.diskplot(
    ##     imgs, CC.in_negπ_π.(φ)', π.-θ; txt=txt, fontsize=14
    ## )

    save_figures && savefig("figure$(fig.number).png", dpi=250, bbox_inches="tight")

    return nothing
end


#-

@sblock let f_cr, φ, θ, hide_plots, save_figures

    hide_plots && return

    imgs = Dict(1=>real(f_cr[:]), 2=>imag(f_cr[:]))
    txt  = Dict(
        1=>"unlensed Q est",     2=>"unlensed U est",
    )

    fig,ax = subplots(2, figsize=(9,8))
    imgs[1] |> imshow(-,fig,ax[1])
    imgs[2] |> imshow(-,fig,ax[2])
    fig.suptitle("unlensed Q (top) and U (bottom)")
    ## fig, ax = CMBrings.diskplot(
    ##     imgs, CC.in_negπ_π.(φ)', π.-θ; txt=txt, fontsize=14
    ## )

    save_figures && savefig("figure$(fig.number).png", dpi=250, bbox_inches="tight")

    return nothing

end



#-

@sblock let f_cr, qu, φ, θ, hide_plots, save_figures

    hide_plots && return

    imgs = Dict(1=>real(f_cr[:] .- qu[:]), 2=>imag(f_cr[:] .- qu[:]))
    txt  = Dict(
        1=>"unlensed Q err",     2=>"unlensed U err",
    )

    fig,ax = subplots(2, figsize=(9,8))
    imgs[1] |> imshow(-,fig,ax[1])
    imgs[2] |> imshow(-,fig,ax[2])
    fig.suptitle("unlensed err Q (top) and U (bottom)")
    ## fig, ax = CMBrings.diskplot(
    ##     imgs, CC.in_negπ_π.(φ)', π.-θ; txt=txt, fontsize=14
    ## )

    save_figures && savefig("figure$(fig.number).png", dpi=250, bbox_inches="tight")

    return nothing

end



#-

@sblock let f_cr, ϕ_cr, ϕ, qu, Ł, M, φ, θ, hide_plots, save_figures

    hide_plots && return

    L_cr = Ł(ϕ_cr)
    L = Ł(ϕ)
    lnf_cr = M*L_cr*f_cr
    lnf = M*L*qu

    imgs = Dict(1=>real(lnf_cr[:] .- lnf[:]), 2=>imag(lnf_cr[:] .- lnf[:]))
    txt  = Dict(
        1=>"lensed Q err (masked)",     2=>"lensed U err (masked)",
    )

    fig,ax = subplots(2, figsize=(9,8))
    imgs[1] |> imshow(-,fig,ax[1])
    imgs[2] |> imshow(-,fig,ax[2])
    fig.suptitle("unlensed err Q (top) and U (bottom)")
    ## fig, ax = CMBrings.diskplot(
    ##     imgs, CC.in_negπ_π.(φ)', π.-θ; txt=txt, fontsize=14
    ## )

    save_figures && savefig("figure$(fig.number).png", dpi=250, bbox_inches="tight")

    return nothing

end

