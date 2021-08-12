## get lensing-spin2 example up and running

## ▪ == "\smblksquare" or  "\vrectangleblack"


# Modules
# ==============================
using LinearAlgebra
using FFTW 
FFTW.set_num_threads(Threads.nthreads())

using XFields
using CMBrings
using CMBsphere     
using FieldLensing 
using Spectra: camb_cls
using CMBflat: PrQr # Eventually remove this

using  FFTransforms
import FFTransforms as FT
import CirculantCov as CC

using DelimitedFiles
using LBblocks: @sblock
using PyPlot
import Dierckx 
import NLopt
using BenchmarkTools
using ProgressMeter

#- 

if isdefined(Main, :IJulia) && Main.IJulia.inited
    hide_plots = false
else 
    hide_plots = true
end

hide_plots = false

# Pixel grid
# ==============================

θ, φ, Ω, Δθ, nθ, nφ, freq_mult, grid_type = @sblock let 

    freq_mult = 4 
    φspan     = (0, 2π/freq_mult)
    φ, φ∂     = CC.φ_grid(;φspan, N=768) # N=768 or N=1024

    type, N, θspan  = :equiθ,    250,  (2.6,2.8)   # (2.5,2.7)
    ## type, N, θspan  = :equicosθ, 250,  (2.6,2.8) 
    ## type, N, θspan  = :healpix,  1200, (2.6,2.8) 
    θ, θ∂ = CC.θ_grid(; θspan, N, type)

    nθ, nφ = length(θ), length(φ)
    Ω  = CC.counterclock_Δφ(φ∂[1], φ∂[2]) .* diff(.- cos.(θ∂))
    Δθ = diff(θ∂)

    collect(θ), φ, Ω, Δθ, nθ, nφ, freq_mult, type
end 

@show (nθ, nφ)

@show extrema(@. rad2deg(√Ω)*60) 

# Plot √Ωpix over ring θ's 

@sblock let θ, φ, Ω, Δθ, hide_plots
    hide_plots && return
    fig,ax = subplots(1)
    ax.plot(θ, (@. rad2deg(√Ω)*60), label="sqrt pixel area (arcmin)")
    ax.plot(θ, (@. rad2deg(Δθ)*60), label="Δθ (arcmin)")
    ax.set_xlabel(L"polar coordinate $\theta$")
    ax.legend()
    return nothing
end


# Transformations, Mask and CMBring observation region
# ==============================


tmUS2, tmUS0 = @sblock let nθ, nφ, freq_mult

    T = Float64

    ## tmUS2  = FT.:⊗(FT.𝕀(nθ), FT.𝕎(Complex{T}, nφ, 2π/freq_mult))
    ## tmUS2 *= FT.unitary_scale(tmUS2) 
    ## tmUS0  = FT.:⊗(FT.𝕀(nθ), FT.𝕎(T, nφ, 2π/freq_mult))
    ## tmUS0 *= FT.unitary_scale(tmUS0) 

    tmUS2 = 𝕀(nθ) ⊗ 𝕌(Complex{T}, nφ, 2π/freq_mult)
    tmUS0 = 𝕀(nθ) ⊗ 𝕌(T, nφ, 2π/freq_mult)

    return tmUS2, tmUS0
end;

#-

data_msk = @sblock let θ, φ
    
    pr_msk  = readdlm(joinpath(CMBrings.module_dir,"examples/artifacts/FastTransform_mask_nθ3072_nφ4095.csv"), ',', Bool)    
    ## pr_msk  = readdlm(joinpath(CMBrings.module_dir,"examples/artifacts/FastTransform_mask_mid2pole_nθ2560_nφ3071.csv"), ',', Bool)    
    ## pr_msk  = readdlm(joinpath(CMBrings.module_dir,"examples/artifacts/FastTransform_mask_spole_nθ3072_nφ4095.csv"), ',', Bool)    
    nθ_msk, nφ_msk = size(pr_msk)
    θ_msk = π*(0.5:nθ_msk-0.5)/nθ_msk |> collect
    φ_msk = 2π*(0:nφ_msk-1)/nφ_msk    |> collect
    spline_mask = Dierckx.Spline2D(θ_msk, φ_msk, pr_msk, kx=1, ky=1, s=0.0)

    data_msk = spline_mask.(θ, φ') .> 0
    data_msk[1:5,:] .= 0
    data_msk[end - 5 + 1:end,:] .= 0

    ## data_msk[:,1:15] .= 0
    ## data_msk[:, end - 15 + 1:end] .= 0

    return data_msk
end;

#- 


Pr, Qr = @sblock let tmUS2, θ, φ, data_msk, QP_bdry=1e-5, fwhmθ′=25, fwhmφ′=200
    tmFlat = FT.𝕎(real(eltype_in(tmUS2)), size(data_msk), (θ[end]-θ[1], φ[end]-φ[1]))
    pr0x, qr0x = PrQr(tmFlat, data_msk, fwhmθ′, fwhmφ′, QP_bdry)
    pr0 = Xmap(tmUS2, pr0x)
    qr0 = Xmap(tmUS2, qr0x)
    DiagOp(pr0), DiagOp(qr0)
end;

# Pr[:] .|> real |> matshow; colorbar()
# Qr[:] .|> real |> matshow; colorbar()

# Localize lensing vector field to data mask.

Mϕ = @sblock let tmUS0, Pr

    ## sqz = 7 # increase sqz to get shaper transition
    ## mϕx = real.(Pr[:]) .|> x-> atan(sqz*(x-1/2))
    ## ---------- or 
    sqz = 8
    sft = 0.5
    mϕx = real.(Pr[:]) .|> x-> clamp((atan(sqz*(x-sft)) + π/2)/π, .05, .95)

    ## make sure it hits zero and 1
    mϕx .-= minimum(mϕx)
    mϕx ./= maximum(mϕx)
    Mϕ    = DiagOp(Xmap(tmUS0, mϕx))
    Mϕ
end;
#=
 Mϕ[:] .|> real |> matshow; colorbar()
 Pr[:] .|> real |> matshow; colorbar()
=# 

# Azimuthal ring mask

@sblock let ma=real.(Pr[:]), dma=data_msk, φ, θ, hide_plots
    hide_plots && return
    imgs = Dict(1=>dma, 2=>ma)
    txt  = Dict(1=>"pre-smoothed mask", 2=>"mask")
    fig, ax = CMBrings.diskplot(
        imgs, φ', π.-θ; 
        txt=txt, 
        figsize=(10,8), nrows=1, fontsize=14
    )
    return nothing
end


# Spectral densities and operators
# ==============================

μK_arcmin       = 2.5
## beamfwhm_arcmin = 0.9maximum(@. rad2deg(√Ω)*60)
beamfwhm_arcmin = mean(@. rad2deg(√Ω)*60)

ℓ, eeℓ, bbℓ, ϕϕℓ, beamℓ, ẽẽℓ, b̃b̃ℓ = @sblock let beamfwhm_arcmin
    
    r  = 0.01

    lmax = 11000
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
    ϕϕl[1] =  0

    beamfwhm_rad = beamfwhm_arcmin |> arcmin -> deg2rad(arcmin/60)
    σ² = beamfwhm_rad^2 / 8 / log(2)
    beaml = @. exp( - σ²*l*(l+1) / 2)

    return l, eel, bbl, ϕϕl, beaml, ẽel, b̃bl 
end;

# Uncertainty for ϕ based on iterative quadratic estimate
## TODO: needs fixing up ...

import CMBflat

N0ℓ, NΦNℓ =  @sblock let n_iter=5, ℓ, eeℓ, bbℓ, ϕϕℓ, beamℓ, nnℓ = deg2rad(μK_arcmin/60)^2 .+ zero(ℓ)

    ## T_fld = Float32
    T_fld = Float64
    
    nθ, nφ  = 512, 512   
    periodθ = T_fld(nθ * deg2rad(3.5 / 60))
    periodφ = T_fld(nφ * deg2rad(3.5 / 60))
    tm    = FT.𝕎(T_fld, (nθ, nφ), (periodθ, periodφ))
    tmΦ   = FT.ordinary_scale(tm) * tm
    tmEB  = CMBflat.QU2EB(T_fld, (nθ, nφ), (periodθ, periodφ))

    Idx  = round.(Int,FT.wavenum(tmΦ)) .+ 1
    ecl  = map(i -> getindex(eeℓ, i), Idx)
    bcl  = map(i -> getindex(bbℓ, i), Idx)
    ϕcl  = map(i -> getindex(ϕϕℓ, i), Idx)
    ncl  = map(i -> getindex(nnℓ, i), Idx)
    bmcl = map(i -> getindex(beamℓ, i), Idx)

    EBcov = DiagOp(Xfourier(tmEB, cat(ecl,bcl;dims=3))) 
    Ncov  = DiagOp(Xfourier(tmEB, cat(ncl,ncl;dims=3))) 
    Bm    = DiagOp(Xfourier(tmEB, cat(bmcl,bmcl;dims=3)))
    Φcov  = DiagOp(Xfourier(tmΦ, ϕcl))

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


# Ring Ops 
# ==============================


EB▪, Phi▪, Beam▪, N▪, Ð▪⁻¹, NΦN▪  = @sblock let ℓ, eeℓ, bbℓ, ẽẽℓ, b̃b̃ℓ, ϕϕℓ, beamℓ, NΦNℓ, μK_arcmin, θ, φ, freq_mult, Ω 

    nθ, nφ = length(θ), length(φ)

    # create the structs for computing the cov diag blocks
    ΓC_EB  = CC.ΓCθ₁θ₂φ₁φ⃗_CMBpol(ℓ, eeℓ, bbℓ; ngrid=50_000)
    ΓC_ẼB̃  = CC.ΓCθ₁θ₂φ₁φ⃗_CMBpol(ℓ, ẽẽℓ, b̃b̃ℓ; ngrid=50_000)
    Γ_Phi  = CC.Γθ₁θ₂φ₁φ⃗_Iso(ℓ, ϕϕℓ;      ngrid=50_000)
    Γ_NΦN  = CC.Γθ₁θ₂φ₁φ⃗_Iso(ℓ, NΦNℓ;     ngrid=50_000)
    Γ_Beam = CC.Γθ₁θ₂φ₁φ⃗_Iso(ℓ, beamℓ;    ngrid=50_000)

    # create the storage for the cov diag blocks
    T     = ComplexF64 # ComplexF32
    rT    = real(T)
    EB▫   = Matrix{T}[zeros(T,2nθ,2nθ)   for ℓ = 1:nφ÷2+1]
    ẼB̃▫   = Matrix{T}[zeros(T,2nθ,2nθ)   for ℓ = 1:nφ÷2+1]
    Phi▫  = Matrix{rT}[zeros(rT,nθ,nθ)   for ℓ = 1:nφ÷2+1]
    NΦN▫  = Matrix{rT}[zeros(rT,nθ,nθ)   for ℓ = 1:nφ÷2+1]
    Beam▫ = Matrix{rT}[zeros(rT,2nθ,2nθ) for ℓ = 1:nφ÷2+1]
    
    # FFTW plan and pre-compute storage
    ptmW    = FFTW.plan_fft(Vector{ComplexF64}(undef, nφ))

    prgss = Progress(nθ, 1, "EB▪, Phi▪, Beam▪, N▪, Ð▪⁻¹, NΦN▪ ")
    for k = 1:nθ
        for j = 1:nθ

            Phiγⱼₖℓ⃗  = CC.γθ₁θ₂ℓ⃗(θ[j], θ[k], φ, Γ_Phi,  ptmW)
            NΦNγⱼₖℓ⃗  = CC.γθ₁θ₂ℓ⃗(θ[j], θ[k], φ, Γ_NΦN,  ptmW)
            Beamγⱼₖℓ⃗ = CC.γθ₁θ₂ℓ⃗(θ[j], θ[k], φ, Γ_Beam, ptmW)
            EBγⱼₖℓ⃗, EBξⱼₖℓ⃗ = CC.γθ₁θ₂ℓ⃗_ξθ₁θ₂ℓ⃗(θ[j], θ[k], φ, ΓC_EB..., ptmW)
            ẼB̃γⱼₖℓ⃗, ẼB̃ξⱼₖℓ⃗ = CC.γθ₁θ₂ℓ⃗_ξθ₁θ₂ℓ⃗(θ[j], θ[k], φ, ΓC_ẼB̃..., ptmW)

            for ℓ = 1:nφ÷2+1
                Phi▫[ℓ][j,k] = real(Phiγⱼₖℓ⃗[ℓ])
                NΦN▫[ℓ][j,k] = real(NΦNγⱼₖℓ⃗[ℓ])

                Jℓ = CC.Jperm(ℓ, nφ)
                
                EB▫[ℓ][j,   k   ]   = EBγⱼₖℓ⃗[ℓ]
                EB▫[ℓ][j,   k+nθ]   = EBξⱼₖℓ⃗[ℓ]
                EB▫[ℓ][j+nθ,k   ]   = conj(EBξⱼₖℓ⃗[Jℓ])
                EB▫[ℓ][j+nθ,k+nθ]   = conj(EBγⱼₖℓ⃗[Jℓ])

                ẼB̃▫[ℓ][j,   k   ]   = ẼB̃γⱼₖℓ⃗[ℓ]
                ẼB̃▫[ℓ][j,   k+nθ]   = ẼB̃ξⱼₖℓ⃗[ℓ]
                ẼB̃▫[ℓ][j+nθ,k   ]   = conj(ẼB̃ξⱼₖℓ⃗[Jℓ])
                ẼB̃▫[ℓ][j+nθ,k+nθ]   = conj(ẼB̃γⱼₖℓ⃗[Jℓ])

                Beam▫[ℓ][j, k   ]   = real(Beamγⱼₖℓ⃗[ℓ])  * Ω[k]
                Beam▫[ℓ][j+nθ,k+nθ] = real(Beamγⱼₖℓ⃗[Jℓ]) * Ω[k]
            end

        end
        next!(prgss)
    end

    @show Base.summarysize(EB▫) / 1e9
    @show Base.summarysize(Phi▫)  / 1e9
    @show Base.summarysize(Beam▫)  / 1e9

    EB▪   = CircOp(map(x->Hermitian(x,:L),EB▫))
    Phi▪  = CircOp(map(x->Symmetric(x,:L),Phi▫))
    NΦN▪  = CircOp(map(x->Symmetric(x,:L),NΦN▫))
    ## Beam▪ = CircOp(map(x->Symmetric(x,:L),Beam▫))
    ## The beam shouldn't be symmetric ... unless equicosθ grid
    Beam▪ = CircOp(Beam▫)

    μKᵒn = μK_arcmin / 60
    σ²   = deg2rad(μKᵒn)^2
    σ²_Ω = T.(σ² ./ Ω)
    Nmat = Diagonal(vcat(σ²_Ω,σ²_Ω))
    N▪   = CircOp([Nmat for ℓ = 1:nφ÷2+1])

    Ð▪⁻¹ =  CircOp(@. sqrt(Hermitian(EB▪,:L)) / sqrt(Hermitian(ẼB̃▫ + 4*N▪,:L)))

    return EB▪, Phi▪, Beam▪, N▪, Ð▪⁻¹, NΦN▪
end;

# 7mins vrs 4mins, slower using CC.γθ₁θ₂ℓ⃗_ξθ₁θ₂ℓ⃗ ...

## EB▪.Σ[2] == EB▪[2]
## EB▪.Σ[2]' == (EB▪')[2]
## (Beam▪')[2] == Beam▪[2]'



# Preconditioner
@time Precon▪⁻¹ = CircOp(@. Hermitian(pinv(Beam▪ * EB▪ * Beam▪' + N▪),:L));


# Gradients Set sparse increment matrices for non-FFT lensing
# ==================================================

## ∇!,  ∇!_ϕ = generate_∇!∇!ϕ(θ, φ;uniformΔθ=true) 
∇!,  ∇!_ϕ = CMBrings.generate_∇!∇!ϕ(θ, φ; uniformΔθ = (grid_type == :equiθ) ? true : false) 

Ł, ϕ2v!, ϕ2vᴴ!, ∇! = CMBrings.generate_lense(;
        θ, mv1x=Mϕ[:], mv2x=Mϕ[:], ∇!,  ∇!_ϕ, 
        nsteps_lensing=14
);

# simulation
# ==============================

@time ϕ = map(Phi▪, Xmap(tmUS0,randn(Float64,nθ,nφ))) do Σ,v
## @time ϕ = map(Phi▪, Xfourier(Xmap(tmUS0,randn(Float64,nθ,nφ)))) do Σ,v
    ## sqrt(Σ)*v
    Matrix(cholesky(Σ).L)*v
end 

@time qu = map(EB▪, Xmap(tmUS2,randn(ComplexF64,nθ,nφ))) do Σ,v
## @time qu = map(EB▪, Xfourier(Xmap(tmUS2,randn(ComplexF64,nθ,nφ)))) do Σ,v
    ## sqrt(Σ)*v
    Matrix(cholesky(Σ).L)*v
end 

@time no = map(N▪, Xmap(tmUS2,randn(ComplexF64,nθ,nφ))) do Σ,v
    ## sqrt(Σ)*v
    Matrix(cholesky(Σ).L)*v
end 

d = Pr * (Beam▪ * Ł(ϕ) * qu + no)


#=

sum(abs2, Xfourier(Xmap(qu))[!] .- qu[!])
sum(abs2, Xfourier(Xmap(qu))[:] .- qu[:])
sum(abs2, Xmap(Xfourier(qu))[!] .- qu[!])
sum(abs2, Xmap(Xfourier(qu))[:] .- qu[:])

=#

#= β
lnqu = Ł(ϕ) * qu

@benchmark $(Ł(ϕ)) * qu

fig, ax = subplots(2)
d[:] .|> real |> imshow(-, fig, ax[1]) 
d[:] .|> imag |> imshow(-, fig, ax[2]) 

fig, ax = subplots(2)
qu[:] .|> real |> imshow(-, fig, ax[1]) 
qu[:] .|> imag |> imshow(-, fig, ax[2]) 

fig, ax = subplots(2)
lnqu[:]  .|> real |> imshow(-, fig, ax[1]) 
lnqu[:]  .|> imag |> imshow(-, fig, ax[2]) 


fig, ax = subplots(2)
(lnqu-qu)[:] .|> real |> imshow(-, fig, ax[1]) 
(lnqu-qu)[:] .|> imag |> imshow(-, fig, ax[2]) 
=#



# Now do some iterations ...
# ==============================

## ------ initalize 
gwf  = 0*d 
ϕ_cr = 0*ϕ

## special for this noise
N▪⁻¹ = map(Nℓ->diagm(1 ./ diag(Nℓ)), N▪.Σ) |> CircOp

## @showprogress for otr = 1:30
@showprogress for otr = 1:10
    global f_cr, gwf, hst
    global f′_cr, ϕ_cr, ∇ϕ_cr

    ## ------ update field
    @time f_cr, gwf, hst = CMBrings.update_f(
        (otr==1) ? DiagOp(Xmap(tmUS2,1)) : Ł(ϕ_cr), # slot for Łϕ
        EB▪; 
        data=Xfourier(d),
        Pr, Qr, 
        Bm=Beam▪, No=N▪, Pc⁻¹=Precon▪⁻¹,
        ginit=Xfourier(gwf),
        pcg_nsteps = (otr==1) ? 400 : 300, 
        pcg_rel_tol=1e-10
    );
    @show hst[end]
    f′_cr =  Ł(ϕ_cr) * (Ð▪⁻¹ \ f_cr) 
    @show CMBrings.ll_ϕf′(ϕ_cr, f′_cr, Phi▪, EB▪; data=d, Ł, Ð⁻¹=Ð▪⁻¹, Pr, Beam_ring=Beam▪, Noise_ring⁻¹=N▪⁻¹)
    
    ## ------ ϕ gradient
    ## @time gradϕ = CMBrings.∇ll_ϕf′(ϕ_cr, f′_cr, Phi▪, EB▪; data=d, Ł, Ð⁻¹=Ð▪⁻¹, Pr, Beam_ring=Beam▪, Noise_ring⁻¹=N▪⁻¹, ϕ2v!, ϕ2vᴴ!, ∇!, grad_nsteps=11)
    @time gradϕ = CMBrings.∇ll_ϕf′_usingf(ϕ_cr, f_cr, Phi▪, EB▪; data=d, Ł, Ð⁻¹=Ð▪⁻¹, Pr, Beam_ring=Beam▪, Noise_ring⁻¹=N▪⁻¹, ϕ2v!, ϕ2vᴴ!, ∇!, grad_nsteps=14)
    @time ∇ϕ_cr = NΦN▪ * gradϕ 
        
    ## ------ linesearch 
    @time β = CMBrings.linesearch_ϕf′(
        ∇ϕ_cr, ϕ_cr, f′_cr, Phi▪, EB▪; 
        data = d, Ł, Ð⁻¹=Ð▪⁻¹, Pr, Beam_ring=Beam▪, Noise_ring⁻¹=N▪⁻¹,
        eval_max = 200, startval = 0.001, ftol_abs = 20, solver = :LN_COBYLA,  
        ## eval_max = 250, startval = 0.001, ftol_abs = 1, solver = :LN_COBYLA,  
    )
    @show β

    ## ------ update ϕ_cr
    ϕ_cr += β * ∇ϕ_cr
end


#=
fig, ax = subplots(2)
ϕ_cr[:]  |> imshow(-, fig, ax[1]) 
ϕ[:]     |> imshow(-, fig, ax[2]) 

fig, ax = subplots(2)
qu[:] .|> real |> imshow(-, fig, ax[1]) 
qu[:] .|> imag |> imshow(-, fig, ax[2]) 
=#


#-

@sblock let ϕtru = ϕ, ϕest = ϕ_cr, ϕ2v!, φ, θ, hide_plots
    hide_plots && return
    viz = function (ϕ0)
        v = (deepcopy(ϕ0[:]), deepcopy(ϕ0[:]))
        ϕ2v!(v, ϕ0[:])
        v 
    end
    imgs = Dict(1=>viz(ϕtru)[1], 2=>viz(ϕest)[1])
    txt  = Dict(1=>L"true $\nabla_\theta \phi$", 2=>L"est $\nabla_\theta \phi$")
    fig, ax = CMBrings.diskplot(
        imgs, φ', π.-θ; txt=txt, fontsize=14
    )
    return nothing
end



#-

@sblock let ϕtru = ϕ, ϕest = ϕ_cr, ϕ2v!, φ, θ, hide_plots
    hide_plots && return
    viz = function (ϕ0)
        v = (deepcopy(ϕ0[:]), deepcopy(ϕ0[:]))
        ϕ2v!(v, ϕ0[:])
        v 
    end
    imgs = Dict(1=>viz(ϕtru)[2], 2=>viz(ϕest)[2])
    txt  = Dict(1=>L"true $\nabla_\varphi \phi$", 2=>L"est $\nabla_\varphi \phi$")
    fig, ax = CMBrings.diskplot(
        imgs, φ', π.-θ; txt=txt, fontsize=14
    )
    return nothing
end


#- 


@sblock let ϕtru = ϕ, ϕest = ϕ_cr, ϕ2v!, φ, θ, hide_plots
    hide_plots && return
    viz = function (ϕ0)
        v = (deepcopy(ϕ0[:]), deepcopy(ϕ0[:]))
        ϕ2v!(v, ϕ0[:])
        v 
    end
    imgs = Dict(1=>ϕtru[:], 2=>ϕest[:])
    txt  = Dict(1=>L"true $\phi$", 2=>L"est $\phi$")
    fig, ax = CMBrings.diskplot(
        imgs, φ', π.-θ; txt=txt, fontsize=14
    )
    return nothing
end


#-


@sblock let d, φ, θ, hide_plots

    hide_plots && return

    imgs = Dict(1=>real(d[:]), 2=>imag(d[:]))
    txt  = Dict(
        1=>"Q data",     2=>"U data",
    )
    fig, ax = CMBrings.diskplot(
        imgs, φ', π.-θ; txt=txt, fontsize=14
    )
    return nothing

end



#-


@sblock let f_cr, φ, θ, hide_plots

    hide_plots && return

    imgs = Dict(1=>real(f_cr[:]), 2=>imag(f_cr[:]))
    txt  = Dict(
        1=>"Q est",     2=>"U est",
    )
    fig, ax = CMBrings.diskplot(
        imgs, φ', π.-θ; txt=txt, fontsize=14
    )
    return nothing

end



# Some testing 
# =============================

#= ############################################

@benchmark $Phi▪  * $(Xfourier(ϕ))  # 9.953 ms down from 262.847 ms
@benchmark $Beam▪ * $(Xfourier(qu)) # 27.339 ms
@benchmark $EB▪   * $(Xfourier(qu)) # 35.575 ms
@benchmark $N▪    * $(Xfourier(qu)) # 3.036 ms
@benchmark $Ð▪⁻¹  \ $(Xfourier(qu)) # 2.423 s

=# ############################################



#= ##################################
loglog(ℓ, ℓ.^4 .* NΦNℓ)
loglog(ℓ, ℓ.^4 .* ϕϕℓ)
=# ##################################



#= ##################################################### 
nℓₒ = exp(mean(log.(eeℓ[4:5000])))
loglog(ℓ, eeℓ)
loglog(ℓ, bbℓ)
loglog(ℓ, fill(nℓₒ, length(ℓ)) )
=# ##################################################### 


#= #####################################################
@time Ðqu = Ð▪⁻¹ \ qu
@time Ð▪⁻¹Ðqu = Ð▪⁻¹ * Ðqu

qu[:] |> real |> matshow; colorbar()
Ð▪⁻¹Ðqu[:]|> real |> matshow; colorbar()
Ð▪⁻¹Ðqu[:] .- qu[:] |> real |> matshow; colorbar()
Ðqu[:] .- qu[:] |> real |> matshow; colorbar()

qu[!] .|> abs |> matshow; colorbar()
Ð▪⁻¹Ðqu[!] .|> abs |> matshow; colorbar()
Ð▪⁻¹Ðqu[!] .- qu[!] .|> abs |> matshow; colorbar()
qu[!] .|> abs |> matshow; colorbar()
Ðqu[!] .|> abs |> matshow; colorbar()
=# #####################################################



#= #######################################
Base.summarysize(Precon▪⁻¹) * 1e-9
Base.summarysize(EB▪) * 1e-9
=# #######################################



#= ##################################################### 
## Tests an azmuthally symmetric mask as part of the preconditioner

Mask_ring = @sblock let pr_col=Pr[:][:,2*end÷10], θ, φ, T = Float64
    
    nθ=length(θ)
    nφ=length(φ)

    Tpr_col = T.(pr_col)
    Γdb  = typeof(Diagonal(Tpr_col))[Diagonal(Tpr_col) for ℓ = 1:nφ]
    Cdb  = typeof(false*I(nθ))[false*I(nθ) for ℓ = 1:nφ]

    return CMBrings.ComplexCircRings(Γdb, Cdb)

end;

ei  = Xmap(tmUS2)
eo  = Xmap(tmUS2)
ei.fd[:] .= im
eo.fd[:] .= 1

@time ei′ = Mask_ring * ei;  
@time eo′ = Mask_ring * eo;  

ei′[:] .|> real |> matshow; colorbar()
ei′[:] .|> imag |> matshow; colorbar()

eo′[:] .|> real |> matshow; colorbar()
eo′[:] .|> imag |> matshow; colorbar()
=# ##################################################### 

#= ####################################
qu[:] .|> real |> matshow; colorbar()
qu[:] .|> imag |> matshow; colorbar()

d[:] .|> real |> matshow; colorbar()
d[:] .|> imag |> matshow; colorbar()

ϕ[:] |> matshow
Łϕ = Ł(ϕ)

@time Łϕqu   = Łϕ * qu
@time Łϕquᴴ   = Łϕ' * qu
@time Beamqu = Beam▪ * qu

Łϕqu[:] .|> real |> matshow; colorbar()
Łϕqu[:] .|> imag |> matshow; colorbar()

Łϕquᴴ[:] .|> real |> matshow; colorbar()
Łϕquᴴ[:] .|> imag |> matshow; colorbar()

Łϕqu[:] .- qu[:] .|> real |> matshow; colorbar()
Łϕqu[:] .- qu[:] .|> imag |> matshow; colorbar()
=# ####################################


#= ############################################
## for test the WF. 
semilogy(hst)

fwf[:][:,1:1000] .|> real |> matshow; colorbar()
fwf[:][:,1:1000] .|> imag |> matshow; colorbar()

(Qr * fwf)[:] .|> real |> matshow; colorbar()
(Qr * fwf)[:] .|> imag |> matshow; colorbar()

fwf[!] .|> real .|> abs |> matshow; colorbar()
fwf[!] .|> imag .|> abs |> matshow; colorbar()

qu[!] .|> real .|> abs |> matshow; colorbar()
qu[!] .|> imag .|> abs |> matshow; colorbar()

(d - fwf)[:][:,1:1000] .|> real .|> abs |> matshow; colorbar()
(d - fwf)[:][:,1:1000] .|> imag .|> abs |> matshow; colorbar()

@sblock let fwf, φ, θ, hide_plots
    hide_plots && return
    imgs = Dict(1=>real.(fwf[:]), 2=>imag.(fwf[:]))
    txt  = Dict(1=>"E(Q|d)", 2=>"E(U|d)")
    fig, ax = CMBrings.diskplot(imgs, φ', π.-θ; txt=txt, nrows=1, fontsize=14)
    return nothing
end
=# ############################################



#=  ############################################
@time qu_test =  @sblock let EB▪, wn
    wnk  = fielddata(FourierField(wn))
    quk = similar(wnk)
    wnℓ = collect(eachcol(wnk))
    quℓ = collect(eachcol(quk))
    J   = Spectra.Jop(EB▪.nblks)
    Threads.@threads for ℓ = 1:J.n
        Ωℓ = sqrt(Hermitian(EB▪[ℓ])) 
        quℓ[ℓ] .= @view(Ωℓ[1:end÷2,:]) * vcat(wnℓ[ℓ], conj.(wnℓ[J(ℓ)]))
    end 
    Xfourier(fieldtransform(wn), quk)
end;

qu[:][:,1:1000]  .|> real |> matshow; colorbar()
qu_test[:][:,1:1000]  .|> real |> matshow; colorbar()
(qu - qu_test)[:][:,1:1000]  .|> real |> matshow; colorbar()

qu[:][:,1:1000]  .|> imag |> matshow; colorbar()
qu_test[:][:,1:1000]  .|> imag |> matshow; colorbar()
(qu - qu_test)[:][:,1:1000]  .|> imag |> matshow; colorbar()
=#  ############################################



#= ##################################################### 
## Beam Test 

ei  = Xmap(tmUS2)
eo  = Xmap(tmUS2)
ei.fd[350,400] = im
eo.fd[350,400] = 1

@time ei′ = Beam▪ * ei;  # 10 times faster than EBcov * ei 
@time eo′ = Beam▪ * eo;  # 10 times faster than EBcov * ei 

ei′[:] .|> real |> matshow; colorbar()
ei′[:] .|> imag |> matshow; colorbar()

eo′[:] .|> real |> matshow; colorbar()
eo′[:] .|> imag |> matshow; colorbar()

ei′[!] .|> abs |> matshow; colorbar()
eo′[!] .|> abs |> matshow; colorbar()

sum(eo′[:]) # ≈ 1
sum(ei′[:]) # ≈ im*1
=# ##################################################### 

#=  #####################################################
## Noise Test 

ei  = Xmap(tmUS2)
ei.fd[end - 50,100] = 1
Nei = N▪ * ei
Nei[:][end - 50,100] # should be approx ...
deg2rad(μK′n / 60)^2 / Ω[end - 50]
=# ##################################################### 

#= #####################################################
d,V = Phi▪[3] |> Symmetric |> eigen
d,V = Phi▪[100] |> Symmetric |> eigen
@time Phi▪[100] |> Symmetric |> sqrt
@time Phi▪[100] |> Symmetric |> cholesky
=# #####################################################



#= ############################################
## Test to make sure the beam has the right size....
(Beam▪ * qu)[:] .|> real |> matshow; colorbar()
(Beam▪ * qu)[:] .|> imag |> matshow; colorbar()

@time Beam▪ * qu # beam takes .1 seconds
=# ############################################



#= ############################################
ei  = Xmap(tmUS2)
ei.fd[end-50,400] = 1
## ei.fd[150,400] = im * 1

@time ei′ = Lcut * ei;
@time ei′ = EB▪ * ei;
@time ei′ = N▪ * ei;
@time ei′ = Beam▪ * ei;  # 10 times faster than EBcov * ei 
@time ei′ = Pr * Beam▪ * EBcov * ei; 

ei′[:] .|> real |> matshow; colorbar()
ei′[:] .|> imag |> matshow; colorbar()
=# ############################################












