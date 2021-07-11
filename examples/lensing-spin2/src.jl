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
import FFTransforms as FT
using Spectra
using FieldLensing 

using BlockArrays
using SparseArrays
using DelimitedFiles
using LBblocks: @sblock
using PyPlot
import Dierckx 
import NLopt
using BenchmarkTools
using ProgressMeter



hide_plots = true

#- 

if isdefined(Main, :IJulia) && Main.IJulia.inited
    hide_plots = false
else 
    hide_plots = true
end


# Extra methods
# ==============================

# TODO: get this in Spectra or spin-off
function periodize(f::Vector{T}, freq_mult::Int) where {T}
    n = length(f)
    nfm = n÷freq_mult
    @assert nfm == n//freq_mult
    f′ = sum( circshift(f, k*nfm) for k=0:freq_mult-1)
    f′[1:nfm]
end


# Pixel grid
# ==============================


θ, φ, Ω, Δθ, nθ, nφ, freq_mult, tmUS2, tmUS0 = @sblock let 

    freq_mult = 3 # 3
    nθ, nφ    = (200, 768)
    θnorth∂ = 2.8 # 2.5 #  2.3784 # 
    θsouth∂ = 3.0 # 2.7 #  2.7694 # 

    ## θpix∂   = θnorth∂ .+ (θsouth∂ - θnorth∂)*(0:nθ)/nθ  |> collect
    ## --- or -------
    znorth = cos.(θnorth∂)
    zsouth = cos.(θsouth∂)
    θpix∂ = acos.(range(znorth, zsouth, length=nθ+1))
    ## --------------
    Δθ = diff(θpix∂)
    θ = θpix∂[2:end] .- Δθ/2    
    
    ## set φ (assuming it is uniform)
    φleft∂  = 0.0          # 2.5 # 2.3784
    φright∂ = 2π/freq_mult # 2.7 # 2.7694
    φ       = φleft∂ .+ (φright∂ - φleft∂)*(0:nφ-1)/nφ  |> collect

    ## set φ (this assumes φ gridding is uniform)
    Ω   = @. (φ[2] - φ[1]) * abs(cos(θpix∂[1:end-1]) - cos(θpix∂[2:end]))


    ## Unitary transforms
    T = Float64
    tmUS2  = FT.:⊗(FT.𝕀(nθ), FT.𝕎(Complex{T}, nφ, 2π/freq_mult))
    tmUS2 *= FT.unitary_scale(tmUS2) 
    
    tmUS0  = FT.:⊗(FT.𝕀(nθ), FT.𝕎(T, nφ, 2π/freq_mult))
    tmUS0 *= FT.unitary_scale(tmUS0) 


    return θ, φ, Ω, Δθ, nθ, nφ, freq_mult, tmUS2, tmUS0
end;

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



# Mask and CMBring observation region
# ==============================



data_msk = @sblock let θ, φ
    
    pr_msk  = readdlm(joinpath(CMBrings.module_dir,"examples/artifacts/FastTransform_mask_nθ3072_nφ4095.csv"), ',', Bool)    
    ## pr_msk  = readdlm(joinpath(CMBrings.module_dir,"examples/artifacts/FastTransform_mask_mid2pole_nθ2560_nφ3071.csv"), ',', Bool)    
    ## pr_msk  = readdlm(joinpath(CMBrings.module_dir,"examples/artifacts/FastTransform_mask_spole_nθ3072_nφ4095.csv"), ',', Bool)    
    nθ_msk, nφ_msk = size(pr_msk)
    θ_msk = π*(0.5:nθ_msk-0.5)/nθ_msk |> collect
    φ_msk = 2π*(0:nφ_msk-1)/nφ_msk    |> collect
    spline_mask = Dierckx.Spline2D(θ_msk, φ_msk, pr_msk, kx=1, ky=1, s=0.0)

    data_msk = spline_mask.(θ, φ') .> 0
    data_msk[1:15,:] .= 0
    data_msk[end - 15 + 1:end,:] .= 0

    return data_msk

end;


#- 

using CMBflat: PrQr # Eventually remove this

Pr, Qr = @sblock let tmUS2, θ, φ, data_msk, QP_bdry=1e-5, fwhm′=150
    tmFlat = FT.𝕎(real(eltype_in(tmUS2)), size(data_msk), ((θ[2] - θ[1])*length(θ), 2π))
    pr0x, qr0x = PrQr(tmFlat, data_msk, fwhm′, fwhm′, QP_bdry)
    pr0 = Xmap(tmUS2, pr0x)
    qr0 = Xmap(tmUS2, qr0x)
    DiagOp(pr0), DiagOp(qr0)
end;

# Localize lensing vector field to data mask.

Mϕ = @sblock let tmUS0, θ, φ, data_msk, QP_bdry=1e-5, fwhm′=75
    tmFlat = FT.𝕎(real(eltype_in(tmUS0)), size(data_msk), ((θ[2] - θ[1])*length(θ), 2π))
    pr0x, qr0x = PrQr(tmFlat, data_msk, fwhm′, fwhm′, QP_bdry)

    ## mϕx = pr0x .+ qr0x
    mϕx = pr0x 

    ## make sure it hits zero and 1
    mϕx .-= minimum(mϕx)
    mϕx ./= maximum(mϕx)
    Mϕ    = DiagOp(Xmap(tmUS0, mϕx))
    Mϕ
end;

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


# Spectral densities
# ==============================

# ϕϕ, EB spectra
eeℓ, bbℓ, ẽeℓ, b̃bℓ, ϕϕℓ, ℓ = @sblock let
    
    r  = 0.01

    lmax = 11000
    l = 0:lmax
    cld = Spectra.camb_cls(;lmax=lmax, r)
    
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

    return eel, bbl, ẽel, b̃bl, ϕϕl, l

end;



# EB and Phi cov operator 
# ==============================
Phi▪, EB▪  = @sblock let  ℓ, eeℓ, bbℓ, ϕϕℓ, θ, φ, freq_mult, nθ, nφ

    covβEB  = Spectra.βcovSpin2(ℓ, eeℓ, bbℓ);
    covβPhi = Spectra.βcovSpin0(ℓ, ϕϕℓ)

    nφ2π  = nφ*freq_mult
    φ2π   = 2π*(0:nφ2π-1)/nφ2π |> collect

    ptmW   = FFTW.plan_fft(Vector{ComplexF64}(undef, nφ), flags=FFTW.MEASURE) 
    EBγⱼₖ  = zeros(ComplexF64, nφ)
    EBξⱼₖ  = zeros(ComplexF64, nφ)
    Phiγⱼₖ = zeros(ComplexF64, nφ)

    T    = ComplexF64 # ComplexF32
    rT   = real(T)
    EB▪  = Matrix{T}[zeros(T,2nθ,2nθ) for ℓ = 1:nφ÷2+1]
    Phi▪ = Matrix{rT}[zeros(rT,nθ,nθ) for ℓ = 1:nφ÷2+1]

    prgss = Progress(nθ, 1, "Computing Phi▪ and EB▪ ")
    for k = 1:nθ
        for j = 1:nθ
            θ1, θ2 = θ[j], θ[k]
            β      = Spectra.geoβ.(θ1, θ2, 0.0, φ2π)
            covIĪ  = complex.(covβPhi(β))  
            covPP̄, covPP = covβEB(β)  
            covPP̄ .*= Spectra.multPP̄.(θ1, θ2, 0.0, φ2π) 
            covPP .*= Spectra.multPP.(θ1, θ2, 0.0, φ2π)
            ## periodize and restrict from φ2π to φ
            covIĪ′ = periodize(covIĪ, freq_mult)   
            covPP̄′ = periodize(covPP̄, freq_mult)       
            covPP′ = periodize(covPP, freq_mult)       
            mul!(Phiγⱼₖ, ptmW, covIĪ′)
            mul!(EBγⱼₖ, ptmW, covPP̄′)
            mul!(EBξⱼₖ, ptmW, covPP′)
            @inbounds for ℓ = 1:nφ÷2+1
                Jℓ = ℓ==1 ? 1 : nφ - ℓ + 2
                Phi▪[ℓ][j,  k   ] = real.(Phiγⱼₖ[ℓ])
                EB▪[ℓ][j,   k   ] = EBγⱼₖ[ℓ]
                EB▪[ℓ][j,   k+nθ] = EBξⱼₖ[ℓ]
                EB▪[ℓ][j+nθ,k   ] = conj(EBξⱼₖ[Jℓ])
                EB▪[ℓ][j+nθ,k+nθ] = conj(EBγⱼₖ[Jℓ])
            end
        end
        next!(prgss)
    end

    @show Base.summarysize(Phi▪) / 1e9
    @show Base.summarysize(EB▪)  / 1e9

    return CircOp(map(Symmetric, Phi▪)), CircOp(map(Hermitian,EB▪))
end;


# EB▪½  = map(M->Array(cholesky(M).L), EB▪.Σ)  |> CircOp
# Phi▪½ = map(M->Array(cholesky(M).L), Phi▪.Σ) |> CircOp

EB▪½  = map(sqrt, EB▪.Σ)  |> CircOp
Phi▪½ = map(sqrt, Phi▪.Σ) |> CircOp

zUS2 = Xmap(tmUS2, randn(ComplexF64, nθ, nφ))
zUS0 = Xmap(tmUS0, randn(Float64, nθ, nφ))

f0    = Phi▪½ * zUS0
f2    = EB▪½  * zUS2

# f0 = ▪2field(tmUS0, map((Σ,w)->sqrt(Σ)*w, Phi▪.Σ, field2▪(zUS0)))
# f2 = ▪2field(tmUS2, map((Σ,w)->sqrt(Σ)*w, EB▪.Σ, field2▪(zUS2)))

f0[:] |> matshow; colorbar()
f2[:] .|> real |> matshow; colorbar()
f2[:] .|> imag |> matshow; colorbar()

@benchmark $Phi▪½ * $(Xfourier(zUS0)) # 9.953 ms down from 262.847 ms
@benchmark $EB▪½  * $(Xfourier(zUS2)) # 34.088 ms


# Beam
# ==============================

beamℓ = @sblock let ℓ
    ## THIS IS A TEST ↯↯↯↯↯↯↯↯
    ## beamfwhm  = 55.0 |> arcmin -> deg2rad(arcmin/60)
    beamfwhm     = 3.5 |> arcmin -> deg2rad(arcmin/60)
    ## beamfwhm  = 25.0 |> arcmin -> deg2rad(arcmin/60)

    σ² = beamfwhm^2 / 8 / log(2)
    bℓ = @. exp( - σ²*ℓ*(ℓ+1) / 2)

    ## ℓcut = 200
    ## bℓ .*= ℓ .< ℓcut

    return bℓ
end;

Beam_ring = @sblock let beamℓ, ℓ, θ, φ, Ω
    
    covBeamβ = Spectra.βcovSpin0(ℓ, beamℓ)

    nθ=length(θ)
    nφ=length(φ)

    ptmW = FT.FFTW.plan_fft(Vector{ComplexF64}(undef, nφ), flags=FT.FFTW.MEASURE) 
    Γdjk = zeros(ComplexF64, nφ)

    ## T    = Float32
    T    = Float64
    Γdb  = Matrix{T}[zeros(T, nθ, nθ) for ℓ = 1:nφ]
    Cdb  = typeof(false*I(nθ))[false*I(nθ) for ℓ = 1:nφ]

    prgss = Progress(nθ, 1, "Computing the beam operator ...")
    for k = 1:nθ
        for j = 1:nθ
            θ1, θ2, φ1 = θ[j], θ[k], φ[1]
            Ωk    = Ω[k] 
            β     =  Spectra.geoβ.(θ1, θ2, φ1, φ) 
            covIĪ = complex.(covBeamβ(β))  
            mul!(Γdjk, ptmW, covIĪ)
            for ℓ = 1:nφ
                @inbounds Γdb[ℓ][j,k] = real(Γdjk[ℓ]) * Ωk
            end
        end
        next!(prgss)
    end

    return CMBrings.ComplexCircRings(Γdb, Cdb)
end;


# Noise
# ==============================

Noise_ring, μK′n = @sblock let μK′n = 1.5, θ, φ, Ω

    ## T = Float32
    T = Float64
    nθ=length(θ)
    nφ=length(φ)

    μKᵒn = μK′n / 60
    σ²   = deg2rad(μKᵒn)^2
    σ²_Ω = T.(σ²./Ω)

    Γdb  = typeof(Diagonal(σ²_Ω))[Diagonal(σ²_Ω) for ℓ = 1:nφ]
    Cdb  = typeof(false*I(nθ))[false*I(nθ) for ℓ = 1:nφ]

    return CMBrings.ComplexCircRings(Γdb, Cdb), μK′n
end;




# Gradients Set sparse increment matrices for non-FFT lensing
# ==================================================

function generate_∇!_∇!ϕ_1storder(tmS0::Transform{Tf,d}, θℝ, φℝ) where {Tf,d}
    Δθℝ, Δφℝ = θℝ[2] - θℝ[1], φℝ[2] - φℝ[1]

    ## ∂θ′ = spdiagm(
    ##         0 => fill(-1,length(θℝ)), 
    ##         1 => fill(1,length(θℝ)-1),
    ##     )
    ## ∂θ′[end,1] =  1
    ## ∂θ = Tf(1 / (Δθℝ)) * ∂θ′
    ∂θ′ = spdiagm(
            -2 => fill( 1,length(θℝ)-2),
            -1 => fill(-8,length(θℝ)-1),
             1 => fill( 8,length(θℝ)-1),
             2 => fill(-1,length(θℝ)-2),
            )
    ∂θ′[1,end]   =  -8
    ∂θ′[1,end-1] =  1
    ∂θ′[2,end]   =  1
    ∂θ′[end,1]   =  8
    ∂θ′[end,2]   = -1
    ∂θ′[end-1,1] = -1
    ∂θ = Tf(1 / (12Δθℝ)) * ∂θ′


    ## ∂φ  = spdiagm(
    ##     0 => fill(-1,length(φℝ)), 
    ##     1 => fill(1,length(φℝ)-1)
    ## )
    ## ∂φ[end,1] =  1
    ## ∂φᵀ = transpose(Tf(1 / (Δφℝ)) * ∂φ)
    ∂φ  = spdiagm(
            -2 => fill( 1,length(φℝ)-2),
            -1 => fill(-8,length(φℝ)-1),
             1 => fill( 8,length(φℝ)-1),
             2 => fill(-1,length(φℝ)-2),
            )
    ∂φ[1,end]   =  -8
    ∂φ[1,end-1] =  1
    ∂φ[2,end]   =  1
    ∂φ[end,1]   =  8
    ∂φ[end,2]   =  -1
    ∂φ[end-1,1] =  -1
    ∂φᵀ = transpose(Tf(1 / (12Δφℝ)) * ∂φ)

    ∇!   = CMBrings.Nabla!((∂θ - ∂θ')/2, (∂φᵀ - ∂φᵀ')/2)
    ∇!_ϕ = CMBrings.Nabla!(∂θ, ∂φᵀ)

    ## ∇!   = CMBrings.Pix1dFFTNabla!((∂θ - ∂θ')/2, Tf, length(φℝ), Tf(2π))
    ## ∇!_ϕ = CMBrings.Pix1dFFTNabla!(∂θ, Tf, length(φℝ), Tf(2π))

    return ∇!, ∇!_ϕ
end  


function generate_lense_sublense(;
        tmS0, θ, mv1x=1, mv2x=1, 
        ∇!,  ∇!_ϕ, ## subidx, sub_∇!, 
        nsteps_lensing=14
        ) 

    ## ∇!_ϕ used in ϕ2v! and ϕ2vᴴ!
    ## ∇! used in Ł
    ## sub_∇! used in sub_Ł
    
    sin⁻²θ = @. csc(θ)^2 
    mvx₁ = ones(size(θ)) .* mv1x
    mvx₂ = sin⁻²θ .* mv2x

    ϕ2v! = function (v::NTuple{2,Array}, ϕ::Array)
        ∇!_ϕ(v, ϕ)
        v[1] .*= mvx₁
        v[2] .*= mvx₂
        v
    end 

    ϕ2vᴴ! = function (ϕ::Array, v::NTuple{2,Array})
        mv = (similar(v[1]), similar(v[2]))
        ∇!_ϕ'(mv, (mvx₁.*v[1], mvx₂.*v[2]) )
        ϕ .= mv[1] .+ mv[2]
        ϕ 
    end 

    Ł = function (ϕ_az::Xfield)
        ϕ = ϕ_az[:]
        v = (similar(ϕ), similar(ϕ))
        ϕ2v!(v,ϕ)
        FieldLensing.ArrayLense(v, ∇!, 0, 1, nsteps_lensing)
    end

    ## sub_Ł = function (ϕ_az::Xfield)
    ##     ϕ = ϕ_az[:]
    ##     v = (similar(ϕ), similar(ϕ))
    ##     ϕ2v!(v,ϕ)
    ##     sub_v  = getindex.(v, Ref(subidx))  
    ##     sub_Łϕ = CMBsphere.SubArrayLense(
    ##         FieldLensing.ArrayLense(sub_v, sub_∇!, 0, 1, nsteps_lensing), 
    ##         subidx
    ##     )
    ##     sub_Łϕ
    ## end

    ## Ł, ϕ2v!, ϕ2vᴴ!, ∇!, sub_Ł
    Ł, ϕ2v!, ϕ2vᴴ!, ∇!
end


##  Subset transform for lensing
## 
##  subidx, θ_sub, φ_sub, mϕ_sub = @sblock let tmAzS0, Mϕ
## 
##      nθ, nφ = size_in(tmAzS0)
##      nθ_sub_range = 1:nθ
##      nφ_sub_range = 1:round(Int, .35 * nφ) 
## 
##      subidx = CartesianIndices((nθ_sub_range, nφ_sub_range))
##      nθ_sub = length(nθ_sub_range)
##      nφ_sub = length(nφ_sub_range)
## 
##      θ, φ = ST.pix(tmAzS0) 
##      θ_sub = θ[nθ_sub_range]
##      φ_sub = φ[nφ_sub_range]
## 
##      mϕ_sub = Mϕ[:][subidx]
## 
##      return subidx, θ_sub, φ_sub, mϕ_sub
##  end;


∇!,  ∇!_ϕ = generate_∇!_∇!ϕ_1storder(tmS0, θ, φ) 
## sub_∇!,   = generate_∇!_∇!ϕ_1storder(θ_sub, φ_sub) 


## Ł, ϕ2v!, ϕ2vᴴ!, ∇!, sub_Ł = generate_lense_sublense(;
Ł, ϕ2v!, ϕ2vᴴ!, ∇! = generate_lense_sublense(;
        tmS0=tmS0, θ=θ, 
        mv1x=Mϕ[:], mv2x=Mϕ[:], 
        ∇!,  ∇!_ϕ, ## subidx, sub_∇!, 
        nsteps_lensing=11
);




# simulation
# ==============================

@time qu = CMBrings.map_ring(
    (fℓ, Σℓ) -> sqrt(Hermitian(Σℓ)) * fℓ,
    Xmap(tmUS2, randn(eltype_in(tmUS2), size_in(tmUS2))),
    EB_ring, 
)

@time no = CMBrings.map_ring(
    (fℓ, Σℓ) -> sqrt(Hermitian(Matrix(Σℓ))) * fℓ,
    Xmap(tmUS2, randn(eltype_in(tmUS2), size_in(tmUS2))),
    Noise_ring, 
)

@time ϕ = CMBrings.map_ring(
    (fℓ, Σℓ) -> sqrt(Symmetric(Matrix(Σℓ))) * fℓ,
    Xmap(tmS0, randn(eltype_in(tmS0), size_in(tmS0))),
    Phi▪, 
)

d = Pr * (Beam_ring * Ł(ϕ) * qu + no)



# MixFlow
# ==============================

Ð⁻¹ = @sblock let ẽeℓ, b̃bℓ, ℓ, θ, φ, EB_ring, Noise_ring

    covβ = Spectra.βcovSpin2(ℓ, ẽeℓ, b̃bℓ)

    nθ = length(θ)
    nφ = length(φ)

    ptmW = FT.FFTW.plan_fft(Vector{ComplexF64}(undef, nφ), flags=FT.FFTW.MEASURE) 
    Γdjk = zeros(ComplexF64, nφ)
    Cdjk = zeros(ComplexF64, nφ)
    ## T    = ComplexF32
    T    = ComplexF64
    Γdb  = Matrix{T}[zeros(T, nθ, nθ) for ℓ = 1:nφ]
    Cdb  = Matrix{T}[zeros(T, nθ, nθ) for ℓ = 1:nφ]

    prgss = Progress(nθ, 1, "Computing EB̃_ring operator ...")
    for k = 1:nθ
        for j = 1:nθ
            θ1, θ2, φ1 = θ[j], θ[k], φ[1]
            β  =  Spectra.geoβ.(θ1, θ2, φ1, φ) 
            covPP̄, covPP = covβ(β)  
            covPP̄ .*= Spectra.multPP̄.(θ1, θ2, φ1, φ) 
            covPP .*= Spectra.multPP.(θ1, θ2, φ1, φ)            
            mul!(Γdjk, ptmW, covPP̄)
            mul!(Cdjk, ptmW, covPP)
            for ℓ = 1:nφ
                @inbounds Γdb[ℓ][j,k] = Γdjk[ℓ]
                @inbounds Cdb[ℓ][j,k] = Cdjk[ℓ]
            end
        end
        next!(prgss)
    end

    EB̃_ring = CMBrings.ComplexCircRings(Γdb, Cdb)

    Ð⁻¹ =  CMBrings.map_ring(
        (EBℓ, EB̃ℓ, Nℓ) -> sqrt(Hermitian(EBℓ)) / sqrt(Hermitian(EB̃ℓ + 4*Nℓ)),
        EB_ring, EB̃_ring, Noise_ring,
    );

    return Ð⁻¹
end;


# Uncertainty for ϕ based on iterative quadratic estimate
# ==============================
## TODO: needs fixing up ...

import CMBflat

N0ℓ, NΦNℓ =  @sblock let n_iter = 5, eeℓ, bbℓ, ϕϕℓ, beamℓ, nnℓ = deg2rad(μK′n / 60)^2 .+ zero(ℓ), ℓ

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

#-

NΦN_ring = @sblock let NΦNℓ, ℓ, θ, φ, Ω

    covΦβ = Spectra.βcovSpin0(ℓ, NΦNℓ)
    nθ=length(θ)
    nφ=length(φ)

    ## ptmW = FT.FFTW.plan_fft(Vector{ComplexF64}(undef, nφ), flags=FT.FFTW.PATIENT) 
    ptmW = FT.FFTW.plan_fft(Vector{ComplexF64}(undef, nφ), flags=FT.FFTW.MEASURE) 
    Γdjk = zeros(ComplexF64, nφ)
    
    ## T    = Float32
    T    = Float64
    Γdb  = Matrix{T}[zeros(T, nθ, nθ) for ℓ = 1:nφ]
    ## Cdb  = Matrix{T}[zeros(T, nθ, nθ) for ℓ = 1:nφ]
    Cdb  = typeof(false*I(nθ))[false*I(nθ) for ℓ = 1:nφ]

    prgss = Progress(nθ, 1, "Computing the NΦN operator ...")
    for k = 1:nθ
        for j = 1:nθ
            θ1, θ2, φ1 = θ[j], θ[k], φ[1]
            Ωk    = Ω[k] 
            β     =  Spectra.geoβ.(θ1, θ2, φ1, φ) 
            covIĪ = complex.(covΦβ(β))  
            mul!(Γdjk, ptmW, covIĪ)
            for ℓ = 1:nφ
                @inbounds Γdb[ℓ][j,k] = real.(Γdjk[ℓ])
            end
        end
        next!(prgss)
    end
    return CMBrings.ComplexCircRings(Γdb, Cdb)
end;


# Preconditioner
# ==============================

@time Precon⁻¹_ring = @sblock let EB_ring, Beam_ring, Noise_ring, pr_col=Pr[:][:,2*end÷10], qr_col=Qr[:][:,2*end÷10]

    ## T = ComplexF32
    T = ComplexF64
    Precon⁻¹ = CMBrings.ComplexCircRings(EB_ring.nblks, EB_ring.nside, Matrix{T}, Matrix{T})

    prgss = Progress(Precon⁻¹.nblks÷2+1, 1, "Computing the inverse preconditioner ...")
    Threads.@threads for ℓ = 1:Precon⁻¹.nblks÷2+1
        Bm = Beam_ring[ℓ] 
        EB = EB_ring[ℓ] 
        No = Noise_ring[ℓ]
        Ωℓ = Bm * EB * Bm' + No
        Precon⁻¹[ℓ] = pinv(Ωℓ) ## pinv(Ωℓ)
        next!(prgss)
    end 

    return Precon⁻¹
end;

# Now do some iterations ...
# ==============================

## ------ initalize 
gwf = 0*d 
ϕ_cr  = 0*ϕ
## special for this noise
Noise_ring⁻¹ = CMBrings.map_ring(Nℓ->diagm(1 ./ diag(Nℓ)), Noise_ring);


@showprogress for otr = 1:2
## @showprogress for otr = 1:3
    global f_cr, gwf, hst
    global f′_cr, ϕ_cr, ∇ϕ_cr

    ## ------ update field
    @time f_cr, gwf, hst = CMBrings.update_f(
        (otr==1) ? DiagOp(Xmap(tmUS2,1)) : Ł(ϕ_cr), # slot for Łϕ
        EB_ring; 
        data=Xmap(d),
        Pr, Qr, 
        Bm=Beam_ring, No=Noise_ring, Pc⁻¹=Precon⁻¹_ring,
        ginit=Xmap(gwf),
        pcg_nsteps = (otr==1) ? 300 : 175, # 175, ## 200, 
        pcg_rel_tol=1e-10
    );
    @show hst[end]
    f′_cr =  Ł(ϕ_cr) * (Ð⁻¹ \ f_cr) 
    @show CMBrings.ll_ϕf′(ϕ_cr, f′_cr, Phi▪, EB_ring; data=d, Ł, Ð⁻¹, Pr, Beam_ring, Noise_ring⁻¹)

    ## ------ ϕ gradient
    ## @time gradϕ = CMBrings.∇ll_ϕf′(ϕ_cr, f′_cr, Phi▪, EB_ring; data=d, Ł, Ð⁻¹, Pr, Beam_ring, Noise_ring⁻¹, ϕ2v!, ϕ2vᴴ!, ∇!, grad_nsteps=11)
    @time gradϕ = CMBrings.∇ll_ϕf′_usingf(ϕ_cr, f_cr, Phi▪, EB_ring; data=d, Ł, Ð⁻¹, Pr, Beam_ring, Noise_ring⁻¹, ϕ2v!, ϕ2vᴴ!, ∇!, grad_nsteps=11)
    @time ∇ϕ_cr = NΦN_ring * gradϕ 
        
    ## ------ linesearch 
    @time β = CMBrings.linesearch_ϕf′(
        ∇ϕ_cr, ϕ_cr, f′_cr, Phi▪, EB_ring; 
        data = d, Ł, Ð⁻¹, Pr, Beam_ring, Noise_ring⁻¹,
        eval_max = 200, startval = 0.001, ftol_abs = 50, solver = :LN_COBYLA,  
        ## eval_max = 250, startval = 0.001, ftol_abs = 1, solver = :LN_COBYLA,  
    )
    @show β

    ## ------ update ϕ_cr
    ϕ_cr += β * ∇ϕ_cr
end

#-

#=
ϕ_cr[:] |> matshow; colorbar()
ϕ[:] |> matshow; colorbar()
=#

#-

@sblock let ϕtru = ϕ, ϕest = ϕ_cr, ϕ2v!, φ, θ, hide_plots
    hide_plots && return
    viz = function (ϕ0)
        v = (deepcopy(ϕ0[:]), deepcopy(ϕ0[:]))
        ϕ2v!(v, ϕ0[:])
        v 
    end
    imgs = Dict(1=>viz(ϕtru)[1], 3=>viz(ϕest)[1],
                2=>viz(ϕtru)[2], 4=>viz(ϕest)[2])
    txt  = Dict(1=>"true", 3=>"est")
    fig, ax = CMBrings.diskplot(
        imgs, φ', π.-θ; txt=txt, 
        figsize=(10,16), nrows=2, fontsize=14
    )
    return nothing
end

#- 


@sblock let ϕtru = ϕ, ϕest = ϕ_cr, ϕ2v!, φ, θ, hide_plots
    hide_plots && return
    imgs = Dict(1=>ϕtru[:], 2=>ϕest[:])
    txt  = Dict(1=>"true", 2=>"est")
    fig, ax = CMBrings.diskplot(
        imgs, φ', π.-θ; txt=txt, 
        figsize=(10,8), nrows=1, fontsize=14
    )
    return nothing
end


#-


@sblock let f_cr, qu, φ, θ, hide_plots

    hide_plots && return

    imgs = Dict(1=>real(f_cr[:]), 2=>imag(f_cr[:]))
    imgs = Dict(
        1=>real(f_cr[:]), 2=>imag(f_cr[:]),
        3=>real(qu[:]),   4=>imag(qu[:])
        )
    txt  = Dict(
        1=>"Q wf",     2=>"U wf",
        3=>"Q true",   4=>"U true",
    )
    fig, ax = CMBrings.diskplot(
        imgs, φ', π.-θ; txt=txt, 
        figsize=(10,16), nrows=2, fontsize=14
    )
    return nothing

end




###################################################
###################################################

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

##  CMBrings.ll_ϕf′(ϕ_cr, f′_cr, Phi▪, EB_ring; data=d, Ł, Ð⁻¹, Pr, Beam_ring, Noise_ring⁻¹)
##  CMBrings.ll_ϕf′(ϕ_cr + .01 * ∇ϕ_cr, f′_cr, Phi▪, EB_ring; data=d, Ł, Ð⁻¹, Pr, Beam_ring, Noise_ring⁻¹)
## 
##  opt = NLopt.Opt(:LN_COBYLA, 1)
##  opt.upper_bounds = Float64[2]
##  opt.lower_bounds = Float64[0]
##  opt.ftol_abs = 10.0
##  ϕₒ, inHgradₒ = promote(ϕ_cr, ∇ϕ_cr)
##  opt.max_objective = function (β, grad)
##      ϕβ = ϕₒ + β[1] * inHgradₒ       
##      return CMBrings.ll_ϕf′(ϕβ, f′_cr, Phi▪, EB_ring; data=d, Ł, Ð⁻¹, Pr, Beam_ring, Noise_ring⁻¹)
##  end
##     
##  ll_opt, β_opt, = NLopt.optimize(opt,  Float64[0.001])
    


#= ############################################
wn   = Xmap(tmUS2, randn(eltype_in(tmUS2), size_in(tmUS2)))
Σwn1 = @time CMBrings.map_ring((fℓ, Σℓ) -> Σℓ*fℓ, wn, EB_ring)
Σwn2 = @time EB_ring * wn 
Σwn1[:] .- Σwn2[:] .|> abs |> matshow; colorbar()
Σwn2[:] .|> abs |> matshow; colorbar()


wn2 = @time EB_ring \ Σwn2
wn2[:] .|> abs |> matshow; colorbar()
wn2[:] .- wn[:] .|> real |> matshow; colorbar()
wn2[:] .- wn[:] .|> imag |> matshow; colorbar()
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



#=  ##################################################### 
d,V = EB_ring[3] |> Hermitian |> eigen
d,V = EB_ring[100] |> Hermitian |> eigen
@time EB_ring[100] |> Hermitian |> sqrt
@time EB_ring[100] |> Hermitian |> cholesky
=#  ##################################################### 


#= #####################################################
@time Ðqu = Ð⁻¹ \ qu
@time Ð⁻¹Ðqu = Ð⁻¹ * Ðqu

qu[:] |> real |> matshow; colorbar()
Ð⁻¹Ðqu[:]|> real |> matshow; colorbar()
Ð⁻¹Ðqu[:] .- qu[:] |> real |> matshow; colorbar()
Ðqu[:] .- qu[:] |> real |> matshow; colorbar()

qu[!] .|> abs |> matshow; colorbar()
Ð⁻¹Ðqu[!] .|> abs |> matshow; colorbar()
Ð⁻¹Ðqu[!] .- qu[!] .|> abs |> matshow; colorbar()
qu[!] .|> abs |> matshow; colorbar()
Ðqu[!] .|> abs |> matshow; colorbar()
=# #####################################################



#= #######################################
Base.summarysize(Precon⁻¹_ring) * 1e-9
Base.summarysize(EB_ring) * 1e-9
d,V = Precon⁻¹_ring[2] |> Hermitian |> eigen
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
@time Beamqu = Beam_ring * qu

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
@time qu_test =  @sblock let EB_ring, wn
    wnk  = fielddata(FourierField(wn))
    quk = similar(wnk)
    wnℓ = collect(eachcol(wnk))
    quℓ = collect(eachcol(quk))
    J   = Spectra.Jop(EB_ring.nblks)
    Threads.@threads for ℓ = 1:J.n
        Ωℓ = sqrt(Hermitian(EB_ring[ℓ])) 
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

@time ei′ = Beam_ring * ei;  # 10 times faster than EBcov * ei 
@time eo′ = Beam_ring * eo;  # 10 times faster than EBcov * ei 

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
Nei = Noise_ring * ei
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
(Beam_ring * qu)[:] .|> real |> matshow; colorbar()
(Beam_ring * qu)[:] .|> imag |> matshow; colorbar()

@time Beam_ring * qu # beam takes .1 seconds
=# ############################################



#= ############################################
ei  = Xmap(tmUS2)
ei.fd[end-50,400] = 1
## ei.fd[150,400] = im * 1

@time ei′ = Lcut * ei;
@time ei′ = EB_ring * ei;
@time ei′ = Noise_ring * ei;
@time ei′ = Beam_ring * ei;  # 10 times faster than EBcov * ei 
@time ei′ = Pr * Beam_ring * EBcov * ei; 

ei′[:] .|> real |> matshow; colorbar()
ei′[:] .|> imag |> matshow; colorbar()
=# ############################################












