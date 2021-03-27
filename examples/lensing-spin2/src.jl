## Spin 2 lensing example which uses CMBsphere transform to handle the QU cov operator

# Modules
# ==============================
using LinearAlgebra
BLAS.set_num_threads(1)

using FFTW 
## FFTW.set_num_threads(Threads.nthreads())
FFTW.set_num_threads(1)

using XFields
using CMBrings
using CMBsphere     # we will use CMBsphere to do the EBcovariance operator
using CMBflat: PrQr # Eventually remove this CMBflat.PrQr dependence ...

import FFTransforms as FT
import SphereTransforms as ST

using Spectra
using FieldLensing 

using  LinearAlgebra
using  SparseArrays
import Dierckx 
import NLopt

using DelimitedFiles
using LBblocks: @sblock
using PyPlot
using BenchmarkTools
using ProgressMeter

#- 

if isdefined(Main,:PlutoRunner)
    import PlutoUI
    hide_plots = false
elseif isdefined(Main, :IJulia) && Main.IJulia.inited
    hide_plots = false
else 
    hide_plots = true
end

# Set ring transforms
# ==============================

tmUS0, tmUS2, θ, φ, Ω, ringidx, tmS0 = @sblock let 

    ## size of the embedding full sphere
    ## 𝕊nθ, 𝕊nφ = (1536, 1536-1)
    ## 𝕊nθ, 𝕊nφ = (1536, 2560-1)
    𝕊nθ, 𝕊nφ = (2048, 1536-1)
    ## 𝕊nθ, 𝕊nφ = (2048, 2048-1)
    ## 𝕊nθ, 𝕊nφ = (2560, 2048-1)
    ## 𝕊nθ, 𝕊nφ = (2560, 2560-1)
    ## 𝕊nθ, 𝕊nφ = (3584, 2560-1)
    ## 𝕊nθ, 𝕊nφ = (3584, 3584-1) # good one here 
    ## 𝕊nθ, 𝕊nφ = (3584, 4096-1) # good one here 
    ## 𝕊nθ, 𝕊nφ = (4096, 3584-1)

    ## grid coords on full sphere
    θ𝕊, φ𝕊  = ST.pix(𝕊nθ, 𝕊nφ) 

    ## north and southern boundaries and the corresponding indices
    ## Default, SPT:
    ## θnorth∂ = 2.4 # (small) # 2.2 (part) # 2.12 (full)
    ## θsouth∂ = 2.85
    ## Further south
    θnorth∂ = 2.7
    θsouth∂ = 3.05

    θrng    = findall(θnorth∂ .<= θ𝕊 .<= θsouth∂)
    ringidx = CartesianIndices((θrng[1]:θrng[end], 1:length(φ𝕊)))
    
    nθ, nφ  = size(ringidx)
    θ, φ  = θ𝕊[ringidx[:,1]], φ𝕊
    Ω     = ST.Ωpix(𝕊nθ, 𝕊nφ)[ringidx[:,1]]

    ## Unitary transforms for spin0 and spin2 
    T = Float64
    tmS0 = FT.:⊗(FT.𝕀(nθ), FT.𝕎(T, nφ, 2π)) |> x -> FT.unitary_scale(x)*x
    tmUS0 = FT.:⊗(FT.𝕀(nθ), FT.𝕎(Complex{T}, nφ, 2π)) |> x -> FT.unitary_scale(x)*x
    tmUS2 = FT.:⊗(FT.𝕀(nθ), FT.𝕎(Complex{T}, nφ, 2π)) |> x -> FT.unitary_scale(x)*x

    return tmUS0, tmUS2, θ, φ, Ω, ringidx, tmS0
end

# Mask and CMBring observation region
# ==============================

data_mask_init = @sblock let θ, φ
    
    ## Default:
    ## pr_mat_init  = readdlm(joinpath(CMBrings.module_dir,"examples/artifacts/FastTransform_mask_nθ3072_nφ4095.csv"), ',', Bool)    
    ## South pole mask:
    pr_mat_init  = readdlm(joinpath(CMBrings.module_dir,"examples/artifacts/FastTransform_mask_spole_nθ3072_nφ4095.csv"), ',', Bool)    
    
    θ_mat_init, φ_mat_init = ST.pix(size(pr_mat_init)...)
    spline_mask = Dierckx.Spline2D(θ_mat_init, φ_mat_init, pr_mat_init, kx=1, ky=1, s=0.0)

    data_mask_init = spline_mask.(θ, φ') .> 0
    data_mask_init[1:20,:] .= 0
    data_mask_init[end - 20 + 1:end,:] .= 0

    return data_mask_init

end;

#- 

Pr, Qr = @sblock let tmUS2, θ, φ, data_mask_init, QP_bdry=1e-5, fwhm′=150

    ## Testing to see if the seg fault can be avoided by doing a direct plan 
    ## nθ, nφ = length(θ), length(φ)
    ## ptmW = FT.FFTW.plan_rfft(Matrix{Float64}(undef, nθ, nφ), flags=FT.FFTW.PATIENT) 
    
    tmFlat = FT.𝕎(real(eltype_in(tmUS2)), size(data_mask_init), ((θ[2] - θ[1])*length(θ), 2π))
    pr0x, qr0x = PrQr(tmFlat, data_mask_init, fwhm′, fwhm′, QP_bdry)
    pr0 = Xmap(tmUS2, pr0x)
    qr0 = Xmap(tmUS2, qr0x)

    DiagOp(pr0), DiagOp(qr0)
end;

# Localize lensing vector field to data mask.

Mϕ = @sblock let tmS0, θ, φ, data_mask_init, QP_bdry=1e-5, fwhm′=75

    tmFlat = FT.𝕎(real(eltype_in(tmS0)), size(data_mask_init), ((θ[2] - θ[1])*length(θ), 2π))
    pr0x, qr0x = PrQr(tmFlat, data_mask_init, fwhm′, fwhm′, QP_bdry)

    ## mϕx = pr0x .+ qr0x
    mϕx = pr0x 

    ## make sure it hits zero and 1
    mϕx .-= minimum(mϕx)
    mϕx ./= maximum(mϕx)
    Mϕ    = DiagOp(Xmap(tmS0, mϕx))

    Mϕ
end;

# Azimuthal ring mask

@sblock let ma=real.(Pr[:]), φ, θ, hide_plots
    hide_plots && return
    imgs = Dict(1=>ma)
    txt  = Dict(1=>"Mask")
    ctxt = Dict(1=>"w")
    ## fig, ax = CMBrings.brickplot(imgs; txt=txt, ctxt=ctxt, fφ=1)
    fig, ax = CMBrings.diskplot(imgs, φ', π.-θ; txt=txt, nrows=1, fontsize=14)
    return fig
end

# Plot √Ωpix over ring θ's 

@sblock let θ, φ, Ω, hide_plots
    hide_plots && return
    fig,ax = subplots(1)
    ax.plot(θ, rad2deg.(sqrt.(Ω)).*60, label="sqrt pixel area (arcmin)")
    ax.plot(θ, zero(θ) .+ rad2deg.(θ[2] - θ[1]).*60, label="Δθ (arcmin)")
    ## ax.plot(θ, zero(θ) .+ rad2deg.(φ[2] - φ[1]).*60, label="Δφ (arcmin)")
    ax.set_xlabel(L"polar coordinate $\theta$")
    ax.legend()
    return fig
end


# Spectral densities
# ==============================

# ϕϕ, EB spectra
eeℓ, bbℓ, ẽeℓ, b̃bℓ, ϕϕℓ, ℓvec = @sblock let
    
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


#= ##################################################### 
nℓₒ = exp(mean(log.(eeℓ[4:5000])))
loglog(ℓvec, eeℓ)
loglog(ℓvec, bbℓ)
loglog(ℓvec, fill(nℓₒ, length(ℓvec)) )
=# ##################################################### 



# EB ring operator 
# ==============================

EB_ring = @sblock let  eeℓ, bbℓ, ℓvec, θ, φ, 

    T = ComplexF64

    covPβ = Spectra.βcovSpin2(ℓvec, eeℓ, bbℓ;
        ## n_grid::Int = 100_000, 
        ## β_grid = range(0, π^(1/3), length=n_grid).^3,
    );

    nθ = length(θ)
    nφ = length(φ)

    ## ptmW = FT.FFTW.plan_fft(Vector{ComplexF64}(undef, nφ), flags=FT.FFTW.PATIENT) 
    ptmW = FT.FFTW.plan_fft(Vector{ComplexF64}(undef, nφ), flags=FT.FFTW.MEASURE) 
    Γdjk = zeros(ComplexF64, nφ)
    Cdjk = zeros(ComplexF64, nφ)
    Γdb  = Matrix{T}[zeros(T, nθ, nθ) for ℓ = 1:nφ]
    Cdb  = Matrix{T}[zeros(T, nθ, nθ) for ℓ = 1:nφ]

    prgss = Progress(nθ, 1, "Computing EB cov operator ...")
    for k = 1:nθ
        for j = 1:nθ
            θ1, θ2, φ1 = θ[j], θ[k], φ[1]
            β  =  Spectra.geoβ.(θ1, θ2, φ1, φ) 
            covPP̄, covPP = covPβ(β)  
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

    return CMBrings.ComplexCircRings(Γdb, Cdb)
end;


#=  ##################################################### 
d,V = EB_ring[3] |> Hermitian |> eigen
d,V = EB_ring[100] |> Hermitian |> eigen
@time EB_ring[100] |> Hermitian |> sqrt
@time EB_ring[100] |> Hermitian |> cholesky
=#  ##################################################### 

# Beam
# ==============================

beamℓ = @sblock let ℓvec

    ## THIS IS A TEST ↯↯↯↯↯↯↯↯
    ## beamfwhm  = 55.0 |> arcmin -> deg2rad(arcmin/60)
    beamfwhm  = 3.5 |> arcmin -> deg2rad(arcmin/60)
    ## beamfwhm  = 25.0 |> arcmin -> deg2rad(arcmin/60)

    σ² = beamfwhm^2 / 8 / log(2)
    bℓ = @. exp( - σ²*ℓvec*(ℓvec+1) / 2)


    ## ℓcut = 2500
    ## bℓ .*= ℓvec .< ℓcut

    return bℓ

end;

Beam_ring = @sblock let beamℓ, ℓvec, θ, φ, Ω
    
    T = Float32

    covBeamβ = Spectra.βcovSpin0(ℓvec, beamℓ)

    nθ=length(θ)
    nφ=length(φ)

    ## ptmW = FT.FFTW.plan_fft(Vector{ComplexF64}(undef, nφ), flags=FT.FFTW.PATIENT) 
    ptmW = FT.FFTW.plan_fft(Vector{ComplexF64}(undef, nφ), flags=FT.FFTW.MEASURE) 
    Γdjk = zeros(ComplexF64, nφ)
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


# Noise
# ==============================

Noise_ring, μK′n = @sblock let μK′n = 2.5, θ, φ, Ω

    T = Float32

    nθ=length(θ)
    nφ=length(φ)

    μKᵒn = μK′n / 60
    σ²   = deg2rad(μKᵒn)^2
    σ²_Ω = T.(σ²./Ω)

    Γdb  = typeof(Diagonal(σ²_Ω))[Diagonal(σ²_Ω) for ℓ = 1:nφ]
    Cdb  = typeof(false*I(nθ))[false*I(nθ) for ℓ = 1:nφ]

    return CMBrings.ComplexCircRings(Γdb, Cdb), μK′n

end

#=  #####################################################
## Noise Test 

ei  = Xmap(tmUS2)
ei.fd[end - 50,100] = 1
Nei = Noise_ring * ei
Nei[:][end - 50,100] # should be approx ...
deg2rad(μK′n / 60)^2 / Ω[end - 50]
=# ##################################################### 




# Φ operator 
# ==============================

Φ_ring = @sblock let ϕϕℓ, ℓvec, θ, φ, Ω

    covΦβ = Spectra.βcovSpin0(ℓvec, ϕϕℓ)

    nθ=length(θ)
    nφ=length(φ)

    ## ptmW = FT.FFTW.plan_fft(Vector{ComplexF64}(undef, nφ), flags=FT.FFTW.PATIENT) 
    ptmW = FT.FFTW.plan_fft(Vector{ComplexF64}(undef, nφ), flags=FT.FFTW.MEASURE) 
    Γdjk = zeros(ComplexF64, nφ)
    Γdb  = Matrix{Float64}[zeros(Float64, nθ, nθ) for ℓ = 1:nφ]
    Cdb  = Matrix{Float64}[zeros(Float64, nθ, nθ) for ℓ = 1:nφ]

    prgss = Progress(nθ, 1, "Computing the Φ operator ...")
    for k = 1:nθ
        for j = 1:nθ
            θ1, θ2, φ1 = θ[j], θ[k], φ[1]
            Ωk    = Ω[k] 
            β     =  Spectra.geoβ.(θ1, θ2, φ1, φ) 
            covIĪ = complex.(covΦβ(β))  
            mul!(Γdjk, ptmW, covIĪ)
            for ℓ = 1:nφ
                ## TODO: double check this is real ....
                @inbounds Γdb[ℓ][j,k] = real(Γdjk[ℓ] / 2)
                @inbounds Cdb[ℓ][j,k] = real(Γdjk[ℓ] / 2)
            end
        end
        next!(prgss)
    end

    return CMBrings.ComplexCircRings(Γdb, Cdb)

end;

#= #####################################################
d,V = Φ_ring[3] |> Symmetric |> eigen
d,V = Φ_ring[100] |> Symmetric |> eigen
@time Φ_ring[100] |> Symmetric |> sqrt
@time Φ_ring[100] |> Symmetric |> cholesky
=# #####################################################

# Gradients Set sparse increment matrices for non-FFT lensing
# ==================================================

function generate_∇!_∇!ϕ_1storder(tmS0::Transform{Tf,d}, θℝ, φℝ) where {Tf,d}
    Δθℝ, Δφℝ = θℝ[2] - θℝ[1], φℝ[2] - φℝ[1]

    ∂θ′ = spdiagm(
            0 => fill(-1,length(θℝ)), 
            1 => fill(1,length(θℝ)-1),
        )
    ∂θ′[end,1] =  1
    ∂θ = Tf(1 / (Δθℝ)) * ∂θ′

    ## ∂φ  = spdiagm(
    ##         0 => fill(-1,length(φℝ)), 
    ##         1 => fill(1,length(φℝ)-1)
    ##     )
    ## ∂φ[end,1] =  1
    ## ∂φᵀ = transpose(Tf(1 / (Δφℝ)) * ∂φ)

    ## ∇!   = CMBrings.Nabla!((∂θ - ∂θ')/2, (∂φᵀ - ∂φᵀ')/2)
    ## ∇!_ϕ = CMBrings.Nabla!(∂θ, ∂φᵀ)

    ∇!   = CMBrings.Pix1dFFTNabla!((∂θ - ∂θ')/2, Tf, length(φℝ), Tf(2π))
    ∇!_ϕ = CMBrings.Pix1dFFTNabla!(∂θ, Tf, length(φℝ), Tf(2π))

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
    
    ## 
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

    # sub_Ł = function (ϕ_az::Xfield)
    #     ϕ = ϕ_az[:]
    #     v = (similar(ϕ), similar(ϕ))
    #     ϕ2v!(v,ϕ)
    #     sub_v  = getindex.(v, Ref(subidx))  
    #     sub_Łϕ = CMBsphere.SubArrayLense(
    #         FieldLensing.ArrayLense(sub_v, sub_∇!, 0, 1, nsteps_lensing), 
    #         subidx
    #     )
    #     sub_Łϕ
    # end

    ## Ł, ϕ2v!, ϕ2vᴴ!, ∇!, sub_Ł
    Ł, ϕ2v!, ϕ2vᴴ!, ∇!
end


# Subset transform for lensing

# subidx, θ_sub, φ_sub, mϕ_sub = @sblock let tmAzS0, Mϕ

#     nθ, nφ = size_in(tmAzS0)
#     nθ_sub_range = 1:nθ
#     nφ_sub_range = 1:round(Int, .35 * nφ) 

#     subidx = CartesianIndices((nθ_sub_range, nφ_sub_range))
#     nθ_sub = length(nθ_sub_range)
#     nφ_sub = length(nφ_sub_range)

#     θ, φ = ST.pix(tmAzS0) 
#     θ_sub = θ[nθ_sub_range]
#     φ_sub = φ[nφ_sub_range]

#     mϕ_sub = Mϕ[:][subidx]

#     return subidx, θ_sub, φ_sub, mϕ_sub
# end;


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


#=
wn   = Xmap(tmUS2, randn(eltype_in(tmUS2), size_in(tmUS2)))
Σwn1 = @time CMBrings.map_ring((fℓ, Σℓ) -> Σℓ*fℓ, wn, EB_ring)
Σwn2 = @time EB_ring * wn 
Σwn1[:] .- Σwn2[:] .|> abs |> matshow; colorbar()
Σwn2[:] .|> abs |> matshow; colorbar()


wn2 = @time EB_ring \ Σwn2
wn2[:] .|> abs |> matshow; colorbar()
wn2[:] .- wn[:] .|> real |> matshow; colorbar()
wn2[:] .- wn[:] .|> imag |> matshow; colorbar()
=#


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
    (fℓ, Σℓ) -> sqrt(Symmetric(Σℓ)) * fℓ,
    # Xmap(tmUS0, randn(eltype_in(tmUS0), size_in(tmUS0))),
    Xmap(tmS0, randn(eltype_in(tmS0), size_in(tmS0))),
    Φ_ring, 
)

d = Pr * (Beam_ring * Ł(ϕ) * qu + no)

#=


qu[:] .|> real |> matshow; colorbar()
qu[:] .|> imag |> matshow; colorbar()

d[:] .|> real |> matshow; colorbar()
d[:] .|> imag |> matshow; colorbar()


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

=#


# MixFlow
# ==============================

EB̃_ring = @sblock let eeℓ, bbℓ, ẽeℓ, b̃bℓ, ℓvec, θ, φ

    T = ComplexF32
    nℓₒ   = exp(mean(log.(eeℓ[4:end])))
    covPβ = Spectra.βcovSpin2(ℓvec, ẽeℓ, b̃bℓ)

    nθ = length(θ)
    nφ = length(φ)

    ptmW = FT.FFTW.plan_fft(Vector{ComplexF64}(undef, nφ), flags=FT.FFTW.MEASURE) 
    Γdjk = zeros(ComplexF64, nφ)
    Cdjk = zeros(ComplexF64, nφ)
    Γdb  = Matrix{T}[zeros(T, nθ, nθ) for ℓ = 1:nφ]
    Cdb  = Matrix{T}[zeros(T, nθ, nθ) for ℓ = 1:nφ]

    prgss = Progress(nθ, 1, "Computing EB̃_ring operator ...")
    for k = 1:nθ
        for j = 1:nθ
            θ1, θ2, φ1 = θ[j], θ[k], φ[1]
            β  =  Spectra.geoβ.(θ1, θ2, φ1, φ) 
            covPP̄, covPP = covPβ(β)  
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

    return CMBrings.ComplexCircRings(Γdb, Cdb)
end;

#= #####################################################
@time Ðqu = CMBrings.map_ring(
    (fℓ, EBℓ, EB̃ℓ, Nℓ) -> sqrt(Hermitian(EB̃ℓ + Nℓ)) * (factorize(sqrt(Hermitian(EBℓ))) \ fℓ),
    qu, EB_ring, EB̃_ring, Noise_ring,
)

@time Ð⁻¹Ðqu = CMBrings.map_ring(
    (fℓ, EBℓ, EB̃ℓ, Nℓ) ->  sqrt(Hermitian(EBℓ)) * (factorize(sqrt(Hermitian(EB̃ℓ + Nℓ))) \ fℓ),
    Ðqu, EB_ring, EB̃_ring, Noise_ring,
)

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



# Preconditioner
# ==============================

@time Precon⁻¹_ring = @sblock let T=ComplexF32, EB_ring, Beam_ring, Noise_ring, pr_col=Pr[:][:,2*end÷10], qr_col=Qr[:][:,2*end÷10]

    ## ΩPrℓ = Diagonal(vcat(pr_col, conj.(pr_col)))
    ## ΩQrℓ = Diagonal(vcat(qr_col, conj.(qr_col)))

    Precon⁻¹ = CMBrings.ComplexCircRings(EB_ring.nblks, EB_ring.nside, Matrix{T}, Matrix{T})

    prgss = Progress(Precon⁻¹.nblks÷2+1, 1, "Computing the inverse preconditioner ...")
    Threads.@threads for ℓ = 1:Precon⁻¹.nblks÷2+1
        Bm = Beam_ring[ℓ] 
        EB = EB_ring[ℓ] 
        No = Noise_ring[ℓ]
        Ωℓ = Bm * EB * Bm' + No
        ## Ωℓ   = ΩPrℓ * (Bm * EB * Bm' + No) * ΩPrℓ' 
        ## Ωℓ .+= ΩQrℓ * (Bm * EB * Bm' + No) * ΩQrℓ' 
        ## Precon⁻¹[ℓ] = pinv(factorize(Hermitian(Ωℓ))) ## pinv(Ωℓ)
        Precon⁻¹[ℓ] = pinv(Ωℓ) ## pinv(Ωℓ)
        next!(prgss)
    end 

    return Precon⁻¹

end;

Base.summarysize(Precon⁻¹_ring) * 1e-9
Base.summarysize(EB_ring) * 1e-9

## d,V = Precon⁻¹_ring[2] |> Hermitian |> eigen

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





# WF pcg
# =====================================


gwf = 0*d 


## @time fwf, gwf, hst =  @sblock let pcg_nsteps=300, Łϕ=Ł(ϕ), ginit=Xmap(gwf), data=Xmap(d),  pcg_rel_tol=1e-10,  Pr, Qr, EB=EB_ring, Bm=Beam_ring, No=Noise_ring, Pc⁻¹=Precon⁻¹_ring
@time fwf, gwf, hst =  @sblock let Łϕ=DiagOp(Xmap(tmUS2,1)), ginit=Xmap(gwf), data=Xmap(d), pcg_nsteps=300, pcg_rel_tol=1e-10,  Pr, Qr, EB=EB_ring, Bm=Beam_ring, No=Noise_ring, Pc⁻¹=Precon⁻¹_ring

    Łϕᴴ = Łϕ'
    C1a = Pr * Bm * Łϕ * EB * Łϕᴴ * Bm'
    C1b = Pr * No
    C2b = Qr * No
    ## C2a = Qr * Bm * Łϕ * EB * Łϕᴴ * Bm' # this one or ....
    C2a = Qr * Bm * EB * Bm' # .... this one

    ## C2a and C2b can be combine into one op.

    A = function (g)
        Prᴴ_g = Pr' * g
        Qrᴴ_g = Qr' * g
        tmp1a = C1a * Prᴴ_g
        tmp1b = C1b * Prᴴ_g
        tmp2a = C2a * Qrᴴ_g
        tmp2b = C2b * Qrᴴ_g
        return tmp1a + tmp1b + tmp2a + tmp2b
    end 

    gwf, hst = CMBrings.pcg(
        g -> Pc⁻¹ * g, A, 
        data, ginit,
        nsteps=pcg_nsteps, rel_tol=pcg_rel_tol,
    )

    # Minv = g -> Pc⁻¹ * g
    # b = deepcopy(data)
    # x = 0*b

    fwf   = EB *  Łϕᴴ * Bm' * Pr' * gwf

    return  fwf, gwf, hst
end


#= ############################################

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
    return fig
end


=# ############################################








# gradient of log likelihood of f′ w.r.t ϕ
# =====================================

function ∇ll_ϕf′(ϕ, f′; data, Ł, Pr, Beam_ring, EB̃_ring, EB_ring, Φ_ring, Noise_ring, ϕ2v!, ϕ2vᴴ!, ∇!, grad_nsteps=14, ds...)
    L    = Ł(ϕ)
    Lᴴ   = Ł(ϕ)'

    f = CMBrings.map_ring(
        (fℓ, EBℓ, EB̃ℓ, Nℓ) ->  sqrt(Hermitian(EBℓ)) * (factorize(sqrt(Hermitian(EB̃ℓ + Nℓ))) \ fℓ),
        L \ f′, EB_ring, EB̃_ring, Noise_ring,
    )
    # f    = D \ (L \ f′)
    
    lnf  = L * f

    tmf    = fieldtransform(f′)
    tmϕ    = fieldtransform(ϕ)

    Ma     = DiagOp(Xmap(tmf, abs.(Pr[:]).>0))
    
    dΔlnf_tmp1 = Pr \ (data - Pr * (Beam_ring * lnf))
    dΔlnf =  Beam_ring' * Ma * CMBrings.map_ring(
        (fℓ, Nℓ) ->  Hermitian(Nℓ) \ fℓ,
         dΔlnf_tmp1, Noise_ring,
    )
    # dΔlnf  = Beam_ring' * Ma * (Ncov \ (Pr \ (data - Pr * (Beam_ring * lnf))))

    # --------------------------
    ϕx  = ϕ[:]
    vx  = (similar(ϕx), similar(ϕx)) # can you do this without hardcoding it?
    ϕ2v!(vx, ϕx)

    g1x = dΔlnf[L] 
    f1x = lnf[L]   
    τvx₀ = FieldLensing.ᴴ∂Łfx_∂vx(g1x, f1x, vx, ∇!, grad_nsteps)
    
    g0x_tmp1 = Lᴴ*dΔlnf - EB_ring\f
    g0x = CMBrings.map_ring(
        (fℓ, EBℓ, EB̃ℓ, Nℓ) ->  factorize(sqrt(Hermitian(EB̃ℓ + Nℓ))) \ (sqrt(Hermitian(EBℓ)) *  fℓ),
         g0x_tmp1, EB_ring, EB̃_ring, Noise_ring,
    )[L]
    # g0x  = (Dᴴ\(Lᴴ*dΔlnf - EB_ring\f))[L]
    
    f0x  = f[L]
    τvx₁ = FieldLensing.ᴴ∂Ł⁻¹fx_∂vx(g0x, f0x, vx, ∇!, grad_nsteps)

    τvx  = τvx₀ .+ τvx₁

    τϕx = similar(τvx[1])
    ϕ2vᴴ!(τϕx, τvx)

    τϕ = Xmap(tmϕ, τϕx) - Φ_ring \ ϕ
    # --------------------------

    return τϕ
end


# log likelihood and quasi-gibbs and optimization updates
# =====================================


function ll_ϕf′(ϕ, f′; data, Ł, Ð, EB_ring, Φ_ring, Noise_ring, Pr, Beam_ring, ds...)
    L    = Ł(ϕ)
    f = CMBrings.map_ring(
        (fℓ, EBℓ, EB̃ℓ, Nℓ) ->  sqrt(Hermitian(EBℓ)) * (factorize(sqrt(Hermitian(EB̃ℓ + Nℓ))) \ fℓ),
        L \ f′, EB_ring, EB̃_ring, Noise_ring,
    )
    # f    = Ð \ (Ł(ϕ) \ f′)
    lnf  = Ł(ϕ) * f
    estn = data - Pr * Beam_ring * lnf
    z1   = (Pr' * Noise_ring * Pr) \ estn
    z2   = Φ_ring \ ϕ
    z3   = EB_ring \ f
    return - FFTransforms.sum_kbn([dot(estn,z1), dot(ϕ,z2), dot(f,z3)]) / 2
end

#  linesearch updates for ϕ
# =====================================

function linesearch_ϕf′(inHgrad, ϕ, f′; 
        data, Ł, Ð, EB_ring, Φ_ring, Noise_ring, Pr, Beam_ring, 
        seconds_max = 0, # seconds 
        eval_max    = 0,
        upper_bound = 2, 
        stopval  = -Inf, #  stop when an objective value of at least stopval is found. 
        ftol_rel = 0,    #  relative tolerance on function value. 
        ftol_abs = 0,    #  absolute tolerance on function value. 
        xtol_rel = 0,    #  relative tolerance on arg value. 
        xtol_abs = 0,    #  absolute tolerance on arg value.
        solver = :LN_COBYLA,  
        ds...)

    # solvers :LN_SBPLX :LN_NELDERMEAD, :LN_COBYLA
    opt = NLopt.Opt(solver, 1)
    opt.maxtime      = seconds_max
    opt.maxeval      = eval_max
    opt.upper_bounds = [upper_bound]
    opt.lower_bounds = [0]
    opt.stopval  = stopval
    opt.ftol_rel = ftol_rel
    opt.ftol_abs = ftol_abs
    opt.xtol_rel = xtol_rel
    opt.xtol_abs = xtol_abs

    ϕₒ, inHgradₒ = promote(ϕ, inHgrad)
    
    opt.max_objective = function (β, grad)
        ϕβ = ϕₒ + β[1] * inHgradₒ       
        return ll_ϕf′(ϕβ, f′; data, Ł, EB̃_ring, EB_ring, Φ_ring, Noise_ring, Pr, Beam_ring)
    end
    
    ll_opt, β_opt, = NLopt.optimize(opt,  [0])
    
    return β_opt[1]
end








# TODO: 
# ===================================

* I need the mixflow transformation for gradient updates... 
* make Precon⁻¹ and Precon both Float32 so some of the conj grad calculations don't take as much storage.
* clean up the consistance of how we handle the types of the fields and the operators
* likelihoods

#=
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
=#


# Test to make sure the beam has the right size....
(Beam_ring * qu)[:] .|> real |> matshow; colorbar()
(Beam_ring * qu)[:] .|> imag |> matshow; colorbar()


@time Beam_ring * qu # beam takes .1 seconds

# Test 
# ========================




ei  = Xmap(tmUS2)
ei.fd[end-50,400] = 1
## ei.fd[150,400] = im * 1

# @time ei′ = Lcut * ei;
@time ei′ = EB_ring * ei;
# @time ei′ = Noise_ring * ei;
# @time ei′ = Beam_ring * ei;  # 10 times faster than EBcov * ei 
# @time ei′ = Pr * Beam_ring * EBcov * ei; 

ei′[:] .|> real |> matshow; colorbar()
ei′[:] .|> imag |> matshow; colorbar()














