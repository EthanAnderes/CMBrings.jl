## Spin 2 lensing example which uses CMBsphere transform to handle the QU cov operator


# Modules
# ==============================
# using FFTW
# FFTW.FFTW.set_num_threads(8)

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

tmAzS0, tmAzS2 = @sblock let 

    ## size of the embedding full sphere
    𝕊nθ, 𝕊nφ = (2048, 1536-1)
    ## 𝕊nθ, 𝕊nφ = (2560, 2560-1)
    ## 𝕊nθ, 𝕊nφ = (3584, 2048-1)

    ## Spin ±2 transform
    tmS2 = ST.𝕊2(𝕊nθ, 𝕊nφ)
    tmS0 = ST.𝕊0(𝕊nθ, 𝕊nφ)

    ## grid coords on full sphere
    θ𝕊, φ𝕊 = ST.pix(tmS0) 

    ## north and southern boundaries and the corresponding indices
    θnorth∂ = 2.2 # 2.12
    θsouth∂ = 2.85
    θrng    = findall(θnorth∂ .<= θ𝕊 .<= θsouth∂)
    ringidxS0 = CartesianIndices((θrng[1]:θrng[end], 1:length(φ𝕊)))
    ringidxS2 = CartesianIndices((θrng[1]:θrng[end], 1:length(φ𝕊), 1:2))
    nθ, nφ  = size(ringidxS0)

    ## Spin 0 ring transform is just inherited from FFTransforms
    Tf = Float64
    tmW0  = FT.:⊗(FT.𝕀(nθ), FT.𝕎(Tf, nφ, 2π)) # 𝕀(nθ) ⊗ 𝕎(Tf, nφ, 2π)
    tmW2  = FT.:⊗(FT.𝕀(nθ), FT.𝕎(Tf, nφ, 2π), FT.𝕀(2)) 

    ## Spin 2 transform includes the ring embedding ...
    tmAzS0 = CMBrings.Az𝕊0(tmW0, tmS0, ringidxS0)
    tmAzS2 = CMBrings.Az𝕊2(tmW2, tmS2, ringidxS2)

    return tmAzS0, tmAzS2
end



# Mask and CMBring observation region
# ==============================


data_mask_init, Ω, θ, φ = @sblock let tmAzS0, tmAzS2, QP_bdry=1e-5, fwhm′=150

    pr_mat_init  = readdlm(joinpath(CMBrings.module_dir,"examples/artifacts/FastTransform_mask_nθ3072_nφ4095.csv"), ',', Bool)
    
    full_sky_tm𝕊0 = ST.𝕊0(size(pr_mat_init)...)
    θ_mat_init, φ_mat_init = ST.pix(full_sky_tm𝕊0)
    spline_mask = Dierckx.Spline2D(θ_mat_init, φ_mat_init, pr_mat_init, kx=1, ky=1, s=0.0)

    nθ, nφ,  = size_in(tmAzS2)
    θ, φ  = ST.pix(tmAzS2)
    Ω     = ST.Ωpix(tmAzS2)

    ## θ = θnorth∂ .+ ((θsouth∂ - θnorth∂) / nθ) .* (0:nθ-1)
    ## φ = (2π / nφ) .* (0:nφ-1)
    ## Ω = ST.Ωpix.(θ, θ[2] - θ[1], φ[2] .- φ[1])

    data_mask_init = spline_mask.(θ, φ') .> 0
    data_mask_init[1:30,:] .= 0
    data_mask_init[end - 30 + 1:end,:] .= 0

    return data_mask_init, Ω, θ, φ

end;

#- 

Pr, Qr = @sblock let tmAzS0, tmAzS2, data_mask_init, QP_bdry=1e-5, fwhm′=150

    θ, φ  = ST.pix(tmAzS2)
    tmFlat = FT.𝕎(Float64, size(data_mask_init), ((θ[2] - θ[1])*length(θ), 2π))
    pr0x, qr0x = PrQr(tmFlat, data_mask_init, fwhm′, fwhm′, QP_bdry)
    pr0 = Xmap(tmAzS2, pr0x, pr0x)
    qr0 = Xmap(tmAzS2, qr0x, qr0x)

    DiagOp(pr0), DiagOp(qr0)
end;

# Localize lensing vector field to data mask.

Mϕ = @sblock let tmAzS0, tmAzS2, data_mask_init, QP_bdry=1e-5, fwhm′=75

    θ, φ  = ST.pix(tmAzS2)
    tmFlat = FT.𝕎(Float64, size(data_mask_init), ((θ[2] - θ[1])*length(θ), 2π))
    pr0x, qr0x = PrQr(tmFlat, data_mask_init, fwhm′, fwhm′, QP_bdry)

    ## mϕx = pr0x .+ qr0x
    mϕx = pr0x 

    ## make sure it hits zero and 1
    mϕx .-= minimum(mϕx)
    mϕx ./= maximum(mϕx)
    Mϕ    = DiagOp(Xmap(tmAzS0, mϕx))

    Mϕ
end;

# Azimuthal ring mask

@sblock let ma=Pr[:Qx], φ, θ, hide_plots
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

eel, bbl, ẽel, b̃bl, ϕϕl = @sblock let
    
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

    return eel, bbl, ẽel, b̃bl, ϕϕl

end;



# Full sphere signal operators
# ==============================


EBcov, Lcut, Φcov = @sblock let tmAzS0, tmAzS2, eel, bbl, ϕϕl, lcut = 2000

    n𝕊θ, n𝕊φ, = size_in(tmAzS2.tm𝕊)
    l2,m2,a2 = ST.lma(-2, n𝕊θ, n𝕊φ)
    l0,m0,a0 = ST.lma(0, n𝕊θ, n𝕊φ)
    
    ECL  = @. getindex((eel,), l2 + 1)
    BCL  = @. getindex((bbl,), l2 + 1)
    ΦCL  = @. getindex((ϕϕl,), l0 + 1)
    LCL  =  (0 .< l2 .<= lcut)
    ECL[.!a2] .= 0
    BCL[.!a2] .= 0
    ΦCL[.!a0] .= 0

    EBcov = DiagOp(Xfourier(tmAzS2.tm𝕊, ECL, BCL))
    Lcut  = DiagOp(Xfourier(tmAzS2.tm𝕊, LCL, LCL))
    Φcov  = DiagOp(Xfourier(tmAzS0.tm𝕊, ΦCL))

    return EBcov, Lcut, Φcov

end




# Can I parameterize q + i*u
# ==========================================


nθ, nφ  = size(tmAzS0.ringidx)
tmW  = FT.:⊗(FT.𝕀(nθ), FT.𝕎(Complex{Float64}, nφ, 2π)) #  |> x -> FT.unitary_scale(x)*x
ptmW = plan(tmW)

## Qθi  = Xmap(tmAzS2)
## Qθi.fd[end - 60, 1, 1] = 1
## Uθi  = Xmap(tmAzS2)
## Uθi.fd[end - 60, 1, 2] = 1
## 
## Qθi′ = EBcov * Qθi;
## Uθi′ = EBcov * Uθi;
## 
## ## Λqq = ptmW * complex.(Qθi′[:Qx], 0)
## ## Λuu = ptmW * complex.(Uθi′[:Ux], 0)
## ## Λqu = ptmW * complex.(Qθi′[:Ux], 0)
## ## Λuq = ptmW * complex.(Uθi′[:Qx], 0)
## ## ΓΛ = @. (Λqq + Λuu + im * (Λqu - Λuq)) / 2
## ## CΛ = @. (Λqq - Λuu + im * (Λqu + Λuq)) / 2
## 
## ΓΛ = ptmW * (@. complex(Qθi′[:Qx] + Uθi′[:Ux], Qθi′[:Ux] - Uθi′[:Qx]) / 2)
## CΛ = ptmW * (@. complex(Qθi′[:Qx] - Uθi′[:Ux], Qθi′[:Ux] + Uθi′[:Qx]) / 2)
## 
## ## ΓΛ .|> real |> matshow; colorbar()
## ## ΓΛ .|> imag |> matshow; colorbar()
## ## 
## ## CΛ .|> real |> matshow; colorbar()
## ## CΛ .|> imag |> matshow; colorbar()

# ---------- template out a function to generate Γand C for az polarization

lengthθ, nblks = size_out(tmW)
Tb = Float64
azΓ = Matrix{Tb}[zeros(Tb, lengthθ, lengthθ) for k = 1:nblks]
azC = Matrix{Tb}[zeros(Tb, lengthθ, lengthθ) for k = 1:nblks]

Qθi  = Xmap(tmAzS2)
Uθi  = Xmap(tmAzS2)

@time begin 

@sblock let azΓ, azC, lengthθ, nblks, ptmW, EBcov, Qθi, Uθi

    @showprogress for i = 1:lengthθ

        Qθi.fd[i, 1, 1] = 1
        Uθi.fd[i, 1, 2] = 1
        # TODO: make a version of the following that doesn't allocate memory
        Qθi′ = EBcov * Qθi
        Uθi′ = EBcov * Uθi
    
        ΓΛ = ptmW * (@. complex(Qθi′[:Qx] + Uθi′[:Ux], Qθi′[:Ux] - Uθi′[:Qx]) / 2)
        CΛ = ptmW * (@. complex(Qθi′[:Qx] - Uθi′[:Ux], Qθi′[:Ux] + Uθi′[:Qx]) / 2)
    
        ## Threads.@threads for k = 1:nblks
        for k = 1:nblks
            azΓ[k][:,i] .= real.(ΓΛ[:,k])
            azC[k][:,i] .= real.(CΛ[:,k])
        end

        Qθi.fd[i, 1, 1] = 0
        Uθi.fd[i, 1, 2] = 0

    end 

end

end


k = 50
M = [
    azΓ[k] azC[k]
    azC[k] azΓ[k]
]

va, Ve = Symmetric( M ) |> eigen

plot(va)

plot(Ve[:,end-15])
plot(Ve[:,end-5])
plot(Ve[:,end])
plot(Ve[:,1])

# Base.summarysize(azΣ) * 1e-9 #-> gigabites







# AzBlock operators for noise, beam phi covariance matrix
# ==============================

# noise

## nnl, wnl, snl = @sblock let 
## 
##     μK′n      = 2.5 
##     ellknee   = 0   
##     alphaknee = 3
## 
##     lmax = 11000
##     l = 0:lmax
##     whitenoisel    = fill(μK′n^2 * (π/60/180)^2, size(l))
##     smoothnoisel   = @. μK′n^2 * (π/60/180)^2 * Spectra.knee(l; ell=ellknee, alpha=alphaknee) 
##     smoothnoisel .-= μK′n^2 * (π/60/180)^2 
##     smoothnoisel[smoothnoisel .< 0] .= 0    
##     noisel = smoothnoisel .+ whitenoisel
##     return noisel, whitenoisel, smoothnoisel
## 
## end;

#-


Naz = @sblock let tmAzS0, Ω, μK′n = 2.5
    μKᵒn = μK′n / 60
    σ²   = deg2rad(μKᵒn)^2
    Vector_M = [Diagonal(σ²./Ω) for k in 1:size_out(tmAzS0)[2]]
    CMBrings.AzBlock(Vector_M)
end

# quick test

#=

ei = Xmap(tmAzS0)
ei.fd[end - 50,100] = 1
Nei = Naz * ei
Nei[:][end - 50,100]
deg2rad(2.5 / 60)^2 / Ω[end - 50]

=#


# beam/transfer

bl = @sblock let 

    beamfwhm  = 5.0 |> arcmin -> deg2rad(arcmin/60)

    lmax = 11000 
    l = 0:lmax
    σ² = beamfwhm^2 / 8 / log(2)
    bl = @. exp( - σ²*l*(l+1) / 2)
    return bl

end;


Baz = @sblock let tmAzS0,  bl, θ, φ, Ω

	tmW=FT.unscale(tmAzS0.tmAz)
    
    dmax = 1.2maximum(CMBrings.geoθ1θ2Δφcol(θ[1], θ[1], φ .- φ[1]))
    θgrid = range(0, dmax^(1/2), length=100_000).^2
    
    covf  = Dierckx.Spline1D(
        θgrid, 
        Spectra.spec2spherecov(bl, θgrid), 
        k=3
    )
    
    covf_θ1θ2Δφℝ = (θ1, θ2, Δφ) -> covf(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφ)) 

    Baz  = CMBrings.AzBlock(covf_θ1θ2Δφℝ, θ, φ, tmW) do Σ, k
        real.(Σ) * LinearAlgebra.Diagonal(Ω)
    end

    return Baz
end;




#=
eiS0 = Xmap(tmAzS0)
eiS0.fd[end - 50,100] = 1
eiS2 = Xmap(tmAzS2)
eiS2.fd[end - 50,100,1] = 1

@time XFields._lmult(Baz, eiS0)
@time XFields._lmult(Baz, eiS2)
=#



#-

Φaz = @sblock let tmAzS0,  ϕϕl, θ, φ

    tmW=FFTransforms.unscale(tmAzS0.tmAz)
    
    dmax = 1.2maximum(CMBrings.geoθ1θ2Δφcol(θ[1], θ[1], φ .- φ[1]))
    θgrid = range(0, dmax^(1/2), length=100_000).^2
    
    covf  = Dierckx.Spline1D(
        θgrid, 
        Spectra.spec2spherecov(ϕϕl, θgrid), 
        k=3
    )
    
    covf_θ1θ2Δφℝ = (θ1, θ2, Δφ) -> covf(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφ)) 

    ## Φaz  = CMBrings.AzBlock(covf_θ1θ2Δφℝ, θ, φ, tmW) do Σ, k
    ##     factorize(Symmetric(real.(Σ)))
    ## end
    ## ------
    ## Φaz  = CMBrings.AzBlock(covf_θ1θ2Δφℝ, θ, φ, tmW) do Σ, k
    ##     C = cholesky(Symmetric(real.(Σ), :L)) # , check=false)
    ##     Cholesky(C.factors, C.uplo, C.info)
    ## end
    ## ------
    Φaz  = CMBrings.AzBlock(covf_θ1θ2Δφℝ, θ, φ, tmW) do Σ, k
        ## B = eigen(Symmetric( real.(Σ) + 1e-9*I, :L))
        B = eigen(Symmetric( real.(Σ), :L))
        B.values[B.values .<= 0] .= 0
        B
    end

    return Φaz
end;



#=
ei  = Xmap(tmAzS0)
ei.fd[150,400] = 1
@time ei′ = Φaz * ei; # this mult takes a long time if the factorization isn't convert to matrix
ei′[:] |> matshow
=#


#=

ei  = Xmap(tmAzS2)
ei.fd[150,400,1] = 1

@time ei′ = Lcut * ei;
@time ei′ = EBcov * ei;
@time ei′ = Naz * ei;
@time ei′ = Baz * ei; # 10 times faster than EBcov * ei 
@time ei′ = Pr * Baz * EBcov * ei; 

ei′[:Qx] |> matshow
ei′[:Ux] |> matshow


ϕ_sim = Xmap(tmAzS0, CMBsphere.simmap(Φcov)[:][tmAzS0.ringidx])
p_sim = Xmap(tmAzS2, CMBsphere.simmap(EBcov)[:][tmAzS2.ringidx])


(Baz * p_sim)[:Qx] |> matshow
(Baz * p_sim)[:Ux] |> matshow

=#





# Gradients Set sparse increment matrices for non-FFT lensing
# ==================================================

import CMBrings: Nabla!


# Subset transform for lensing

subidx, θ_sub, φ_sub, mϕ_sub = @sblock let tmAzS0, Mϕ

    nθ, nφ = size_in(tmAzS0)
    nθ_sub_range = 1:nθ
    nφ_sub_range = 1:round(Int, .35 * nφ) 

    subidx = CartesianIndices((nθ_sub_range, nφ_sub_range))
    nθ_sub = length(nθ_sub_range)
    nφ_sub = length(nφ_sub_range)

    θ, φ = ST.pix(tmAzS0) 
    θ_sub = θ[nθ_sub_range]
    φ_sub = φ[nφ_sub_range]

    mϕ_sub = Mϕ[:][subidx]

    return subidx, θ_sub, φ_sub, mϕ_sub
end;




function generate_∇!_∇!ϕ_1storder(θℝ::Vector{T_fld}, φℝ::Vector{T_fld}) where T_fld
    Δθℝ, Δφℝ = θℝ[2] - θℝ[1], φℝ[2] - φℝ[1]

    ∂θ′ = spdiagm(
            0 => fill(-1,length(θℝ)), 
            1 => fill(1,length(θℝ)-1),
        )
    ∂θ′[end,1] =  1
    ∂θ = T_fld(1 / (Δθℝ)) * ∂θ′

    ∂φ  = spdiagm(
            0 => fill(-1,length(φℝ)), 
            1 => fill(1,length(φℝ)-1)
        )
    ∂φ[end,1] =  1
    ∂φᵀ = transpose(T_fld(1 / (Δφℝ)) * ∂φ)

    ∇!   = CMBrings.Nabla!((∂θ - ∂θ')/2, (∂φᵀ - ∂φᵀ')/2)
    ∇!_ϕ = CMBrings.Nabla!(∂θ, ∂φᵀ)

    return ∇!, ∇!_ϕ
end  


function generate_lense_sublense(;
        tmS0, subidx, mv1x=1, mv2x=1, 
        ∇!,  ∇!_ϕ, sub_∇!, 
        nsteps_lensing=14
        ) 

    ## ∇!_ϕ used in ϕ2v! and ϕ2vᴴ!
    ## ∇! used in Ł
    ## sub_∇! used in sub_Ł
    
    ## need to adjust for curvature 
    θ      = ST.pix(tmS0)[1]
    sin⁻²θ = @. csc(θ)^2 
    maθ = ones(size(θ))
    maφ = ones(size(θ))
    mvx₁_init = maθ
    mvx₂_init = sin⁻²θ .* maφ

    ## 
    mvx₁ = mvx₁_init .* mv1x
    mvx₂ = mvx₂_init .* mv2x


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

    sub_Ł = function (ϕ_az::Xfield)
        ϕ = ϕ_az[:]
        v = (similar(ϕ), similar(ϕ))
        ϕ2v!(v,ϕ)
        sub_v  = getindex.(v, Ref(subidx))  
        sub_Łϕ = CMBsphere.SubArrayLense(
            FieldLensing.ArrayLense(sub_v, sub_∇!, 0, 1, nsteps_lensing), 
            subidx
        )
        sub_Łϕ
    end

    Ł, ϕ2v!, ϕ2vᴴ!, ∇!, sub_Ł
end

#-

∇!,  ∇!_ϕ = generate_∇!_∇!ϕ_1storder(ST.pix(tmAzS0)...) 
sub_∇!,   = generate_∇!_∇!ϕ_1storder(θ_sub, φ_sub) 

#-

Ł, ϕ2v!, ϕ2vᴴ!, ∇!, sub_Ł = generate_lense_sublense(;
        tmS0=tmAzS0, subidx, 
        mv1x=Mϕ[:], mv2x=Mϕ[:], 
        ∇!,  ∇!_ϕ, sub_∇!,
        nsteps_lensing=11
);

#-
## ϕ_ring = Xmap(tmAzS0, CMBsphere.simmap(Φcov)[:][tmAzS0.ringidx])
## v = (ϕ_ring[:], ϕ_ring[:]) .|> deepcopy
## ∇!_ϕ(v, ϕ_ring[:])
## ∇!(v, ϕ_ring[:])

@sblock let hide_plots, plot_field=:Qx, tmAzS0, tmAzS2, Ł, sub_Ł, Φcov, EBcov
    hide_plots && return


    ϕ_ring = Xmap(tmAzS0, CMBsphere.simmap(Φcov)[:][tmAzS0.ringidx])
    p_ring = Xmap(tmAzS2, CMBsphere.simmap(EBcov)[:][tmAzS2.ringidx])

    lnp_ring     = Ł(ϕ_ring) * p_ring
    sub_lnp_ring = sub_Ł(ϕ_ring) * p_ring

    time_Ł     = @belapsed $(Ł(ϕ_ring))    * $(Xmap(p_ring))
    time_sub_Ł = @belapsed $(sub_Ł(ϕ_ring)) * $(Xmap(p_ring))

    imgs = Dict(
        1 => lnp_ring[plot_field],
        2 => sub_lnp_ring[plot_field],
    )
    txt =  Dict(
        1 => "full lense with M, time=$time_Ł",
        2 => "sub lense with M, time=$time_sub_Ł",
    )
    fig, ax = CMBrings.brickplot(
        imgs; 
        txt=txt,
        fφ   = 1/2,  # fraction of azimuth
    )
    ## fig, ax = CMBrings.diskplot(imgs, φ', π.-θ; txt=txt, nrows=1, fontsize=14)


    fig
end;






#  
# ==========================================


Qθi  = Xmap(tmAzS2)
Qθi.fd[end - 60, 1, 1] = 1
Uθi  = Xmap(tmAzS2)
Uθi.fd[end - 60, 1, 2] = 1

@time Qθi′ = EBcov * Qθi;
@time Uθi′ = EBcov * Uθi;

# Qθi′[:Qx] |> matshow
# Qθi′[:Ux] |> matshow

Qθik = Xfourier(Qθi′)
Uθik = Xfourier(Uθi′)

Qθik[!][:,:,1] .|> real |> maximum # *
Qθik[!][:,:,1] .|> imag |> maximum

Qθik[!][:,:,2] .|> real |> maximum
Qθik[!][:,:,2] .|> imag |> maximum # *


Uθik[!][:,:,1] .|> real |> maximum
Uθik[!][:,:,1] .|> imag |> maximum # *

Uθik[!][:,:,2] .|> real |> maximum # *
Uθik[!][:,:,2] .|> imag |> maximum


# * 
Qθik[!][:,:,1] .|> real |> matshow; colorbar()
Uθik[!][:,:,2] .|> real |> matshow; colorbar()

Uθik[!][:,:,1] .|> imag |> matshow; colorbar()
Qθik[!][:,:,2] .|> imag |> matshow; colorbar()




