
# TODO
# ==============================
#=

• Using EAZTransforms for all fields

=#


# Modules
# ==============================
# using FFTW
# FFTW.FFTW.set_num_threads(12)
## FFTW.FFTW.set_num_threads(5)

using CMBrings
import CMBsphere as CS
import CMBflat as CF

using XFields
using Spectra
using FFTransforms
using FieldLensing 

using LinearAlgebra
using SparseArrays
using DelimitedFiles
using Statistics
using LBblocks: @sblock
using PyPlot
using BenchmarkTools
using ProgressMeter


import Dierckx 
import NLopt

# 

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


tmS0 = @sblock let 

    T_fld   = Float64

    nθ, nφ  = 800, 2048-1
    # nθ, nφ  = 308, 3072
    # nθ, nφ  = 308, 4095
    
    tmW  = 𝕀(nθ) ⊗ 𝕎(T_fld, nφ, 2π)
    tmS0  = unitary_scale(tmW) * tmW

    return tmS0
end



# Mask and CMBring observation region
# ==============================


data_mask_init, Ω, θ, φ, θnorth∂, θsouth∂ = @sblock let tmS0, QP_bdry=1e-5, fwhm′=150

    pr_mat_init  = readdlm(joinpath(CMBrings.module_dir,"examples/artifacts/FastTransform_mask_nθ3072_nφ4095.csv"), ',', Bool)
    
    full_sky_tm𝕊0 = CS.𝕊0(size(pr_mat_init)...)
    θ_mat_init, φ_mat_init = CS.SphereTransforms.pix(full_sky_tm𝕊0)
    spline_mask = Dierckx.Spline2D(θ_mat_init, φ_mat_init, pr_mat_init, kx=1, ky=1, s=0.0)

    nθ, nφ  = size_in(tmS0)
    ## θnorth∂ = 2.12
    θnorth∂ = 2.2
    θsouth∂ = 2.85
    θ = θnorth∂ .+ ((θsouth∂ - θnorth∂) / nθ) .* (0:nθ-1)
    φ = (2π / nφ) .* (0:nφ-1)
    Ω = CS.SphereTransforms.Ωpix.(θ, θ[2] - θ[1], φ[2] .- φ[1])

    data_mask_init = spline_mask.(θ, φ') .> 0
    data_mask_init[1:30,:] .= 0
    data_mask_init[end - 30 + 1:end,:] .= 0

    return data_mask_init, Ω, θ, φ, θnorth∂, θsouth∂

end;


# 


Pr, Qr = @sblock let tmS0, data_mask_init, θnorth∂, θsouth∂,  QP_bdry=1e-5, fwhm′=150

    ## --------
    tmFlat = CF.𝕎(Float64, size(data_mask_init), (θsouth∂ - θnorth∂, 2π))
    pr0x, qr0x = CF.PrQr(tmFlat, data_mask_init, fwhm′, fwhm′, QP_bdry)
    pr0 = Xmap(tmS0, pr0x)
    qr0 = Xmap(tmS0, qr0x)
    ## ----------------


    DiagOp(pr0), DiagOp(qr0)
end;

# Localize lensing vector field to data mask.

Mϕ = @sblock let tmS0, data_mask_init, θnorth∂, θsouth∂,  QP_bdry=1e-5, fwhm′=75

    tmFlat = CF.𝕎(Float64, size(data_mask_init), (θsouth∂ - θnorth∂, 2π))
    pr0x, qr0x = CF.PrQr(tmFlat, data_mask_init, fwhm′, fwhm′, QP_bdry)
    pr0 = Xmap(tmS0, pr0x)
    qr0 = Xmap(tmS0, qr0x)

    ## mϕx = pr0x .+ qr0x
    mϕx = pr0x 

    ## make sure it hits zero and 1
    mϕx .-= minimum(mϕx)
    mϕx ./= maximum(mϕx)
    Mϕ    = DiagOp(Xmap(tmS0, mϕx))

    Mϕ
end;



# Azimuthal ring mask

@sblock let ma=Pr[:], φ, θ, hide_plots
    hide_plots && return
    imgs = Dict(1=>ma)
    txt  = Dict(1=>"Mask")
    ctxt = Dict(1=>"w")
    ## CMBrings.brickplot(imgs; txt=txt, ctxt=ctxt, fφ=1)
    
    CMBrings.diskplot(imgs, φ', π.-θ; txt=txt, nrows=1, fontsize=14)
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
end;



# ϕϕ, TT covariance
# ================================

## rcϕ   = 1e4
rcϕ = 1.0

# TODO: figure out what to do about low ell modes

ttl, t̃tl, ϕϕl = @sblock let rcϕ, lmax = 11000
    l = 0:lmax
    cld = Spectra.camb_cls(lmax=lmax)

    ttl = cld[:unlen_scalar] |> x->(x[:Ctt] ./ x[:factor_on_cl_cmb])
    ttl[1] = ttl[3] /100
    ttl[2] = ttl[3] /10

    t̃tl = cld[:len_scalar] |> x->(x[:Ctt] ./ x[:factor_on_cl_cmb])
    t̃tl[1] = t̃tl[3] /100
    t̃tl[2] = t̃tl[3] /10

    ϕϕl    = cld[:phi] |> x->(x[:Cϕϕ] ./ x[:factor_on_cl_phi])
    ϕϕl[1] =  0 # ϕϕl[3]/100
    ϕϕl[2] =  0 # ϕϕl[3]/10

    ttl, t̃tl, rcϕ^2 .* ϕϕl
end;

# Note: ϕ is now the "new" rescaled version which need to be adjusted when 
# converted to a displacement (and the transpose of that ...)

#-

Φaz = @sblock let tmW=unscale(tmS0), ϕϕl, θ, φ

    ## θgrid = range(0, π^(1/2), length=100_000).^2
    ## --------------
    dmax = 1.2maximum(CMBrings.geoθ1θ2Δφcol(θ[1], θ[1], φ .- φ[1]))
    θgrid = range(0, dmax^(1/2), length=100_000).^2

    covf  = Dierckx.Spline1D(
        θgrid, 
        Spectra.spec2spherecov(ϕϕl, θgrid), 
        k=3
    )

    covf_θ1θ2Δφℝ = (θ1,θ2,Δφ) -> covf(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφ)) 

    Φaz = CMBrings.AzBlock(covf_θ1θ2Δφℝ, θ, φ, tmW) do A, k
        ## -------------
        ## A = Symmetric(real.(A) + 1e-9*I ,:L)
        ## A = Symmetric(real.(A),:L)
        ## C = cholesky(A, Val(false), check=false)
        ## return Cholesky(C.factors, C.uplo, C.info)
        ## -------------
        ## B = eigen(Symmetric( real.(A) + 1e-9*I, :L))
        B = eigen(Symmetric( real.(A), :L))
        B.values[B.values .<= 0] .= 0
        return B
    end

    Φaz
end;

#-

Σaz = @sblock let tmW=unscale(tmS0), ttl, θ, φ

	##θgrid = range(0, π^(1/2), length=100_000).^2
    ## --------------
    dmax = 1.2maximum(CMBrings.geoθ1θ2Δφcol(θ[1], θ[1], φ .- φ[1]))
    θgrid = range(0, dmax^(1/2), length=100_000).^2

    covf  = Dierckx.Spline1D(
        θgrid, 
        Spectra.spec2spherecov(ttl, θgrid), 
        k=3
    )

    covf_θ1θ2Δφℝ = (θ1,θ2,Δφ) -> covf(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφ)) 

    Σaz = CMBrings.AzBlock(covf_θ1θ2Δφℝ, θ, φ, tmW) do A, k
        ## A = Symmetric(real.(A) + 1e-8*I ,:L)
        ## A = Symmetric(real.(A),:L)
        ## C = cholesky(A, Val(false)) #, check=false)
        ## return Cholesky( C.factors, C.uplo, C.info)
        ## -------------
        B = eigen(Symmetric(real.(A), :L))
        B.values[B.values .<= 0] .= 0
        return B 
    end 
    
    Σaz 
end;


# Beam/Transfer function
# ================================

beamfwhm    = 3.0 |> arcmin -> deg2rad(arcmin/60)
azmuth_transfer_k = (k, θ) -> 1
## azmuth_transfer_k = (k, θ) -> inv(1 + (k/cos(θ)/250)^2)
## azmuth_transfer_k = (k, θ) -> inv(1 + (k/75)^2)

#-

bl = @sblock let beamfwhm, lmax = 11000, lcut = 2000  
    l = 0:lmax
    σ² = beamfwhm^2 / 8 / log(2)
    bl = @. exp( - σ²*l*(l+1) / 2)
    #bl[l .< lcut] .= 0
    return bl
end;

#-

Baz = @sblock let tmW=unscale(tmS0),  bl, θ, φ, Ω, azmuth_transfer_k
    
    ## θgrid = range(0, π^(1/2), length=100_000).^2
    ## -------------
    dmax = 1.2maximum(CMBrings.geoθ1θ2Δφcol(θ[1], θ[1], φ .- φ[1]))
    θgrid = range(0, dmax^(1/2), length=100_000).^2
    
    covf  = Dierckx.Spline1D(
        θgrid, 
        Spectra.spec2spherecov(bl, θgrid), 
        k=3
    )
    
    covf_θ1θ2Δφℝ = (θ1,θ2,Δφ) -> covf(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφ)) 

    Baz  = CMBrings.AzBlock(covf_θ1θ2Δφℝ, θ, φ, tmW) do Σ, k
        real.(Σ) * LinearAlgebra.Diagonal(azmuth_transfer_k.(k, θ) .* Ω)
    end


    Baz
end;



# Use this to turn off the Beam 

## Baz = map(Σaz) do Σ
##     Matrix(I(size(Matrix(Σ),1)))
## end |> CMBrings.AzBlock;


# Now add a low pass filter in azimuth

## for k = 1:length(Baz)
##     if k > (length(Baz)*3)÷4
##         Baz[k] .*= 0
##     end 
## end



## Baz = 1

# Noise with weights weight and mask/projection
# ==============================

μK′n      = 3.5 # 10.0
ellknee   = 0   # 150
alphaknee = 3
## weight_θ  = θ -> 1 + 0.15 * sin(300 * θ) # θ -> 1
weight_θ  = θ -> 1
## weight_θ  = θ -> 1 + 1 ./ sin(θ).^2 # θ -> 1
#-

nnl, wnl, snl = @sblock let μK′n, ellknee, alphaknee, lmax = 11000
    l = 0:lmax
    whitenoisel    = fill(μK′n^2 * (π/60/180)^2, size(l))
    smoothnoisel   = @. μK′n^2 * (π/60/180)^2 * Spectra.knee(l; ell=ellknee, alpha=alphaknee) 
    smoothnoisel .-= μK′n^2 * (π/60/180)^2 
    smoothnoisel[smoothnoisel .< 0] .= 0    
    noisel = smoothnoisel .+ whitenoisel
    return noisel, whitenoisel, smoothnoisel
end;

#-

Naz = @sblock let tmW=unscale(tmS0),  μK′n, snl, weight_θ, θ, φ, Δθ = θ[2]-θ[1], Δφ = φ[2]-φ[1]
    
    ## θgrid = range(0, π^(1/2), length=100_000).^2
    ## ----------
    dmax = 1.2maximum(CMBrings.geoθ1θ2Δφcol(θ[1], θ[1], φ .- φ[1]))
    θgrid = range(0, dmax^(1/2), length=100_000).^2

    covf  = Dierckx.Spline1D(
        θgrid, 
        Spectra.spec2spherecov(snl, θgrid), 
        k=3
    )
    
    covf_θ1θ2Δφℝ = function (θ1, θ2, Δφ′)
        rtn   = covf(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφ′))
        if θ1 == θ2
            cc = μK′n^2 * (π/60/180)^2
            pa = CS.SphereTransforms.Ωpix(θ1, Δθ, Δφ) # sin(θ1) * Δθ * Δφ
            rtn[Δφ′ .== 0] .+= cc / pa # <- since we are using ST grid
        end
        rtn
    end

    Naz = CMBrings.AzBlock(covf_θ1θ2Δφℝ, θ, φ, tmW) do N, k
        WD = Diagonal(weight_θ.(θ))
        ## -------------
        ## A = Symmetric(WD*(real.(N))*WD',:L)        
        ## C = cholesky(A, Val(false)) #, check=false)
        ## Cholesky(C.factors, C.uplo, C.info)
        ## -------------
        A = Symmetric(WD*(real.(N))*WD', :L)
        B = eigen(A)
        B.values[B.values .<= 0] .= 0
        return B 
    end

    Naz
end;



# negative Hessian  for ϕ gradient -> newton update
# ==============================

## n2s_ratio = 0.2
n2s_ratio = 0.1

ϕnnl = @sblock let ϕϕl, n2s_ratio, lmax = 11000
    l        = 0:lmax
    lpeak    = 40
    ϕnnl     = @. n2s_ratio * lpeak^4 * ϕϕl[lpeak+1] / l^4
    ϕnnl     = @. 1 / (1 / ϕnnl + 1 / ϕϕl)
    ϕnnl[1] = ϕnnl[2] = 0
    ϕnnl
end;

#-

## figure()
## (0:11000).^4 .* ϕϕl |> loglog
## (0:11000).^4 .* ϕnnl |> loglog
## (0:11000).^4 .* inv.(inv.(ϕnnl) .+ inv.(ϕϕl)) |> loglog
## (0:11000).^4 .* ϕnnl .* inv.(ϕnnl .+ ϕϕl) |> loglog

#-

NΦNaz = @sblock let tmW=unscale(tmS0), ϕnnl, θ, φ 
    
    ## θgrid = range(0, π^(1/2), length=100_000).^2
    ## ----------------
    dmax = 1.2maximum(CMBrings.geoθ1θ2Δφcol(θ[1], θ[1], φ .- φ[1]))
    θgrid = range(0, dmax^(1/2), length=100_000).^2

    covϕnn  = Dierckx.Spline1D(
        θgrid, 
        Spectra.spec2spherecov(ϕnnl, θgrid), 
        k=3
    )

    covϕnn_θ1θ2Δφℝ = (θ1,θ2,Δφ) -> covϕnn(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφ)) 

    NΦNaz  = CMBrings.AzBlock(covϕnn_θ1θ2Δφℝ, θ, φ, tmW) do A, k
        ## A = Symmetric(real.(A), :L)
        ## return A
        ## -------------
        A = Symmetric(real.(A), :L)
        B = eigen(A)
        B.values[B.values .<= 0] .= 0
        return Matrix(B) 
    end 

    NΦNaz 
end;


# Preconditioner (via g -> Precon_fctr \ g)
# ==============================

Precon_fctr = map(Σaz, Naz, Baz) do Σ, N, B
    A = B*Matrix(Σ)*B' + Matrix(N)
    ## --------------------
    C = cholesky(Symmetric(A, :L)) # , check=false)
    return Cholesky(C.factors, C.uplo, C.info)
    ## ---------------------
    ## C = eigen(Symmetric(A,:L))
    ## C.values[C.values .<= 0] .= 0
    ## return C 
end |> CMBrings.AzBlock;

# Use this when Baz is set to 1

## Precon_fctr = map(Σaz, Naz) do Σ, N
##     A = Matrix(Σ) + Matrix(N)
##     ## --------------------
##     C = cholesky(Symmetric(A,:L)) # , check=false)
##     return Cholesky(T_Precon.(C.factors), C.uplo, C.info)
##     ## ---------------------
##     ## B = eigen(Symmetric(A,:L))
##     ## B.values[B.values .<= 0] .= 0
##     ## return B 
## end |> AzBlock;



# Lensing
# ==================================================



∂θ, ∂φᵀ = @sblock let tmS0, θ, φ, T_fld=Float64

    Δθ = θ[3] - θ[2]
    # ∂θ′ = spdiagm(
    #         0 => fill(-1,length(θ)), 
    #         1 => fill(1,length(θ)-1),
    #     )
    # ∂θ′[end,1] =  1
    # ∂θ = T_fld(1 / (Δθ)) * ∂θ′
    ∂θ′ = spdiagm(
            -2 => fill( 1,length(θ)-2),
            -1 => fill(-8,length(θ)-1),
             1 => fill( 8,length(θ)-1),
             2 => fill(-1,length(θ)-2),
            )
    ∂θ′[1,end]   =  -8
    ∂θ′[1,end-1] =  1
    ∂θ′[2,end]   =  1
    ∂θ′[end,1]   =  8
    ∂θ′[end,2]   = -1
    ∂θ′[end-1,1] = -1
    ∂θ = T_fld(1 / (12Δθ)) * ∂θ′


    Δφ = φ[3] - φ[2]
    ∂φ  = spdiagm(
            -2 => fill( 1,length(φ)-2),
            -1 => fill(-8,length(φ)-1),
             1 => fill( 8,length(φ)-1),
             2 => fill(-1,length(φ)-2),
            )
    ∂φ[1,end]   =  -8
    ∂φ[1,end-1] =  1
    ∂φ[2,end]   =  1
    ∂φ[end,1]   =  8
    ∂φ[end,2]   =  -1
    ∂φ[end-1,1] =  -1
    ∂φᵀ = transpose(T_fld(1 / (12Δφ)) * ∂φ)


    ∂θ, ∂φᵀ
end

## ------- or alternatively ----------
## ∇!   = CMBrings.Nabla!((∂θ - ∂θ')/2, (∂φᵀ - ∂φᵀ')/2)
## ∇!_ϕ = CMBrings.Nabla!(∂θ, ∂φᵀ)
## ------- or ------------
∇!   = CMBrings.Pix1dFFTNabla!((∂θ - ∂θ')/2, tmS0)
∇!_ϕ = CMBrings.Pix1dFFTNabla!((∂θ - ∂θ')/2, tmS0)
## ∇!_ϕ = CMBrings.Pix1dFFTNabla!(∂θ, tmS0)
## ------- or ------------
## sz     = (length(θ), length(φ))
## period = (length(θ)*(θ[2]-θ[1]), length(φ)*(φ[2]-φ[1]))
## ∇!   = CMBrings.FFTNabla!(Float64, sz, period)
## ∇!_ϕ = CMBrings.FFTNabla!(Float64, sz, period)



Ł_fixd∂, ϕ2v!_fixd∂, ϕ2vᴴ!_fixd∂, Ł_free∂ = @sblock let tmS0, Mϕ, rcϕ, ∇!, ∇!_ϕ, nsteps_lensing = 14,  θ, φ 

    ## -------------
    sin⁻²θ = @. csc(θ)^2 

    ## leftlink =  n::Int -> ((cos.(range(-π,0,length=n)) .+ 1)./2).^2
    ## rightlink = n::Int -> ((cos.(range(0,π,length=n)) .+ 1)./2).^2
    ## nbθ, nbφ  = 20, 20

    maθ = ones(size(θ))
    ## maθ[2:nbθ+1]        =  leftlink(nbθ)
    ## maθ[end-nbθ:end-1]  =  rightlink(nbθ)
    ##  maθ[1] = 0
    ## maθ[end] = 0
    
    maφ = ones(size(θ))
    ## maφ[2:nbφ+1]        =  leftlink(nbφ)
    ## maφ[end-nbφ:end-1]  =  rightlink(nbφ)
    ## maφ[1] = 0
    ## maφ[end] = 0
    
    mvx₁_init = maθ ./ rcϕ
    mvx₂_init = sin⁻²θ .* maφ ./ rcϕ
    ## -------------

    mv1x = Mϕ[:]
    mv2x = Mϕ[:]

    mvx₁ = mvx₁_init .* mv1x
    mvx₂ = mvx₂_init .* mv2x

    ϕ2v!_fixd∂ = function (v::NTuple{2,Array}, ϕ::Array)
        ∇!_ϕ(v, ϕ)
        v[1] .*= mvx₁
        v[2] .*= mvx₂
        v
    end 

    ϕ2vᴴ!_fixd∂ = function (ϕ::Array, v::NTuple{2,Array})
        mv = (similar(v[1]), similar(v[2]))
        ∇!_ϕ'(mv, (mvx₁.*v[1], mvx₂.*v[2]) )
        ϕ .= mv[1] .+ mv[2]
        ϕ 
    end 

    Ł_fixd∂ = function (ϕ_az::Xfield)
        ϕ = ϕ_az[:]
        v = (similar(ϕ), similar(ϕ))
        ϕ2v!_fixd∂(v,ϕ)
        FieldLensing.ArrayLense(v, ∇!, 0, 1, nsteps_lensing)
    end

    Ł_free∂ = function (ϕ_az::Xfield)
        ϕ = ϕ_az[:]
        v = (similar(ϕ), similar(ϕ))
        ∇!_ϕ(v, ϕ)
        v[1] .*= mvx₁_init
        v[2] .*= mvx₂_init
        FieldLensing.ArrayLense(v, ∇!, 0, 1, nsteps_lensing)
    end

    Ł_fixd∂, ϕ2v!_fixd∂, ϕ2vᴴ!_fixd∂, Ł_free∂
end;






# Show lensing (zoomed into 1/2 of azimuth band).

@sblock let Ł=Ł_fixd∂, tmS0, Σaz, Φaz, φ, θ, fφ=1/2, hide_plots
    hide_plots && return

    ϕ_az = CMBrings.az_sim(tmS0, Φaz)
    Ln         = Ł(ϕ_az)
    t_az       = Xmap(CMBrings.az_sim(tmS0, Σaz))
    lnt_az     = Ln * t_az
    lense_time = @belapsed $Ln * $t_az
    t_az′      = Ln \ lnt_az

    imgs = Dict(
        1 => ϕ_az[:],
        2 => lnt_az[:],
        3 => (t_az - lnt_az)[:],
        4 => abs.(t_az[:] .- t_az′[:]), 
    )
    txt =  Dict(
        1 => "lensing potential",
        2 => "lense(CMB) ($(lense_time) seconds)",
        3 => "CMB - lense(CMB)",
        4 => "abs(CMB - unlense(lense(CMB)))", 
    )
    ctxt = Dict(
        4 => "w"
    )
    ## brickplot(imgs; txt=txt, ctxt=ctxt, fφ=fφ)
    CMBrings.diskplot(imgs, φ', π.-θ; txt=txt, nrows=2, fontsize=12)
end;


# t_az  = Xmap(CMBrings.az_sim(tmS0, Σaz))
# bt_az = Baz * t_az
# bt_az[:] |> matshow; colorbar()
# t_az[:] |> matshow; colorbar()
# (t_az - bt_az)[:][:,1:308] |> matshow

# @benchmark $Baz * $t_az
## 27 ms for 308 (308, 2048)


# Mixflow operator
# ==============================================

# perhaps make Ð a operator type ... 
# then you can define logabsdet(Ð, θ) and Ð(θ) -> linear op. 

# Ð = @sblock let tmT, Tcov, T̃cov, Wcov
#     # Ð = sqrt(T̃cov / Tcov)
#     Ð = sqrt((T̃cov + 2Wcov) / Tcov)
#     return Ð
# end


# Ð = @sblock let T=T_Σaz, tmW=unscale(tmS0), ttl, t̃tl, wnl, θ, φ

#     cl    =  (t̃tl .+ 2 .* wnl) ./ ttl 

#     dmax  = 1.2maximum(CMBrings.geoθ1θ2Δφcol(θ[1], θ[1], φ .- φ[1]))
#     θgrid = range(0, dmax^(1/2), length=100_000).^2
#     covf  = Dierckx.Spline1D(
#         θgrid, 
#         Spectra.spec2spherecov(cl, θgrid), 
#         k=3
#     )
#     covf_θ1θ2Δφℝ = (θ1,θ2,Δφ) -> covf(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφ)) 

#     Daz = CMBrings.AzBlock(covf_θ1θ2Δφℝ, θ, φ, tmW) do A, k
#         ## A = Symmetric(real.(A) + 1e-8*I ,:L)
#         ## A = Symmetric(real.(A),:L)
#         ## C = cholesky(A, Val(false)) #, check=false)
#         ## return Cholesky(T.(C.factors), C.uplo, C.info)
#         ## -------------
#         B = eigen(Symmetric(T.(real.(A)),:L))
#         B.values[B.values .<= 0] .= 0
#         return B 
#     end 
    
#     Daz 
# end;




Ð = map(Σaz) do Σ
    Matrix(I(size(Matrix(Σ),1)))
end |> CMBrings.AzBlock;


# Simulate data 
# ================================================

ϕ_az  = CMBrings.az_sim(tmS0, Φaz) |> Xmap
t_az  = CMBrings.az_sim(tmS0, Σaz) |> Xmap

# For the data we are currently using fixed boundary lensing

d_az  = Pr * (Baz * (Ł_fixd∂(ϕ_az)*t_az) + CMBrings.az_sim(tmS0, Naz)) |> Xmap;


# @time fsim_out, gwf, hst = CMBrings.update_f(
#     Ł_fixd∂(ϕ_az) ; 
#     data = d_az,
#     Pr, Qr, CMBcov = Σaz, Ncov = Naz , Bm = Baz, Precon = Precon_fctr,
#     pcg_nsteps = 100, 
# )


# @time fsim_out, gwf, hst = CMBrings.update_f(
#     DiagOp(Xfourier(tmS0, 1));
#     data = d_az,
#     Pr, Qr, CMBcov = Σaz, Ncov = Naz , Bm = Baz, Precon = Precon_fctr,
#     pcg_nsteps = 300, 
# )


# Put settings and needed parameters in ds ...
# ===========================================

ds = (;
    data   = d_az, 
    Ł      = Ł_fixd∂, 
    ϕ2v!   = ϕ2v!_fixd∂, 
    ϕ2vᴴ!  = ϕ2vᴴ!_fixd∂, ### but here we need the full gradient etc for transpose flow
    CMBcov = Σaz, 
    Φcov   = Φaz, 
    Ncov   = Naz, 
    Bm     = Baz, 
    NΦNcov = NΦNaz, 
    Precon = Precon_fctr, 
    ∇!, 
    tmS0, 
    Pr, 
    Qr, 
    Ð, 
    ## -------------- pcg ....
    grad_nsteps = 14, # seems sensable to match the lensing nsteps 
    ## -------------- pcg ....
    pcg_nsteps = 150, #200, 
    pcg_rel_tol = 1e-10,
    ## ------------- For NLopt ....
    upper_bound = 5.0,
    ftol_abs = 10,  ## within 50 loglikelihood points is just fine
    xtol_abs = 1e-2,
    solver = :LN_COBYLA, # :LN_NELDERMEAD, :LN_COBYLA, :LN_SBPLX, 
);







# newton/gibbs iterations
# ================================================

ϕ_cr  = Xmap(tmS0)
∇ϕ_cr_array = typeof(ϕ_cr)[]
gradϕ_array = typeof(ϕ_cr)[]
β_array     = Float64[]

# Warm up the pcg
@time p_cr, ginit, hst = CMBrings.update_f(
    DiagOp(Xmap(tmS0, 1));
    ds...,
    pcg_nsteps=300, 
)
p′_cr = Ð * p_cr;


# Fixme: needs to adjust for rcϕ in the likelihood or the gradient ...

# Gradient iterations

@showprogress for otr = 1:7
    global p′_cr, ϕ_cr, ginit, hst
    global ∇ϕ_cr_array, gradϕ_array, β_array

    ## WF update p_cr, p′_cr and ginit for pcg warm start
    # if otr == 1
    #       pcg_steps = 10
    # else 
        pcg_steps = 200
    # end
    @time p_cr, ginit, hst = CMBrings.update_f(
        ds.Ł(ϕ_cr); 
        ds..., 
        ginit=ginit, 
        pcg_nsteps=pcg_steps,
    )
    p′_cr = ds.Ł(ϕ_cr) * Ð * p_cr 
    @show hst[end], length(hst)
    @show CMBrings.ll_ϕf′(ϕ_cr, p′_cr; ds...)

    ## ϕ gradient
    @time gradϕ = CMBrings.∇ll_ϕf′(ϕ_cr, p′_cr; ds...)
    @time ∇ϕ_cr = ds.NΦNcov * gradϕ
    push!(gradϕ_array, gradϕ)
    push!(∇ϕ_cr_array, ∇ϕ_cr)
    
    ## linesearch 
    @time β = CMBrings.linesearch_ϕf′(∇ϕ_cr, ϕ_cr, p′_cr; ds...)
    @show β
    push!(β_array, β)

    ## update ϕ_cr
    ϕ_cr += β * ∇ϕ_cr
 
end




v_cr = (similar(ϕ_cr[:]), similar(ϕ_cr[:]))
v_az = (similar(ϕ_az[:]), similar(ϕ_az[:]))
ϕ2v!_fixd∂(v_cr, ϕ_cr[:])
ϕ2v!_fixd∂(v_az, ϕ_az[:])

v_az[1] |> matshow; colorbar()
v_cr[1] |> matshow; colorbar()

v_az[2] |> matshow; colorbar()
v_cr[2] |> matshow; colorbar()



# Test out the fourier basis for block matrices 
# ===========================

# nθ   = size_in(tmS0)[1]
# tmUθ = 𝕎(Float64, nθ, θ[2] - θ[1]) |> x->unitary_scale(x)*x

∇θ = (∂θ - ∂θ')/2
Δθ = ∇θ^2 / 1e5

Σmt = Matrix(Σaz[3])
# Σmt = Δθ * Matrix(Σaz[3]) * Δθ'
# Σmt = Σmt[10:end-10+1, 10:end-10+1]
Γ_pre = Matrix(rfft(Σmt,(1,))')
C_pre = Matrix(transpose(rfft(Σmt,(1,))))
Γr = rfft( real.(Γ_pre), (1,) ) 
Γi = rfft( imag.(Γ_pre), (1,) ) 
Cr = rfft( real.(C_pre), (1,) ) 
Ci = rfft( imag.(C_pre), (1,) ) 

Γ = Γr .+ im .* Γi
C = Cr .+ im .* Ci



Σnew = [ 
    Γ         C 
    conj.(C)  conj.(Γ)
] |> Hermitian    


# Γ .|> real .|> abs .|> log |> matshow; colorbar()
# Γ .|> imag .|> abs .|> log |> matshow; colorbar()
diag(Γ,1) .|> real |> plot
diag(C,1) .|> real |> plot

A = [   Γ .+ conj.(Γ)         im.*(Γ .- conj.(Γ))
      - im.*(Γ .- conj.(Γ))   Γ .+ conj.(Γ)        ]  .|> real |> Symmetric
B = [  (C.+conj.(C))        im.*(conj.(C) .- C)
      im.*(conj.(C) .- C)   .-(C.+conj.(C))  ]     .|> real |> Symmetric

Σnew = A + B


Σnew = [ 
    Γ         C 
    conj.(C)  conj.(Γ)
] |> Hermitian    

eigen(Σnew).values .|>  abs |> semilogy
eigen(Γ).values |> semilogy

Γuv = eigen(Γ)
Γuv.values |> semilogy # these should be the same ...

Γuv.vectors[:,end]    |> plot
Γuv.vectors[:,end-2]  |> plot
Γuv.vectors[:,end-10] |> plot
Γuv.vectors[:,end-20] |> plot

Γcut = Γ[1:100,:100]

x   = fft(Σmt[:,1])
pp  = vcat(real(x[1]), x[2:end] .+ x[end:-1:2])
pm  = vcat(imag(x[1]), x[2:end] .- x[end:-1:2])

e   = real.(pp) .+ im .* imag(pm)
b   = imag.(pp) .- im .* real(pm)

plot(real.(e))
plot(imag.(e))

real.(ifft(e)) |> plot
real.(ifft(b)) |> plot

## 

Σmt[:,10] |> plot

x = rfft(Σmt[:,1])


x1 = randn(15)
x2 = randn(15)
x  = x1 .+ im .* x2


y = fft(x1) .+ im .* fft(x2)
-> w = fft(x)[1:end÷2+1] holds all DoF if x1 or x2 is zero
-> w = fft(x)



# Old 
# ===========================

#- 
@sblock let fest = ϕ_cr, ftru = ϕ_az, tmS0, φ, θ, ∇!, Pr, hide_plots
    hide_plots && return

    ## set mask
    𝕄 = Pr
    ## 𝕄 = I

    ##------- raw potential
    fest_raw = fest  |> 
                    ## x -> x - mean(x[:][Pr[:] .> 0.9]) |> 
                    x->𝕄*x
    ftru_raw = ftru  |> 
                    ## x -> x - mean(x[:][Pr[:] .> 0.9]) |> 
                    x->𝕄*x
    ##------- smoothed laplace 
    fest_sΔ = fest  |>  x->CMBrings.laplace(x, θ, ∇!; padpix=5) |> 
                        x->CMBrings.smooth(x, θ, φ; fwhm′θ=15, fwhm′φ = 15) |>
                        x->𝕄*x 
    ftru_sΔ = ftru  |>  x->CMBrings.laplace(x, θ, ∇!; padpix=5) |> 
                        x->CMBrings.smooth(x, θ, φ; fwhm′θ=15, fwhm′φ = 15) |>
                        x->𝕄*x 
            
    ##---------------- Fourier filter
    k   = CMBrings.fullfreq(tmS0)
    fltr = abs.(k[2])
    ## fltr = ones(eltype_out(tmS0), size_out(tmS0))
    ## fltr[:,1:10] .= 0    
    𝔽 = Xfourier(tmS0,fltr) |> DiagOp
    fest_F = 𝕄 * 𝔽 * fest            
    ftru_F = 𝕄 * 𝔽 * ftru

    imgs = Dict(
        1 => ftru_raw[:],
        2 => fest_raw[:],
        3 => ftru_sΔ[:],
        4 => fest_sΔ[:],
        5 => ftru_F[:],
        6 => fest_F[:],
    )
    txt =  Dict(
        1 => "true ϕ",
        2 => "est ϕ",
        3 => "smoothed true Δϕ",
        4 => "smoothed est Δϕ",
        5 => "true ∂ϕ / ∂az",
        6 => "est ∂ϕ / ∂az",
    )

    diskplot(
        imgs, φ', π .- θ; 
        txt=txt, 
        nrows=2, fontsize=12 , vcenter=0, vmin_quantile=1e-6,
    )

    brickplot(
        imgs, 
        txt=txt,
        fφ=1/2
    )

end

#-




@sblock let Ł, Baz, lnt_cr, t_cr, t_az, d_az, ϕ_cr, ϕ_az, θ, φ, Pr, hide_plots
    hide_plots && return

    lnt_az = Ł(ϕ_az)*t_az
    imgs = Dict(

        1 => (Pr * (t_az - t_cr))[:],
        2 => (Pr * (lnt_az - t_az))[:],
    )
    txt =  Dict(
        1 => "f_true - f_est",
        2 => "lnf_true - f_true",
    )
    diskplot(imgs, φ', π.-θ; txt=txt, nrows=1, fontsize=12)
end;


#-


@sblock let Ł, Baz, lnt_cr, t_cr, t_az, d_az, ϕ_cr, ϕ_az, θ, φ, Pr, hide_plots
    hide_plots && return

    lnt_az = Ł(ϕ_az)*t_az
    imgs = Dict(

        1 => (Pr * (t_az - t_cr))[:],
        2 => (Pr * (lnt_az - lnt_cr))[:],
        3 =>  (Pr *( d_az - Pr * (Baz * lnt_cr) ))[:],
        4 =>  (Pr *( d_az - Pr * (Baz * lnt_az) ))[:]
    )
    txt =  Dict(
        1 => "Mask*(f_true - f_est)",
        2 => "Mask*(lnf_true - lnf_est)",
        3 => "data - M * B * lnf_est",
        4 => "data - M * B * lnf_true"
    )
    diskplot(imgs, φ', π.-θ; txt=txt, nrows=2, fontsize=12)
end;


#-

@sblock let Ł, Baz, lnt_cr, t_cr, t_az, d_az, ϕ_cr, ϕ_az, θ, φ, Pr, hide_plots
    hide_plots && return

    lnt_az = Ł(ϕ_az)*t_az
    imgs = Dict(
        1 => t_az[:],
        2 => t_cr[:], 
        3 => lnt_az[:], 
        4 => lnt_cr[:],
    )
    txt =  Dict(
        1 => "unlensed truth",
        2 => "unlensed estimate",
        3 => "lensed truth",
        4 => "lensed estimate",
    )
    diskplot(imgs, φ', π.-θ; txt=txt, nrows=2, fontsize=12)
end;


#-


ln_az = length(t_az[:])

zll_t_az = (dot(t_az, Σaz \ t_az) - ln_az) / sqrt(2*ln_az) # PCG sim
zll_t_cr = (dot(t_cr, Σaz \ t_cr) - ln_az) / sqrt(2*ln_az) # PCG sim
@show (zll_t_az, zll_t_cr)

zll_ϕ_az = (dot(ϕ_az, Φaz \ ϕ_az) - ln_az) / sqrt(2*ln_az) # PCG sim
zll_ϕ_cr = (dot(ϕ_cr, Φaz \ ϕ_cr) - ln_az) / sqrt(2*ln_az) # PCG sim
@show (zll_ϕ_az, zll_ϕ_cr)


# with the mean field we get this ....

## (zll_ϕ_az, zll_ϕ_cr) = (0.3685636660817692, -648.133252134063)
## (zll_t_az, zll_t_cr) = (1.4497029404433848, -3.545465681176255)

#-

CMBrings.llfield(ϕ_cr, ds.Φaz_fctr)[!] .|> abs .|> log |> matshow

CMBrings.llfield(t_cr, ds.Σaz_fctr)[!] .|> abs .|> log |> matshow


#-
figure()

plot(φ, ϕ_cr[:][75,:])
plot(φ, ϕ_cr[:][100,:])
plot(φ, ϕ_cr[:][150,:])
plot(φ, ϕ_cr[:][200,:])
plot(φ, ϕ_cr[:][210,:])

plot(φ,3.0 * 75 .* cos.(φ).*sin.(θ[200]))

matshow(ϕ_cr[:] .- 75 .* cos.(φ').* sin.(θ))
matshow(ϕ_az[:])

75 .* cos.(φ')

cosfield  = Xmap(tmS0, cos.(φ') .* sin.(θ))
Δcosfield = CMBrings.laplace(cosfield, θ, ∇!; padpix=5)
Δcosfield[:] |> matshow


ϕ_cr_test = deepcopy(ϕ_cr)

for k=1:2
    ϕ_cr_test.fd[:,k] .-= mean(ϕ_cr_test.fd[:,k])
end

ϕ_cr_test[:] |> matshow

ϕ_az[:] |> matshow