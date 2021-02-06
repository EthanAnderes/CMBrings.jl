
# Modules
# ==============================
using CMBrings
import CMBsphere as CS

using XFields
using Spectra
using FFTransforms
using FieldLensing 

using LinearAlgebra
using SparseArrays
using DelimitedFiles
using Statistics
using Dierckx: Spline1D
using LBblocks: @sblock
using PyPlot
using BenchmarkTools
using ProgressMeter
import Dierckx 
import NLopt

# 

if isdefined(Main, :IJulia) && Main.IJulia.inited
    hide_plots = false
else 
    hide_plots = true
end



# Set ring transforms
# ==============================


tmS0 = @sblock let 

    T_fld   = Float64

    nθ, nφ  = 308, 2048
    # nθ, nφ  = 308, 3072
    # nθ, nφ  = 308, 4095
    
    tmW  = 𝕀(nθ) ⊗ 𝕎(T_fld, nφ, 2π)
    tmS0  = unitary_scale(tmW) * tmW

    return tmS0
end



# Mask and CMBring observation region
# ==============================

# TODO: get rid of this stuff ...
T_fld = Float64 
T_Φaz    = Float64 
T_NΦNaz  = Float64 
T_Σaz    = Float64
T_Naz    = Float64
T_Baz    = Float64
T_Precon = Float64


#-

Pr, Qr, Mϕ, Ωℝ, θℝ, φℝ, Ωℝ64, θℝ64, φℝ64 = @sblock let  tmS0, QP_bdry=1e-5, fwhm′=150, T_fld

    ## --------- FIXME: make this work for any nθ, nφ ----
    data_mask_init = readdlm(joinpath(CMBrings.module_dir,"examples/lensing-spin0/FastTransform_mask_nθ3072_nφ4095.csv"), ',', Bool)
    nθ, nφ = size_in(tmS0)
    sθ_clip = (87*3072÷100 - nθ + 1):(87*3072÷100)
    sφ_clip = 1:2:4095

    CStmS0 = CS.𝕊0(size(data_mask_init)...)
    Ωℝ64 = CS.Ωpix(CStmS0)[sθ_clip]
    θℝ64, φℝ64 = CS.pix(CStmS0) |> x->(x[1][sθ_clip], x[2][sφ_clip])
    Ωℝ, θℝ, φℝ = T_fld.(Ωℝ64), T_fld.(θℝ64), T_fld.(φℝ64)

    ### ------------- data masking
    d_pr0x, d_qr0x = CS.PrQr(CStmS0, data_mask_init, fwhm′, QP_bdry)
    d_pr0x, d_qr0x = d_pr0x[sθ_clip,sφ_clip], d_qr0x[sθ_clip,sφ_clip]
    Pr = DiagOp(Xmap(tmS0, d_pr0x)) 
    Qr = DiagOp(Xmap(tmS0, d_qr0x))

    ### ------------- lensing displacement mask
    ϕ_pr0x, ϕ_qr0x = CS.PrQr(CStmS0, data_mask_init, fwhm′÷3, QP_bdry)
    mϕx   =  ϕ_pr0x[sθ_clip,sφ_clip] .+ ϕ_qr0x[sθ_clip,sφ_clip]
    mϕx .-= minimum(mϕx)
    mϕx ./= maximum(mϕx)
    Mϕ    = DiagOp(Xmap(tmS0, mϕx))

    Pr, Qr, Mϕ, Ωℝ, θℝ, φℝ, Ωℝ64, θℝ64, φℝ64
end;  




# Azimuthal ring mask

@sblock let ma=Pr[:], φℝ, θℝ, hide_plots
    hide_plots && return
    imgs = Dict(1=>ma)
    txt  = Dict(1=>"Mask")
    ctxt = Dict(1=>"w")
    CMBrings.brickplot(imgs; txt=txt, ctxt=ctxt, fφ=1)
    
    ## Fixme: ...
    ## CMBrings.diskplot(imgs, φℝ', π.-θℝ; txt=txt, nrows=1, fontsize=14)
end

# Plot √Ωpix over ring θℝ's 

@sblock let θℝ, φℝ, Ωℝ, hide_plots
    hide_plots && return
    fig,ax = subplots(1)
    ax.plot(θℝ, rad2deg.(sqrt.(Ωℝ)).*60, label="sqrt pixel area (arcmin)")
    ax.plot(θℝ, zero(θℝ) .+ rad2deg.(θℝ[2] - θℝ[1]).*60, label="Δθ (arcmin)")
    ## ax.plot(θℝ, zero(θℝ) .+ rad2deg.(φℝ[2] - φℝ[1]).*60, label="Δφ (arcmin)")
    ax.set_xlabel(L"polar coordinate $\theta$")
    ax.legend()
end;



# ϕϕ, TT covariance
# ================================

rcϕ = T_fld(1e2)

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
    ϕϕl[1] =  ϕϕl[3]/100
    ϕϕl[2] =  ϕϕl[3]/10

    ttl, t̃tl, rcϕ^2 .* ϕϕl
end;

# Note: ϕ is now the "new" rescaled version which need to be adjusted when 
# converted to a displacement (and the transpose of that ...)

#-
# Note: to get this positive definite it appears we need 
# twice precision for θℝ and φℝ
# Also appears we need 64 for Φaz cov 

Φaz = @sblock let T_cov=T_Φaz, tmW=unscale(tmS0), ϕϕl, θℝ=θℝ64, φℝ=φℝ64
    ## θgrid = range(0, π^(1/2), length=100_000).^2
    dmax = 1.2maximum(CMBrings.geoθ1θ2Δφcol(θℝ[1], θℝ[1], φℝ .- φℝ[1]))
    θgrid = range(0, dmax^(1/2), length=100_000).^2
    covf  = Spline1D(
        θgrid, 
        Spectra.spec2spherecov(ϕϕl, θgrid), 
        k=3
    )
    covf_θ1θ2Δφℝ = (θ1,θ2,Δφℝ) -> covf(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφℝ)) 

    Φaz = CMBrings.AzBlock(covf_θ1θ2Δφℝ, θℝ, φℝ, tmW) do A, k
        ## -------------
        ## A = Symmetric(real.(A) + 1e-9*I ,:L)
        ## A = Symmetric(real.(A),:L)
        ## C = cholesky(A, Val(false), check=false)
        ## return Cholesky(T_cov.(C.factors), C.uplo, C.info)
        ## -------------
        ## B = eigen(Symmetric(T_cov.(real.(A) + 1e-9*I),:L))
        B = eigen(Symmetric(T_cov.(real.(A)),:L))
        B.values[B.values .<= 0] .= 0
        return B 
    end

    Φaz
end;

#

Σaz = @sblock let T_cov=T_Σaz, tmW=unscale(tmS0), ttl, θℝ=θℝ64, φℝ=φℝ64
	##θgrid = range(0, π^(1/2), length=100_000).^2
    dmax = 1.2maximum(CMBrings.geoθ1θ2Δφcol(θℝ[1], θℝ[1], φℝ .- φℝ[1]))
    θgrid = range(0, dmax^(1/2), length=100_000).^2
    covf  = Spline1D(
        θgrid, 
        Spectra.spec2spherecov(ttl, θgrid), 
        k=3
    )
    covf_θ1θ2Δφℝ = (θ1,θ2,Δφℝ) -> covf(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφℝ)) 

    Σaz = CMBrings.AzBlock(covf_θ1θ2Δφℝ, θℝ, φℝ, tmW) do A, k
        ## A = Symmetric(real.(A) + 1e-8*I ,:L)
        ## A = Symmetric(real.(A),:L)
        ## C = cholesky(A, Val(false)) #, check=false)
        ## return Cholesky(T_cov.(C.factors), C.uplo, C.info)
        ## -------------
        B = eigen(Symmetric(T_cov.(real.(A)),:L))
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

bl = @sblock let beamfwhm, lmax = 11000  
    l = 0:lmax
    σ² = beamfwhm^2 / 8 / log(2)
    bl = @. exp( - σ²*l*(l+1) / 2)
    return bl
end;

#-

Baz = @sblock let T_cov=T_Baz, tmW=unscale(tmS0),  bl, θℝ=θℝ64, φℝ=φℝ64, Ωℝ=Ωℝ64, azmuth_transfer_k
    ##θgrid = range(0, π^(1/2), length=100_000).^2
    dmax = 1.2maximum(CMBrings.geoθ1θ2Δφcol(θℝ[1], θℝ[1], φℝ .- φℝ[1]))
    θgrid = range(0, dmax^(1/2), length=100_000).^2
    
    covf  = Spline1D(
        θgrid, 
        Spectra.spec2spherecov(bl, θgrid), 
        k=3
    )
    
    covf_θ1θ2Δφℝ = (θ1,θ2,Δφℝ) -> covf(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφℝ)) 

    Baz  = CMBrings.AzBlock(covf_θ1θ2Δφℝ, θℝ, φℝ, tmW) do Σ, k
        T_cov.(real.(Σ) * Diagonal(azmuth_transfer_k.(k, θℝ) .* Ωℝ))
    end

    Baz
end;

## Baz = 1

# Noise with weights weight and mask/projection
# ==============================

μK′n      = 2.5 # 10.0
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

Naz = @sblock let T_cov=T_Naz, tmW=unscale(tmS0),  μK′n, snl, weight_θ, θℝ=θℝ64, φℝ=φℝ64, Δθ = θℝ64[2]-θℝ64[1], Δφ = φℝ64[2]-φℝ64[1]
    ## θgrid = range(0, π^(1/2), length=100_000).^2
    dmax = 1.2maximum(CMBrings.geoθ1θ2Δφcol(θℝ[1], θℝ[1], φℝ .- φℝ[1]))
    θgrid = range(0, dmax^(1/2), length=100_000).^2
    covf  = Spline1D(
        θgrid, 
        Spectra.spec2spherecov(snl, θgrid), 
        k=3
    )
    
    covf_θ1θ2Δφℝ = function (θ1, θ2, Δφℝ)
        rtn   = covf(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφℝ))
        if θ1 == θ2
            cc = μK′n^2 * (π/60/180)^2
            pa = CS.Ωpix(θ1, Δθ, Δφ) # sin(θ1) * Δθ * Δφ
            rtn[Δφℝ .== 0] .+= cc / pa # <- since we are using ST grid
        end
        rtn
    end

    Naz = CMBrings.AzBlock(covf_θ1θ2Δφℝ, θℝ, φℝ, tmW) do N, k
        WD = Diagonal(weight_θ.(θℝ))
        ## -------------
        ## A = Symmetric(WD*(real.(N))*WD',:L)        
        ## C = cholesky(A, Val(false)) #, check=false)
        ## Cholesky(T_cov.(C.factors), C.uplo, C.info)
        ## -------------
        A = Symmetric(T_cov.(WD*(real.(N))*WD'),:L)
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
    ϕnnl   = @. 1 / (1 / ϕnnl + 1 / ϕϕl)
    ϕnnl
end;


## figure()
## (0:11000).^4 .* ϕϕl |> loglog
## (0:11000).^4 .* ϕnnl |> loglog
## (0:11000).^4 .* inv.(inv.(ϕnnl) .+ inv.(ϕϕl)) |> loglog
## (0:11000).^4 .* ϕnnl .* inv.(ϕnnl .+ ϕϕl) |> loglog

NΦNaz = @sblock let T_cov=T_NΦNaz, tmW=unscale(tmS0), ϕnnl, θℝ=θℝ64, φℝ=φℝ64 
    
    ##θgrid = range(0, π^(1/2), length=100_000).^2
    dmax = 1.2maximum(CMBrings.geoθ1θ2Δφcol(θℝ[1], θℝ[1], φℝ .- φℝ[1]))
    θgrid = range(0, dmax^(1/2), length=100_000).^2

    covϕnn  = Spline1D(
        θgrid, 
        Spectra.spec2spherecov(ϕnnl, θgrid), 
        k=3
    )

    covϕnn_θ1θ2Δφℝ = (θ1,θ2,Δφℝ) -> covϕnn(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφℝ)) 

    NΦNaz  = CMBrings.AzBlock(covϕnn_θ1θ2Δφℝ, θℝ, φℝ, tmW) do A, k
        ## A = Symmetric(T_cov.(real.(A)),:L)
        ## return A
        ## -------------
        A = Symmetric(T_cov.(real.(A)),:L)
        B = eigen(A)
        B.values[B.values .<= 0] .= 0
        return Matrix(B) 
    end 

    NΦNaz 
end;



# Band limit the updates 

## for i = length(NΦNaz)÷2:length(NΦNaz)
##     NΦNaz[i] .*= 0
## end

#-

## ϕnnl = @sblock let ϕϕl, n2s_ratio, lmax = 11000
##     l        = 0:lmax
##     lpeak    = 40
##     ϕnnl     = @. n2s_ratio * lpeak^4 * ϕϕl[lpeak+1] / l^4
##     ϕnnl[1] *= 1e5
##     ϕnnl[2] *= 1e5
##     ## ϕnnl   = @. 1 / (1 / nnl + 1 / ϕϕl)
##     ϕnnl
## end;
## 
## NΦNaz = @sblock let T_cov=T_NΦNaz, tmW, Φaz, ϕnnl, θℝ=θℝ64, φℝ=φℝ64 
## 	
## 	##θgrid = range(0, π^(1/2), length=100_000).^2
##     dmax = 1.2maximum(CMBrings.geoθ1θ2Δφcol(θℝ[1], θℝ[1], φℝ .- φℝ[1]))
##     θgrid = range(0, dmax^(1/2), length=100_000).^2
## 
##     covϕnn  = Spline1D(
##         θgrid, 
##         Spectra.spec2spherecov(ϕnnl, θgrid), 
##         k=3
##     )
## 
##     covϕnn_θ1θ2Δφℝ = (θ1,θ2,Δφℝ) -> covϕnn(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφℝ)) 
## 
##     Naz = AzBlock(covϕnn_θ1θ2Δφℝ, θℝ, φℝ, tmW) do A, k
##         ## A = Symmetric(T_cov.(real.(A)),:L)
##         ## return A
##         ## -------------
##         A = Symmetric(T_cov.(real.(A)),:L)
##         B = eigen(A)
##         B.values[B.values .<= 0] .= 0
##         return B 
##     end 
## 
## 	NΦNaz  = map(Φaz, Naz) do Φ, N
##         ## N * inv(cholesky(Symmetric(Matrix(Φ) + N))) # worked well with float64 
##         ## N / Symmetric(Matrix(Φ) + N) ## testing ... !!!!!
##         ## pinv(pinv(Matrix(Φ)) + pinv(Matrix(N))) ## try this too
## 	    A = pinv(eigen(Symmetric(Matrix(pinv(Φ)) + Matrix(pinv(N)))))
##         A.values[A.values .<= 0] .= 0
##         ## return A 
##         return Matrix(A) 
##     end |> AzBlock
## 
## 
##     NΦNaz 
## end;

## Note that in an earlier version that worked ... Φaz and NΦNaz where both kept at 
## Float64 resolution

# Preconditioner (via g -> Precon_fctr \ g)
# ==============================

Precon_fctr = map(Σaz, Naz, Baz) do Σ, N, B
    A = B*Matrix(Σ)*B' + Matrix(N)
    ## --------------------
    C = cholesky(Symmetric(A,:L)) # , check=false)
    return Cholesky(T_Precon.(C.factors), C.uplo, C.info)
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



∂θ = @sblock let tmS0, θℝ, T_fld

    Δθℝ = θℝ[2] - θℝ[1]
    ∂θ′ = spdiagm(
            0 => fill(-1,length(θℝ)), 
            1 => fill(1,length(θℝ)-1),
        )
    ∂θ′[end,1] =  1
    ∂θ = T_fld(1 / (Δθℝ)) * ∂θ′

    ∂θ
end

## ------- or alternatively ----------
## ∇!   = CMBrings.Nabla!((∂θ - ∂θ')/2, (∂φᵀ - ∂φᵀ')/2)
## ∇!_ϕ = CMBrings.Nabla!(∂θ, ∂φᵀ)
## ------- or ------------
∇!   = CMBrings.Pix1dFFTNabla!((∂θ - ∂θ')/2, tmS0)
∇!_ϕ = CMBrings.Pix1dFFTNabla!(∂θ, tmS0)


Ł_fixd∂, ϕ2v!_fixd∂, ϕ2vᴴ!_fixd∂, Ł_free∂ = @sblock let tmS0, Mϕ, rcϕ, ∇!, ∇!_ϕ, nsteps_lensing = 14,  θℝ=θℝ64, φℝ=φℝ64 

    ## -------------
    sin⁻²θ = @. csc(θℝ)^2 

    leftlink =  n::Int -> ((cos.(range(-π,0,length=n)) .+ 1)./2).^2
    rightlink = n::Int -> ((cos.(range(0,π,length=n)) .+ 1)./2).^2
    nbθ, nbφ  = 20, 5
    maθ = ones(size(θℝ))
    maθ[2:nbθ+1]        =  leftlink(nbθ)
    maθ[end-nbθ:end-1]  =  rightlink(nbθ)
    maθ[1] = maθ[end] = 0
    maφ = ones(size(θℝ))
    maφ[2:nbφ+1]        =  leftlink(nbφ)
    maφ[end-nbφ:end-1]  =  rightlink(nbφ)
    maφ[1] = maφ[end] = 0
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

@sblock let Ł=Ł_fixd∂, tmS0, Σaz, Φaz, φℝ, θℝ, fφ=1/2, hide_plots
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
    CMBrings.diskplot(imgs, φℝ', π.-θℝ; txt=txt, nrows=2, fontsize=12)
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


Ð = @sblock let T=T_Σaz, tmW=unscale(tmS0), ttl, t̃tl, wnl, θℝ=θℝ64, φℝ=φℝ64

    cl    =  t̃tl .+ 2 .* wnl ./ ttl 

    dmax  = 1.2maximum(CMBrings.geoθ1θ2Δφcol(θℝ[1], θℝ[1], φℝ .- φℝ[1]))
    θgrid = range(0, dmax^(1/2), length=100_000).^2
    covf  = Spline1D(
        θgrid, 
        Spectra.spec2spherecov(cl, θgrid), 
        k=3
    )
    covf_θ1θ2Δφℝ = (θ1,θ2,Δφℝ) -> covf(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφℝ)) 

    Daz = CMBrings.AzBlock(covf_θ1θ2Δφℝ, θℝ, φℝ, tmW) do A, k
        ## A = Symmetric(real.(A) + 1e-8*I ,:L)
        ## A = Symmetric(real.(A),:L)
        ## C = cholesky(A, Val(false)) #, check=false)
        ## return Cholesky(T.(C.factors), C.uplo, C.info)
        ## -------------
        B = eigen(Symmetric(T.(real.(A)),:L))
        B.values[B.values .<= 0] .= 0
        return B 
    end 
    
    Daz 
end;



# Simulate data 
# ================================================

ϕ_az  = CMBrings.az_sim(tmS0, Φaz) |> Xmap
t_az  = CMBrings.az_sim(tmS0, Σaz) |> Xmap
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
    Ł      = Ł_fixd∂, ### here we are using sub_Ł
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
    upper_bound = 0.2,
    ftol_abs = 10,  ## within 50 loglikelihood points is just fine
    xtol_abs = 1e-7,
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
    DiagOp(Xfourier(tmS0, 1));
    ds...,
    pcg_nsteps=300, 
)
p′_cr = Ð * p_cr;



# Gradient iterations

#@showprogress for otr = 1:25
@showprogress for otr = 1:3
    global p′_cr, ϕ_cr, ginit, hst
    global ∇ϕ_cr_array, gradϕ_array, β_array

    ## WF update p_cr, p′_cr and ginit for pcg warm start
    if otr == 1
         pcg_steps = 10
    else 
        pcg_steps = 150
    end
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










# Old 
# ===========================




# TODO: see if you can adjust the hessian with these samples 
# Wouldn't a wishart type draw work? 

ϕ_cr  = Xmap(tmS0)
ginit = Xmap(tmS0)
∇ϕ_cr = Xmap(tmS0)
∇ϕ_cr_array  = typeof(∇ϕ_cr)[]

# iterate ...
@showprogress for otr = 1:4
    global lnt_cr, t_cr, hst
    global ∇ϕ_cr_array, gradϕ
    
    ## for itr = 1:4
        n′  = CMBrings.az_sim(tmS0, Naz) |> Xfourier
        f′  = CMBrings.az_sim(tmS0, Σaz) |> Xfourier
        data′ = Pr * (Baz * (Ł(ϕ_cr) * f′)) +  Pr * n′ |> Xfourier

        data′ *= 0
        f′ *=0
        @time lnt_cr, t_cr, ginit, hst = CMBrings.update_lnf_f(ϕ_cr, d_az; data′, f′, ginit, ds...)
        # @show hst[end]

        @time gradϕ   = CMBrings.∇ϕ(ϕ_cr, lnt_cr, d_az; ds...)
        ## @time gradϕ = ∇ϕ(ϕ_cr, lnt_cr, d_az; ds...)

        @time ∇ϕ_cr = NΦNaz * gradϕ - NΦNaz * (Φaz \ ϕ_cr)  |> Xmap
        @time β = linesearchϕ(∇ϕ_cr, ϕ_cr, lnt_cr, d_az; ds...)

        push!(∇ϕ_cr_array, β * ∇ϕ_cr)
    ## end

    ϕ_cr += mean(∇ϕ_cr_array) 
    ∇ϕ_cr_array = typeof(∇ϕ_cr)[]
end


  
## βs = collect(range(0., .05, length = 25))
## lls1 = zeros(T_fld, length(βs))
## lls2 = zeros(T_fld, length(βs))
## for i=1:length(βs)
##     ϕβ = ϕ_cr + βs[i] * ∇ϕ_cr
##     ## ϕβ.fd[:,1] .= 0
##     
##     t_test = Ł(ϕβ) \ lnt_cr |> Xfourier
##     t_test.fd[:,end-1:end] .= 0 
##     lls1[i] = CMBrings.llϕ(t_test, ds.Σaz_fctr)
## 
##     ## lls1[i] = CMBrings.lllnf(ϕβ, lnt_cr,  ds.Ł, ds.Σaz_fctr) 
##     lls2[i] = CMBrings.llϕ(ϕβ, ds.Φaz_fctr)
## end 
## plot(βs, lls1)
## plot(βs, lls2)
## 
## plot(βs, lls1 .+ lls2)
## 
## hcat(βs, lls1, lls2, lls1 .+ lls2 ./ 100)
## 
## ϕβ = ϕ_cr + 0.015 * ∇ϕ_cr
## t_test = Ł(ϕβ) \ lnt_cr |> Xfourier
## t_test.fd[:,end] .= 0 
## CMBrings.llϕ(t_test, ds.Σaz_fctr)
## CMBrings.lllnf(ϕβ, lnt_cr,  ds.Ł, ds.Σaz_fctr) 


## CMBrings.llfield(ϕβ, ds.Φaz_fctr)[!] .|> abs .|> log |> matshow
## CMBrings.llfield(t_test, ds.Σaz_fctr)[!] .|> abs .|> log |> matshow


## ϕ_cr     = Xfourier(tmS0)
## ginit_cr = Xfourier(tmS0)
## ∇ϕ_cr    = Xfourier(tmS0)
## 
## 
## # iterate ...
## @showprogress for otr = 1:10
##     global lnt_cr, t_cr, inHgrad, hst 
##     global ∇ϕ_cr_array, lnt_cr_array, ginit_array
## 
## 	∇ϕ_cr_array   = typeof(∇ϕ_cr)[]
## 	lnt_cr_array  = typeof(ϕ_cr)[]
## 	ginit_array   = typeof(ginit_cr)[]
## 
##     f′  = CMBrings.az_sim(tmU, Σaz) |> Xfourier
##     n′  = CMBrings.az_sim(tmU, Naz) |> Xfourier
##     
##     for itr = 1:1
##         data′ = Pr * (Baz * (Ł(ϕ_cr) * f′)) +  Pr * n′ |> Xfourier;
##         @time lnt_cr, t_cr, ginit_wf, hst = CMBrings.update_lnf_f(ϕ_cr, d_az; data′, f′, ginit=ginit_cr, ds...)
##         ## @show hst[end]
##         ∇ϕ_cr = CMBrings.∇ϕ(ϕ_cr, lnt_cr, d_az; ds...)
##         push!(∇ϕ_cr_array,  ∇ϕ_cr)
##         push!(lnt_cr_array, lnt_cr)
##         push!(ginit_array,  ginit_wf)
##     end
## 
##     @time inHgrad, β = update_ϕ_maxlllnf(mean(∇ϕ_cr_array), ϕ_cr, lnt_cr_array, d_az; ds...)
##     ## @time inHgrad, β = update_ϕ_meanlllnf(mean(∇ϕ_cr_array), ϕ_cr, lnt_cr_array, d_az; ds...)
##     ϕ_cr += β * inHgrad
##     ginit_cr = mean(ginit_array)
## 
## 
## end
## 

## dΔlnf     = Baz' * (Ma * (Naz_fctr \ (Pr \ (data - Pr * (Baz * (Ł(ϕ_az)*t_az))))))
## 
## 
## 
## 

## ϕβ = ϕ_cr + 0.05 * inHgrad;
## 
## ϕβ[:] |> matshow
## ϕ_az[:] |> matshow
## 
## 
## CMBrings.llfield(ϕβ, ds.Φaz_fctr)[!] .|> abs .|> log |> matshow
## CMBrings.lllnf(ϕβ, lnt_cr,  ds.Ł, ds.Σaz_fctr) 
## CMBrings.llϕ(ϕβ, ds.Φaz_fctr)
## 
## (ds.Ł(ϕβ) \ lnt_cr)[:] |> matshow
## lnt_cr[:] |> matshow
## t_cr[:] |> matshow

##  t_cr[:] .- lnt_cr[:] |> matshow; colorbar()
##  t_az[:] .- (Ł(ϕ_az)*t_az)[:]  |> matshow; colorbar()

##  t_cr[:] .- t_az[:] |> matshow; colorbar()
##  (Pr*(t_cr - t_az))[:] |> matshow; colorbar()
##  (Pr*(lnt_cr - (Ł(ϕ_az)*t_az)))[:] |> matshow; colorbar()

##  d_az[:] .- lnt_cr[:] |> matshow
##  d_az[:] .- (Ł(ϕ_az)*t_az)[:] |> matshow

## t_cr[!] .|> abs .|> log |> matshow; colorbar() 
## t_az[!] .|> abs .|> log |> matshow; colorbar() 
## 
## ϕ_cr[!] .|> abs .|> log |> matshow; colorbar() 
## ϕ_az[!] .|> abs .|> log |> matshow; colorbar() 
## 
## figure()
## for i = 2:4
##     semilogy(abs.(ϕ_sumi[!][:,i]))
## end
## figure()
## for i = 2:4
##     semilogy(abs.(ϕ_az[!][:,i]), ":")
## end
## 
## ϕ_sumi.fd[:,2:3] .= ϕ_az.fd[:,2:3]



#- 
@sblock let fest = ϕ_cr, ftru = ϕ_az, tmU, φℝ, θℝ, ∇!, Pr, hide_plots
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
    fest_sΔ = fest  |>  x->CMBrings.laplace(x, θℝ, ∇!; padpix=5) |> 
                        x->CMBrings.smooth(x, θℝ, φℝ; fwhm′θ=15, fwhm′φ = 15) |>
                        x->𝕄*x 
    ftru_sΔ = ftru  |>  x->CMBrings.laplace(x, θℝ, ∇!; padpix=5) |> 
                        x->CMBrings.smooth(x, θℝ, φℝ; fwhm′θ=15, fwhm′φ = 15) |>
                        x->𝕄*x 
            
    ##---------------- Fourier filter
    k   = CMBrings.fullfreq(tmU)
    fltr = abs.(k[2])
    ## fltr = ones(eltype_out(tmU), size_out(tmU))
    ## fltr[:,1:10] .= 0    
    𝔽 = Xfourier(tmU,fltr) |> DiagOp
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
        imgs, φℝ', π .- θℝ; 
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




@sblock let Ł, Baz, lnt_cr, t_cr, t_az, d_az, ϕ_cr, ϕ_az, θℝ, φℝ, Pr, hide_plots
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
    diskplot(imgs, φℝ', π.-θℝ; txt=txt, nrows=1, fontsize=12)
end;


#-


@sblock let Ł, Baz, lnt_cr, t_cr, t_az, d_az, ϕ_cr, ϕ_az, θℝ, φℝ, Pr, hide_plots
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
    diskplot(imgs, φℝ', π.-θℝ; txt=txt, nrows=2, fontsize=12)
end;


#-

@sblock let Ł, Baz, lnt_cr, t_cr, t_az, d_az, ϕ_cr, ϕ_az, θℝ, φℝ, Pr, hide_plots
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
    diskplot(imgs, φℝ', π.-θℝ; txt=txt, nrows=2, fontsize=12)
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

plot(φℝ, ϕ_cr[:][75,:])
plot(φℝ, ϕ_cr[:][100,:])
plot(φℝ, ϕ_cr[:][150,:])
plot(φℝ, ϕ_cr[:][200,:])
plot(φℝ, ϕ_cr[:][210,:])

plot(φℝ,3.0 * 75 .* cos.(φℝ).*sin.(θℝ[200]))

matshow(ϕ_cr[:] .- 75 .* cos.(φℝ').* sin.(θℝ))
matshow(ϕ_az[:])

75 .* cos.(φℝ')

cosfield  = Xmap(tmU, cos.(φℝ') .* sin.(θℝ))
Δcosfield = CMBrings.laplace(cosfield, θℝ, ∇!; padpix=5)
Δcosfield[:] |> matshow


ϕ_cr_test = deepcopy(ϕ_cr)

for k=1:2
    ϕ_cr_test.fd[:,k] .-= mean(ϕ_cr_test.fd[:,k])
end

ϕ_cr_test[:] |> matshow

ϕ_az[:] |> matshow