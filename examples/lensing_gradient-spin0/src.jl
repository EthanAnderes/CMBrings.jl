
# Modules
# ==============================
using FFTW
## FFTW.set_num_threads(1)

using LinearAlgebra
## BLAS.set_num_threads(1)
## BLAS.vendor() 
## perhaps try and use system blas?? 


using SparseArrays

using CMBrings
using CMBrings: pcg, brickplot, diskplot
using CMBrings: AzBlock, check_factorization, az_sim
using CMBrings: Nabla!, FFTNabla!
using CMBrings.FieldLensing: ArrayLense
using XFields
using Spectra

import FFTransforms
using FFTransforms: 𝕀, 𝕎, r𝕎, ⊗, unitary_scale, ordinary_scale, fullfreq
import CMBsphere
const ST = CMBsphere.SphereTransforms

using DelimitedFiles
using Statistics
using Dierckx: Spline1D
using LBblocks: @sblock
using PyPlot
using NLopt
using BenchmarkTools
using ProgressMeter

## These are for the definition of τpotential 
using LoopVectorization
using FieldLensing 

hide_plots = true

# Mask and CMBring observation region
# ==============================


T_fld = Float64 

T_Φaz    = Float64 
T_NΦNaz  = Float64 
T_Σaz    = Float64

T_Naz    = Float64
T_Baz    = Float64
T_Precon = Float64

QP_boundry_clearance = 1e-5 

#-

ma, maᶜ, Ωℝ, θℝ, φℝ, Ωℝ64, θℝ64, φℝ64 = @sblock let QP_boundry_clearance, T_fld

    ## ------------------
    ## ma𝕊 = readdlm("FastTransform_mask_nθ3072_nφ4095.csv", ',', Bool)
    ## nθ𝕊, nφ𝕊 = size(ma𝕊)
    ## sθ_clip = (80*nθ𝕊÷100):(90*nθ𝕊÷100)
    ## sθ_clip = (78*nθ𝕊÷100):(87*nθ𝕊÷100) # default
    ## # sθ_clip = (75*nθ𝕊÷100):(85*nθ𝕊÷100)
    ## # sθ_clip = (72*nθ𝕊÷100):(87*nθ𝕊÷100)
    ## # sθ_clip = (69*nθ𝕊÷100):(90*nθ𝕊÷100)
    ## # sθ_clip = (69*nθ𝕊÷100):(90*nθ𝕊÷100)
    ## ------------------
    ## # ma𝕊      = readdlm("FastTransform_mask_spole_nθ3072_nφ4095.csv", ',', Bool)
    ## ma𝕊      = readdlm("FastTransform_mask_spole_nθ3072_nφ3071.csv", ',', Bool)
    ## nθ𝕊, nφ𝕊 = size(ma𝕊)
    ## sθ_clip  = (87*nθ𝕊÷100):(98*nθ𝕊÷100) 
    ## sθ_clip  = (84*nθ𝕊÷100):(98*nθ𝕊÷100) # default
    ## # sθ_clip  = (82*nθ𝕊÷100):(97*nθ𝕊÷100)
    ## # sθ_clip  = (87*nθ𝕊÷100):(985*nθ𝕊÷1000)
    ## # sθ_clip  = (82*nθ𝕊÷100):(99*nθ𝕊÷100)
    ## ------------------
    ## ma𝕊      = readdlm("FastTransform_mask_nearpole_nθ3072_nφ3071.csv", ',', Bool)
    ## nθ𝕊, nφ𝕊 = size(ma𝕊)
    ## sθ_clip  = (82*nθ𝕊÷100):(98*nθ𝕊÷100)
    ## ---------------------
    ma𝕊  = readdlm("FastTransform_mask_mid2pole_nθ2560_nφ3071.csv", ',', Bool)
    ## ma𝕊  = readdlm("FastTransform_mask_mid2pole_nθ2560_nφ4095.csv", ',', Bool)
    nθ𝕊, nφ𝕊 = size(ma𝕊)
    ## sθ_clip  = (79*nθ𝕊÷100):(96*nθ𝕊÷100)
    ## sθ_clip  = (80*nθ𝕊÷100):(90*nθ𝕊÷100)
    sθ_clip  = (84*nθ𝕊÷100):(95*nθ𝕊÷100)


    s0 = ST.𝕊(Float64, nθ𝕊, nφ𝕊, 0)
    Ωℝ64 = ST.Ωpix(s0)[sθ_clip]
    θℝ64, φℝ64 = ST.pix(s0) |> x->(x[1][sθ_clip], x[2])
    ## regardless of the types T_fld and T_cov it appears 
    ## we need full resolution versions of θℝ, φℝ and Ωℝ

    ## Here are the field storage versions
    Ωℝ, θℝ, φℝ = T_fld.(Ωℝ64), T_fld.(θℝ64), T_fld.(φℝ64)


    𝕨 = r𝕎(nθ𝕊, π) ⊗ r𝕎(nφ𝕊, 2π) |> x-> ordinary_scale(x)*x
    beamfwhm1 = (arcmin=200.0; deg2rad(arcmin/60))
    beamfwhm2 = (arcmin=200.0; deg2rad(arcmin/60))
    ## beamfwhm1 = (arcmin=500.0; deg2rad(arcmin/60))
    ## beamfwhm2 = (arcmin=500.0; deg2rad(arcmin/60))
    σ²1 = beamfwhm1^2 / 8 / log(2)
    σ²2 = beamfwhm2^2 / 8 / log(2)
    k   = fullfreq(𝕨)
    bk  = @. exp( - σ²1 * k[1]^2 / 2) * exp( - σ²2 * k[2]^2 / 2)
    Bt  = DiagOp(Xfourier(𝕨, bk)) 

    ps_qs = ma𝕊 .- .!(ma𝕊 .> 0)
    Bps_qs =  (Bt * Xmap(𝕨, ps_qs))[:]
    psBool = @. Bps_qs > 0

    Aps_qs   = @. abs(Bps_qs)
    Aps_qs .+= QP_boundry_clearance
    Aps_qs ./= maximum(Aps_qs)
    ps = Aps_qs .* psBool
    qs = Aps_qs .* .!psBool

    @assert all(abs.(qs.*ps) .== 0)
    @assert all(abs.(qs) .+ abs.(ps) .> 0)

    Ps  = DiagOp(Xmap(s0, ps))
    Qs  = DiagOp(Xmap(s0, qs))


    T_fld.(Ps[:][sθ_clip,:]), T_fld.(Qs[:][sθ_clip,:]), Ωℝ, θℝ, φℝ,  Ωℝ64, θℝ64, φℝ64
end;  

# Turn off the mask 

## ma  = ones(size(ma)) 
## maᶜ = zeros(size(ma)) 

# Azimuthal ring mask

@sblock let ma, φℝ, θℝ, hide_plots
    hide_plots && return
    imgs = Dict(1=>ma)
    txt  = Dict(1=>"Mask")
    ctxt = Dict(1=>"w")
    ## brickplot(imgs; txt=txt, ctxt=ctxt, fφ=1)
    diskplot(imgs, φℝ', π.-θℝ; txt=txt, nrows=1, fontsize=14)
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


# Set ring transforms
# ==============================


tmU, tmW = @sblock let T_fld, nθ=length(θℝ), nφ=length(φℝ)
    tmW64    = 𝕀(nθ) ⊗ 𝕎(Float64, nφ, 2π)
    tmW_fld  = 𝕀(nθ) ⊗ 𝕎(T_fld, nφ, 2π)
    tmU_fld  = unitary_scale(tmW_fld)*tmW_fld
    tmU_fld, tmW64
end


# masking 
# ==========================

Pr, Qr = @sblock let tmU, ma, maᶜ
    Pr = Xmap(tmU, ma)
    Qr = Xmap(tmU, maᶜ)
    DiagOp(Pr), DiagOp(Qr)
end;



# ϕϕ, TT covariance
# ================================

rcϕ = T_fld(1e5)

ttl, ϕϕl = @sblock let rcϕ, lmax = 11000
    l = 0:lmax
    cld = Spectra.camb_cls(lmax=lmax)
    ctlvec = cld[:unlen_scalar] |> x->(x[:Ctt] ./ x[:factor_on_cl_cmb])
    ctlvec[2] = 1e-1 * ctlvec[3]
    ctlvec[1] = 1e-2 * ctlvec[3]
    cϕlvec = cld[:phi] |> x->(x[:Cϕϕ] ./ x[:factor_on_cl_phi])
    ## cϕlvec = Spectra.cϕl_approx.(l)
    cϕlvec[2] =  1e-2 *  cϕlvec[3]
    cϕlvec[1] =  1e-2 *  cϕlvec[3]

    ctlvec, rcϕ^2 .* cϕlvec
end;

# Note: ϕ is now the "new" rescaled version which need to be adjusted when 
# converted to a displacement (and the transpose of that ...)

#-
# Note: to get this positive definite it appears we need 
# twice precision for θℝ and φℝ
# Also appears we need 64 for Φaz cov 

Φaz = @sblock let T_cov=T_Φaz, tmW, ϕϕl, θℝ=θℝ64, φℝ=φℝ64
    ## θgrid = range(0, π^(1/2), length=100_000).^2
    dmax = 1.2maximum(CMBrings.geoθ1θ2Δφcol(θℝ[1], θℝ[1], φℝ .- φℝ[1]))
    θgrid = range(0, dmax^(1/2), length=100_000).^2
    covf  = Spline1D(
        θgrid, 
        Spectra.spec2spherecov(ϕϕl, θgrid), 
        k=3
    )
    covf_θ1θ2Δφℝ = (θ1,θ2,Δφℝ) -> covf(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφℝ)) 

    Φaz = AzBlock(covf_θ1θ2Δφℝ, θℝ, φℝ, tmW) do A, k
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

#-

Σaz = @sblock let T_cov=T_Σaz, tmW, ttl, θℝ=θℝ64, φℝ=φℝ64
	##θgrid = range(0, π^(1/2), length=100_000).^2
    dmax = 1.2maximum(CMBrings.geoθ1θ2Δφcol(θℝ[1], θℝ[1], φℝ .- φℝ[1]))
    θgrid = range(0, dmax^(1/2), length=100_000).^2
    covf  = Spline1D(
        θgrid, 
        Spectra.spec2spherecov(ttl, θgrid), 
        k=3
    )
    covf_θ1θ2Δφℝ = (θ1,θ2,Δφℝ) -> covf(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφℝ)) 

    Σaz = AzBlock(covf_θ1θ2Δφℝ, θℝ, φℝ, tmW) do A, k
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

beamfwhm    = 3.5 |> arcmin -> deg2rad(arcmin/60)
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

Baz = @sblock let T_cov=T_Baz, tmW, bl, θℝ=θℝ64, φℝ=φℝ64, Ωℝ=Ωℝ64, azmuth_transfer_k
    ##θgrid = range(0, π^(1/2), length=100_000).^2
    dmax = 1.2maximum(CMBrings.geoθ1θ2Δφcol(θℝ[1], θℝ[1], φℝ .- φℝ[1]))
    θgrid = range(0, dmax^(1/2), length=100_000).^2
    
    covf  = Spline1D(
        θgrid, 
        Spectra.spec2spherecov(bl, θgrid), 
        k=3
    )
    
    covf_θ1θ2Δφℝ = (θ1,θ2,Δφℝ) -> covf(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφℝ)) 

    Baz  = AzBlock(covf_θ1θ2Δφℝ, θℝ, φℝ, tmW) do Σ, k
        T_cov.(real.(Σ) * Diagonal(azmuth_transfer_k.(k, θℝ) .* Ωℝ))
    end

    Baz
end;




# Noise with weights weight and mask/projection
# ==============================

μK′n      = 3.5 # 10.0
ellknee   = 0   # 150
alphaknee = 3
## weight_θ  = θ -> 1 + 0.15 * sin(300 * θ) # θ -> 1
weight_θ  = θ -> 1
## weight_θ  = θ -> 1 + 1 ./ sin(θ).^2 # θ -> 1
#-

nnl, snl = @sblock let μK′n, ellknee, alphaknee, lmax = 11000
    l = 0:lmax
    whitenoisel    = fill(μK′n^2 * (π/60/180)^2, size(l))
    smoothnoisel   = @. μK′n^2 * (π/60/180)^2 * Spectra.knee(l; ell=ellknee, alpha=alphaknee) 
    smoothnoisel .-= μK′n^2 * (π/60/180)^2 
    smoothnoisel[smoothnoisel .< 0] .= 0    
    noisel = smoothnoisel .+ whitenoisel
    return noisel, smoothnoisel
end;

#-

Naz = @sblock let T_cov=T_Naz, tmW, μK′n, snl, weight_θ, θℝ=θℝ64, φℝ=φℝ64, Δθ = θℝ64[2]-θℝ64[1], Δφ = φℝ64[2]-φℝ64[1]
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
            pa = ST.Ωpix(θ1, Δθ, Δφ) # sin(θ1) * Δθ * Δφ
            rtn[Δφℝ .== 0] .+= cc / pa # <- since we are using ST grid
        end
        rtn
    end

    Naz = AzBlock(covf_θ1θ2Δφℝ, θℝ, φℝ, tmW) do N, k
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
n2s_ratio = 0.2

ϕnnl = @sblock let ϕϕl, n2s_ratio, lmax = 11000
    l        = 0:lmax
    lpeak    = 40
    ϕnnl     = @. n2s_ratio * lpeak^4 * ϕϕl[lpeak+1] / l^4
    ϕnnl[1] *= 1e5
    ϕnnl[2] *= 1e5
    ## ϕnnl   = @. 1 / (1 / nnl + 1 / ϕϕl)
    ϕnnl
end;

## figure()
## (0:11000).^4 .* ϕϕl |> loglog
## (0:11000).^4 .* ϕnnl |> loglog
## (0:11000).^4 .* inv.(inv.(ϕnnl) .+ inv.(ϕϕl)) |> loglog
## (0:11000).^4 .* ϕnnl .* inv.(ϕnnl .+ ϕϕl) |> loglog


#-

NΦNaz = @sblock let T_cov=T_NΦNaz, tmW, Φaz, ϕnnl, θℝ=θℝ64, φℝ=φℝ64 
	
	##θgrid = range(0, π^(1/2), length=100_000).^2
    dmax = 1.2maximum(CMBrings.geoθ1θ2Δφcol(θℝ[1], θℝ[1], φℝ .- φℝ[1]))
    θgrid = range(0, dmax^(1/2), length=100_000).^2

    covϕnn  = Spline1D(
        θgrid, 
        Spectra.spec2spherecov(ϕnnl, θgrid), 
        k=3
    )

    covϕnn_θ1θ2Δφℝ = (θ1,θ2,Δφℝ) -> covϕnn(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφℝ)) 

    Naz = AzBlock(covϕnn_θ1θ2Δφℝ, θℝ, φℝ, tmW) do A, k
        ## A = Symmetric(T_cov.(real.(A)),:L)
        ## return A
        ## -------------
        A = Symmetric(T_cov.(real.(A)),:L)
        B = eigen(A)
        B.values[B.values .<= 0] .= 0
        return B 
    end 

	NΦNaz  = map(Φaz, Naz) do Φ, N
        ## N * inv(cholesky(Symmetric(Matrix(Φ) + N))) # worked well with float64 
        ## N / Symmetric(Matrix(Φ) + N) ## testing ... !!!!!
        ## pinv(pinv(Matrix(Φ)) + pinv(Matrix(N))) ## try this too
	    A = pinv(eigen(Symmetric(Matrix(pinv(Φ)) + Matrix(pinv(N)))))
        A.values[A.values .<= 0] .= 0
        ## return A 
        return Matrix(A) 
    end |> AzBlock


    NΦNaz 
end;

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
    ## B = eigen(Symmetric(A,:L))
    ## B.values[B.values .<= 0] .= 0
    ## return B 
end |> AzBlock;



# Lensing
# ==================================================


# Gradients with respect to polar: acts by left mult.

∂θaz = @sblock let T_fld, θℝ=θℝ64
    Δθℝ = θℝ[2] - θℝ[1]
    onesnθm1 = fill(1,length(θℝ)-1)
    ∂θ′ = spdiagm(-1 => .-onesnθm1, 1 => onesnθm1)
    ∂θ′[1,end] = -1 # make periodic boundar conditions even though we will attinuate the boundary later
    ∂θ′[end,1] =  1
    ∂θ = T_fld(1 / (2Δθℝ)) * ∂θ′
    return (∂θ - ∂θ') / 2 
end

# Gradients with respect to azimuth: acts by right mult.

∂φᵀaz = @sblock let T_fld, φℝ=φℝ64
    Δφℝ= φℝ[2] - φℝ[1]
    onesnφm1 = fill(1,length(φℝ)-1)
    ∂φ       = spdiagm(-1 => .-onesnφm1, 1 => onesnφm1)
    ## for the periodic boundary conditions
    ∂φ[1,end] = -1
    ∂φ[end,1] =  1
    ## now as a right operator
    ## (∂φ * f')' == ∂/∂φ f == f * ∂φᵀ
    ∂φᵀ = transpose(T_fld(1 / (2Δφℝ)) * ∂φ)
    ## return ∂φᵀ
    return (∂φᵀ - ∂φᵀ') / 2 
end;



# Test two different gradients 

## ∇!FFT = im .* fullfreq(tmW)[2] |> x -> FFTNabla!(∂θaz, plan(tmW), x, similar(x)) 
## ∇! = Nabla!(∂θaz, ∂φᵀaz) 
## 
## ϕ_az      = az_sim(tmU, Φaz) |> Xfourier
## ∇!ϕ_az_map = ∇!(ϕ_az[:])
## ∇!FFTϕ_az_map = ∇!FFT(ϕ_az[:])
## 
## ∇!FFTϕ_az_map[2] |> matshow; colorbar()
## ∇!ϕ_az_map[2] |> matshow; colorbar()


# Now construct the lense (attinuate the lense near the upper and lower boundaries)
# with increments for φ

Ł, ϕ2v!, ϕ2vᴴ!, ϕ2v, ϕ2vᴴ, ∇!, maθ = @sblock let T_fld, rcϕ = rcϕ, nsteps=14, tmU, θℝ=θℝ64, φℝ=φℝ64, ∂θaz, ∂φᵀaz, ∇! = Nabla!(∂θaz, ∂φᵀaz) 
    
    ## smooth out the transition to the polar boundaries
    leftlink =  n::Int -> (cos.(range(-π,0,length=n)) .+ 1)./2
    rightlink = n::Int -> (cos.(range(0,π,length=n))  .+ 1)./2
    maθ = ones(T_fld,size(θℝ))
    nup = 10  #<--- edge buffer which attinuates lensing
    nlw = 25  #<--- edge buffer which attinuates lensing
    maθ[1:nup]         =  leftlink(nup)
    maθ[end-nlw+1:end] =  rightlink(nlw)
    maθ = T_fld.(maθ)

    sin⁻²θℝ = @. T_fld(1 + cot(θℝ)^2) # = cscθ^2

    ϕ2v! = function (v::NTuple{2,Array}, ϕ::Array)
        ∇!(v, ϕ)
        v[1] .*= maθ ./ rcϕ
        v[2] .*= maθ .* sin⁻²θℝ ./ rcϕ
        v
    end 

    # ϕ2vᴴ = function (v)
    #     vθ, vφ = v
    #     mvθ = transpose(∂θaz) * (maθ .* vθ ./ rcϕ) 
    #     mvφ = (maθ .* sin⁻²θℝ .* vφ ./ rcϕ) * transpose(∂φᵀaz)  
    #     Xmap(tmU, mvθ + mvφ) 
    # end 
    ## TODO: you should probably automate ∇!ᵀ ...
    ϕ2vᴴ! = function (ϕ::Array, v::NTuple{2,Array})
        mv = (similar(v[1]), similar(v[2]))
        ∇!(mv, (maθ.*v[1]./rcϕ, maθ.*sin⁻²θℝ.*v[2]./rcϕ) )
        mv[1] .*= -1 # for the transpose
        mv[2] .*= -1 # for the transpose
        ϕ .= mv[1] + mv[2]
        ϕ 
    end 

    # perhaps not needed
    ϕ2v = function (ϕ_az::Xfield)
        ϕ = ϕ_az[:]
        v = (similar(ϕ), similar(ϕ))
        ϕ2v!(v,ϕ)
    end 

    # perhaps not needed 
    ϕ2vᴴ = function (v)
        ϕ = similar(v[1])
        ϕ2vᴴ!(ϕ, v)
        Xmap(tmU, ϕ) 
    end 


    Ł = function (ϕ_az::Xfield)
        v = ϕ2v(ϕ_az)
        ArrayLense(v, ∇!, 0, 1, nsteps)
    end

    Ł, ϕ2v!, ϕ2vᴴ!, ϕ2v, ϕ2vᴴ, ∇!, maθ
end;

# FFT based grradients

## Ł, ϕ2v!, ϕ2vᴴ!, ϕ2v, ϕ2vᴴ, ∇!, maθ = @sblock let tmW, tmU, T_fld, rcϕ = rcϕ, nsteps=14,  θℝ=θℝ64, φℝ=φℝ64, ∂θaz
## 
##     ∇! = im .* fullfreq(tmW)[2] |> x -> FFTNabla!(∂θaz, plan(tmW), x, similar(x)) 
##     
##     ## smooth out the transition to the polar boundaries
##     leftlink =  n::Int -> (cos.(range(-π,0,length=n)) .+ 1)./2
##     rightlink = n::Int -> (cos.(range(0,π,length=n))  .+ 1)./2
##     maθ = ones(T_fld,size(θℝ))
##     nup = 10  #<--- edge buffer which attinuates lensing
##     nlw = 10  #<--- edge buffer which attinuates lensing
##     maθ[1:nup]         =  leftlink(nup)
##     maθ[end-nlw+1:end] =  rightlink(nlw)
##     maθ = T_fld.(maθ)
## 
##     ## sin⁻²θℝ = @. T_fld(1 + cot(θℝ)^2) # = cscθ^2
##     ## sin⁻²θℝ = @. T_fld(csc(θℝ)^2) # test against this ...
##     sin⁻²θℝ = @. T_fld(1 / sin(θℝ)^2) # test against this ...
## 
##     ϕ2v! = function (v::NTuple{2,Array}, ϕ::Array)
##         ∇!(v, ϕ)
##         v[1] .*= maθ ./ rcϕ
##         v[2] .*= maθ .* sin⁻²θℝ ./ rcϕ
##         v
##     end 
## 
##     # perhaps not needed
##     ϕ2v = function (ϕ_az::Xfield)
##         ϕ = ϕ_az[:]
##         v = (similar(ϕ), similar(ϕ))
##         ϕ2v!(v,ϕ)
##     end 
## 
## 
##     ϕ2vᴴ! = function (ϕ::Array, v::NTuple{2,Array})
##         mv = (similar(v[1]), similar(v[2]))
##         ∇!(mv, (maθ.*v[1]./rcϕ, maθ.*sin⁻²θℝ.*v[2]./rcϕ) )
##         mv[1] .*= -1 # for the transpose
##         mv[2] .*= -1 # for the transpose
##         ϕ .= mv[1] + mv[2]
##         ϕ 
##     end 
## 
##     # perhaps not needed 
##     ϕ2vᴴ = function (v)
##         ϕ = similar(v[1])
##         ϕ2vᴴ!(ϕ, v)
##         Xmap(tmU, ϕ) 
##     end 
## 
## 
##     Ł = function (ϕ_az::Xfield)
##         v = ϕ2v(ϕ_az)
##         ArrayLense(v, ∇!, 0, 1, nsteps)
##     end
## 
##     Ł, ϕ2v!, ϕ2vᴴ!, ϕ2v, ϕ2vᴴ, ∇!, maθ
## end;


#- 



# Show lensing (zoomed into 1/2 of azimuth band).

@sblock let Ł, ϕ_az=az_sim(tmU, Φaz), Σaz, φℝ, θℝ, fφ=1/2, hide_plots
    hide_plots && return

    Ln         = Ł(ϕ_az)
    t_az       = Xmap(az_sim(fieldtransform(ϕ_az), Σaz))
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
    diskplot(imgs, φℝ', π.-θℝ; txt=txt, nrows=2, fontsize=12)
end;





# Other Methods 
# ==============================================


function update_ϕ_maxlllnf(gradϕ, ϕ, lnf_array, data; Pr, NΦNaz, Σaz_fctr,  Φaz_fctr, Ł, ∇!, tmU, linesearch_time_max, solver = :LN_COBYLA,  ds...)
    
    vmax, imax = findmax(map(lnf -> CMBrings.lllnf(ϕ, lnf, Ł, Σaz_fctr), lnf_array))
    lnf        = lnf_array[imax]
    sc_lllnf   = vmax

    ## here are a couple other solvers :LN_SBPLX :LN_NELDERMEAD, :LN_COBYLA
    inHgrad = NΦNaz * gradϕ - NΦNaz * (Φaz_fctr \ ϕ) 

    T   = eltype_in(tmU)
    opt = NLopt.Opt(solver, 1)
    opt.maxtime      = linesearch_time_max
    opt.upper_bounds = T[1.0]
    opt.lower_bounds = T[0]
    opt.max_objective = function (β, grad)
        ϕβ = ϕ + β[1] * inHgrad
        CMBrings.sum_kbn([CMBrings.lllnf(ϕβ, lnf, Ł, Σaz_fctr), CMBrings.llϕ(ϕβ, Φaz_fctr), -sc_lllnf])
    end

    ll_opt, β_opt, = NLopt.optimize(opt,  T[0])
    @show ll_opt, β_opt[1]
    
    return inHgrad, β_opt[1]
end


function update_ϕ_meanlllnf(gradϕ, ϕ, lnf_array, data; Pr, NΦNaz, Σaz_fctr,  Φaz_fctr, Ł, ∇!, tmU, linesearch_time_max, solver = :LN_COBYLA,  ds...)
    
    ## here are a couple other solvers :LN_SBPLX :LN_NELDERMEAD, :LN_COBYLA
    inHgrad = NΦNaz * gradϕ - NΦNaz * (Φaz_fctr \ ϕ) 

    T   = eltype_in(tmU)
    opt = NLopt.Opt(solver, 1)
    opt.maxtime      = linesearch_time_max
    opt.upper_bounds = T[1.0]
    opt.lower_bounds = T[0]
    opt.max_objective = function (β, grad)
        ϕβ = ϕ + β[1] * inHgrad
        rtn  = mean(map(lnf -> CMBrings.lllnf(ϕβ, lnf, Ł, Σaz_fctr), lnf_array))
        rtn += CMBrings.llϕ(ϕβ, Φaz_fctr)
        rtn 
    end

    ll_opt, β_opt, = NLopt.optimize(opt,  T[0])
    @show ll_opt, β_opt[1]
    
    return inHgrad, β_opt[1]
end


function linesearchϕ(inHgrad, ϕ, lnf, data; tmU, Σaz_fctr, Φaz_fctr, linesearch_time_max, solver = :LN_COBYLA,  ds...)
    # solvers :LN_SBPLX :LN_NELDERMEAD, :LN_COBYLA
    T   = eltype_in(tmU)
    opt = NLopt.Opt(solver, 1)
    opt.maxtime      = linesearch_time_max
    opt.upper_bounds = T[1.0]
    opt.lower_bounds = T[0]
    opt.max_objective = function (β, grad)
        ϕβ = ϕ + β[1] * inHgrad
        CMBrings.lllnf(ϕβ, lnf, Ł, Σaz_fctr) + CMBrings.llϕ(ϕβ, Φaz_fctr) 
    end
    ll_opt, β_opt, = NLopt.optimize(opt,  T[0])
    @show ll_opt, β_opt
    return β_opt[1]
end



function ∇ϕ(ϕ, lnf, data; Pr, Σaz_fctr, Naz_fctr, Baz, ϕ2v, ϕ2v!, ϕ2vᴴ!, Ł, ∇!, tmU, grad_nsteps, ds...)
    ## Remark: for the next line to be correct Naz_fctr must be diagonal in pixel space
    ##dΔlnf     = Baz' * (Pr' * (Naz_fctr \ (data - Pr * (Baz * lnf))))
    Ma        = DiagOp(Xmap(tmU, abs.(Pr[:]).>0))
    dΔlnf     = Baz' * (Ma * (Naz_fctr \ (Pr \ (data - Pr * (Baz * lnf)))))
    v         = ϕ2v(ϕ)
    f         = Ł(ϕ) \ lnf 
    τŁ₀₁      = CMBrings.FieldLensing.τArrayLense(v, (f[:],), ∇!, 0, 1, grad_nsteps)
    τŁ₁₀      = CMBrings.FieldLensing.τArrayLense(v, (lnf[:],), ∇!, 1, 0, grad_nsteps)        
    ## τv₀, τf   = τŁ₁₀(map(zero,v),  (dΔlnf[:],))
    τϕ₀, τf   = τpotential(τŁ₁₀, zero(ϕ[:]), (dΔlnf[:],), ϕ2v!, ϕ2vᴴ!)
    ∇f        = Xmap(tmU, τf[1]) - Σaz_fctr \ f
    ## τv₁, τlnf = τŁ₀₁(τv₀,  (∇f[:],))
    τϕ₁, τlnf = τpotential(τŁ₀₁, τϕ₀,  (∇f[:],), ϕ2v!, ϕ2vᴴ!)
    ## return ϕ2vᴴ(τv₁) #  - Φaz_fctr \ ϕ # this last term is added later
    return Xmap(tmU, τϕ₁)
end



function τpotential(
        τL::FieldLensing.τArrayLense{m,n,Tf,d,Tg,Tt},
        τϕ::A, 
        τf::NTuple{n,A}, 
        ϕ2v!, ϕ2vᴴ!, 
    )::Tuple{A, NTuple{n,A}} where {m,n,Tf,d,Tg,Tt<:Real,A<:Array{Tf,d}}

    pτL!  = FieldLensing.plan(τL) 

    # these are just storage containers
    y′ = deepcopy(tuple(τL.v..., τf..., τL.f...))
    ẏ′ = deepcopy(tuple(τL.v..., τf..., τL.f...))

    f! = function (ẏ,t,y)

        # fill y′ ≡ (τv,τf,f) from  y ≡ (τϕ,τf,f) 
        # ------------------
        # first y′[1:m] = ϕ2v(y[1])
        τvₜ = tuple(y′[Base.OneTo(m)]...)
        ϕ2v!(τvₜ, first(y)) 

        # now y′[(m+1):(m+2n)] directly from tail of y
        ytail  = Base.tail(y)
        y′tail = y′[(m+1):end]
        for i = 1:2n
            @avx @. y′tail[i] = ytail[i]
        end

        # now compute ẏ′ from y′
        # ------------------
        pτL!(ẏ′, t, y′)

        # finally compute ẏ via compression of ẏ′
        # -----------------------
        # compute τ̇vₜ (alisased to ẏ′[1:m])
        τ̇vₜ = tuple(ẏ′[Base.OneTo(m)]...)
        ϕ2vᴴ!(first(ẏ), τ̇vₜ)

        # compute (τ̇fₜ, ḟₜ) (alisased to ẏ′[m+1:end])
        ẏtail  = Base.tail(ẏ)
        ẏ′tail = ẏ′[(m+1):end]
        for i = 1:2n
            @avx @. ẏtail[i] = ẏ′tail[i]
        end
    end

    rtn   = FieldLensing.odesolve_RK4(f!, tuple(τϕ, τf..., τL.f...), τL.t₀, τL.t₁, τL.nsteps)
    
    return first(rtn), tuple(Base.tail(rtn)[Base.OneTo(n)]...)

end





# Benchmarks 
# ==============================

## ## f = Xmap(tmU, randn(eltype_in(tmU), size_in(tmU)))
## f = Xfourier(tmU, randn(eltype_out(tmU), size_out(tmU)))
## ## f = Xmap(tmU32, randn(eltype_in(tmU32), size_in(tmU32)))
## ## f = Xfourier(tmU32, randn(eltype_out(tmU32), size_out(tmU32)))
## 
## 
## @benchmark $Σaz * $f # 430 ms
## #-
## @benchmark $Σaz \ $f # 50 ms
## #- 
## @benchmark map(Matrix, $Σaz) # 2 s
## #-
## @benchmark $Baz * $f # 54.728 ms
## #-
## @benchmark $(Baz') * $f # 
## #- 
## 
## @benchmark $(Ł(az_sim(tmU, Φaz))) * $f # 1s
## @benchmark $∂θaz * $(f[:])    # 4ms
## @benchmark $(f[:]) * $(∂φᵀaz) # 5ms



# Simulate data 
# ================================================


ϕ_az  = az_sim(tmU, Φaz) |> Xmap
t_az  = az_sim(tmU, Σaz) |> Xfourier
d_az  = Pr * (Baz * (Ł(ϕ_az)*t_az) + az_sim(tmU, Naz)) |> Xfourier;


@sblock let Ł, Baz, t_az, d_az, ϕ_az, θℝ, φℝ, Pr, hide_plots
    hide_plots && return
    imgs = Dict(
        1 => d_az[:],
        2 => t_az[:],
        3 => abs.((d_az - Pr * (Baz * (Ł(ϕ_az)*t_az)))[:])
    )
    txt =  Dict(
        1 => "data",
        2 => "signal",
        3 => "abs(noise)"
    )
    ctxt = Dict(
        3 => "w"
    )
    brickplot(imgs; txt=txt, ctxt=ctxt, fφ=1)
    ## diskplot(imgs, φℝ', π.-θℝ; txt=txt, nrows=2, fontsize=12)
end;




# Put settings and needed parameters in ds ...
# ===========================================


ds = (;  
    tmU, Ł, ∇!, Pr, Qr, 
    Σaz_fctr=Σaz, Φaz_fctr=Φaz, Naz_fctr=Naz, Baz, 
    Precon_fctr, NΦNaz, 
    ϕ2v!, ϕ2vᴴ!,  ϕ2v, ϕ2vᴴ, # not sure the last two are needed
    grad_nsteps = 14, pcg_nsteps=125, 
    linesearch_time_max = 60*3,
    solver = :LN_COBYLA, # :LN_SBPLX, ##  :LN_NELDERMEAD, 
);



# newton/gibbs iterations
# ================================================


# TODO: see if you can adjust the hessian with these samples 
# Wouldn't a wishart type draw work? 

ϕ_cr  = Xmap(tmU)
ginit = Xfourier(tmU)
∇ϕ_cr = Xmap(tmU)
∇ϕ_cr_array  = typeof(∇ϕ_cr)[]

# iterate ...
@showprogress for otr = 1:5
    global lnt_cr, t_cr, hst
    global ∇ϕ_cr_array, gradϕ
    
    ## for itr = 1:4
        n′  = az_sim(tmU, Naz) |> Xfourier
        f′  = az_sim(tmU, Σaz) |> Xfourier
        data′ = Pr * (Baz * (Ł(ϕ_cr) * f′)) +  Pr * n′ |> Xfourier
        @time lnt_cr, t_cr, ginit, hst = CMBrings.update_lnf_f(ϕ_cr, d_az; data′, f′, ginit, ds...)
        @show hst[end]

        ## gradϕ   = CMBrings.∇ϕ(ϕ_cr, lnt_cr, d_az; ds...)
        @time gradϕ = ∇ϕ(ϕ_cr, lnt_cr, d_az; ds...)
        @time ∇ϕ_cr = NΦNaz * gradϕ - NΦNaz * (Φaz \ ϕ_cr)  |> Xmap
        ## ∇ϕ_cr = Φaz * gradϕ - ϕ_cr 
        ## ∇ϕ_cr.fd[:,1:10] .*= 1/6
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


## ϕ_cr     = Xfourier(tmU)
## ginit_cr = Xfourier(tmU)
## ∇ϕ_cr    = Xfourier(tmU)
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
##     f′  = az_sim(tmU, Σaz) |> Xfourier
##     n′  = az_sim(tmU, Naz) |> Xfourier
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
    fltr[:,1:10] .= 0    
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