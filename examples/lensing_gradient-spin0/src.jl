
# Modules
# ==============================
using FFTW
FFTW.set_num_threads(8)

using CMBrings
using CMBrings: pcg, brickplot, diskplot
using CMBrings: AzBlock, check_factorization, az_sim
using CMBrings: Nabla!
using CMBrings.FieldLensing: ArrayLense
using XFields
using Spectra

using FFTransforms: 𝕀, r𝕎, ⊗, unitary_scale, ordinary_scale, fullfreq
import CMBsphere
const ST = CMBsphere.SphereTransforms

using DelimitedFiles
using LinearAlgebra
using SparseArrays
using Statistics
using Dierckx: Spline1D
using LBblocks: @sblock
using PyPlot
using NLopt
using BenchmarkTools


hide_plots = true

# Mask and CMBring observation region
# ==============================

QP_boundry_clearance = 1e-4 

#-

ma, maᶜ, Ωℝ, θℝ, φℝ, s0, s0_clip = @sblock let QP_boundry_clearance

    ## ------------------
    ma𝕊 = readdlm("FastTransform_mask_nθ3072_nφ4095.csv", ',', Bool)
    nθ𝕊, nφ𝕊 = size(ma𝕊)
    s0_clip = (78*nθ𝕊÷100):(87*nθ𝕊÷100) # default
    ## # s0_clip = (75*nθ𝕊÷100):(85*nθ𝕊÷100)
    ## # s0_clip = (72*nθ𝕊÷100):(87*nθ𝕊÷100)
    ## # s0_clip = (69*nθ𝕊÷100):(90*nθ𝕊÷100)
    ## ------------------
    ## # ma𝕊      = readdlm("FastTransform_mask_spole_nθ3072_nφ4095.csv", ',', Bool)
    ## ma𝕊      = readdlm("FastTransform_mask_spole_nθ3072_nφ3071.csv", ',', Bool)
    ## nθ𝕊, nφ𝕊 = size(ma𝕊)
    ## s0_clip  = (84*nθ𝕊÷100):(98*nθ𝕊÷100) # default
    ## # s0_clip  = (82*nθ𝕊÷100):(97*nθ𝕊÷100)
    ## # s0_clip  = (87*nθ𝕊÷100):(985*nθ𝕊÷1000)
    ## # s0_clip  = (82*nθ𝕊÷100):(99*nθ𝕊÷100)
    ## ------------------
    ## ma𝕊      = readdlm("FastTransform_mask_nearpole_nθ3072_nφ3071.csv", ',', Bool)
    ## nθ𝕊, nφ𝕊 = size(ma𝕊)
    ## s0_clip  = (84*nθ𝕊÷100):(97*nθ𝕊÷100)
    ## ---------------------
    ## ma𝕊  = readdlm("FastTransform_mask_mid2pole_nθ2560_nφ3071.csv", ',', Bool)
    ## ma𝕊  = readdlm("FastTransform_mask_mid2pole_nθ2560_nφ4095.csv", ',', Bool)
    ## nθ𝕊, nφ𝕊 = size(ma𝕊)
    ## s0_clip  = (79*nθ𝕊÷100):(96*nθ𝕊÷100)


    s0 = ST.𝕊(Float64, nθ𝕊, nφ𝕊, 0)
    Ωℝ = ST.Ωpix(s0)[s0_clip]
    θℝ, φℝ = ST.pix(s0) |> x->(x[1][s0_clip], x[2])

    𝕨 = r𝕎(nθ𝕊, π) ⊗ r𝕎(nφ𝕊, 2π) |> x-> ordinary_scale(x)*x
    beamfwhm1 = (arcmin=100.0; deg2rad(arcmin/60))
    beamfwhm2 = (arcmin=200.0; deg2rad(arcmin/60))
    ## beamfwhm1 = (arcmin=200.0; deg2rad(arcmin/60))
    ## beamfwhm2 = (arcmin=400.0; deg2rad(arcmin/60))
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


    Ps[:][s0_clip,:], Qs[:][s0_clip,:], Ωℝ, θℝ, φℝ, s0, s0_clip
end;  



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

tmU, tmW = let nθ=length(θℝ), nφ=length(φℝ)
    tmW = 𝕀(nθ) ⊗ r𝕎(nφ, 2π)
    tmU  = unitary_scale(tmW)*tmW
    tmU, tmW
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

ttl, ϕϕl = @sblock let lmax = 8000
    l = 0:lmax
    cld = Spectra.camb_cls(lmax=lmax)
    ctlvec = cld[:unlen_scalar] |> x->(x[:Ctt] ./ x[:factor_on_cl_cmb])
    ctlvec[1:2] .= 0
    cϕlvec = cld[:phi] |> x->(x[:Cϕϕ] ./ x[:factor_on_cl_phi])
    cϕlvec[1:2] .= 0
    ctlvec, cϕlvec
end;

#-

covϕ_θ1θ2Δφℝ = @sblock let ϕϕl, θℝ, φℝ
    ##θgrid = range(0, π^(1/2), length=100_000).^2
    dmax = 1.2maximum(CMBrings.geoθ1θ2Δφcol(θℝ[1], θℝ[1], φℝ .- φℝ[1]))
    θgrid = range(0, dmax^(1/2), length=100_000).^2
    covt  = Spline1D(
        θgrid, 
        Spectra.spec2spherecov(ϕϕl, θgrid), 
        k=3
    )
    return (θ1,θ2,Δφℝ) -> covt(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφℝ))  
end;

#-

covt_θ1θ2Δφℝ = @sblock let ttl, θℝ, φℝ
	##θgrid = range(0, π^(1/2), length=100_000).^2
    dmax = 1.2maximum(CMBrings.geoθ1θ2Δφcol(θℝ[1], θℝ[1], φℝ .- φℝ[1]))
    θgrid = range(0, dmax^(1/2), length=100_000).^2
    covt  = Spline1D(
        θgrid, 
        Spectra.spec2spherecov(ttl, θgrid), 
        k=3
    )
    return (θ1,θ2,Δφℝ) -> covt(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφℝ))  
end;

#-

Φaz = AzBlock(covϕ_θ1θ2Δφℝ, θℝ, φℝ, tmW) do A, k
    ##real.(A) + 1e-8*I(length(θℝ))
    A = Symmetric(real.(A),:L)
    cholesky(A, Val(false)) #, check=false)
end; 

#-

Σaz = AzBlock(covt_θ1θ2Δφℝ, θℝ, φℝ, tmW) do A, k
    ##real.(A) + 1e-8*I(length(θℝ))
    A = Symmetric(real.(A),:L)
    cholesky(A, Val(false)) #, check=false)
end; 


# Beam/Transfer function
# ================================

beamfwhm    = 3.0 |> arcmin -> deg2rad(arcmin/60)
## azmuth_transfer_k = k -> 1
azmuth_transfer_k = k -> inv(1 + (k/200)^2)
## azmuth_transfer_k = k -> inv(1 + (k/75)^2)

#-

bl = @sblock let beamfwhm, lmax = 8000
    l = 0:lmax
    σ² = beamfwhm^2 / 8 / log(2)
    bl = @. exp( - σ²*l*(l+1) / 2)
    return bl
end;

#-

covb_θ1θ2Δφℝ = @sblock let bl, θℝ, φℝ
    ##θgrid = range(0, π^(1/2), length=100_000).^2
    dmax = 1.2maximum(CMBrings.geoθ1θ2Δφcol(θℝ[1], θℝ[1], φℝ .- φℝ[1]))
    θgrid = range(0, dmax^(1/2), length=100_000).^2
    covb  = Spline1D(
        θgrid, 
        Spectra.spec2spherecov(bl, θgrid), 
        k=3
    )
    return (θ1,θ2,Δφℝ) -> covb(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφℝ))  
end;

#-

Baz  = AzBlock(covb_θ1θ2Δφℝ, θℝ, φℝ, tmW) do Σ, k
    azmuth_transfer_k(k) * real.(Σ) * Diagonal(Ωℝ)
end; 

## This turns the beam off
## Baz  = AzBlock(covb_θ1θ2Δφℝ, θℝ, φℝ, tmW) do Σ, k
##     (0 .* real.(Σ)) + I
## end; 



# Noise with weights weight and mask/projection
# ==============================

μK′n      = 5.0 # 10.0
ellknee   = 0   # 150
alphaknee = 3
weight_θ  = θ -> 2 + 0.5 * sin(300 * θ) # θ -> 1
## weight_θ  = θ -> 1

#-

nnl, snl = @sblock let μK′n, ellknee, alphaknee, lmax = 8000
    l = 0:lmax
    whitenoisel    = fill(μK′n^2 * (π/60/180)^2, size(l))
    smoothnoisel   = @. μK′n^2 * (π/60/180)^2 * Spectra.knee(l; ell=ellknee, alpha=alphaknee) 
    smoothnoisel .-= μK′n^2 * (π/60/180)^2 
    smoothnoisel[smoothnoisel .< 0] .= 0    
    noisel = smoothnoisel .+ whitenoisel
    return noisel, smoothnoisel
end;

#-

covn_θ1θ2Δφℝ = @sblock let μK′n, snl, θℝ, φℝ, Δθ = θℝ[2]-θℝ[1], Δφ = φℝ[2]-φℝ[1]
    ## θgrid = range(0, π^(1/2), length=100_000).^2
    dmax = 1.2maximum(CMBrings.geoθ1θ2Δφcol(θℝ[1], θℝ[1], φℝ .- φℝ[1]))
    θgrid = range(0, dmax^(1/2), length=100_000).^2
    covsn  = Spline1D(
        θgrid, 
        Spectra.spec2spherecov(snl, θgrid), 
        k=3
    )
    return function (θ1, θ2, Δφℝ)
        rtn   = covsn(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφℝ))
        if θ1 == θ2
            cc = μK′n^2 * (π/60/180)^2
            pa = ST.Ωpix(θ1, Δθ, Δφ) # sin(θ1) * Δθ * Δφ
            rtn[Δφℝ .== 0] .+= cc / pa # <- since we are using ST grid
        end
        rtn
    end
end;

#-

# Note `Naz` includes the weight multiplier
Naz = AzBlock(covn_θ1θ2Δφℝ, θℝ, φℝ, tmW) do N, k
    WD = Diagonal(weight_θ.(θℝ))
    A = Symmetric(WD*(real.(N))*WD',:L)
    cholesky(A, Val(false)) #, check=false)
end;


# negative Hessian  for ϕ gradient -> newton update
# ==============================

nhϕl = @sblock let n2s_ratio = 0.5 , ϕϕl, lmax = 8000
    l = 0:lmax
    nhl    = (n2s_ratio * maximum(l.^4 .* ϕϕl)) ./ (l.^4)
    nhϕl       = inv.(inv.(ϕϕl) .+ inv.(nhl))
    nhϕl[1:2] .= 0

    return nhϕl
end;

## figure()
## (0:8000).^4 .* ϕϕl |> loglog
## (0:8000).^4 .* nhϕl |> loglog

#-

cov_nhϕ_θ1θ2Δφℝ = @sblock let nhϕl, θℝ, φℝ
    ##θgrid = range(0, π^(1/2), length=100_000).^2
    dmax = 1.2maximum(CMBrings.geoθ1θ2Δφcol(θℝ[1], θℝ[1], φℝ .- φℝ[1]))
    θgrid = range(0, dmax^(1/2), length=100_000).^2
    covf  = Spline1D(
        θgrid, 
        Spectra.spec2spherecov(nhϕl, θgrid), 
        k=3
    )
    return (θ1,θ2,Δφℝ) -> covf(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφℝ))  
end;

#-

bHϕaz  = AzBlock(cov_nhϕ_θ1θ2Δφℝ, θℝ, φℝ, tmW) do Σ, k
    real.(Σ)
end; 


# Preconditioner (via g -> Precon_fctr \ g)
# ==============================

Precon_fctr = map(Σaz, Naz, Baz) do Σ, N, B
    A = B*Matrix(Σ)*B' + Matrix(N)
    cholesky(Symmetric(A,:L)) # , check=false)
end |> AzBlock;



# Lensing
# ==================================================


# Gradients with respect to polar: acts by left mult.

∂θaz = @sblock let θℝ
    Δθℝ = θℝ[2] - θℝ[1]
    onesnθm1 = fill(1,length(θℝ)-1)
    ∂θ′ = spdiagm(-1 => .-onesnθm1, 1 => onesnθm1)
    ∂θ′[1,end] = -1 # make periodic boundar conditions even though we will attinuate the boundary later
    ∂θ′[end,1] =  1
    ∂θ = (1 / (2Δθℝ)) * ∂θ′
    ## return ∂θ
    return (∂θ - ∂θ')/2 
end

# Gradients with respect to azimuth: acts by right mult.

∂φᵀaz = @sblock let φℝ
    Δφℝ= φℝ[2] - φℝ[1]
    onesnφm1 = fill(1,length(φℝ)-1)
    ∂φ       = spdiagm(-1 => .-onesnφm1, 1 => onesnφm1)
    ## for the periodic boundary conditions
    ∂φ[1,end] = -1
    ∂φ[end,1] =  1
    ## now as a right operator
    ## (∂φ * f')' == ∂/∂φ f == f * ∂φᵀ
    ∂φᵀ = transpose((1 / (2Δφℝ)) * ∂φ)
    ## return ∂φᵀ
    return (∂φᵀ - ∂φᵀ')/2 
end;



# Now construct the lense (attinuate the lense near the upper and lower boundaries)

Ł, ϕ2v, ϕ2vᴴ, ∇!, maθ = @sblock let nsteps=14, tmU, θℝ, φℝ, ∂θaz, ∂φᵀaz, ∇! = Nabla!(∂θaz, ∂φᵀaz) 
    
    ## smooth out the transition to the polar boundaries
    leftlink =  n::Int -> (cos.(range(-π,0,length=n)) .+ 1)./2
    rightlink = n::Int -> (cos.(range(0,π,length=n)) .+ 1)./2
    maθ = ones(size(θℝ))
    nup = 5 # 10  #<--- edge buffer which attinuates lensing
    nlw = 5 # 25  #<--- edge buffer which attinuates lensing
    maθ[1:nup]         =  leftlink(nup)
    maθ[end-nlw+1:end] =  rightlink(nlw)

    sin⁻²θℝ = @. 1 + cot(θℝ)^2 # = cscθ^2

    ϕ2v = function (ϕ_az::Xfield)
        ϕ  = ϕ_az[:]
        vθ = maθ .* (∂θaz * ϕ)
        vφ = maθ .* sin⁻²θℝ .* (ϕ * ∂φᵀaz) 
        ## vφ = sin⁻²θℝ .* (ϕ * ∂φᵀaz) 
        vθ, vφ
    end 

    ϕ2vᴴ = function (v)
        vθ, vφ = v
        mvθ = transpose(∂θaz) * (maθ .* vθ) 
        mvφ = (maθ .* sin⁻²θℝ .* vφ) * transpose(∂φᵀaz) 
        ## mvφ = (sin⁻²θℝ .* vφ) * transpose(∂φᵀaz) 
        Xmap(tmU, mvθ + mvφ) 
    end 

    Ł = function (ϕ_az::Xfield)
        v = ϕ2v(ϕ_az)
        ArrayLense(v, ∇!, 0, 1, nsteps)
    end

    Ł, ϕ2v, ϕ2vᴴ, ∇!, maθ
end;

# Show lensing (zoomed into 1/2 of azimuth band).

@sblock let Ł, ϕ_az=az_sim(tmU, Φaz), Σaz, φℝ, θℝ, fφ=1, hide_plots
    hide_plots && return

    Ln = Ł(ϕ_az)
    t_az   = Xmap(az_sim(fieldtransform(ϕ_az), Σaz))
    lnt_az = Ln * t_az
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
    diskplot(imgs, φℝ', π.-θℝ; txt=txt, nrows=2, fontsize=14)
end;



# Simulate data 
# ================================================


ϕ_az  = az_sim(tmU, Φaz)
t_az  = az_sim(tmU, Σaz)
d_az  = Pr * (Baz * (Ł(ϕ_az)*t_az) + az_sim(tmU, Naz));


@sblock let Ł, Baz, t_az, d_az, ϕ_az, θℝ, φℝ, hide_plots
    hide_plots && return
    imgs = Dict(
        1 => d_az[:],
        2 => t_az[:],
        3 => abs.((d_az - Baz * (Ł(ϕ_az)*t_az))[:])
    )
    txt =  Dict(
        1 => "data",
        2 => "signal",
        3 => "abs(noise)"
    )
    ctxt = Dict(
        3 => "w"
    )
    ## brickplot(imgs; txt=txt, ctxt=ctxt, fφ=1)
    diskplot(imgs, φℝ', π.-θℝ; txt=txt, nrows=2, fontsize=14)
end;




# Put settings and needed parameters in ds ...
# ===========================================

ds = (;  
    tmU, Ł, ∇!, ϕ2v, ϕ2vᴴ, 
    Σaz_fctr=Σaz, Φaz_fctr=Φaz, Naz_fctr=Naz, Precon_fctr,
    Baz, bHϕaz, Pr, Qr, 
    grad_nsteps = 14, pcg_nsteps=75, 
    linesearch_time_max = 60*5,
)



# newton/gibbs iterations
# ================================================

# initalize ϕ_cr, t_cr, lnt_cr
ϕ_cr   = Xfourier(tmU)
lnt_cr = Xfourier(tmU)

#=
@time CMBrings.update_lnf_f(ϕ_cr, d_az; ds...)
@time CMBrings.update_ϕ(ϕ_cr, lnt_cr, d_az; ds...)
@time CMBrings.ll(ϕ_cr, lnt_cr, d_az; ds...)
@time CMBrings.∇ϕ(ϕ_cr, lnt_cr, d_az; ds...)
=#


# iterate ...
for itr = 1:50
    global ϕ_cr, lnt_cr, t_cr, hst 
    @time lnt_cr, t_cr, hst = CMBrings.update_lnf_f(ϕ_cr, d_az; ds...)
    @time ϕ_cr              = CMBrings.update_ϕ(ϕ_cr, lnt_cr, d_az; ds...)
end


#- 

@sblock let fest = ϕ_cr, ftru = ϕ_az, tmU, φℝ, θℝ, M = Pr
    fltr = CMBrings.fullfreq(tmU)[2]
    #fltr = ones(eltype_out(tmU), size_out(tmU))
    fltr[:,1:6] .= 0
    𝔽 = Xfourier(tmU,fltr) |> DiagOp
    #𝔽 = I
    diskplot(
        Dict(1=> (M * (𝔽 * fest))[:], 2 =>(M * (𝔽 * ftru))[:]), 
        φℝ', π.-θℝ; nrows=1, fontsize=14
    )
end


#-

@sblock let fest = ϕ2vᴴ(ϕ2v(ϕ_cr )), ftru = ϕ2vᴴ(ϕ2v(ϕ_az)), φℝ, θℝ, Pr

    diskplot(
        Dict(1=>(Pr * fest)[:], 2=>(Pr * ftru)[:]),
        φℝ', π.-θℝ; nrows=1, fontsize=14
    )

    brickplot(
        Dict(1=>(Pr * fest)[:], 2=>(Pr * ftru)[:]), 
        fφ=1/2
    )

end 







# newton updates within gibbs iterations















