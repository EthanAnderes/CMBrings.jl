
# Modules
# ==============================
using FFTW
FFTW.set_num_threads(5)

using CMBrings
using CMBrings: pcg, brickplot
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
## using PyCall
using PyPlot
using BenchmarkTools


hide_plots = false

# Mask and CMBring observation region
# ==============================

QP_boundry_clearance = 1e-5 

#-

ma, maᶜ, Ωℝ, θℝ, φℝ, s0, s0_clip = @sblock let QP_boundry_clearance

    ma𝕊 = readdlm("FastTransform_mask_nθ3072_nφ4095.txt", ',', Bool)
    nθ𝕊, nφ𝕊 = size(ma𝕊)

    ## s0_clip = (69*nθ𝕊÷100):(90*nθ𝕊÷100)
    ## s0_clip = (72*nθ𝕊÷100):(87*nθ𝕊÷100)
    ## s0_clip = (75*nθ𝕊÷100):(85*nθ𝕊÷100)
    s0_clip = (77*nθ𝕊÷100):(87*nθ𝕊÷100)

    s0 = ST.𝕊(Float64, nθ𝕊, nφ𝕊, 0)
    Ωℝ = ST.Ωpix(s0)[s0_clip]
    θℝ, φℝ = ST.pix(s0) |> x->(x[1][s0_clip], x[2])

    𝕨 = r𝕎(nθ𝕊, π) ⊗ r𝕎(nφ𝕊, 2π) |> x-> ordinary_scale(x)*x
    beamfwhm1 = (arcmin=200.0; deg2rad(arcmin/60))
    beamfwhm2 = (arcmin=500.0; deg2rad(arcmin/60))
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



# Azimuthal rings, with mask

@sblock let ma, hide_plots
    hide_plots && return
    matshow(ma)
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

# Spectra noise, signal, beam 
# ================================

μK′n      = 7.0 # 10.0
ellknee   = 100   # 150
alphaknee = 3
beamfwhm  = 3.5 |> arcmin -> deg2rad(arcmin/60)

#-

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

bl = @sblock let beamfwhm, lmax = 8000
    l = 0:lmax
    σ² = beamfwhm^2 / 8 / log(2)
    bl = @. exp( - σ²*l*(l+1) / 2)
    return bl
end;


# TT covariance, NN covariance and beam kernel
# ================================

covt_θ1θ2Δφℝ = @sblock let ttl
	θgrid = range(0, π^(1/2), length=100_000).^2
    covt  = Spline1D(
        θgrid, 
        Spectra.spec2spherecov(ttl, θgrid), 
        k=3
    )
    return (θ1,θ2,Δφℝ) -> covt(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφℝ))  
end;

covb_θ1θ2Δφℝ = @sblock let bl 
    θgrid = range(0, π^(1/2), length=100_000).^2
    covb  = Spline1D(
        θgrid, 
        Spectra.spec2spherecov(bl, θgrid), 
        k=3
    )
    return (θ1,θ2,Δφℝ) -> covb(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφℝ))  
end;

covn_θ1θ2Δφℝ = @sblock let μK′n, snl, Δθ = θℝ[2]-θℝ[1], Δφ = φℝ[2]-φℝ[1]
    θgrid = range(0, π^(1/2), length=100_000).^2
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


# AzBlocks and Ops
# ================================

azmuth_transfer_k = k -> inv(1 + (k/75)^2)

#-

@time Σaz = AzBlock(covt_θ1θ2Δφℝ, θℝ, φℝ, tmW) do Σ, k
    ## A = Symmetric(real.(Σ),:L)
    ## cholesky(A, Val(false), check=false)
    real.(Σ)
end; 
## Note: if Σaz.Σ is set to symmetric then it takes a hit on mult


@time Naz = AzBlock(covn_θ1θ2Δφℝ, θℝ, φℝ, tmW) do Σ, k
    ## A = Symmetric(real.(Σ),:L)
    ## cholesky(A, Val(false), check=false)
    real.(Σ)
end; 

@time Baz  = AzBlock(covb_θ1θ2Δφℝ, θℝ, φℝ, tmW) do Σ, k
    azmuth_transfer_k(k) * real.(Σ) * Diagonal(Ωℝ)
end; 

# Some benchmarks 

f = Xmap(tmU, randn(eltype_in(tmU), size_in(tmU)))
@benchmark $Σaz * $f
#-
@benchmark $Naz * $f
#-
@benchmark $Baz * $f
#-
@benchmark $(AzBlock(map(cholesky,Σaz))) \ $f


# Noise weight and mask/projection
# ==============================

weight_θ = θ -> 1 + 0.5 * sin(300 * θ)

#-

Wt = @sblock let tmU, weight_θ, θℝ, φℝ 
    wt  = weight_θ.(θℝ) .+ fill(0,(1,length(φℝ)))
    DiagOp(Xmap(tmU, wt))
end;

#-

Pr, Qr = @sblock let tmU, ma, maᶜ
    Pr = Xmap(tmU, ma)
    Qr = Xmap(tmU, maᶜ)
    DiagOp(Pr), DiagOp(Qr)
end;



# Lensing
# ==================================================


# Gradients with respect to polar: acts by left mult.

∂θaz = @sblock let θℝ
    Δθℝ = θℝ[2] - θℝ[1]
    onesnθm1 = fill(1,length(θℝ)-1)
    ∂θ = (1 / (2Δθℝ)) * spdiagm(-1 => .-onesnθm1, 1 => onesnθm1)
    ∂θ[1,:] .= 0
    ∂θ[end,:] .= 0
    ∂θ
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
    ∂φᵀ
end;


# Now construct the lense (attinuate the lense near the upper and lower boundaries)

Ln, ϕ_az = @sblock let nsteps=14, tmU, s0, s0_clip, ϕϕl, θℝ, φℝ, ∂θaz, ∂φᵀaz, ∇! = Nabla!(∂θaz, ∂φᵀaz) 
    
    ls0, ms0 = ST.lm(s0)
    Cϕ       = DiagOp(Xfourier(s0, ϕϕl[ls0 .+ 1])) 
    ϕ        = CMBsphere.simfourier(Cϕ)

    ϕaz = ϕ[:][s0_clip,:]
    sin⁻²θℝ = @. 1 + cot(θℝ)^2 # = cscθ^2
    vθ = ∂θaz * ϕaz
    vφ = (ϕaz * ∂φᵀaz) .* sin⁻²θℝ

    ## smooth out the transition to the polar boundaries
    leftlink =  n::Int -> (cos.(range(-π,0,length=n)) .+ 1)./2
    rightlink = n::Int -> (cos.(range(0,π,length=n)) .+ 1)./2
    maθ = ones(size(θℝ))
    n = 10  #<--- edge buffer which attinuates lensing
    maθ[1:n]      =  leftlink(n)
    maθ[end-n+1:end] =  rightlink(n)
    vθ .*= maθ
    vφ .*= maθ

    t₀ = 0
    t₁ = 1
    L = ArrayLense((vθ, vφ), ∇!, t₀, t₁, nsteps)
    L, Xmap(tmU, ϕaz)
end;


# Show lensing (zoomed into 1/2 of azimuth band).

@sblock let Ln, ϕ_az, Σaz, hide_plots
    hide_plots && return

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
    brickplot(imgs; txt=txt, ctxt=ctxt, fφ=1/2)
end




# AzCov preconditioned conjugate gradient
# ==================================================

PCG = @sblock let  Pr, Qr, Wt, Ln, Σaz, Naz, Baz
                    
    Baz′ = Baz'

    BΣBᴴWNWᴴ_mat = map(Σaz, Naz, Baz) do Σ, N, B 
        WD = Diagonal(Wt[:][:,1])
        B*Σ*B' + WD*N*WD'
    end |> AzBlock;

    BΣBᴴWNWᴴ_chol = map(BΣBᴴWNWᴴ_mat) do Σ
        cholesky(Symmetric(Σ,:L), Val(false)) # , check=false)
    end |> AzBlock;

    A_noL = function (g)
        tmp1  = Pr*(BΣBᴴWNWᴴ_mat*(Pr'*g))
        tmp2  = Qr*(BΣBᴴWNWᴴ_mat*(Qr'*g))    
        return tmp1 + tmp2
    end 

    A_wL = function (g)
        tmp0  = Pr*(Baz*(Ln*(Σaz*(Ln'*(Baz′*(Pr'*g))))))
        tmp1  = Pr*(Wt*(Naz*(Wt'*(Pr'*g))))
        tmp2  = Qr*(BΣBᴴWNWᴴ_mat*(Qr'*g))    
        return tmp0 + tmp1 + tmp2
    end 

    PCG = function (data; lense=true, nsteps, rel_tol=1e-12)
        gwf, hist = pcg(
            g -> BΣBᴴWNWᴴ_chol \ g, 
            lense ? A_wL : A_noL, 
            data, 
            nsteps=nsteps, rel_tol=rel_tol,
        )
        if lense
            return Σaz*(Ln'*(Baz′*(Pr'*gwf))), hist
        else 
            return Σaz*(Baz′*(Pr'*gwf)), hist
        end
    end

    return PCG

end; 




# Simulate AzCov data
# =======================================

t_az  = az_sim(tmU, Σaz)
n_az  = az_sim(tmU, Naz)
d_az  = Pr * (Baz*(Ln*t_az) + Wt*n_az);

# Second simulation for generating conditional fluctuations

t_az′  = az_sim(tmU, Σaz)
n_az′  = az_sim(tmU, Naz)
d_az′  = Pr * (Baz*(Ln*t_az′) + Wt*n_az′);

#  Plot the data, the signal and noise (full azimuthal band)

@sblock let t_az, d_az, n_az, Pr, Wt, hide_plots
    hide_plots && return
    imgs = Dict(
        1 => d_az[:],
        2 => t_az[:],
        3 => abs.((Pr*Wt*n_az)[:])
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
end;


# Run PCG for WF
# =======================================

# WF (not accounting for the lensing in the data)

@time twf_1, hwf_1 = PCG(d_az, lense=false, nsteps=150, rel_tol = 1e-4);

# WF (modeling the lensing)  

@time twf_2, hwf_2 = PCG(d_az, lense=true, nsteps=150, rel_tol = 1e-4);

# Plot the wiener filters

@sblock let twf_1, twf_2, t_az, hide_plots
    hide_plots && return
    imgs = Dict(
        1 => t_az[:],
        2 => twf_1[:],
        3 => twf_2[:],
    )
    txt =  Dict(
        1 => "CMB simulation truth",
        2 => "wiener filter (not modeling lensing)",
        3 => "wiener filter (modeling lensing)",
    )
    ctxt = Dict(
    )
    brickplot(imgs; txt=txt, ctxt=ctxt, fφ=1/2)
end

# Plot the errors

@sblock let twf_1, twf_2, t_az, Pr, hide_plots
    hide_plots && return
    imgs = Dict(
        1 => t_az[:],
        2 => twf_1[:] .- Pr[:] .* t_az[:],
        3 => twf_2[:] .- Pr[:] .* t_az[:],
    )
    txt =  Dict(
        1 => "CMB simulation truth",
        2 => "wiener filter error (not modeling lensing)",
        3 => "wiener filter error (modeling lensing)",
    )
    ctxt = Dict(
    )
    brickplot(imgs; txt=txt, ctxt=ctxt, fφ=1/2)
end

# Here are the residuals from PCG

@sblock let hwf_1, hwf_2, hide_plots
    hide_plots && return
    fig,ax = subplots(1)
    ax.semilogy(hwf_1, label="PCG residuals (lensing=false)")
    ax.semilogy(hwf_2, label="PCG residuals (lensing=true)")
    ax.legend()
end


# Run PCG for conditional simulation
# =======================================

# Conditional simulation (not accounting for the lensing in the data)

@time tsim_1, hsim_1 = PCG(d_az + d_az′, lense=false, nsteps=150, rel_tol = 1e-4);
tsim_1 -= t_az′; 

# Conditional simulation  (modeling the lensing)  

@time tsim_2, hsim_2 = PCG(d_az + d_az′, lense=true, nsteps=150, rel_tol = 1e-4);
tsim_2 -= t_az′; 

# Plot the conditional simulations from PCG

@sblock let tsim_1, tsim_2, t_az, hide_plots
    hide_plots && return
    imgs = Dict(
        1 => t_az[:],
        2 => tsim_1[:],
        3 => tsim_2[:],
    )
    txt =  Dict(
        1 => "CMB simulation truth",
        2 => "conditional sim (not modeling lensing)",
        3 => "conditional sim (modeling lensing)",
    )
    ctxt = Dict(
    )
    brickplot(imgs; txt=txt, ctxt=ctxt, fφ=1/2)
end

# Plot the errors 

@sblock let tsim_1, tsim_2, t_az, Pr, hide_plots
    hide_plots && return
    imgs = Dict(
        1 => t_az[:],
        2 => (tsim_1[:] .-  Pr[:] .* t_az[:]),
        3 => (tsim_2[:] .-  Pr[:] .* t_az[:]),
        4 => tsim_1[:] .- tsim_2[:],
    )
    txt =  Dict(
        1 => "CMB simulation truth",
        2 => "conditional sim error (not modeling lensing)",
        3 => "conditional sim error (modeling lensing)",
        4 => "diff of the two sims "
    )
    ctxt = Dict(
    )
    brickplot(imgs; txt=txt, ctxt=ctxt, fφ=1/2)
end


# Here are the residuals from PCG

@sblock let hsim_1, hsim_2, hide_plots
    hide_plots && return
    fig,ax = subplots(1)
    ax.semilogy(hsim_1, label="PCG residuals (lensing=false)")
    ax.semilogy(hsim_2, label="PCG residuals (lensing=true)")
    ax.legend()
end

# Check to see that the conditional sims have the right likelihood.
# These should behave like ≈ N(0,1)

ln_az      = length(d_az[:])
zll_t_az   = (dot(t_az[:], (Σaz \ t_az)[:]) - ln_az) / sqrt(2*ln_az) # PCG sim
zll_tsim_1 = (dot(tsim_1[:], (Σaz \ tsim_1)[:]) - ln_az) / sqrt(2*ln_az) # PCG sim
zll_tsim_2 = (dot(tsim_2[:], (Σaz \ tsim_2)[:]) - ln_az) / sqrt(2*ln_az) # PCG sim
@show zll_t_az  
@show zll_tsim_1
@show zll_tsim_2;



#-

## if all(abs.(Csn[!]) .== 0)
##     dfd  = sum(abs.(d_az[:]) .> 0)
##     Δdf  = (Pr*Wt) \ (d_az - Pr*(Baz*(Ln*twf_2))) 
##     nvar =  abs2(μK′n*π/60/180) ./ Ωℝ
##     nll  = dot(Δdf[:], Δdf[:] ./ nvar) 
##     fll  = dot(twf_2[:], (Σaz \ twf_2)[:])
##     zll  = (nll + fll - dfd) / sqrt(2*dfd)
##     @show zll
## end