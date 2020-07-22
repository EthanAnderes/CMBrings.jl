
# Modules
# ==============================
using Distributed
addprocs(2)

@everywhere using FFTW
@everywhere FFTW.set_num_threads(1)
@everywhere using CMBrings
using CMBrings: AzCov, az2op, az3op, az2az, kazmap
using CMBrings: flatnoisemap, simfourier, pcg
using CMBrings: brickplot

using Spectra
using XFields
using FieldLensing

using FFTransforms: r𝕎, 𝕎, ordinary_scale, ⊗, fullfreq
using SphereTransforms
const ST = SphereTransforms

using DelimitedFiles
using LinearAlgebra
using SparseArrays
using Statistics
using Dierckx: Spline1D
using LBblocks: @sblock
using PyCall
using PyPlot
using BenchmarkTools
using JLD2

hide_plots = true

# Methods and structs
# ==============================

function azc_sim(::Type{T}, azc::AzCov) where T<:Number
    wx = randn(T, CMBrings.size_arg(azc))
    az2op((Σ, x) -> Σ.L * x, azc, wx)
end

azc_sim(azc::AzCov) = azc_sim(Float64, azc::AzCov)

s0_sim(Cl::DiagOp) = simfourier(Cl)

struct Nabla!{Tθ,Tφ}
    ∂θ::Tθ
    ∂φᵀ::Tφ
end

function (∇!::Nabla!{Tθ,Tφ})(∇y::NTuple{2,A}, y::NTuple{2,A}) where {Tθ,Tφ,Tf,A<:Array{Tf,2}}
    mul!(∇y[1], ∇!.∂θ, y[1])
    mul!(∇y[2], y[2], ∇!.∂φᵀ)
    ∇y
end

function (∇!::Nabla!{Tθ,Tφ})(∇y::NTuple{2,A}, y::A) where {Tθ,Tφ,Tf,A<:Array{Tf,2}}
    ∇!(∇y, (y,y))
end

function (∇!::Nabla!{Tθ,Tφ})(y::A) where {Tθ,Tφ,Tf,A<:Array{Tf,2}}
    ∇y = (similar(y), similar(y))
    ∇!(∇y, (y,y))
    ∇y
end

# Set SphereTransform 
# ==============================

s0 = @sblock let 
    nθ, nφ, spin = 6*512, 8*512-1, 0 
    ST.𝕊(Float64, nθ, nφ, spin)
end


# Mask and CMBring observation region
# ==============================

## s0_clip = (69*s0.nθ÷100):(90*s0.nθ÷100)
## s0_clip = (72*s0.nθ÷100):(87*s0.nθ÷100)
## s0_clip = (75*s0.nθ÷100):(85*s0.nθ÷100)
s0_clip = (77*s0.nθ÷100):(87*s0.nθ÷100)

#-

ma𝕊, maℝ, Ω𝕊, Ωℝ, θ𝕊, θℝ, φ𝕊, φℝ = @sblock let s0, s0_clip

    ma𝕊 = readdlm("FastTransform_mask_nθ3072_nφ4095.txt", '\t', Bool)
    maℝ = ma𝕊[s0_clip,:]
    ## -------- option: strip from north to south pole
    ## ma𝕊 = falses(s0.nθ, s0.nφ)
    ## ma𝕊[:,(s0.nφ÷25):(s0.nφ÷3)] .= true
    ## maℝ = ma𝕊[s0_clip,:]

    Ω𝕊 = ST.Ωpix(s0)
    Ωℝ = Ω𝕊[s0_clip]
    θ𝕊, φ𝕊 = ST.pix(s0) 
    θℝ, φℝ = θ𝕊[s0_clip], φ𝕊 

    ma𝕊, maℝ, Ω𝕊, Ωℝ, θ𝕊, θℝ, φ𝕊, φℝ
end  

# ### Full sky mask view 

@sblock let ma𝕊, hide_plots
    hide_plots && return
    matshow(ma𝕊)
end

# ### Restriction to subset of rings

@sblock let maℝ, hide_plots
    hide_plots && return
    matshow(maℝ)
end

# ### Plot √Ωpix over ring θℝ's 

@sblock let θℝ, Ωℝ, hide_plots
    hide_plots && return
    fig,ax = subplots(1)
    ax.plot(θℝ, rad2deg.(sqrt.(Ωℝ)).*60)
    ax.set_xlabel(L"polar coordinate $\theta$")
    ax.set_ylabel("sqrt pix area (arcmin)")
end



# Set azimuthal frequency blocks
# ==================================

kidx_blk = @sblock let φℝ
    ## FIXME: the periodic sims leak to modes set to zero for some reason

    ## Full range of frequency indices
    kidx = 1:(length(φℝ)÷2+1)
    ## kidx = 1:2:(length(φcol)÷2+1) 

    ## Divided into blocks
    kidx_blk = [
         kidx[1:end÷2],
         kidx[(1(end÷2)+1):end],
    ]
    ## kidx_blk = [
    ##     kidx[1:end÷4],
    ##     kidx[(1(end÷4)+1):(2(end÷4))],
    ##     kidx[(2(end÷4)+1):(3(end÷4))],
    ##    kidx[(3(end÷4)+1):end],
    ## ]

    kidx_blk
end 



# Signal model (Σaz, Σs)
# ================================

# ### Spectra and XFields Op

ttl, ϕϕl, Σs, Cϕ = @sblock let s0

    lmax = 8000
    l = 0:lmax
    ls0, ms0 = ST.lm(s0)

    cld = Spectra.camb_cls(lmax=lmax)
    ctlvec = cld[:unlen_scalar] |> x->(x[:Ctt] ./ x[:factor_on_cl_cmb])
    ctlvec[1:2] .= 0
    ct_s0 = ctlvec[ls0 .+ 1]
    Ct_s0 = DiagOp(Xfourier(s0, ct_s0)) 

    cϕlvec = cld[:phi] |> x->(x[:Cϕϕ] ./ x[:factor_on_cl_phi])
    cϕlvec[1:2] .= 0
    cϕ_s0 = cϕlvec[ls0 .+ 1]
    Cϕ_s0 = DiagOp(Xfourier(s0, cϕ_s0)) 

    ctlvec, cϕlvec, Ct_s0, Cϕ_s0
end;

# ### Pixel space covariance function (z-rotation invariant)

covt_θ1θ2Δφℝ = @sblock let ttl
	θgrid = range(0, π^(1/2), length=100_000).^2
    covt  = Spline1D(
        θgrid, 
        Spectra.spec2spherecov(ttl, θgrid), 
        k=3
    )
    return (θ1,θ2,Δφℝ) -> covt(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφℝ))  
end

# ### now compute the corresponding AzCov

Σaz = AzCov(covt_θ1θ2Δφℝ, θℝ, φℝ, kidx_blk) do k, Σ
    cholesky(Σ, Val(false), check=false)
end; 
## Check that the cholesky's where successful
CMBrings.check_factorization(Σaz)


# Also check the Mmapped size

run(`ls -lh $(Σaz.filenm)`)





# Noise model  (Naz, Ns)
# =============================

μK′n      = 7.0 # 10.0
ellknee   = 0 # 150
alphaknee = 3

# ### Spectra (white and smooth component separated) and XFields Op

nnl, snl, Ns, Sns = @sblock let μK′n, ellknee, alphaknee, s0

    lmax = 8000
    l = 0:lmax
    ls0, ms0 = ST.lm(s0)

    whitenoisel    = fill(μK′n^2 * (π/60/180)^2, size(l))
    smoothnoisel   = @. μK′n^2 * (π/60/180)^2 * Spectra.knee(l; ell=ellknee, alpha=alphaknee) 
    smoothnoisel .-= μK′n^2 * (π/60/180)^2 
    ## smoothnoisel[l .> 1000] .= 0
    ## smoothnoisel[l .< 2]  .= 0
    smoothnoisel[smoothnoisel .< 0] .= 0    
    noisel = smoothnoisel .+ whitenoisel

    ## construct spectral operators
    Csmoothnoisel = smoothnoisel[ls0 .+ 1] |> c->DiagOp(Xfourier(s0, c)) 
    Cwhitenoisel  = whitenoisel[ls0 .+ 1]  |> c->DiagOp(Xfourier(s0, c))
    Cn = Csmoothnoisel + Cwhitenoisel

    return noisel, smoothnoisel, Cn, Csmoothnoisel
end

# ### Pixel space covariance function (z-rotation invariant)

covn_θ1θ2Δφℝ = @sblock let μK′n, snl, s0
    θgrid = range(0, π^(1/2), length=100_000).^2
    covsn  = Spline1D(
        θgrid, 
        Spectra.spec2spherecov(snl, θgrid), 
        k=3
    )
    covn_θ1θ2Δφℝ = function (θ1, θ2, Δφℝ)
        rtn   = covsn(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφℝ))
        if θ1 == θ2
            cc = μK′n^2 * (π/60/180)^2
            pa = sin(θ1) * ST.Δθ(s0) * ST.Δφ(s0)
            rtn[Δφℝ .== 0] .+= cc / pa # <- since we are using ST grid
        end
        rtn
    end
    return covn_θ1θ2Δφℝ
end

# ### now compute the corresponding AzCov

Naz = AzCov(covn_θ1θ2Δφℝ,  θℝ, φℝ, kidx_blk) do k, Σ
    cholesky(Σ, Val(false), check=false)
end 
## Check that the cholesky's where successful
CMBrings.check_factorization(Naz)



# ### Plot signal and noise spectra

@sblock let cls=(ttl, nnl), leg=("signal", "noise"), hide_plots 
    hide_plots && return
    fig,ax = subplots(1)
    l = 0:(length(cls[1])-1)
    for (s,cl) ∈ zip(leg,cls)
        ax.loglog(l[9:end],l[9:end].^2 .* cl[9:end], label=s)
    end
    ax.set_xlabel(L"\ell")
    ax.set_ylabel(L"\ell^2 C_\ell")
    ax.legend()
end




# Noise pixel weight (Wt, Ws)
# ==============================

## w_fun  = θ -> 1
w_fun = θ -> 1 + 0.5 * sin(300 * θ)

# `Ws` is the `SphereTransform` operator for XFields. `Wt` operates on ring maps.

Wt, Wtᴴ, Ws = @sblock let w_fun, θℝ, θ𝕊, φ𝕊, s0
    w_s0 = w_fun.(θ𝕊) .+ fill(0,(1,length(φ𝕊)))
    Ws   = DiagOp(Xmap(s0, w_s0))
    Wt   = Diagonal(w_fun.(θℝ)) # when operating on a column indexed by θ for fixed φ
    Wtᴴ  = Wt # when operating on a column indexed by θ for fixed φ
    Wt, Wtᴴ, Ws
end

# Show the weight effect on a noise simulation (zoomed into 1/2 of azimuth band).

@sblock let Wt, Naz, hide_plots
    hide_plots && return

    n_az  = azc_sim(Naz)
    wn_az = Wt * n_az

    imgs = Dict(
        1 => n_az,
        2 => abs.(wn_az),
    )
    txt =  Dict(
        1 => "noise",
        2 => "abs(weight * noise)",
    )
    ctxt = Dict(2=>"w")
    brickplot(imgs; txt=txt, ctxt=ctxt,fφ=1/2)
end




# Beam/Transfer function (Baz, Bs)
# ============================

beamfwhm = 3.0 |> arcmin -> deg2rad(arcmin/60)
## beamfwhm = 3.0 |> arcmin -> deg2rad(arcmin/60)

# ### Spectra and Pixel space covariance function (z-rotation invariant)

blm, Bs, covb_θ1θ2Δφℝ = @sblock let beamfwhm, s0
    
    lmax = 8000
    l = 0:lmax
    ls0, ms0 = ST.lm(s0)
    ms0max = maximum(ms0)
    ls0max = maximum(ls0)

    σ² = beamfwhm^2 / 8 / log(2)
    bl = @. exp( - σ²*l*(l+1) / 2)
    blm   = bl[ls0 .+ 1]
    Bl_s0 = DiagOp(Xfourier(s0, blm)) 

    θgrid = range(0, π^(1/2), length=100_000).^2
    covb  = Spline1D(
        θgrid, 
        Spectra.spec2spherecov(bl, θgrid), 
        k=3
    )
 
    return blm, Bl_s0, (θ1,θ2,Δφℝ) -> covb(CMBrings.geoθ1θ2Δφcol(θ1, θ2, Δφℝ))  
end

# ### now compute the corresponding AzCov

# Note the additional Ω pre-factor which mimics the 
# required surface area element

Baz  = AzCov(covb_θ1θ2Δφℝ, θℝ, φℝ, kidx_blk) do k, Σ
    ## Σ * Diagonal(Ωℝ)
    ## inv(1 + (k/50)^2) * Σ * Diagonal(Ωℝ)
    inv(1 + (k/75)^2) * Σ * Diagonal(Ωℝ)
end; 

# ### wrap Baz and transpose(Baz) with functions

Be, Beᴴ = @sblock let Baz
    Be  = x -> Baz * x
    Beᴴ = x -> az2op((Σ,g)->Σ'*g, Baz, x)
    Be, Beᴴ
end;

# Show the beam effect on a simulation (zoomed into 1/2 of azimuth band)

@sblock let Be, Σaz, hide_plots
    hide_plots && return

    t_az = azc_sim(Σaz)
	bt_az = Be(t_az)

    imgs = Dict(
        1 => t_az,
        2 => bt_az,
    )
    txt =  Dict(
        1 => "CMB simulation",
        2 => "Beam * CMB simulation",
    )
    brickplot(imgs; txt=txt, fφ=1/2)

end




# Mask/Projection 
# ==============================

# This and the lense is the only operator that isn't azmuthally symmetric.

Pr, Prᴴ, Qr, Qrᴴ, Ps, Qs = @sblock let ma𝕊, s0, s0_clip, QP_boundry_clearance = 1e-3 # 1e-3

    nθ𝕊, nφ𝕊 = size(ma𝕊)
    𝕨 = r𝕎(nθ𝕊, π) ⊗ 𝕎(nφ𝕊, 2π) |> x-> ordinary_scale(x)*x
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
    ## ----------
    ## ps = @. Bps_qs * (Bps_qs > 0) 
    ## qs = @. (- Bps_qs + 0.001) * (Bps_qs <= 0) 
    ## qs ./= maximum(qs)

    @assert all(abs.(qs.*ps) .== 0)
    @assert all(abs.(qs) .+ abs.(ps) .> 0)


    Ps  = DiagOp(Xmap(s0, ps))
    Qs  = DiagOp(Xmap(s0, qs))

	pr = Ps[:][s0_clip,:]
	qr = Qs[:][s0_clip,:]
    Qr   = x -> qr .* x
    Qrᴴ  = x -> qr .* x
    Pr   = x -> pr  .* x
    Prᴴ  = x -> pr  .* x
    
    Pr, Prᴴ, Qr, Qrᴴ, Ps, Qs
end



# Plots of the mask (zoomed into 1/2 of azimuth band)

@sblock let Qs, Ps, s0_clip, hide_plots
    hide_plots && return

    m_az  = Ps[:][s0_clip,:]
    mᶜ_az = Qs[:][s0_clip,:]

    imgs = Dict(
        1 => m_az,
        2 => mᶜ_az,
    )
    txt =  Dict(
        1 => "mask",
        2 => "mask complement",
    )
    ctxt = Dict(
        1 => "w", 2 => "w"
    )
    brickplot(imgs; txt=txt, ctxt=ctxt,fφ=1/2)
end




# Lensing
# ==================================================

# Gradients with respect to polar: acts by left mult.
∂θaz = @sblock let θℝ, Δθℝ=ST.Δθ(s0)
    onesnθm1 = fill(1,length(θℝ)-1)
    ∂θ = (1 / (2Δθℝ)) * spdiagm(-1 => .-onesnθm1, 1 => onesnθm1)
    ∂θ[1,:] .= 0
    ∂θ[end,:] .= 0
    ∂θ
end

# Gradients with respect to azimuth: acts by right mult.
∂φᵀaz = @sblock let φℝ, Δφℝ=ST.Δφ(s0)
    onesnφm1 = fill(1,length(φℝ)-1)
    ∂φ       = spdiagm(-1 => .-onesnφm1, 1 => onesnφm1)
    ## for the periodic boundary conditions
    ∂φ[1,end] = -1
    ∂φ[end,1] =  1
    ## now as a right operator
    ## (∂φ * f')' == ∂/∂φ f == f * ∂φᵀ
    ∂φᵀ = transpose((1 / (2Δφℝ)) * ∂φ)
    ∂φᵀ
end


# Now construct the lense (attinuate the lense near the upper and lower boundaries)

Ln, ϕ_az = @sblock let Cϕ, s0_clip, θℝ, ∂θaz, ∂φᵀaz, ∇! = Nabla!(∂θaz, ∂φᵀaz), nsteps=14
    
    ϕ   = s0_sim(Cϕ)

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
    L = FieldLensing.ArrayLense((vθ, vφ), ∇!, t₀, t₁, nsteps)
    L, ϕaz
end;


# Show lensing (zoomed into 1/2 of azimuth band).

@sblock let Ln, ϕ_az, Σs, s0_clip, hide_plots
    hide_plots && return

    t_az   = s0_sim(Σs)[:][s0_clip,:]
    lnt_az = Ln * t_az
    lense_time = @belapsed $Ln * $t_az
    t_az′      = Ln \ lnt_az

    imgs = Dict(
        1 => ϕ_az,
        2 => lnt_az,
        3 => t_az .- lnt_az,
        4 => abs.(t_az .- t_az′), 
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

# ### Pre-mask AzCov for the data.

BΣBᴴ_WNWᴴ_az = az2az(Σaz, Naz, Baz) do Σ, N, B 
    BΣBᴴ = Symmetric(B  * Matrix(Σ) * B')
    WNWᴴ = Symmetric(Wt * Matrix(N) * Wtᴴ)
    cholesky(BΣBᴴ + WNWᴴ)
end
CMBrings.check_factorization(BΣBᴴ_WNWᴴ_az)


# ### Precon Conj Grad closure

PCG = @sblock let   Ln, Lnᴴ=Ln', Naz, Σaz, BΣBᴴ_WNWᴴ_az, 
                    Be, Beᴴ, Wt, Wtᴴ, Pr, Prᴴ, Qr, Qrᴴ
                    

    
    P = g -> BΣBᴴ_WNWᴴ_az \ g
    B = P

    ## A_noL and A_wL are the operators we want to invert
    A_noL = function (g)
        tmp1  = Pr(BΣBᴴ_WNWᴴ_az * Prᴴ(g))
        tmp2  = Qr(B(Qrᴴ(g)))    
        return tmp1 .+ tmp2
    end 

    A_wL = function (g)
        tmp0  = Pr(Be(Ln * (Σaz * (Lnᴴ * Beᴴ(Prᴴ(g))))))
        tmp1  = Pr(Wt * (Naz * (Wtᴴ * Prᴴ(g))))
        tmp2  = Qr(B(Qrᴴ(g)))    
        return tmp0 .+ tmp1 .+ tmp2
    end 

    PCG = function (data; lense=true, nsteps, rel_tol=1e-12)
        gwf, hist = pcg(
            P, 
            lense ? A_wL : A_noL, 
            data, 
            nsteps=nsteps, rel_tol=rel_tol,
        )
        @show hist[end] 
        if lense
            return Σaz*(Lnᴴ*Beᴴ(Prᴴ(gwf))), hist
        else 
            return Σaz*(Beᴴ(Prᴴ(gwf))), hist
        end
    end

    return PCG

end; 




# Simulate AzCov data
# =======================================

t_az  = azc_sim(Σaz)
n_az  = azc_sim(Naz)
d_az  = Pr(Be(Ln*t_az) + Wt * n_az);

# Second simulation for generating conditional fluctuations

t_az′  = azc_sim(Σaz)
n_az′  = azc_sim(Naz)
d_az′  = Pr(Be(Ln*t_az′) + Wt * n_az′);

#  Plot the data and the signal (full azimuthal band)

@sblock let t_az, d_az, hide_plots
    hide_plots && return
    imgs = Dict(
        1 => d_az,
        2 => t_az,
    )
    txt =  Dict(
        1 => "data",
        2 => "signal",
    )
    ctxt = Dict(
    )
    brickplot(imgs; txt=txt, ctxt=ctxt, fφ=1)
end


# Run PCG for WF
# =======================================

# WF (not accounting for the lensing in the data)
@time twf_1, hwf_1 = PCG(d_az, lense=false, nsteps=250, rel_tol = 2e-2);
## @time twf_1, hwf_1 = PCG(d_az, lense=false, nsteps=250, rel_tol = 9e-2);
## @time twf_2, hwf_2 = PCG(d_az, lense=false, nsteps=500, rel_tol = 1e-4);


# WF (modeling the lensing)  
@time twf_2, hwf_2 = PCG(d_az, lense=true, nsteps=250, rel_tol = 2e-2);


# Plot the wiener filters
@sblock let twf_1, twf_2, t_az, hide_plots
    hide_plots && return
    imgs = Dict(
        1 => t_az,
        2 => twf_1,
        3 => twf_2,
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
@sblock let twf_1, twf_2, t_az, maℝ, hide_plots
    hide_plots && return
    imgs = Dict(
        1 => t_az,
        2 => twf_1 .- maℝ .* t_az,
        3 => twf_2 .- maℝ .* t_az,
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


# If noise is white ... i.e. snl .== 0 
if all(snl .== 0)
    ## σn² = abs2(μK′n*π/60/180)./Ωℝ # white noise level
    pr   = Ps[:][s0_clip,:]
    dfd  = sum(abs.(pr) .> 0)
    Δdf1 = Wt \ (pinv.(pr) .* (d_az .- Pr(Be(twf_1))))
    Δdf2 = Wt \ (pinv.(pr) .* (d_az .- Pr(Be(Ln*twf_2))))
    nll1 = dot(Δdf1, Naz \ Δdf1)
    nll2 = dot(Δdf2, Naz \ Δdf2)
    fll1 = dot(twf_1, Σaz \ twf_1)
    fll2 = dot(twf_2, Σaz \ twf_2)

    zll_1 = (nll1 + fll1 - dfd) / sqrt(2*dfd) 
    zll_2 = (nll2 + fll2 - dfd) / sqrt(2*dfd) 
    @show zll_1
    @show zll_2
end



# Run PCG for conditional simulation
# =======================================

## Conditional simulation (not accounting for the lensing in the data)
@time tsim_1, hsim_1 = PCG(d_az + d_az′, lense=false, nsteps=250, rel_tol = 2e-2);
tsim_1 -= t_az′; 

## Conditional simulation  (modeling the lensing)  
@time tsim_2, hsim_2 = PCG(d_az + d_az′, lense=true, nsteps=250, rel_tol = 2e-2);
tsim_2 -= t_az′; 


# Plot the conditional simulations from PCG
@sblock let tsim_1, tsim_2, t_az, hide_plots
    hide_plots && return
    imgs = Dict(
        1 => t_az,
        2 => tsim_1,
        3 => tsim_2,
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
@sblock let tsim_1, tsim_2, t_az, maℝ, hide_plots
    hide_plots && return
    imgs = Dict(
        1 => t_az,
        2 => (tsim_1 .-  maℝ .* t_az),
        3 => (tsim_2 .-  maℝ .* t_az),
        4 => tsim_1 .- tsim_2,
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

ln_az      = length(d_az)
zll_t_az   = (dot(t_az, Σaz \ t_az) - ln_az) / sqrt(2*ln_az) # PCG sim
zll_tsim_1 = (dot(tsim_1, Σaz \ tsim_1) - ln_az) / sqrt(2*ln_az) # PCG sim
zll_tsim_2 = (dot(tsim_2, Σaz \ tsim_2) - ln_az) / sqrt(2*ln_az) # PCG sim
@show zll_t_az  
@show zll_tsim_1
@show zll_tsim_2;




# Full sky
# ==============================
# No lensing or non-stationary beam/transfer. Using FastTransforms

t_s0 = s0_sim(Σs)
n_s0 = s0_sim(Ns)
d_s0 = Ps * (Bs * t_s0 + Ws * n_s0)

t_s0′ = s0_sim(Σs)
n_s0′ = s0_sim(Ns)
d_s0′ = Ps * (Bs * t_s0′ + Ws * n_s0′)

## σn²   = abs2(μK′n*π/60/180) ./ Ω𝕊
## σn²Op = DiagOp(Xmap(s0, σn² .* ones(s0.nθ, s0.nφ)))
DP = Bs * Σs * Bs' + Ns
DB = DP
MA₁ = Ps * Bs * Σs * Bs' * Ps'
MA₂ = Ps * Ws * Ns * Ws' * Ps'
MA₃ = Qs * DB * Qs'
MG  = Σs * Bs' * Ps'

@time g1s0, hist0s0 = pcg(
        w -> DP \ w, 
        w -> MA₁ * w + MA₂ * w + MA₃ * w,
        d_s0,
        nsteps  = 100,
        rel_tol = 1e-10,
)
t1_cs0sim = MG * g1s0

t1_cs0sim[:] |> matshow
t1_cs0sim[:] .- t_s0[:] |> matshow






# Noise fill full sky 
# ==============================
# No lensing or non-stationary beam/transfer. Using FastTransforms

t_s0 = s0_sim(Σs)
n_s0 = s0_sim(Ns)
d_s0 = Ps * Bs * t_s0 + Ws * n_s0

t_s0′ = s0_sim(Σs)
n_s0′ = s0_sim(Ns)

MP₁ = Σs * Bs' / (Bs * Σs * Bs' + Ns) * Ns / Bs'
MP₂ = Bs' / Ns * Bs + inv(Σs)
MA  = Bs' * Ps' / Ws' / Ns / Ws * Ps * Bs
DA  = Σs
MD  = Bs' * Ps' / Ws' / Ns / Ws

@time t0_cs0sim, hist0s0 = pcg(
        w -> MP₂ * w,
        w -> MA * w + DA \ w,
        MD * d_s0, # MD * (d_s0 + Ws * n_s0′) + DA \ t_s0′,
        nsteps  = 100,
        rel_tol = 1e-10,
)


t0_cs0sim[:] |> matshow