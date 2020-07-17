
# Modules
# ==============================
using Distributed
addprocs(2)

@everywhere using FFTW
@everywhere FFTW.set_num_threads(1)
@everywhere using CMBrings
using CMBrings: AzCov, kAzCov, az2op, az3op, az2az, kazmap
using CMBrings: flatnoisemap, simfourier, pcg
using Spectra
using XFields
using FieldLensing
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



function brickplot(imgs::Dict{Int,T};
            txt  = Dict{Int,String}(), # overlay text
            ctxt = Dict{Int,String}(), # color of text
            fφ = 1/2, # fraction of azimuth 
            sz = 2,   # Overall size scale
            hmlt = 0.875, # Hight adjust
        ) where T

    nimg = maximum(keys(imgs))
    nr = size(imgs[nimg])[1]
    nc = size(imgs[nimg])[2] * fφ |> x->round(Int,x)

    fig, ax = subplots(nimg,1,figsize=(sz*(nc/nr), sz*nimg*hmlt))
    ax = nimg==1 ? [ax] : ax

    for (i,f) ∈ imgs
        img = ax[i].imshow(f[:,1:nc]) 
        fig.colorbar(img, ax=ax[i], shrink=0.8, extend="both", pad=0.015)
    end
    for i=1:nimg-1
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
    end
    for (i,s) ∈ txt
        ax[i].text(
            nc*0.98, nr*0.95, s, 
            color=i ∈ keys(ctxt) ? ctxt[i] : "k",
            horizontalalignment = "right",
        )
    end
    fig.subplots_adjust(hspace=0.01, bottom = 0.1, top = 0.98, left = 0.05, right=0.98)
    ## fig.tight_layout()

    fig, ax
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

Σaz = AzCov(covt_θ1θ2Δφℝ, θℝ, φℝ, kidx_blk) do Σ
    cholesky(Σ, Val(false), check=false)
end; 
## Check that the cholesky's where successful
CMBrings.check_factorization(Σaz)


# Also check the Mmapped size

run(`ls -lh $(Σaz.filenm)`)





# Noise model  (Naz, Ns)
# =============================

μK′n      = 7.0 # 10.0
ellknee   = 150 # 0
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

Naz = AzCov(covn_θ1θ2Δφℝ,  θℝ, φℝ, kidx_blk) do Σ
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

## Baz  = AzCov(covb_θ1θ2Δφℝ, θℝ, φℝ, kidx_blk) do Σ
##     Σ * Diagonal(Ωℝ)
## end 
## --- or make make some beam smoothing in azimuth 
Baz  = kAzCov(covb_θ1θ2Δφℝ, θℝ, φℝ, kidx_blk) do k, Σ
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

Pr, Prᴴ, Qr, Qrᴴ, Ps, Qs = @sblock let ma𝕊, s0, s0_clip

    leftlink =  n::Int -> (cos.(range(-π,0,length=n+2)[2:end-1]) .+ 1)./2
    rightlink = n::Int -> (cos.(range(0,π,length=n+2)[2:end-1]) .+ 1)./2
    nwdt′   = 75
    nwdt′ᶜ  = 4 # 45
    ma𝕊′  = zeros(size(ma𝕊))
    ma𝕊′ᶜ = ones(size(ma𝕊))
    for (i,rw) ∈ enumerate(eachrow(ma𝕊))
        ma1 = findfirst(rw .> 0)
        ma2 = findlast(rw .> 0)
        if !isnothing(ma1)
            ma𝕊′[i,ma1:ma2] .= 1
            ma𝕊′ᶜ[i,ma1:ma2] .= 0

            ma𝕊′ᶜ[i,(ma1-1-nwdt′ᶜ):(ma1-1)] .= rightlink(nwdt′ᶜ+1)
            ma𝕊′ᶜ[i,(ma2+1):end] .= 1
            ma𝕊′ᶜ[i,(ma2+1):(ma2+1+nwdt′ᶜ)] .= leftlink(nwdt′ᶜ+1)

            ma𝕊′[i,(ma1):(ma1+nwdt′)] .= leftlink(nwdt′+1)
            ma𝕊′[i,(ma2-nwdt′):(ma2)] .= rightlink(nwdt′+1)
        end
    end
    @assert all((ma𝕊′.>0) .| (ma𝕊′ᶜ.>0))

 	Qs   = DiagOp(Xmap(s0, ma𝕊′ᶜ));
 	Ps   = DiagOp(Xmap(s0, ma𝕊′));
    
    maℝ′ = Ps[:][s0_clip,:]
    maℝᶜ = Qs[:][s0_clip,:]
    Qr   = x -> maℝᶜ .* x
    Qrᴴ  = x -> maℝᶜ .* x
    Pr  = x -> maℝ′  .* x
    Prᴴ = x -> maℝ′  .* x

    Pr, Prᴴ, Qr, Qrᴴ, Ps, Qs
end
## ----- or use a smoother mask
## 
## Pr, Prᴴ, Qr, Qrᴴ, Ps, Qs = @sblock let maℝ, ma𝕊, s0
## 
##     maℝᶜ = .!maℝ
##     ma𝕊ᶜ = .!ma𝕊
##     Qs   = DiagOp(Xmap(s0, ma𝕊ᶜ));
##     Ps  = DiagOp(Xmap(s0, ma𝕊));
##     
##     Qr   = x -> maℝᶜ .* x
##     Qrᴴ  = x -> maℝᶜ .* x
##     Pr  = x -> maℝ  .* x
##     Prᴴ = x -> maℝ  .* x
## 
##     Pr, Prᴴ, Qr, Qrᴴ, Ps, Qs
## end
## ----- or use a smoother mask
## Pr, Prᴴ, Qr, Qrᴴ, Ps, Qs = @sblock let ma𝕊, Bs, s0, s0_clip
## 
## 	ma𝕊′ = (Bs^50 * Xmap(s0,   ma𝕊))[:]
##     ma𝕊ᶜ = (Bs^50 * Xmap(s0, 1 .- ma𝕊′))[:]
##     ma𝕊ᶜ[ma𝕊ᶜ .< 1e-5] .= 0
##     
## 	ma𝕊′[abs.(ma𝕊ᶜ) .> 0.0]  .= 0
##     Ps  = DiagOp(Xmap(s0, ma𝕊′));
##     Qs  = DiagOp(Xmap(s0, ma𝕊ᶜ))
## 
## 	maℝ′ = Ps[:][s0_clip,:]
## 	maℝᶜ = Qs[:][s0_clip,:]
##     Qr   = x -> maℝᶜ .* x
##     Qrᴴ  = x -> maℝᶜ .* x
##     Pr   = x -> maℝ′  .* x
##     Prᴴ  = x -> maℝ′  .* x
##     
##     Pr, Prᴴ, Qr, Qrᴴ, Ps, Qs
## end



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
                    Be, Beᴴ, Wt, Wtᴴ, Pr, Prᴴ, Qr, Qrᴴ, 
                    σn² = abs2(μK′n*π/60/180)./Ωℝ
    
    ## A_noL and A_wL are the operators we want to invert
    A_noL = function (g)
        tmp1  = Pr(BΣBᴴ_WNWᴴ_az * Prᴴ(g))
        tmp2  = Qr(σn² .* Qrᴴ(g))    
        return tmp1 .+ tmp2
    end 

    A_wL = function (g)
        tmp0  = Pr(Be(Ln * (Σaz * (Lnᴴ * Beᴴ(Prᴴ(g))))))
        tmp1  = Pr(Wt * (Naz * (Wtᴴ * Prᴴ(g))))
        tmp2  = Qr(σn² .* Qrᴴ(g))    
        return tmp0 .+ tmp1 .+ tmp2
    end 

    PCG = function (data; lense=true, nsteps, rel_tol=1e-12)
        gwf, hist = pcg(
            g -> BΣBᴴ_WNWᴴ_az \ g, 
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
@time twf_1, hwf_1 = PCG(d_az, lense=false, nsteps=250, rel_tol = 0.1);


# WF (modeling the lensing)  
@time twf_2, hwf_2 = PCG(d_az, lense=true, nsteps=250, rel_tol = 0.1);


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



# Run PCG for conditional simulation
# =======================================

## Conditional simulation (not accounting for the lensing in the data)
@time tsim_1, hsim_1 = PCG(d_az + d_az′, lense=false, nsteps=250)
tsim_1 -= t_az′; 

## Conditional simulation  (modeling the lensing)  
@time tsim_2, hsim_2 = PCG(d_az + d_az′, lense=true, nsteps=250);
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




# (Under construction) DoF tests for conditional expected value and samples
# ====================================

DoF_d = sum(abs.(d_az) .> 0)
DoF_f = length(d_az)

fsim   = tsim_2

Δdfsim = (d_az .- Pr(Be(Ln*fsim))) 


ll2f = dot(d_az .- fsim, Σaz \ fsim)
ll2n = Δdfsim ./ Wt.^2 ./ ... # this can be computed when white noise...  # should have likelihood like Pr(Wt * n_az)


@time twf_1, hwf_1 = PCG(d_az, lense=false, nsteps=250, rel_tol = 0.001);

@sblock let Ln, Pr, Ps, Be, Wt, d_az, n_az, n_az′, fwf=twf_1, s0_clip, hide_plots=false
    hide_plots && return

    ## 
    ## Prᵒ = Ps[:][s0_clip,:]
    Prᵒ = Ps[:][s0_clip,:] .> 0.99
    ## Prᵒ = falses(size(d_az))
    ## Prᵒ[:,400:1000] .= true
    Δ   = Prᵒ .* (d_az .- Pr(Be(Ln*fwf)))
    wn1 = Prᵒ .* Pr(Wt * n_az)
    wn2 = Prᵒ .* Pr(Wt * n_az′)

    imgs = Dict(
        1 => Δ  ,  
        2 => wn1, 
        3 => wn2, 
    )
    txt =  Dict(
        1 => "data - Pr * Be * Ln * wf",
        2 => "Pr * Wt * n_az",
        3 => "Pr * Wt * n_az′",
    )
    brickplot(imgs; txt=txt, fφ=1/2)
end



# Full sky
# ==============================
# No lensing or non-stationary beam/transfer. Using FastTransforms

t_s0 = s0_sim(Σs)
n_s0 = s0_sim(Ns)
d_s0 = Ps * (Bs * t_s0 + Ws * n_s0)

t_s0′ = s0_sim(Σs)
n_s0′ = s0_sim(Ns)
d_s0′ = Ps * (Bs * t_s0′ + Ws * n_s0′)

σn²   = abs2(μK′n*π/60/180) ./ Ω𝕊
σn²Op = DiagOp(Xmap(s0, σn² .* ones(s0.nθ, s0.nφ)))
DP₁ = Σs + Ns
DP₂ = Bs * Σs * Bs' + Ns
MA₁ = Ps * Bs * Σs * Bs' * Ps'
MA₂ = Ps * Ws * Ns * Ws' * Ps'
MA₃ = Qs * σn²Op * Qs'
MG  = Σs * Bs' * Ps'

@time g1s0, hist0s0 = pcg(
        w -> DP₂ \ w, # w -> DP₁ \ w,
        w -> MA₁ * w + MA₂ * w + MA₃ * w,
        d_s0,
        nsteps  = 100,
        rel_tol = 1e-10,
)
t1_cs0sim = MG * g1s0

t1_cs0sim[:] |> matshow






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