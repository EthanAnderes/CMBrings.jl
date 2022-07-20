

# map space view
# ==============================

function map_plot_QU(
    QU; 
    θ, φ,
    imag_fun = x->x,
    title1 = L"Q(\theta,\varphi)", 
    title2 = L"U(\theta,\varphi)",
    vmin = nothing, vmax = nothing,
    )

    Q, U = QU[:] |> x->(real(x), imag(x))
    
    nθ = length(θ)
    nφ = length(φ)

    f1k = Q |> imag_fun
    f2k = U |> imag_fun
    
    fig, ax = subplots(2,1,dpi=147) # , figsize=(8,5))
    ax[1].set_aspect("auto") # , adjustable="box")
    ax[2].set_aspect("auto") # , adjustable="box")
    ax[1].set_xlim(1, length(φ))
    ax[2].set_xlim(1, length(φ))

    img1 = ax[1].imshow(f1k, vmin=vmin, vmax=vmax, origin="upper")
    
    img2 = ax[2].imshow(f2k, vmin=vmin, vmax=vmax,  origin="upper")
    
    θ_trng = round.(Int,range(1, nθ, 7))
    # push!(θ_trng, findmin(abs.(θ .- 2.445))[2])
    # sort!(θ_trng)
    for axx in ax
        axx.set_yticks(θ_trng)
        axx.set_yticklabels(round.(θ[θ_trng], digits=2),fontsize=6)
        axx.set_ylabel(L"polar $\theta$ [rad]",fontsize=7)
    end 

    φ′ = rad2deg.(CC.in_negπ_π.(φ))
    φi0 = length(φ′)÷2 + 1
    tti = round.(Int,range(0,length(φ′)÷2,6)[2:end-1])
    φ_trng = vcat(φi0 .- tti[end:-1:1], φi0, φi0 .+ tti)

    for axx in ax
        axx.set_xticks(φ_trng)
        axx.set_xticklabels(round.(Int, φ′[φ_trng]),fontsize=6)
        axx.set_xlabel(L"azimuth $\varphi$ [deg]",fontsize=7)
    end 

    ax[1].set_title(title1, fontsize=8)  
    ax[2].set_title(title2, fontsize=8)  

    cbar1 = fig.colorbar(img1, ax=ax[1], fraction=0.046*(nθ/nφ), pad=0.04)
    cbar2 = fig.colorbar(img2, ax=ax[2], fraction=0.046*(nθ/nφ), pad=0.04)
    cbar1.ax.tick_params(labelsize=6)
    cbar2.ax.tick_params(labelsize=6)

    fig.tight_layout()

    return fig, ax
end


function map_plot_I(
    Ifield;
    θ, φ, 
    imag_fun = x->x,
    title1 = L"I(\theta,\varphi)", 
    vmin = nothing, vmax = nothing,
    )

    nθ = length(θ)
    nφ = length(φ)

    fx = Ifield[:] |> imag_fun
    
    fig, ax = subplots(1,dpi=147) # , figsize=(8,5))
    ax.set_aspect("auto") # , adjustable="box")
    ax.set_xlim(1, length(φ))

    img1 = ax.imshow(fx, vmin=vmin, vmax=vmax, origin="upper")

    θ_trng = round.(Int,range(1, nθ, 7))
    ## push!(θ_trng, findmin(abs.(θ .- 2.445))[2])
    ## sort!(θ_trng)
    ax.set_yticks(θ_trng)
    ax.set_yticklabels(round.(θ[θ_trng], digits=2),fontsize=6)
    ax.set_ylabel(L"polar $\theta$ [rad]",fontsize=7)

    φ′ = rad2deg.(CC.in_negπ_π.(φ))
    φi0 = length(φ′)÷2 + 1
    tti = round.(Int,range(0,length(φ′)÷2,6)[2:end-1])
    φ_trng = vcat(φi0 .- tti[end:-1:1], φi0, φi0 .+ tti)
    ax.set_xticks(φ_trng)
    ax.set_xticklabels(round.(Int, φ′[φ_trng]),fontsize=6)
    ax.set_xlabel(L"azimuth $\varphi$ [deg]",fontsize=7)

    ax.set_title(title1, fontsize=8)  

    cbar1 = fig.colorbar(img1, ax=ax, fraction=0.046*(nθ/nφ), pad=0.04)
    cbar1.ax.tick_params(labelsize=6)

    fig.tight_layout()

    return fig, ax
end




# for example 
# using ImageFiltering
# imag_fun = x -> imfilter(x, Kernel.gaussian(blur.*(1,(nφ÷2)/nθ)), "circular")
function fourier_power(
    T; 
    θ, φ, 
    imag_fun = x->abs2.(x),
    ℓs = Int[], 
    vmin=nothing, vmax=nothing, 
    title1=L"$|I\,(\theta,\ell_\varphi)|$", 
    xaxis_units = :Hz, # or :m
    )

    nφ = length(φ)
    nθ = length(θ)
    Δθₒ = abs(θ[2] - θ[1]) 

    if eltype_in(fieldtransform(T)) <: Real
        k = FFTransforms.freq(fieldtransform(T))[2]
        f1k = T[!] |> imag_fun
    else 
        k = FFTransforms.freq(fieldtransform(T))[2] |> fftshift
        k[1] *= iseven(nφ) ? -1 : 1 # the nyquist should signflip to negative when nφ is even
        f1k = T[!] |> x->fftshift(x,2) |> imag_fun 
    end

    if xaxis_units == :Hz
        Hz_or_m = m -> deg2rad(1) / (2π / m)
        xlabel_hz_or_m = L"f [Hz] (for scan rate of $1^o$ per second)"
        # (every second the telescope traverses deg2rad(1) rad) / (wave length of k)
    elseif xaxis_units == :m 
        Hz_or_m = m -> m
        xlabel_hz_or_m = L"azmuthal frequency $m$"
    end

    fig, ax = subplots(1,dpi=147)
    img = ax.imshow(f1k, cmap="viridis", 
        vmin=vmin, vmax=vmax, 
        extent=[Hz_or_m(k[1]), Hz_or_m(k[end]), θ[end], θ[1]],
        origin="upper"
    )
    ax.set_aspect("auto") 

    
    ms_trng = round.(Int,range(1, length(k), 10)[2:end])
    ms = Int.(k[ms_trng])
    if isempty(ℓs)
        ℓs1 = round.(Int, ms[ms .> 0] * csc.(θ[end÷4]))
        ℓs2 = round.(Int, ms[ms .> 0] * csc.(θ[3*end÷4]))
        ℓs = vcat(ℓs1, ℓs2[ℓs2 .> maximum(ℓs1)])
    end
    for ℓₒ in ℓs
        mv = 1:ℓₒ
        θv = π .- acsc.(ℓₒ ./ mv)
        rng = (θ[1] .<= θv .<= θ[end]) .& (k[1] .<= mv .<= k[end]) 
        if any(rng)   
            ax.plot(Hz_or_m.(mv[rng]), θv[rng], c="0.5")
            ax.annotate(
                L"$\,\,\ell=%$ℓₒ\,\,$", 
                xy=(Hz_or_m(mv[rng][end]), θv[rng][end]),  
                xycoords="data", ha="center", va="bottom",
                fontsize=7
            )
            if !(eltype_in(fieldtransform(T)) <: Real)
                ax.plot(.- Hz_or_m.(mv[rng]), θv[rng], c="0.5")
                ax.annotate(
                    L"$\,\,\ell=%$ℓₒ\,\,$", 
                    xy=(.-Hz_or_m(mv[rng][end]), θv[rng][end]),  
                    xycoords="data", ha="center", va="bottom",
                    fontsize=7
                )            
            end
        end
    end

    cbar = fig.colorbar(img, ax=ax, location="bottom", shrink = 0.5) #, fraction=0.046*(nθ/nφ))
    cbar.ax.tick_params(labelsize=7)

    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("θ",fontsize=7)
    ax.set_xlabel(xlabel_hz_or_m,fontsize=7)

    ax.set_title(title1, fontsize=8, pad=20)

    fig.tight_layout()

    return fig, ax
end





####################
## TODO: update these and merge with above



# The following allows 
# fig, ax = subplots(2,dpi=147)
# A |> imshow(-, fig, ax[1])
# A |> imshow(-, fig, ax[2])
#
# ... or ...
# fig, ax = subplots(2,dpi=147)
# imshow(A, fig, ax[1])
# imshow(A, fig, ax[2])

function PyPlot.imshow(A::Matrix, fig::Figure, ax; vmin=nothing, vmax=nothing, shrink=0.7, pad=0.015, tight_layout=true)
    PyPlot.imshow(-, fig, ax; vmin, vmax, shrink, pad, tight_layout)(A)
end


function PyPlot.imshow(::typeof(-), fig::Figure, ax; vmin=nothing, vmax=nothing, shrink=0.7, pad=0.015, tight_layout=true)
    function (A::Matrix)
        img = ax.imshow(A, vmin=vmin, vmax=vmax)
        ax.axis("off")
        fig.colorbar(img, ax=ax, shrink=shrink, pad=pad)
        tight_layout && fig.tight_layout()
        img
    end
end



function brickplot(imgs::Dict{Int,T};
            txt  = Dict{Int,String}(), # overlay text
            ctxt = Dict{Int,String}(), # color of text
            fφ = 1/2,     # fraction of azimuth 
            sz = 2,       # Overall size scale
            hmlt = 0.875, # Hight adjust
        ) where T

    nimg = maximum(keys(imgs))
    nr = size(imgs[nimg])[1]
    nc = size(imgs[nimg])[2] * fφ |> x->round(Int,x)

    fig, ax = subplots(nimg,1,figsize=(sz*(nc/nr), sz*nimg*hmlt),dpi=147)
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
            fontsize=14,
        )
    end
    fig.subplots_adjust(hspace=0.01, bottom = 0.1, top = 0.98, left = 0.05, right=0.98)
    
    fig, ax
end



function diskplot(imgs::Dict{Int,T}, φ, θ;
            txt  = Dict{Int,String}(), # overlay text
            figsize = (),
            nrows   = 1, 
            sz      = 1,   
            fontsize=14,
            vcenter=nothing,
            vmin_quantile=1e-5, # when this is zero extends to max
        ) where T

    clrs = pyimport("matplotlib.colors")

    nimg  = maximum(keys(imgs))
    # how many columnes do we need to fit 
    ncols = ceil(Int, nimg/nrows)

    fig, ax = subplots(
        nrows, ncols, 
        figsize= length(figsize)==0 ? (sz*(4.5*ncols), sz*(5*nrows)) : figsize,
        subplot_kw=Dict(:projection=>"polar"),
        dpi=147
    )

    ax = nimg==1 ? [ax] : ax

    for (i,f) ∈ imgs
        # if isnothing(vcenter)
        #     amin,amax = quantile(f[:],vmin_quantile), quantile(f[:],1-vmin_quantile)
        #     divnorm = clrs.TwoSlopeNorm(
        #         vmin=amin,
        #         vcenter=(amin + amax)/2,
        #         vmax=amax,
        #     )
        # else
        #     maxdist = quantile((abs.(f .- vcenter))[:], 1-vmin_quantile)
        #     divnorm = clrs.TwoSlopeNorm(
        #         vmin=vcenter - maxdist,
        #         vcenter=vcenter,
        #         vmax=vcenter + maxdist,
        #     )
        # end
        img = ax[i].pcolormesh(φ, θ, f) # , norm=divnorm)
        fig.colorbar(
            img, ax=ax[i], 
            shrink=0.6, extend="both", pad=0.015,
            orientation="horizontal"
        )
    end
    
    for i=1:(nrows*ncols)
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        if i>nimg 
            ax[i].grid(false)
            ax[i].axis(false)
        end
    end

    for (i,s) ∈ txt
        ax[i].set_title(s, fontsize=fontsize, loc="left")
    end
    # fig.subplots_adjust(top=0.95)
    fig.tight_layout()
    # fig.subplots_adjust(hspace=0.02, wspace=0.02)
    
    fig, ax
end

