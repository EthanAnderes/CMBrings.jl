# The following allows 
# fig, ax = subplots(2)
# A |> imshow(-, fig, ax[1])
# A |> imshow(-, fig, ax[2])
#
# ... or ...
# fig, ax = subplots(2)
# imshow(A, fig, ax[1])
# imshow(A, fig, ax[2])

function PyPlot.imshow(A::Matrix, fig::Figure, ax; vmin=nothing, vmax=nothing, shrink=0.7)
    PyPlot.imshow(-, fig, ax; vmin, vmax, shrink)(A)
end


function PyPlot.imshow(::typeof(-), fig::Figure, ax; vmin=nothing, vmax=nothing, shrink=0.7)
    function (A::Matrix)
        img = ax.imshow(A, vmin=vmin, vmax=vmax)
        ax.axis("off")
        fig.colorbar(img, ax=ax, shrink=shrink)
        fig.tight_layout()
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
        subplot_kw=Dict(:projection=>"polar")
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

