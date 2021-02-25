### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 338bddac-7633-48f1-935f-ddf6a6bfbbdd
begin
  using CMBrings
  import CMBsphere as CS
  import CMBflat as CF
  
  using XFields
  using Spectra
  using FFTransforms
  using FieldLensing
  
  using LinearAlgebra
  using SparseArrays
  import Dierckx
  import NLopt
  
  using DelimitedFiles
  using LBblocks: @sblock
  using PyPlot
  using BenchmarkTools
  using ProgressMeter
end

# ╔═╡ cd890772-f14d-4ba1-9bca-7c69b794f8ce
# Spin 2 lensing example which uses CMBsphere transform to handle the QU cov operator

# ╔═╡ 24d1abe0-504c-40bc-b795-033f6f2a0674
md"""
Modules
==============================
using FFTW
FFTW.FFTW.set_num_threads(8)
"""

# ╔═╡ 46906fd3-28a2-472b-af2a-25235e544a31
if isdefined(Main,:PlutoRunner)
    import PlutoUI
    hide_plots = false
elseif isdefined(Main, :IJulia) && Main.IJulia.inited
    hide_plots = false
else
    hide_plots = true
end

# ╔═╡ 8261d1f4-de62-4ca6-8af5-148d95ea2900
md"""
Set ring transforms
==============================
"""

# ╔═╡ d89ea090-827d-4d07-a6bf-47194d7e8e12
tmS0, tmS2 = @sblock let

    T_fld   = Float64

    nθ, nφ  = 500, 2048-1

    tmW0  = 𝕀(nθ) ⊗ 𝕎(T_fld, nφ, 2π)
    tmS0  = unitary_scale(tmW0) * tmW0

    # FIXME: not yet sure that this is the correct way to handle polarization
    tmW2  = 𝕀(nθ) ⊗ 𝕎(T_fld, nφ, 2π) ⊗ 𝕀(2)
    tmS2  = unitary_scale(tmW2) * tmW2

    return tmS0, tmS2
end

# ╔═╡ f379eec8-da9e-40ed-828a-4cc485149963
md"""
Mask and CMBring observation region
==============================
"""

# ╔═╡ cde8b28c-136e-46c6-9e54-3f7b3ca87ecb
data_mask_init, Ω, θ, φ, θnorth∂, θsouth∂ = @sblock let tmS0, QP_bdry=1e-5, fwhm′=150

    pr_mat_init  = readdlm(joinpath(CMBrings.module_dir,"examples/artifacts/FastTransform_mask_nθ3072_nφ4095.csv"), ',', Bool)

    full_sky_tm𝕊0 = CS.𝕊0(size(pr_mat_init)...)
    θ_mat_init, φ_mat_init = CS.pix(full_sky_tm𝕊0)
    spline_mask = Dierckx.Spline2D(θ_mat_init, φ_mat_init, pr_mat_init, kx=1, ky=1, s=0.0)

    nθ, nφ  = size_in(tmS0)
    # θnorth∂ = 2.12
    θnorth∂ = 2.25
    θsouth∂ = 2.85
    θ = θnorth∂ .+ ((θsouth∂ - θnorth∂) / nθ) .* (0:nθ-1)
    φ = (2π / nφ) .* (0:nφ-1)
    Ω = CS.Ωpix.(θ, θ[2] - θ[1], φ[2] .- φ[1])

    data_mask_init = spline_mask.(θ, φ') .> 0
    data_mask_init[1:30,:] .= 0
    data_mask_init[end - 30 + 1:end,:] .= 0

    return data_mask_init, Ω, θ, φ, θnorth∂, θsouth∂

end;

# ╔═╡ 8148f0fe-db16-4a55-ba1f-6363f43ec697
Pr, Qr = @sblock let tmS0, data_mask_init, θnorth∂, θsouth∂,  QP_bdry=1e-5, fwhm′=150

    tmFlat = CF.𝕎(Float64, size(data_mask_init), (θsouth∂ - θnorth∂, 2π))
    pr0x, qr0x = CF.PrQr(tmFlat, data_mask_init, fwhm′, fwhm′, QP_bdry)
    pr0 = Xmap(tmS0, pr0x)
    qr0 = Xmap(tmS0, qr0x)

    DiagOp(pr0), DiagOp(qr0)
end;

# ╔═╡ 9005764e-16a5-4c9b-95ea-2955abd45275
md"Localize lensing vector field to data mask."

# ╔═╡ 6311f7c7-ec09-4a75-b1b3-dd47e367ba54
Mϕ = @sblock let tmS0, data_mask_init, θnorth∂, θsouth∂,  QP_bdry=1e-5, fwhm′=75

    tmFlat = CF.𝕎(Float64, size(data_mask_init), (θsouth∂ - θnorth∂, 2π))
    pr0x, qr0x = CF.PrQr(tmFlat, data_mask_init, fwhm′, fwhm′, QP_bdry)
    pr0 = Xmap(tmS0, pr0x)
    qr0 = Xmap(tmS0, qr0x)

    # mϕx = pr0x .+ qr0x
    mϕx = pr0x

    # make sure it hits zero and 1
    mϕx .-= minimum(mϕx)
    mϕx ./= maximum(mϕx)
    Mϕ    = DiagOp(Xmap(tmS0, mϕx))

    Mϕ
end;

# ╔═╡ 254c932c-6671-4caf-8d7e-8e449afd1c48
md"Azimuthal ring mask"

# ╔═╡ 7be01eda-23df-44ad-a949-7f3bd292fe31
@sblock let ma=Pr[:], φ, θ, hide_plots
    hide_plots && return
    imgs = Dict(1=>ma)
    txt  = Dict(1=>"Mask")
    ctxt = Dict(1=>"w")
    # CMBrings.brickplot(imgs; txt=txt, ctxt=ctxt, fφ=1)
    ## fig, ax = CMBrings.brickplot(imgs; txt=txt, ctxt=ctxt, fφ=1)
    fig, ax = CMBrings.diskplot(imgs, φ', π.-θ; txt=txt, nrows=1, fontsize=14)
    return fig
end

# ╔═╡ 76019b6c-753c-11eb-0bbb-7b613ce3b5b4
begin
fig = figure()
imshow(Pr[:])
fig
end

# ╔═╡ 8bdc8a26-99dd-4d0a-8514-99938a2932db
md"Plot √Ωpix over ring θ's"

# ╔═╡ 35089a3c-5394-4967-a0de-1721c1bc2df2
@sblock let θ, φ, Ω, hide_plots
    hide_plots && return
    fig,ax = subplots(1)
    ax.plot(θ, rad2deg.(sqrt.(Ω)).*60, label="sqrt pixel area (arcmin)")
    ax.plot(θ, zero(θ) .+ rad2deg.(θ[2] - θ[1]).*60, label="Δθ (arcmin)")
    # ax.plot(θ, zero(θ) .+ rad2deg.(φ[2] - φ[1]).*60, label="Δφ (arcmin)")
    ax.set_xlabel(L"polar coordinate $\theta$")
    ax.legend()
	return fig
end

# ╔═╡ a6b7bfeb-1b19-48fb-bca8-7eec7950fd82
md"""
Spectral densities
==============================
"""

# ╔═╡ d0cb98d9-aba2-4dc4-9873-1b1430e635cc
md"ϕϕ, EB spectra"

# ╔═╡ 4ab9d54a-4b2d-48eb-b43e-d0ebe87da176
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
    # note: bbsl == 0
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

# ╔═╡ fbbf5de5-e211-4a91-9009-4cfb2012998f
md"beam/transfer"

# ╔═╡ 59191ba7-0462-494e-abd3-cb77d7a79683
bl = @sblock let

    beamfwhm  = 4.0 |> arcmin -> deg2rad(arcmin/60)

    lmax = 11000
    l = 0:lmax
    σ² = beamfwhm^2 / 8 / log(2)
    bl = @. exp( - σ²*l*(l+1) / 2)
    return bl

end;

# ╔═╡ a21676f2-f211-4548-879e-dc128f3db7b3
md"noise"

# ╔═╡ f2101d66-13e7-4da5-a367-9aa3c6cf34d0
nnl, wnl, snl = @sblock let

    μK′n      = 2.5
    ellknee   = 0
    alphaknee = 3

    lmax = 11000
    l = 0:lmax
    whitenoisel    = fill(μK′n^2 * (π/60/180)^2, size(l))
    smoothnoisel   = @. μK′n^2 * (π/60/180)^2 * Spectra.knee(l; ell=ellknee, alpha=alphaknee)
    smoothnoisel .-= μK′n^2 * (π/60/180)^2
    smoothnoisel[smoothnoisel .< 0] .= 0
    noisel = smoothnoisel .+ whitenoisel
    return noisel, whitenoisel, smoothnoisel

end;

# ╔═╡ b8dc4e8a-d93f-4521-bf32-94657e65284e
@sblock let hide_plots, nnl, eel, bbl, ϕϕl, bl, lmax=tmS0.nθ-1
    hide_plots && return

    l = 0:length(nnl)-1
    rng = 2:5000

    fig,ax = subplots(1)
    ax.plot(l[rng], eel[rng], label="ee")
    ax.plot(l[rng], bbl[rng], label="bb")
    ax.plot(l[rng], nnl[rng], ":", label="noise")
    ax.plot(l[rng], nnl[rng]./bl[rng], ":", label="noise/beam")
    ax.axvline(x=lmax, label="data lmax")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(L"\ell")
    ax.set_xlabel(L"\ell^2 C_\ell")
    ax.legend()

    fig
end

# ╔═╡ Cell order:
# ╠═cd890772-f14d-4ba1-9bca-7c69b794f8ce
# ╟─24d1abe0-504c-40bc-b795-033f6f2a0674
# ╠═338bddac-7633-48f1-935f-ddf6a6bfbbdd
# ╠═46906fd3-28a2-472b-af2a-25235e544a31
# ╟─8261d1f4-de62-4ca6-8af5-148d95ea2900
# ╠═d89ea090-827d-4d07-a6bf-47194d7e8e12
# ╟─f379eec8-da9e-40ed-828a-4cc485149963
# ╠═cde8b28c-136e-46c6-9e54-3f7b3ca87ecb
# ╠═8148f0fe-db16-4a55-ba1f-6363f43ec697
# ╟─9005764e-16a5-4c9b-95ea-2955abd45275
# ╠═6311f7c7-ec09-4a75-b1b3-dd47e367ba54
# ╟─254c932c-6671-4caf-8d7e-8e449afd1c48
# ╠═7be01eda-23df-44ad-a949-7f3bd292fe31
# ╠═76019b6c-753c-11eb-0bbb-7b613ce3b5b4
# ╟─8bdc8a26-99dd-4d0a-8514-99938a2932db
# ╠═35089a3c-5394-4967-a0de-1721c1bc2df2
# ╟─a6b7bfeb-1b19-48fb-bca8-7eec7950fd82
# ╟─d0cb98d9-aba2-4dc4-9873-1b1430e635cc
# ╠═4ab9d54a-4b2d-48eb-b43e-d0ebe87da176
# ╟─fbbf5de5-e211-4a91-9009-4cfb2012998f
# ╠═59191ba7-0462-494e-abd3-cb77d7a79683
# ╟─a21676f2-f211-4548-879e-dc128f3db7b3
# ╠═f2101d66-13e7-4da5-a367-9aa3c6cf34d0
# ╠═b8dc4e8a-d93f-4521-bf32-94657e65284e
