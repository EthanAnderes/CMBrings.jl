
# load into CMBLensing
# ===========================

using CMBLensing
using PyPlot
import CirculantCov as CC
import JLD2

gi  = JLD2.jldopen("grid_info.jld2")
fes = JLD2.jldopen("field_estimates.jld2")
fdc = JLD2.jldopen("fields_data_components.jld2")
msk  = JLD2.jldopen("mask.jld2")


# import CMBrings
# prφ = CMBrings.pixweight.(Float64.(1:gi["nφ"]); ▮l=2,▯l=25,▮r=gi["nφ"]-2+1,▯r=gi["nφ"]-25+1)


κ_cr = EquiRectMap(
	## fes["κ_cr_pix"] .* msk["Mϕ_pix"], 
	fes["κ_cr_pix"] .* msk["prθ"] .* msk["prφ"]', 
	## fes["κ_cr_pix"] .* msk["prθ"] .* prφ', 
	Ny=gi["nθ"], 
	Nx=gi["nφ"], 
	θspan=gi["θ∂"] |> extrema, 
	φspan=gi["φ∂"] .|> CC.in_negπ_π |> extrema,
)

κ = EquiRectMap(
	## fdc["κ_pix"] .* msk["Mϕ_pix"], 
	fdc["κ_pix"] .* msk["prθ"] .* msk["prφ"]', 
	## fdc["κ_pix"] .* msk["prθ"] .* prφ', 
	Ny=gi["nθ"], 
	Nx=gi["nφ"], 
	θspan=gi["θ∂"] |> extrema, 
	φspan=gi["φ∂"] .|> CC.in_negπ_π |> extrema,
)

# plot(κ)
# plot(κ_cr)
# plot(κ - κ_cr)

# convert to healpix pixels

import PyCall as PC
HP = PC.pyimport("healpy") 

Nside = 2*1024

κ_hpx    = project(κ => ProjHealpix(Nside))
κ_cr_hpx = project(κ_cr => ProjHealpix(Nside))
κ_err_hpx = project(κ - κ_cr => ProjHealpix(Nside))

# cross correlation bandpowers

lmax = 5000

κ_cross_out = HP.sphtfunc.anafast(κ_hpx.arr, κ_cr_hpx.arr, lmax=lmax, pol=false)
κ_pwr_out     = HP.sphtfunc.anafast(κ_hpx.arr, lmax=lmax, pol=false)
κ_cr_pwr_out  = HP.sphtfunc.anafast(κ_cr_hpx.arr, lmax=lmax, pol=false)

let 
	rng = 20:500
	ℓ  = (0:lmax)[rng]
	ρℓ = (κ_cross_out ./ .√(κ_pwr_out .* κ_cr_pwr_out))[rng]
	figure()
	semilogx(ℓ, 1 .- ρℓ.^2, label="1 - ρℓ^2,  tru_κ, est_κ")
	xlabel("ℓ")
	legend()
end


# error bandpowers

κ_err_pwr_out = HP.sphtfunc.anafast(κ_err_hpx.arr, lmax=lmax, pol=false)

let 
	ℓ     = (0:lmax)[20:3000]
	truκℓ = κ_pwr_out[20:3000]
	estκℓ = κ_cr_pwr_out[20:3000]
	errκℓ = κ_err_pwr_out[20:3000]
	figure()
	loglog(ℓ, truκℓ, label="Ĉℓ for tru_κ")
	loglog(ℓ, estκℓ, label="Ĉℓ for est_κ")
	loglog(ℓ, errκℓ, label="Ĉℓ for tru_κ - est_κ")
	xlabel("ℓ")
	ylabel("bandpower")
	legend()
end






# import JLD2

# vb = JLD2.jldopen("vecchia_blocks.jld2")
# vb["permθ"]
# vb["block_sizesθ"]

# gi   = JLD2.jldopen("grid_info.jld2")
# gi["θ"]
# gi["φ"]
# gi["θ∂"]
# gi["φ∂"]
# gi["Ω"]
# gi["Δθ"]
# gi["nθ"]
# gi["nφ"]
# gi["freq_mult"]
# gi["grid_type"]
# gi["bsd_nθ"]

# spra = JLD2.jldopen("spectra.jld2")
# spra["ℓ"]
# spra["ϕϕℓ"]
# spra["eeℓ"]
# spra["bbℓ"]
# spra["ẽẽℓ"]
# spra["b̃b̃ℓ"]
# spra["nnℓ"]
# spra["N0ℓ"]
# spra["NΦNℓ"]
# spra["μK_arcmin"]
# spra["mult_nnℓ"]

# msk  = JLD2.jldopen("mask.jld2")
# msk["prθ"]
# msk["prφ"]
# msk["Mϕ_pix"]

# vb = JLD2.jldopen("vecchia_blocks.jld2")
# vb["permθ"]
# vb["block_sizesθ"]

# fdc = JLD2.jldopen("fields_data_components.jld2")
# fdc["d_pix"] 
# fdc["no_pix"] 
# fdc["qu_pix"] 
# fdc["ϕ_pix"]  
# fdc["κ_pix"]  

# fes = JLD2.jldopen("field_estimates.jld2")
# fes["f_cr_pix"]
# fes["g_cr_pix"]
# fes["ϕ_cr_pix"]
# fes["κ_cr_pix"]





# # Misc Test
# # =========================

# import HealpixTransforms as HT
# ϕ_hpx_mat = HT.rings2rows(ϕ_hpx.arr, Nside)

# using PyPlot
# fes["ϕ_cr_pix"] |> matshow; colorbar()
# fdc["ϕ_pix"] |> matshow; colorbar()


