
# Native proj9 curved sky Bayes MAP lensing template estimation
# =============================================

# E. Anderes (y22-m06-d14)
# 
# Goal: Proof of concept, Bayesian, curved sky, native proj9, lensing template construction. 
# 
# #### Curved sky simulation model on proj9 pixels 
# $$
# d = M \cdot \big(B \cdot L(\phi) \cdot qu + n\big)
# $$
# where 
# $(d, qu, n)$ corresponds to  (data, CMB polarization, noise) and  $(M, B, L(\phi))$ corresponds to (mask, beam, lense). 
# 
# #### Notable features
# 
# * Fully curved CMB and $\phi$ simulation (and lensing operator) directly on proj9 pixels using a block diagonalization factorization available vias rotation symmetry about the poles.
# * Two stage Wiener filter done directly on proj9 pixels.
# 
# #### Non-physical features of the sims
# 
# * The likelihood uses a new matrix approximation/factorization for the CMB and $\phi$ cov matrix.
# * Simplistic transfer function (only a beam is used)
# * Beam fwhm changes with Dec (set to arclength of pixel diagonal)
# * Simplistic noise model (white noise is used)
# * The masking isn't realistic. Doesn't have point sources. Periodic boundary conditions in RA over the obs region ( RA in [-60deg, 60deg] ) which makes the PCG very fast. 
# * No uncertainty in CMB theory spectra
# 
# #### ToDo:
# 
# * quasi-low-ell masking data on native proj9 (and the corresponding updates with PCG Wiener filtering)
# * More realistic TF and noise models (and the corresponding updates with PCG Wiener filtering)
# * Full sampling prototype (-> to be used for de-lensing error quantification on the BK side)

#-

using LinearAlgebra
using FFTW
FFTW.set_num_threads(BLAS.get_num_threads())

using XFields
using CMBrings

using  FFTransforms
import FFTransforms as FT
import HealpixTransforms as HT

import CirculantCov as CC
using FieldLensing 

using LBblocks: @sblock
import JLD2
using PyPlot
import PyCall as PC
HP = PC.pyimport("healpy")

include("LocalMethods.jl")
import .LocalMethods as LM


# Load Xfields.jl sims/estimates
# ===================================

# Xfields.jl transformations/coordinates

gi  = JLD2.jldopen("grid_info.jld2")
θ, φ, θ∂, φ∂                 = gi["θ"], gi["φ"], gi["θ∂"], gi["φ∂"]
Ω, Δθ, nθ, nφ                = gi["Ω"], gi["Δθ"], gi["nθ"], gi["nφ"]
freq_mult, grid_type, bsd_nθ = gi["freq_mult"], gi["grid_type"], gi["bsd_nθ"]
tmUS2, tmUS0, T = @sblock let nθ, nφ, freq_mult
    T  = ComplexF64
    Tr = real(T)
    tmUS2 = 𝕀(nθ) ⊗ 𝕌(T, nφ, 2π/freq_mult)
    tmUS0 = 𝕀(nθ) ⊗ 𝕌(Tr, nφ, 2π/freq_mult)
    return tmUS2, tmUS0, T
end;

# Pixel mask

msk  = JLD2.jldopen("mask.jld2")
prθ, prφ, Mϕ_pix = msk["prθ"], msk["prφ"], msk["Mϕ_pix"]
M = DiagOp(Xmap(tmUS2, prθ .* prφ' ))
Mϕ = DiagOp(Xmap(tmUS0, Mϕ_pix));

# Theory spectra

spc = JLD2.jldopen("spectra.jld2")
ℓ, ϕϕℓ   = spc["ℓ"], spc["ϕϕℓ"]
eeℓ, bbℓ = spc["eeℓ"], spc["bbℓ"]
ẽẽℓ, b̃b̃ℓ = spc["ẽẽℓ"], spc["b̃b̃ℓ"]
nnℓ = spc["nnℓ"];

# Data and Signal components

fdc = JLD2.jldopen("fields_data_components.jld2")

no  = Xmap(tmUS2, fdc["no_pix"])
d   = Xmap(tmUS2, fdc["d_pix"])
ϕ  = Xmap(tmUS0, fdc["ϕ_pix"])
κ  = Xmap(tmUS0, fdc["κ_pix"])
qu  = Xmap(tmUS2, fdc["qu_pix"])
Lqu = Xmap(tmUS2, fdc["Lqu_pix"]);

# Estimates

fes = JLD2.jldopen("field_estimates.jld2")
ϕ̂   = Xmap(tmUS0, fes["ϕ_cr_pix"])
κ̂   = Xmap(tmUS0, fes["κ_cr_pix"])
qû  = Xmap(tmUS2, fes["qu_cr_pix"])
Lqû = Xmap(tmUS2, fes["Lqu_cr_pix"]);


# proj9 plots
# ==========================================

# ### Lensing template simulation truth

# TODO: add titles...

#-

fig, ax = LM.map_plot_QU(d; θ, φ);
#-
fig, ax = LM.map_plot_QU(no; θ, φ);

#-

fig, ax = LM.map_plot_QU(qu; θ, φ);
#-
fig, ax = LM.map_plot_QU(qû; θ, φ);

#-

fig, ax = LM.map_plot_QU(Lqu; θ, φ);
#-
fig, ax = LM.map_plot_QU(Lqû; θ, φ);

#-

fig, ax = LM.map_plot_QU(Lqu - qu; θ, φ);
#-
fig, ax = LM.map_plot_QU(Lqû - qû; θ, φ);


#-

fig, ax = LM.map_plot_I(κ; θ, φ);
#-
fig, ax = LM.map_plot_I(κ̂; θ, φ);




# Load into CMBLensing.EquiRect
# ==========================================

import CMBLensing

κ′, qu′, Lqu′, κ̂′, qû′, Lqû′, no′ = let T=Float64

	θspan = θ∂ |> extrema
	φspan = φ∂ .|> CC.in_negπ_π |> extrema
	Ny    = nθ
	Nx    = nφ
	proj = CMBLensing.ProjEquiRect(;Ny, Nx, T, θspan, φspan)

	extra_prφ = CMBrings.pixweight.(T.(1:nφ); ▮l=2,▯l=25,▮r=nφ-2+1,▯r=nφ-25+1)
	mask =  real(M[:]) .* extra_prφ'

	## simulation truth

	κ′ = CMBLensing.EquiRectMap(κ[:] .* mask, proj)
	qu′ = CMBLensing.EquiRectQUMap(
		real(qu[:]) .* mask, 
		imag(qu[:]) .* mask, 
		proj,
	)
	Lqu′ = CMBLensing.EquiRectQUMap(
		real(Lqu[:]) .* mask, 
		imag(Lqu[:]) .* mask, 
		proj, 
	)

	## estimates

	κ̂′ = CMBLensing.EquiRectMap(κ̂[:] .* mask, proj)
	qû′ = CMBLensing.EquiRectQUMap(
		real(qû[:]) .* mask, 
		imag(qû[:]) .* mask, 
		proj,
	)
	Lqû′ = CMBLensing.EquiRectQUMap(
		real(Lqû[:]) .* mask, 
		imag(Lqû[:]) .* mask, 
		proj, 
	)

	## noise 
	no′ = CMBLensing.EquiRectQUMap(
		real(no[:]) .* mask, 
		imag(no[:]) .* mask, 
		proj,
	)


	κ′, qu′, Lqu′, κ̂′, qû′, Lqû′, no′
end;



#-

plot(qu′)

#-

plot(qû′)

#-

plot(κ′)

#-

plot(κ̂′)


# Project to healpix
# ==========================================

Nside = 2048

qu′2q′ = function (f)
	q = CMBLensing.Map(f).arr[:,:,1]
	CMBLensing.EquiRectMap(q, f.proj)
end 

qu′2u′ = function (f)
	u = CMBLensing.Map(f).arr[:,:,2]
	## notice the sign change on u for healpix convention
	CMBLensing.EquiRectMap(.- u, f.proj)
end 

nqₕ  = CMBLensing.project(qu′2q′(no′) => CMBLensing.ProjHealpix(Nside))
nuₕ  = CMBLensing.project(qu′2u′(no′)  => CMBLensing.ProjHealpix(Nside))

κₕ  = CMBLensing.project(κ′ => CMBLensing.ProjHealpix(Nside))
qₕ  = CMBLensing.project(qu′2q′(qu′) => CMBLensing.ProjHealpix(Nside))
uₕ  = CMBLensing.project(qu′2u′(qu′)  => CMBLensing.ProjHealpix(Nside))
Lqₕ = CMBLensing.project(qu′2q′(Lqu′) => CMBLensing.ProjHealpix(Nside))
Luₕ = CMBLensing.project(qu′2u′(Lqu′) => CMBLensing.ProjHealpix(Nside))

κ̂ₕ  = CMBLensing.project(κ̂′ => CMBLensing.ProjHealpix(Nside))
q̂ₕ  = CMBLensing.project(qu′2q′(qû′) => CMBLensing.ProjHealpix(Nside))
ûₕ  = CMBLensing.project(qu′2u′(qû′)  => CMBLensing.ProjHealpix(Nside))
Lq̂ₕ = CMBLensing.project(qu′2q′(Lqû′) => CMBLensing.ProjHealpix(Nside))
Lûₕ = CMBLensing.project(qu′2u′(Lqû′) => CMBLensing.ProjHealpix(Nside))

δLqₕ = Lqₕ .- qₕ
δLuₕ = Luₕ .- uₕ
δLq̂ₕ = Lq̂ₕ .- q̂ₕ
δLûₕ = Lûₕ .- ûₕ;



# Bandpowers
# =======================

lmax = 3000

# noise × noise

nE_nEᵦₚ, nB_nBᵦₚ = HP.sphtfunc.anafast(
	map(x->x.arr, (κₕ, nqₕ, nuₕ)), 
	lmax=lmax, pol=true, alm=false,
) |> x->(x[2,:], x[3,:]);

# truth × truth

κ_κᵦₚ, E_Eᵦₚ, B_Bᵦₚ = HP.sphtfunc.anafast(
	map(x->x.arr, (κₕ, qₕ, uₕ)), 
	lmax=lmax, pol=true, alm=false,
) |> x->(x[1,:], x[2,:], x[3,:]);

δLE_δLEᵦₚ, δLB_δLBᵦₚ = HP.sphtfunc.anafast(
	map(x->x.arr, (κₕ, δLqₕ, δLuₕ)), 
	lmax=lmax, pol=true, alm=false,
) |> x->(x[2,:], x[3,:]);

# estimate × estimate

κ̂_κ̂ᵦₚ, Ê_Êᵦₚ, B̂_B̂ᵦₚ = HP.sphtfunc.anafast(
	map(x->x.arr, (κ̂ₕ, q̂ₕ, ûₕ)), 
	lmax=lmax, pol=true, alm=false,
) |> x->(x[1,:], x[2,:], x[3,:]);

δLÊ_δLÊᵦₚ, δLB̂_δLB̂ᵦₚ = HP.sphtfunc.anafast(
	map(x->x.arr, (κ̂ₕ, δLq̂ₕ, δLûₕ)), 
	lmax=lmax, pol=true, alm=false,
) |> x->(x[2,:], x[3,:]);

# estimate × truth

κ̂_κᵦₚ, Ê_Eᵦₚ, B̂_Bᵦₚ = HP.sphtfunc.anafast(
	map(x->x.arr, (κₕ, qₕ, uₕ)), 
	map(x->x.arr, (κ̂ₕ, q̂ₕ, ûₕ)), 
	lmax=lmax, pol=true, alm=false,
) |> x->(x[1,:], x[2,:], x[3,:]);

δLÊ_δLEᵦₚ, δLB̂_δLBᵦₚ   = HP.sphtfunc.anafast(
	map(x->x.arr, (κₕ, δLqₕ, δLuₕ)), 
	map(x->x.arr, (κ̂ₕ, δLq̂ₕ, δLûₕ)), 
	lmax=lmax, pol=true, alm=false,
) |> x->(x[2,:], x[3,:]);

# estimate × truth, correlation scale

corr_κ̂_κᵦₚ     = κ̂_κᵦₚ     ./ .√(κ̂_κ̂ᵦₚ .* κ_κᵦₚ)
corr_Ê_Eᵦₚ     = Ê_Eᵦₚ     ./ .√(Ê_Êᵦₚ .* E_Eᵦₚ)
corr_B̂_Bᵦₚ     = B̂_Bᵦₚ     ./ .√(B̂_B̂ᵦₚ .* B_Bᵦₚ)
corr_δLB̂_δLBᵦₚ = δLB̂_δLBᵦₚ ./ .√(δLB̂_δLB̂ᵦₚ .* δLB_δLBᵦₚ);

# Band powers

C = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

fig,ax = subplots(1)

ℓx = 0:lmax
ℓr = 10:lmax-10

ax.loglog(ℓx[ℓr], ℓx[ℓr].^2 .* E_Eᵦₚ[ℓr], color=C[1] , label=L"$E$ bandpowers")
ax.loglog(ℓx[ℓr], ℓx[ℓr].^2 .* Ê_Êᵦₚ[ℓr], color=C[1] , linestyle=":", label=L"$\hat E$ bandpowers")
## ax.loglog(ℓ[ℓrng], ℓ[ℓrng].^2 .* eeℓ[ℓrng], label=L"$E\times E$ theory")

ax.loglog(ℓx[ℓr], ℓx[ℓr].^2 .* B_Bᵦₚ[ℓr], color=C[2] , label=L"$B$ bandpowers")
ax.loglog(ℓx[ℓr], ℓx[ℓr].^2 .* B̂_B̂ᵦₚ[ℓr], color=C[2] , linestyle=":" ,label=L"$\hat B$ bandpowers")
## ax.loglog(ℓ[ℓrng], ℓ[ℓrng].^2 .* bbℓ[ℓrng], label=L"$E\times E$ theory")

ax.loglog(ℓx[ℓr], ℓx[ℓr].^2 .* δLB_δLBᵦₚ[ℓr],color=C[3], label=L"$LensB - B$ bandpowers")
ax.loglog(ℓx[ℓr], ℓx[ℓr].^2 .* δLB̂_δLB̂ᵦₚ[ℓr],color=C[3], linestyle=":", label=L"$\widehat{LensB - B}$ bandpowers")

ax.loglog(ℓx[ℓr], ℓx[ℓr].^2 .* nE_nEᵦₚ[ℓr], color=C[4] , label=L"noise bandpowers (E)")

ax.set_xlabel(L"\ell")
ax.set_ylabel(L"\ell^2 C_\ell")
ax.legend()

# Compare this with theory spectra 


C = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

fig,ax = subplots(1)

ℓx = 0:lmax
ℓr = 10:lmax-10

ax.loglog(ℓ[ℓr], ℓ[ℓr].^2 .* eeℓ[ℓr], color=C[1], label=L"$E$ theory")
ax.loglog(ℓ[ℓr], ℓ[ℓr].^2 .* bbℓ[ℓr], color=C[2] ,label=L"$B$ theory")
## ax.loglog(ℓ[ℓr], ℓ[ℓr].^2 .* (b̃b̃ℓ - bbℓ)[ℓr], color=C[3], label=L"$LensB - B$ theory")
ax.loglog(ℓ[ℓr], ℓ[ℓr].^2 .* nnℓ, color=C[4], label=L"$n$ theory")

ax.set_xlabel(L"\ell")
ax.set_ylabel(L"\ell^2 C_\ell")
ax.legend()


# cross power

C = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

fig,ax = subplots(1)

ℓx = 0:lmax
ℓr = 10:lmax-1000

ax.plot(ℓx[ℓr], corr_κ̂_κᵦₚ[ℓr],    label=L"$\kappa \times \hat\kappa$ corr")
ax.plot(ℓx[ℓr], corr_Ê_Eᵦₚ[ℓr],    label=L"$E \times \hatE$ corr")
ax.plot(ℓx[ℓr], corr_δLB̂_δLBᵦₚ[ℓr],label=L"$\delta L B \times \widehat{\delta L B}$ corr")
ax.set_ylim(bottom=0.7, top=1.0)
ax.set_xlabel(L"\ell")
ax.set_ylabel("correlation")
ax.legend()



