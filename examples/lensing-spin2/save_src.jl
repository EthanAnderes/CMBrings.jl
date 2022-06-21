import JLD2 

JLD2.jldsave("grid_info.jld2"; 
	θ, φ, θ∂, φ∂, Ω, Δθ, nθ, nφ, freq_mult, grid_type, bsd_nθ
)

JLD2.jldsave("spectra.jld2";
	ℓ, ϕϕℓ, eeℓ, bbℓ, ẽẽℓ, b̃b̃ℓ, nnℓ, N0ℓ, NΦNℓ, μK_arcmin, mult_nnℓ
)

JLD2.jldsave("mask.jld2";
	prθ, prφ, Mϕ_pix=Mϕ[:]
)

JLD2.jldsave("vecchia_blocks.jld2";
	permθ, block_sizesθ
)

JLD2.jldsave("fields_data_components.jld2";
	d_pix  = d[:],
	no_pix = no[:],
	qu_pix = qu[:],
	Lqu_pix = (Ł(ϕ) * qu)[:],
	ϕ_pix = ϕ[:],
	κ_pix = kappa(ϕ),
)

JLD2.jldsave("field_estimates.jld2";
	qu_cr_pix = f_cr[:],
	g_cr_pix = g_cr[:],
	Lqu_cr_pix = (Ł(ϕ_cr) * f_cr)[:],
	ϕ_cr_pix = ϕ_cr[:],
	κ_cr_pix = kappa(ϕ_cr),
)


