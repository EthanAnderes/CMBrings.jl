# Beam operator
# ==========================================

# pix_diag_rad   = CC.geoβ.(tm0.θ∂[2:end], θ∂[1:end-1], φ[1], φ[2]) # arclength of the pixel diagonals


function B̃eam1(θ₁, θ₂, σ²θ₁, σ²θ₂, Δφ)
    sinθ₁, cosθ₁ = sincos(θ₁)
    sinθ₂, cosθ₂ = sincos(θ₂)
    sinΔφ, cosΔφ = sincos(Δφ)
    Δx = sinθ₁ * cosθ₂ * cosΔφ - sinθ₂ * cosθ₁
    Δy = sinθ₁ * sinΔφ
    σ²θ₁θ₂ = (σ²θ₁ + σ²θ₂ ) / 2
    return exp( - (Δx^2 + Δy^2) / σ²θ₁θ₂ / 2 ) / σ²θ₁θ₂ / 2 / π
end 

function B̃eam2(θ₁, θ₂, σ²θ₁, σ²θ₂, Δφ)
    sinθ₁ = sin(θ₁)
    sinθ₂ = sin(θ₂)
    sinΔθ = sin((θ₁-θ₂)/2)
    sinΔφ = sin(Δφ/2)
    σ²θ₁θ₂ = (σ²θ₁ + σ²θ₂ ) / 2
    return exp( - 2 * (sinΔθ^2 + sinθ₁*sinθ₂*sinΔφ^2) / σ²θ₁θ₂) / σ²θ₁θ₂ / 2 / π
end 

function beam_Γ(eaz::EAZ{T}; fwhmθ_rad=EZ.pix_diag_rad(eaz)) where {T}
    σ²θ = @. fwhmrad2σ².(fwhmθ_rad)
    σ²θ_spl = CC.Spline1D(EZ.θ(eaz), σ²θ, k=2)
    (θ₁, θ₂, φ₁, φ⃗) -> complex.(B̃eam1.(θ₁, θ₂, σ²θ_spl(θ₁), σ²θ_spl(θ₂), φ₁ .- φ⃗))
end

# TODO: is it worth it to add perm argument here?
function beam▫(eaz0::EAZ0{T}; fwhmθ_rad=EZ.pix_diag_rad(eaz0), block_sizesθ, normalizeθ = :row_ave) where {T}

    Γ = beam_Γ(eaz0; fwhmθ_rad)

    Σ_pre▫ = eaz_cov_btridiag(eaz0, Γ; block_sizesθ)
    Σ▫     = map(Σ_pre▫) do Σ
        VF.vecchia_general(Σ, block_sizesθ)
    end

    if normalizeθ == :none
        return Σ▫ 
    elseif normalizeθ == :row_ave
        ## Adjust so row mean of the pixel kernel is 1
        bws  = beamθ_weight_sum(eaz0; fwhmθ_rad)
        Dw⁻¹ = Diagonal(inv.(bws))
        return map(Σ▫i -> Dw⁻¹ * Σ▫i, Σ▫)
    elseif normalizeθ == :Ω
        ## Adjust so left mult behaves like an integral operator
        dΩ = EZ.Ωpix(eaz0)
        DΩ = Diagonal(dΩ)
        return map(Σ▫i -> Σ▫i * DΩ, Σ▫)
    else 
        error("normalizeθ ∉ {:row_ave, :Ω, :none}")
    end
end

# TODO: is it worth it to add perm argument here?
function beam▫(eaz2::EAZ2{T}; fwhmθ_rad=EZ.pix_diag_rad(eaz2), block_sizesθ, normalizeθ = :row_ave) where {T}

    Σ0▫ = beam▫(EZ.spin0(eaz2); fwhmθ_rad, block_sizesθ, normalizeθ=:none)

    # TODO: either make this so it shares memory with Σ0▫
    # or allow spin0 operators in fourier to multiply on q,u fields separately
    nθ = eaz2.nθ

    Σ2▫ = map(Σ0▫) do B
        # M -> M2
        M = B[2]
        M2 = vcat(M.data, M.data) |> VF.Midiagonal
        
        # R -> invR2
        R = inv(B[1])
        invR2 = vcat(
            R.data, 
            [zeros(eltype(M.data[1]), size(M.data[1],1), size(M.data[end],2))], 
            R.data
        ) |> VF.Ridiagonal |> inv
        # put everything back together
        invR2 * M2 * invR2'
    end

    if normalizeθ == :none
        return Σ2▫ 
    elseif normalizeθ == :row_ave
        ## Adjust so row mean of the pixel kernel is 1
        bws     = beamθ_weight_sum(eaz2; fwhmθ_rad)
        inv_bws = inv.(bws)
        Dw⁻¹    = Diagonal(vcat(inv_bws,inv_bws))
        return map(Σ▫i -> Dw⁻¹ * Σ▫i, Σ2▫)
    elseif normalizeθ == :Ω
        ## Adjust so left mult behaves like an integral operator
        dΩ = EZ.Ωpix(eaz2)
        DΩ = Diagonal(vcat(dΩ,dΩ))
        return map(Σ▫i -> Σ▫i * DΩ, Σ2▫)
    else 
        error("normalizeθ ∉ {:row_ave, :Ω, :none}")
    end

end  


function beamθ_weight_sum(eaz::EAZ{T}; fwhmθ_rad) where {T}

    Γ = beam_Γ(eaz; fwhmθ_rad)

    # use fwhmθ_rad to give an approximate sub-grid for computing the sum..
    θ  = EZ.θ(eaz)
    Δθ = EZ.Δθ(eaz)

    Δφᵢ = T(EZ.Δφ(eaz))
    φᵢ  = T(0)
    φ   = T(-π/2):Δφᵢ:T(π/2)

    weight_sum_at_θ = fill(T(0), length(θ))

    for i in eachindex(θ)

        θᵢ     = θ[i]

        # create subgrid patch around (θᵢ,φᵢ)
        fwhmrᵢ = fwhmθ_rad[i]
        maxΔθᵢ = 3*max(2*fwhmrᵢ, Δθ[i])
        maxΔφᵢ = 3*max(2*fwhmrᵢ, Δφᵢ)
        θs     = θ[(θᵢ - maxΔθᵢ) .≤ θ .≤ (θᵢ + maxΔθᵢ)]
        φs     = φ[(φᵢ - maxΔφᵢ) .≤ φ .≤ (φᵢ + maxΔφᵢ)]

        weights_around_θᵢφᵢ = real.(Γ.(θᵢ, θs, φᵢ, Ref(φs)))
        weight_sum_at_θ[i]  = EZ.sum_kbn(hcat(weights_around_θᵢφᵢ...))

    end

    return weight_sum_at_θ
end


