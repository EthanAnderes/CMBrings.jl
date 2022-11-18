# Pixel window function and pixel decimation
# TODO

# • add decimation operator. 



# Healpix pixel window function 
# ===============================================
# healpix_pwf▫ constructs the block diagonals for the 
# conv operator


function healpix_pwf_Γ(Nside::Int)
    θhpx, φhpx, idxhpx, Δφhpx, nφhpx = HT.θ_φ_idx_4_rings(Nside)
    function (θ₁, θ₂, φ₁, φ⃗)
        # we need find an approx θ spacing to the nearest healpix rings
        # to the north and south of θ₁
        # ic = findfirst(θhpx .> θ₁) # index of the nearest ring to θ₁
        ic = findmin(abs2.(θhpx .- θ₁))[2] # index of the nearest ring to θ₁
        Δθ_north  = abs(θhpx[ic] - θhpx[ic-1])
        Δθ_south  = abs(θhpx[ic+1] - θhpx[ic])
        Δφ_center = Δφhpx[ic]
        pixtf = HT.pixel.(
            θ₂, φ⃗;
            φ_center = φ₁, 
            Δφ_center, 
            θ_center = θ₁,
            θ_north  = θ₁ - Δθ_north, 
            θ_south  = θ₁ + Δθ_south,
        )
        return complex.(1.0.*pixtf,0)
    end
end


function healpix_count_θ(eaz::EAZ; Nside::Int)
    θ = EZ.θ(eaz)
    φ = EZ.φ(eaz)
    θhpx, φhpx, idxhpx, Δφhpx, nφhpx = HT.θ_φ_idx_4_rings(Nside)
    npixel_at_θ = fill(0, length(θ))

    for i in eachindex(θ)

        θᵢ = θ[i]
        # ic = findfirst(θhpx .> θᵢ) # index of the nearest ring to θ₁
        ic = findmin(abs2.(θhpx .- θᵢ))[2] # index of the nearest ring to θ₁
        Δθ_north  = abs(θhpx[ic] - θhpx[ic-1])
        Δθ_south  = abs(θhpx[ic+1] - θhpx[ic])
        θ_center = θᵢ
        θ_north  = θᵢ - Δθ_north 
        θ_south  = θᵢ + Δθ_south

        φ_center  = φ[3end÷4] # this needs to be in the interior ...
        Δφ_center = Δφhpx[ic]

        θs = θ[θ_north .≤ θ .≤ θ_south]
        φs = φ[φ_center-Δφ_center .≤ φ .≤ φ_center+Δφ_center]


        pixel_at_dec = HT.pixel.(θs, φs'; φ_center, Δφ_center, θ_center, θ_north, θ_south)
        npixel_at_θ[i] = sum(pixel_at_dec)

    end

    return npixel_at_θ
end


function healpix_pwf▫(eaz0::EAZ0{T}; Nside::Int, normalizeθ = :row_ave) where {T}
    # Nside determines the size of the healpix pixels
    # eaz0 determines the grid that will get healpix conv

    θ  = EZ.θ(eaz0)
    nθ = length(θ)

    # Figure out bandwidths ...
    θhpx, φhpx, idxhpx, Δφhpx, nφhpx = HT.θ_φ_idx_4_rings(Nside)
    θhpx_obs = θhpx[minimum(θ) .≤ θhpx .≤ maximum(θ)]
    max2Δθ   = 2*maximum(diff(θhpx_obs)) 
    # this ↑ is the max bandwidth of the pixel along the pole.

    # compute the sparsity pattern
    # tile a block banded matrix to cover it.
    sparse_pattern  = @. abs(θ - θ') ≤ max2Δθ
    nrow_each_block = ceil(Int, maximum(map(sum, eachcol(sparse_pattern)))/2)
    block_sizesθ′ = VF.block_split(nθ, nrow_each_block) 
    block_sizesθ  = vcat(block_sizesθ′[1:end-2], sum(block_sizesθ′[end-1:end]))
    @assert sum(block_sizesθ) == nθ
    # block_sizesθ is a vector of block sizes.

    Σ▫ = eaz_cov_btridiag(eaz0, healpix_pwf_Γ(Nside); block_sizesθ)

    # now we normalize
    if normalizeθ == :none
        return Σ▫ 
    elseif normalizeθ == :row_ave
        ## Adjust so row mean of the pixel kernel is 1
        dnpix   = healpix_count_θ(eaz0; Nside)
        Dnpix⁻¹ = 0 * Σ▫[1] # faster mult if its the same block type
        for i in axes(Dnpix⁻¹, 1)
            Dnpix⁻¹[i,i] = 1 / dnpix[i]
        end
        return map(Σ▫i -> Dnpix⁻¹ * Σ▫i, Σ▫)
    elseif normalizeθ == :Ω
        ## Adjust so left mult behaves like an integral operator
        dΩ = EZ.Ωpix(eaz0)
        DΩ = 0 * Σ▫[1]
        for i in axes(DΩ, 1)
            DΩ[i,i] = dΩ[i]
        end
        return map(Σ▫i -> Σ▫i * DΩ, Σ▫)
    else 
        error("normalizeθ ∉ {:row_ave, :Ω, :none}")
    end

end

function healpix_pwf▫(eaz2::EAZ2{T}; Nside::Int, normalizeθ = :none) where {T}
    Σ0▫   = healpix_pwf▫(EZ.spin0(eaz2); Nside, normalizeθ)
    block_sizesθ0 = blocksizes(Σ0▫[1],1)
    block_sizesθ2 = vcat(block_sizesθ0, block_sizesθ0)
    nθ    = eaz2.nθ
    Σ2▫   = [BlockBandedMatrix{T}(Zeros(2nθ, 2nθ), block_sizesθ2, block_sizesθ2, (1,1)) for i in eachindex(Σ0▫)]
    for i in eachindex(Σ2▫)    
        for J = blockaxes(Σ0▫[i],2), K = blockcolsupport(Σ0▫[i],J)
            view(Σ2▫[i], K, J)       .= Σ0▫[i][K, J]
            view(Σ2▫[i], K+nθ, J+nθ) .= Σ0▫[i][K, J]
        end
    end
    return Σ2▫
end



