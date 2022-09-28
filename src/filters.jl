# TOD like filters on the rows of EAZ fourier

# Mp:  construct point source pixel mask from pnt src list 
# ==========================================

function pix_point_src_mask(tm0::EAZ, point_src_file_; smooth_border_Δ′ = 4.0) 

    hole_map = ones(size_in(tm0)) 

    point_src_list = readdlm(point_src_file_, '\t', skipstart=22)
    point_src_φ  = RA2φ.(point_src_list[:,2])
    point_src_θ  = Dec2θ.(point_src_list[:,3])
    point_src_Δβ = deg2rad.(point_src_list[:,4] / 60)

    θ, φ     = EZ.pix(tm0)
    for (θ₀, φ₀, Δβ₀) in zip(point_src_θ, point_src_φ, point_src_Δβ)
        radius_hole = Δβ₀
        radius_ramp = arcmin2rad(smooth_border_Δ′) 
        radius_tot  = (radius_hole + radius_ramp) * 1.1 # expand this a bit

    
        θ_idx_cut = findall(@. abs(θ - θ₀) <= radius_tot)
        φ_idx_cut = findall(@. min(CC.counterclock_Δφ(φ₀, φ), CC.counterclock_Δφ(φ, φ₀)) <= (radius_tot/sin(θ₀+radius_tot)))
   
        if !isempty(θ_idx_cut) & !isempty(φ_idx_cut)
            θ_val_cut = θ[θ_idx_cut]
            φ_val_cut = φ[φ_idx_cut]

            hole_map_mini = hole_punch.(θ_val_cut, φ_val_cut'; θ₀, φ₀, radius_hole, radius_ramp)
            # @assert all(hole_map_mini[1,:]   .== 1)
            # @assert all(hole_map_mini[end,:] .== 1)
            # @assert all(hole_map_mini[:,end] .== 1)
            # @assert all(hole_map_mini[:,1]   .== 1)
            hole_map[θ_idx_cut, φ_idx_cut] .*= hole_map_mini
        end 
    end

    DiagOp(Xmap(tm0, hole_map))
end


θ2Dec(θ)   = 90 - rad2deg(θ)
Dec2θ(dec) = deg2rad(90 - dec)

φ2RA(φ)   = rad2deg(CC.in_negπ_π(φ))
RA2φ(ra)  = CC.in_0_2π(deg2rad(ra))

function hole_punch(θ, φ; θ₀, φ₀, radius_hole, radius_ramp)
    β      = CC.geoβ(θ, θ₀, φ, φ₀)
    r1, r2 = radius_hole, radius_hole + radius_ramp
    Δr     = r2 - r1

    if β <= r1 
        return zero(β)
    elseif r1 < β <= r2
        return (1+cospi((β-r2)/Δr))/2
    else
        return one(β)
    end
end



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


function healpix_pwf▫(eaz0::EAZ0{T}; Nside::Int, normalizeθ = :none) where {T}
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
    bnθs′           = VF.block_split(nθ, nrow_each_block) 
    bnθs            = vcat(bnθs′[1:end-2], sum(bnθs′[end-1:end]))
    @assert sum(bnθs) == nθ
    # bnθs is a vector of block sizes.

    Σ▫ = block_tridiag_Σ▫(eaz0, healpix_pwf_Γ(Nside), bnθs)

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
    bnθs0 = blocksizes(Σ0▫[1],1)
    bnθs2 = vcat(bnθs0, bnθs0)
    nθ    = eaz2.nθ
    Σ2▫   = [BlockBandedMatrix{T}(Zeros(2nθ, 2nθ), bnθs2, bnθs2, (1,1)) for i in eachindex(Σ0▫)]
    for i in eachindex(Σ2▫)    
        for J = blockaxes(Σ0▫[i],2), K = blockcolsupport(Σ0▫[i],J)
            view(Σ2▫[i], K, J)       .= Σ0▫[i][K, J]
            view(Σ2▫[i], K+nθ, J+nθ) .= Σ0▫[i][K, J]
        end
    end
    return Σ2▫
end

# ===============================================
