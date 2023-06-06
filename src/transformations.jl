

# Linear Algebra extensions general ring encoded as Xfield{<:𝕎} 
# =====================================

# TODO: check these ...

function LinearAlgebra.dot(f::Xfield{T},g::Xfield{T}) where T<:𝕎 
    real.(FT.sum_kbn(f[:] .* conj.(g[:])))
end

function LinearAlgebra.dot(f::Xfield{T},g::Xfield{T}) where T<:EAZ 
    real.(EZ.sum_kbn(f[:] .* conj.(g[:])))
end



# =========



# methods for GL pixel projection
#region ================================================

# Healpix -> EAZ grid projection


function hpix2equirect_patch(QU_hpix::Xmap{<:HT.ℍ2}; ring_idx_rng, φ, φ_full, lb, rb, Δl, Δr)
    tm = fieldtransform(QU_hpix)
    φ_hpix = HT.pix(tm)[2]
    Nside = tm.nside
    hpix2equirect_patch(QU_hpix[:]; Nside, ring_idx_rng, φ_hpix, φ, φ_full, lb, rb, Δl, Δr)
end 
function hpix2equirect_patch(QU_hpix::Matrix; Nside, ring_idx_rng, φ_hpix, φ, φ_full, lb, rb, Δl, Δr)
    Qx = hpix2equirect_patch(QU_hpix[:,1]; Nside, ring_idx_rng, φ_hpix, φ, φ_full, lb, rb, Δl, Δr)
    Ux = hpix2equirect_patch(QU_hpix[:,2]; Nside, ring_idx_rng, φ_hpix, φ, φ_full, lb, rb, Δl, Δr)
    complex.(Qx, .- Ux) 
end


function hpix2equirect_patch(I_hpix::Xmap{<:HT.ℍ0}; ring_idx_rng, φ, φ_full, lb, rb, Δl, Δr)
    tm = fieldtransform(I_hpix)
    φ_hpix = HT.pix(tm)[2]
    Nside = tm.nside
    hpix2equirect_patch(I_hpix[:]; Nside, ring_idx_rng, φ_hpix, φ, φ_full, lb, rb, Δl, Δr)
end
function hpix2equirect_patch(I_hpix::Vector; Nside, ring_idx_rng, φ_hpix, φ, φ_full, lb, rb, Δl, Δr)
    nφ, nφ_full = length(φ), length(φ_full)
    I_eqrt = hpix2GLstrip(φ_hpix, I_hpix, Nside; ring_idx_rng, nφ_full, lb, rb, Δl, Δr)
    if φ == φ_full
        return I_eqrt
    else 
        cshift = nφ_full + 1 - findmin(abs.(φ_full .- φ[1]))[2]
        I_eqrt′ = I_eqrt |> x-> circshift(x, (0,round(Int,cshift))) |> x->x[:, 1:nφ]
        return I_eqrt′
    end
end

function  hpix2GLstrip(hpix_φmap, hpix_map::Vector{T}, Nside; ring_idx_rng, nφ_full, lb, rb, Δl, Δr) where {T}
    ##  south cap ring index: 3*Nside+1 <= i <= 4*Nside-1
    @assert all(3*Nside+1 .<= ring_idx_rng .<= 4*Nside-1)
    Nrings = 4*Nside - 1
    map_φ   = HT.rings2rows(hpix_φmap, Nside)
    map_cmb = HT.rings2rows(hpix_map, Nside)
    map_new′ = zeros(T, nφ_full, length(ring_idx_rng))
    ## the above is transpose of what we want

    r1θ, r1φ, r1idx, r1Δφ, r1nφ = HT.θ_φ_idx_4_rings(Nside;T)

    for (i,ridx) in enumerate(ring_idx_rng)
        ring_idx_from_bottom = Nrings - ridx + 1 # so we can use the north cap hpx formula
        nφ = r1nφ[ridx] # 4*ring_idx_from_bottom
        map_cmb_ringi = map_cmb[ridx, 1:nφ]
        map_φ_ringi   = map_φ[ridx, 1:nφ]
        map_φ_mask    = cosφ°Mask.(rad2deg.(map_φ_ringi); lb, rb, Δl, Δr)
        map_new′[:,i] = regrid_hpix_ring(map_φ_mask .* map_cmb_ringi, r1φ[ridx], nφ_full)
    end

    Array(transpose(map_new′))
end

function regrid_hpix_ring(ring_map, first_φ, nφ_full)
    nφ     = length(ring_map) # should be 4*i, where i is the ring index counting from bottom
    krng   = 0:(nφ÷2)
    shft   = @. cis(- first_φ * krng) # shift left by the az location of the first ring pixel
    ## shft   = @. cis((π/nφ) * krng) # shift left by "half a pix width" == 2π/nφ/2 
    mapk   = rfft(ring_map)
    mapk .*= shft 
    nk_pad   = nφ_full÷2 + 1 # the biggest you want nφ_full is 4*(Nside-1)
    mapk_pad = zeros(eltype(mapk), nk_pad)
    nφ_ovrlp = min(length(mapk_pad), length(mapk))
    mapk_pad[1:nφ_ovrlp] .= mapk[1:nφ_ovrlp]
    ## mapx_pad = irfft(mapk_pad, nφ_full)
    mapx_pad = brfft(mapk_pad, nφ_full) ./ nφ # to make sure we reproduce the same real space function 
    return mapx_pad 
end

