

# Linear Algebra extensions general ring encoded as Xfield{<:𝕎} 
# =====================================

# TODO: check these ...

function LinearAlgebra.dot(f::Xfield{T},g::Xfield{T}) where T<:𝕎 
    real.(FT.sum_kbn(f[:] .* conj.(g[:])))
end

function LinearAlgebra.dot(f::Xfield{T},g::Xfield{T}) where T<:EAZ 
    real.(EZ.sum_kbn(f[:] .* conj.(g[:])))
end


# methods for slicing ... Xfield(EAZ)[5:end-10,:]
# ===============================================
# TODO: perhaps we should make these generic ...


# methods for Healpix -> EAZ grid projection
# ================================================
# TODO: add methods which accept polarization ... and perhaps z-bounds
# TODO: update examples ...
# TODO: test eaz2hpx (take a look at Scratch/2023/m07_d01_bk_viz for comparision possibility)

function rings_θs_φs(Nside::Int; θspan=(0,π))
    θ, φ, idx, Δφ, nφ = HT.θ_φ_idx_4_rings(Nside)

    ring_slices_full = map(idx, nφ) do firstPixIdx, numOfPixels
        (firstPixIdx):(firstPixIdx+numOfPixels-1)
    end

    rings_θspan = θspan[1] .<= θ .<= θspan[2]
    return ring_slices_full[rings_θspan], θ[rings_θspan], φ[rings_θspan]
end

function hpx2eaz(fhpx::Xfield{HT.ℍ0{T}}; θspan=(0,π)) where {T}
    Nside = fieldtransform(fhpx).nside 
    hpx   = fhpx[:]
    @assert length(hpx) == 12Nside^2
    rings, θs, φs = rings_θs_φs(Nside; θspan)
    
    φspan   = (0,2π)
    nφ_eaz  = maximum(length.(rings))
    eaz0    = EAZ0(T, θs, φspan, nφ_eaz)
    mrng    = 0:(nφ_eaz÷2) # fourier frequencies for real rings

    fft_plans = map(rings) do map_slice
        npix = length(map_slice)
        plan_rfft(Array{T}(undef, npix), num_threads=1)
    end
    
    fft_hpx_rings = Vector{Vector{Complex{T}}}(undef,length(rings))
    Threads.@threads for i=1:length(rings)
        Nφ_hpx   = length(rings[i])
        Nm_hpx  = Nφ_hpx÷2 + 1 
        fft_mult = (√nφ_eaz ./ Nφ_hpx) .* cis.(- φs[i] .* mrng)  
        # ↑ scale then shift left by the az location of the first ring pixel
        fft_hpx_rings[i] = fft_mult[1:Nm_hpx] .* (fft_plans[i] * hpx[rings[i]])
    end

    # now stack these as rows of an eaz ...
    feaz = Xfourier(eaz0,0)
    @assert size(feaz.fd, 1) == length(fft_hpx_rings)
    for i in 1:length(fft_hpx_rings)
        nr   = length(fft_hpx_rings[i])
        feaz.fd[i,1:nr] = fft_hpx_rings[i]
    end

    return feaz
end

function eaz2hpx(feaz::Xfield{EAZ0{T}}; Nside, lmax=2Nside) where {T}
    eaz0   = fieldtransform(feaz)
    feaz_m = feaz[!]
    nφ_eaz = eaz0.nφ
    mrng    = 0:(nφ_eaz÷2) # fourier frequencies for real rows of eaz

    rings, θs, φs = rings_θs_φs(Nside; θspan=extrema(eaz0.θ∂))
    φspan   = (0,2π)
    
    @assert isapprox(θs, eaz0.θ)
    @assert φspan  == eaz0.φspan

    ifft_plans = map(rings) do map_slice
        npix = length(map_slice)
        plan_brfft(Array{Complex{T}}(undef, npix÷2+1), npix, num_threads=1)
    end
    
    tmℍ0 = HT.ℍ0{T}(Nside; lmax)
    hpx_map = zeros(T, HT.n_pix(tmℍ0))
    Threads.@threads for i=1:length(rings)
        Nφ_hpx  = length(rings[i])
        Nm_hpx  = Nφ_hpx÷2 + 1 
        Nm_eaz  = nφ_eaz÷2 + 1 
        ifft_mult = inv.((√nφ_eaz ./ Nφ_hpx) .* cis.(- φs[i] .* mrng)) ./ Nφ_hpx
        
        eaz_row_fft = ifft_mult.*feaz_m[i,:]     # length is Nm_eaz 
        hpx_row_fft = zeros(Complex{T}, Nm_hpx)  # length is Nm_hpx
        if Nm_eaz ≥ Nm_hpx
            hpx_row_fft .= eaz_row_fft[1:Nm_hpx] # case 1, just truncate the eaz fft
        else
            hpx_row_fft[1:Nm_eaz] .= eaz_row_fft # case 2, padd the eaz fft
        end
        hpx_map[rings[i]] .= ifft_plans[i] * hpx_row_fft
    end

    return Xmap(tmℍ0, hpx_map)
end

# TODO: make this not such a hack
# TODO: also make a hpx2eaz version
function eaz2hpx(feaz::Xfield{EAZ2{T}}; Nside, lmax=2Nside) where {T}
    eaz2 = EZ.fieldtransform(feaz)
    eaz0 = EZ.spin0(eaz2)
    qmap, umap = reim(feaz[:])
    qeaz = Xmap(eaz0, qmap) 
    ueaz = Xmap(eaz0, umap)
    qhpx = eaz2hpx(qeaz; Nside, lmax=2Nside)
    uhpx = eaz2hpx(ueaz; Nside, lmax=2Nside)
    tmℍ2 = HT.ℍ2{T}(Nside; lmax)
    return Xmap(tmℍ2, hcat(qhpx[:], uhpx[:]))
end

function eaz2hpx(feaz::Xfield{EAZ02{T}}; Nside, lmax=2Nside) where {T}
    eaz02 = EZ.fieldtransform(feaz)
    eaz0  = EZ.spin0(eaz2)
    fmap  = feaz[:]
    tmap, qmap, umap = fmap[:,:,1], fmap[:,:,2], fmap[:,:,3]
    teaz = Xmap(eaz0, tmap) 
    qeaz = Xmap(eaz0, qmap) 
    ueaz = Xmap(eaz0, umap)
    thpx = eaz2hpx(teaz; Nside, lmax=2Nside)
    qhpx = eaz2hpx(qeaz; Nside, lmax=2Nside)
    uhpx = eaz2hpx(ueaz; Nside, lmax=2Nside)
    tmℍ02 = HT.ℍ02{T}(Nside; lmax)
    return Xmap(tmℍ2, hcat(thpx[:], qhpx[:], uhpx[:]))
end




#=
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

=#