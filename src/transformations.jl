

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

# ----------------------
# hpx2eaz
# ----------------------

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

function hpx2eaz(fhpx::Xfield{HT.ℍ2{T}}; θspan=(0,π)) where {T}
    H2  = fieldtransform(fhpx)
    H0  = HT.spin0(H2)
    qu  = Array.(eachcol(fhpx[:]))
    qmap, umap = qu[1], qu[2]
    qH0  = Xmap(H0, qmap) 
    uH0  = Xmap(H0, umap)
    qeaz = hpx2eaz(qH0; θspan)
    ueaz = hpx2eaz(uH0; θspan)
    eaz0 = fieldtransform(qeaz)
    return Xmap(EZ.spin2(eaz0), complex.(qeaz[:], ueaz[:]))
end

function hpx2eaz(fhpx::Xfield{HT.ℍ02{T}}; θspan=(0,π)) where {T}
    H02  = fieldtransform(fhpx)
    H0   = HT.spin0(H02)
    tqu  = Array.(eachcol(fhpx[:]))
    tmap, qmap, umap = tqu[1], tqu[2], tqu[3]
    tH0  = Xmap(H0, tmap) 
    qH0  = Xmap(H0, qmap) 
    uH0  = Xmap(H0, umap)
    teaz = hpx2eaz(tH0; θspan)
    qeaz = hpx2eaz(qH0; θspan)
    ueaz = hpx2eaz(uH0; θspan)
    eaz0 = fieldtransform(teaz)
    return Xmap(EZ.spin02(eaz0), stack((teaz[:], qeaz[:], ueaz[:])))
end


# ----------------------
# eaz2hpx
# ----------------------


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


