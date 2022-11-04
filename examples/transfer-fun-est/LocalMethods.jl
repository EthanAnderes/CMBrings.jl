
module LocalMethods

using XFields

using EAZTransforms
# using EAZTransforms: pix, freq, nyq, Ωpix
# import EAZTransforms as EZ 

# using FFTransforms: 𝕀, ⊗, 𝕎
# import FFTransforms as FT

# import HealpixTransforms as HT
# import CirculantCov as CC


## ----------------------------------------

using LinearAlgebra
import LinearAlgebra: *

export RingDeprojector


function deproject_Xm(f::AbstractVector, Xm::AbstractMatrix, factXm::Factorization) 
    f - Xm * (factXm \ f)
end 

function deproject_Xm_eachrow_qr(f::AbstractMatrix, m::AbstractMatrix, X::AbstractMatrix)
    # f is a matrix representing the CMB (θ, φ) pixel values. (θ, φ) <-> (row, col)
    # m is a matrix representing the (θ, φ) pixel mask (0 => not-observed)
    # X is a matrix with columns representing the unmasked modes to be removed from the rows of f.
    Xmr      = similar(X)
    Xmr_copy = similar(X)
    g        = similar(f)
    for (fr, mr, gr) in zip(eachrow(f), eachrow(m), eachrow(g))
        Xmr      .= mr .* X # mask the columns of X
        Xmr_copy .= Xmr
        factXmr = qr!(Xmr_copy, ColumnNorm()) # modifies Xm_copy
        copyto!(gr, deproject_Xm(fr, Xmr, factXmr))
    end
    return g
end

function deproject_Xm_eachrow_svd(
        f::AbstractMatrix, 
        m::AbstractMatrix, 
        X::AbstractMatrix;
        alg=LinearAlgebra.DivideAndConquer()
    )
    # f is a matrix representing the CMB (θ, φ) pixel values. (θ, φ) <-> (row, col)
    # m is a matrix representing the (θ, φ) pixel mask (0 => not-observed)
    # X is a matrix with columns representing the unmasked modes to be removed from the rows of f.
    Xmr      = similar(X)
    g        = similar(f)
    for (fr, mr, gr) in zip(eachrow(f), eachrow(m), eachrow(g))
        Xmr    .= mr .* X # mask the columns of X
        factXmr = svd(Xmr; full=false, alg)
        copyto!(gr, deproject_Xm(fr, Xmr, factXmr))
    end
    return g
end

struct RingDeprojector{T<:AbstractMatrix, U<:AbstractMatrix} <: AbstractLinearOp
    X::T
    m::U
    alg::Symbol 
    function RingDeprojector(X::T, m::U; alg=:qr) where {T,U}
        # alg ∈ {:qr, :svg_divide_conquer, :svg_qr_iteration}
        new{T,U}(X,m,alg)
    end
end

function *(D::RingDeprojector, f::T) where {T<:Xfield{<:EAZ0}}
    fmat = f[:]
    if D.alg == :qr
        Dfmat = deproject_Xm_eachrow_qr(fmat, D.m, D.X)  
    elseif D.alg == :svg_divide_conquer
        Dfmat = deproject_Xm_eachrow_svd(fmat, D.m, D.X; alg=LinearAlgebra.DivideAndConquer()) 
    elseif  D.alg == :svg_qr_iteration
        Dfmat = deproject_Xm_eachrow_svd(fmat, D.m, D.X; alg=LinearAlgebra.QRIteration()) 
    else 
        error("RingDeprojector.alg not a valid option")
    end
    Xmap(fieldtransform(f), Dfmat)
end

function *(D::RingDeprojector, f::T) where {T<:Xfield{<:EAZ2}}
    fmat = f[:]
    q, u = real(fmat), imag(fmat)
    if D.alg == :qr
        Dqmat = deproject_Xm_eachrow_qr(q, D.m, D.X)  
        Dumat = deproject_Xm_eachrow_qr(u, D.m, D.X)  
    elseif D.alg == :svg_divide_conquer
        Dqmat = deproject_Xm_eachrow_svd(q, D.m, D.X; alg=LinearAlgebra.DivideAndConquer())   
        Dumat = deproject_Xm_eachrow_svd(u, D.m, D.X; alg=LinearAlgebra.DivideAndConquer())   
    elseif  D.alg == :svg_qr_iteration
        Dqmat = deproject_Xm_eachrow_svd(q, D.m, D.X; alg=LinearAlgebra.QRIteration())   
        Dumat = deproject_Xm_eachrow_svd(u, D.m, D.X; alg=LinearAlgebra.QRIteration())   
    else 
        error("RingDeprojector.alg not a valid option")
    end
    Xmap(fieldtransform(f), complex.(Dqmat, Dumat))
end








# Slated for removal since it has been included in 
# ================================================

"""
• first test removing the dependence on lb, rb, Δl, Δr ...
"""
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
    shft   = @. cis(first_φ * krng) # shift left by the az location of the first ring pixel
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


end

