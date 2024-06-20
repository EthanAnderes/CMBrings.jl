# eaz_cov
# ====================================

function eaz_cov(
            eaz0::EAZ0{T}, Γ; 
            θ=EZ.θ(eaz0), φ=EZ.φ(eaz0), ℓrange=1:EZ.lengthφ(eaz0)÷2+1
        ) where {T}
    CC.Γ2cov_blks(Γ; θ, φ, ℓrange)
end

function eaz_cov(
            eaz2::EAZ2{T}, Γ, C; 
            θ=EZ.θ(eaz2), φ=EZ.φ(eaz2), 
            ℓrange=1:EZ.lengthφ(eaz2)÷2+1
        ) where {T}
    CC.ΓC2cov_blks(Γ, C; θ, φ, ℓrange)
end

function eaz_cov(
        eaz0::EAZ0{T}, ℓ::AbstractVector, ffℓ::Vector;
        θ=EZ.θ(eaz0), φ=EZ.φ(eaz0),  
        ℓrange=1:EZ.lengthφ(eaz0)÷2+1, 
        ngrid=100_000, 
    ) where {T}
    Γ  = CC.Γθ₁θ₂φ₁φ⃗_Iso(ℓ, ffℓ; ngrid)
    eaz_cov(eaz0, Γ; θ, φ, ℓrange)
end

function eaz_cov(
        eaz2::EAZ2{T}, ℓ::AbstractVector, eeℓ::Vector, bbℓ::Vector; 
        θ=EZ.θ(eaz2), φ=EZ.φ(eaz2),  
        ℓrange=1:EZ.lengthφ(eaz2)÷2+1, 
        ngrid=100_000,
    ) where {T}
    Γ, C   = CC.ΓCθ₁θ₂φ₁φ⃗_CMBpol(ℓ, eeℓ, bbℓ; ngrid)
    eaz_cov(eaz2, Γ, C; θ, φ, ℓrange)
end


# eaz_cov_vecchia and eaz_½cov_vecchia
# ====================================

function eaz_cov_vecchia(
        eaz0::EAZ0{T}, ℓ::AbstractVector, ffℓ::Vector;
        block_sizesθ,
        chol_atol=0, eig_vmin=0, eig_val=0, 
        ngrid=100_000
    ) where {T}
    Γ      = CC.Γθ₁θ₂φ₁φ⃗_Iso(ℓ, ffℓ; ngrid)
    Σ_pre▫ = eaz_cov_btridiag(eaz0, Γ; block_sizesθ)
    Σ▫ = map(Σ_pre▫) do Σ
        VF.vecchia_pdeigen(Σ, block_sizesθ; chol_atol, eig_vmin, eig_val)
    end
    return Σ▫
end


function eaz_cov_vecchia(
        eaz2::EAZ2{T}, ℓ::AbstractVector, eeℓ::Vector, bbℓ::Vector; 
        block_sizesθ,
        chol_atol=0, eig_vmin=0, eig_val=0, 
        ngrid=100_000
    ) where {T}
    Γ, C       = CC.ΓCθ₁θ₂φ₁φ⃗_CMBpol(ℓ, eeℓ, bbℓ; ngrid)
    Σ_pre▫, P  = eaz_cov_btridiag(eaz2, Γ, C; block_sizesθ)
    block_sizesθ′ = VF.blocksizes(Σ_pre▫[1],1) # for spin2 block sizes get doubled ...
    Σ▫ = map(Σ_pre▫) do Σ
        P' * VF.vecchia_pdeigen(Σ, block_sizesθ′; chol_atol, eig_vmin, eig_val) * P
    end
    return Σ▫
end


# eaz_½cov_vecchia
# ====================================


function eaz_½cov_vecchia(
        eaz0::EAZ0{T}, ℓ::AbstractVector, ffℓ::Vector;
        block_sizesθ::AbstractVector{<:Integer},
        chol_atol=0, eig_vmin=0, eig_val=0, 
        ngrid=100_000
    ) where {T}
    Σ_pre▫ = eaz_cov_vecchia(eaz0, ℓ, ffℓ; block_sizesθ, chol_atol, eig_vmin, eig_val, ngrid) 
    Σ▫ = map(Σ_pre▫) do Σ
        invR, M,  = Σ # Σ is a tuple of vecchia operators
        M½        = VF.Midiagonal(map(sqrt, M.data)) 
        invR * M½
    end
    return Σ▫
end

function eaz_½cov_vecchia(
        eaz2::EAZ2{T}, ℓ::AbstractVector, eeℓ::Vector, bbℓ::Vector; 
        block_sizesθ,
        chol_atol=0, eig_vmin=0, eig_val=0, 
        ngrid=100_000
    ) where {T}
    Σ_pre▫ = eaz_cov_vecchia(eaz2, ℓ, eeℓ, bbℓ; block_sizesθ, chol_atol, eig_vmin, eig_val, ngrid) 
    Σ▫ = map(Σ_pre▫) do Σ
        Pᵀ, invR, M, = Σ # Σ is a tuple of vecchia operators
        M½  = VF.Midiagonal(map(sqrt, M.data)) 
        Pᵀ * invR * M½ * Pᵀ' 
    end
    return Σ▫
end


# eaz_cov_btridiag and 
# ==========================================


function eaz_cov_btridiag(
        eaz0::EAZ0{T}, Γ;
        block_sizesθ::AbstractVector{<:Integer},
        ℓrange=1:size_out(eaz0)[2],
    ) where {T}
    # block_sizesθ looks like this [20, 10, 5, 5, ...]
    # which means the first diag block is 20x20, 
    # next diag block is 10x10, ... 

    θ, φ, nθ, nφ = EZ.θ(eaz0), EZ.φ(eaz0), EZ.lengthθ(eaz0), EZ.lengthφ(eaz0)
    
    cT    = Complex{T}
    ptmW  = [plan_fft(Vector{cT}(undef, nφ), num_threads=1) for i=1:Threads.nthreads()]

    @assert sum(block_sizesθ) == nθ
    Σ▫ = [BlockBandedMatrix{T}(Zeros(nθ, nθ), block_sizesθ, block_sizesθ, (1,1)) for ℓ′ in ℓrange]
    # Σ▫ = [zeros(Float64, nθ, nθ) for ℓ′ in ℓrange]

    # just to make it easier lets create a bool to record the support
    # of the blockedBanded array.
    Supp = BlockBandedMatrix{Bool}(Ones(nθ, nθ), block_sizesθ, block_sizesθ, (1,1))

    setΣ! = function (M▫,j,k,pln)
        if Supp[j,k]
            Mγⱼₖℓ⃗  = CC.γθ₁θ₂ℓ⃗(θ[j], θ[k], φ, Γ, pln)
            for (i,ℓ′) in enumerate(ℓrange)
                M▫[i][j,k] = real(Mγⱼₖℓ⃗[ℓ′])
            end
            return nothing
        else
            return nothing
        end
    end
    
    pbar = Progress(nθ^2, "Constructing block diagonals")
    Threads.@threads for (k,j) in collect(Iterators.product(1:nθ, 1:nθ))
        setΣ!(Σ▫, j, k, ptmW[Threads.threadid()])
        next!(pbar)
    end 
    finish!(pbar)

    return Σ▫
end


function eaz_cov_btridiag(
        eaz2::EAZ2{T}, Γ, C;
        block_sizesθ::AbstractVector{<:Integer},
        ℓrange=1:size_out(eaz2)[2],
    ) where {T}
    # block_sizesθ looks like this [20, 10, 5, 5, ...]
    # which means the first diag block is 20x20, 
    # next diag block is 10x10, ... 

    θ, φ, nθ, nφ = EZ.θ(eaz2), EZ.φ(eaz2), EZ.lengthθ(eaz2), EZ.lengthφ(eaz2)

    cT    = Complex{T}
    ptmW  = [plan_fft(Vector{cT}(undef, nφ), num_threads=1) for i=1:Threads.nthreads()]

    ### First part
    @assert sum(block_sizesθ) == nθ
    Mγ▫   = [BlockBandedMatrix{cT}(Zeros(nθ, nθ), block_sizesθ, block_sizesθ, (1,1)) for ℓ′ in ℓrange]
    Mξ▫   = [BlockBandedMatrix{cT}(Zeros(nθ, nθ), block_sizesθ, block_sizesθ, (1,1)) for ℓ′ in ℓrange]
    cMγJ▫ = [BlockBandedMatrix{cT}(Zeros(nθ, nθ), block_sizesθ, block_sizesθ, (1,1)) for ℓ′ in ℓrange]
    cMξJ▫ = [BlockBandedMatrix{cT}(Zeros(nθ, nθ), block_sizesθ, block_sizesθ, (1,1)) for ℓ′ in ℓrange]
    # create a bool to record the support of the blockedBanded array.
    Supp = BlockBandedMatrix{Bool}(Ones(nθ, nθ), block_sizesθ, block_sizesθ, (1,1))

    setΣ! = function (Mγ▫,Mξ▫,cMγJ▫,cMξJ▫,j,k,pln)
        if Supp[j,k]
            Mγⱼₖℓ⃗, Mξⱼₖℓ⃗ = CC.γθ₁θ₂ℓ⃗_ξθ₁θ₂ℓ⃗(θ[j], θ[k], φ, Γ, C, pln)
            for (i,ℓ′) in enumerate(ℓrange)
                Jℓ′ = CC.Jperm(ℓ′, nφ)
                Mγ▫[i][j,k]   = Mγⱼₖℓ⃗[ℓ′]
                Mξ▫[i][j,k]   = Mξⱼₖℓ⃗[ℓ′]
                cMξJ▫[i][j,k] = conj(Mξⱼₖℓ⃗[Jℓ′])
                cMγJ▫[i][j,k] = conj(Mγⱼₖℓ⃗[Jℓ′])
            end
            return nothing
        else
            return nothing
        end
    end

    pbar = Progress(nθ^2, "Constructing block diagonals")    
    Threads.@threads for (k,j) in collect(Iterators.product(1:nθ, 1:nθ))
        setΣ!(Mγ▫,Mξ▫,cMγJ▫,cMξJ▫, j, k, ptmW[Threads.threadid()])
        next!(pbar)
    end 
    finish!(pbar)

    ### Second part
    N = length(block_sizesθ)
    # Put Mγ▫,Mξ▫,cMγJ▫,cMξJ▫  toghether for the full Spin2 operator
    Σ▫ = map(Mγ▫,Mξ▫,cMγJ▫,cMξJ▫) do Mγ,Mξ,cMγJ,cMξJ
        M = BlockBandedMatrix{cT}(Zeros(2nθ, 2nθ), 2 .* block_sizesθ, 2 .* block_sizesθ, (1,1))
        for ic=1:N 
            M[Block(ic,ic)] = [ Mγ[Block(ic,ic)]   Mξ[Block(ic,ic)]
                              cMξJ[Block(ic,ic)] cMγJ[Block(ic,ic)] ]
            if ic < N
                M[Block(ic+1,ic)] = [ Mγ[Block(ic+1,ic)]   Mξ[Block(ic+1,ic)]
                                    cMξJ[Block(ic+1,ic)] cMγJ[Block(ic+1,ic)] ]
                M[Block(ic,ic+1)] = M[Block(ic+1,ic)]'
            end 
        end
        return M
    end

    ### Third part, put the permuation together so the blocks are interlaced
    a2    = blocks(BlockedArray(collect(1:2nθ), vcat(block_sizesθ, block_sizesθ))) # divide into blocks
    perm2 = a2 |> x->reshape(x,N,2) |> x->permutedims(x) |> vec |> x->vcat(x...) # interlace the blocks
    P     = VF.Piv(perm2)

    return Σ▫, P
end

