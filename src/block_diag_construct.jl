# Constructors for Block diagonals in AzEqui coordinates
# ====================================

function az_cov_blks(Γ; θ, φ, ℓrange=1:length(φ)÷2+1)
    CC.Γ2cov_blks(Γ; θ, φ, ℓrange)
end

function az_cov_blks(Γ, C; θ, φ, ℓrange=1:length(φ)÷2+1)
    CC.ΓC2cov_blks(Γ, C; θ, φ, ℓrange)
end

function az_cov_blks(
        ℓ::AbstractVector, ffℓ::Vector; 
        θ, φ, 
        ℓrange=1:length(φ)÷2+1,
        ngrid=100_000, 
    )
    Γ  = CC.Γθ₁θ₂φ₁φ⃗_Iso(ℓ, ffℓ; ngrid)
    CC.Γ2cov_blks(Γ; θ, φ, ℓrange)
end

function az_cov_blks(
        ℓ::AbstractVector, eeℓ::Vector, bbℓ::Vector; 
        θ, φ, 
        ℓrange=1:length(φ)÷2+1, 
        ngrid=100_000,
    )
    Γ, C   = CC.ΓCθ₁θ₂φ₁φ⃗_CMBpol(ℓ, eeℓ, bbℓ; ngrid)
    CC.ΓC2cov_blks(Γ, C; θ, φ, ℓrange)
end


# New tridiagonal spin0 constructor 
# ====================================
# TODO: add spin2 and tests

function block_tridiag_Σ▫(
        eaz0::EAZ0{T}, 
        Γ,
        bnθs::AbstractVector{<:Integer};
        ℓrange=1:size_out(eaz0)[2],
    ) where {T}
    # bnθs looks like this [20, 10, 5, 5, ...]
    # which means the first diag block is 20x20, 
    # next diag block is 10x10, ... 

    θ, φ, nθ, nφ = EZ.θ(eaz0), EZ.φ(eaz0), eaz0.nθ, eaz0.nφ
    ptmW   = FFTW.plan_fft(Vector{Complex{T}}(undef, nφ))

    @assert sum(bnθs) == nθ
    Σ▫ = [BlockBandedMatrix{T}(Zeros(nθ, nθ), bnθs, bnθs, (1,1)) for ℓ′ in ℓrange]
    # Σ▫ = [zeros(Float64, nθ, nθ) for ℓ′ in ℓrange]

    # just to make it easier lets create a bool to record the support
    # of the blockedBanded array.
    Supp = BlockBandedMatrix{Bool}(Ones(nθ, nθ), bnθs, bnθs, (1,1))

    setΣ! = function (M▫,j,k)
        if Supp[j,k]
            Mγⱼₖℓ⃗  = CC.γθ₁θ₂ℓ⃗(θ[j], θ[k], φ, Γ, ptmW)
            for (i,ℓ′) in enumerate(ℓrange)
                M▫[i][j,k] = real(Mγⱼₖℓ⃗[ℓ′])
            end
            return nothing
        else
            return nothing
        end
    end
    
    prgss  = Progress(nθ, dt=1, desc="Constructing block diagonals")
    for k in 1:nθ # loop over column block
        for j in 1:nθ
            setΣ!(Σ▫, j, k)
        end
        next!(prgss)
    end 

    return Σ▫
end


# az_cov_vecchia_blks is similar to az_cov_blks but the AzEqui blocks
# are approximated with Vecchia 
# ===============================================

# Spin0
function spin0_az_cov_vecchia_blks(
    ℓ::AbstractVector, ffℓ::Vector,
    blk_sizes::AbstractVector{<:Integer}, 
    perm::AbstractVector{<:Integer}=1:sum(blk_sizes);
    θ, φ, ℓrange=1:length(φ)÷2+1
    )
    Γ = CC.Γθ₁θ₂φ₁φ⃗_Iso(ℓ, ffℓ; ngrid=100_100)
    Σ_pre▫, P = spin0_az_bidiagΣ▫_P(Γ, blk_sizes, perm; θ, φ, ℓrange)
    Σ▫ = map(Σ_pre▫) do Σ
        P' * VF.vecchia(Σ, blk_sizes) * P
    end
    return Σ▫
end
function spin0_az_cov_vecchia_blks(
    Γ,
    blk_sizes::AbstractVector{<:Integer}, 
    perm::AbstractVector{<:Integer}=1:sum(blk_sizes);
    θ, φ, ℓrange=1:length(φ)÷2+1
    )
    Σ_pre▫, P = spin0_az_bidiagΣ▫_P(Γ, blk_sizes, perm; θ, φ, ℓrange)
    Σ▫ = map(Σ_pre▫) do Σ
        P' * VF.vecchia(Σ, blk_sizes) * P
    end
    return Σ▫
end



# Spin2
function spin2_az_cov_vecchia_blks(
    ℓ::AbstractVector, eeℓ::Vector, bbℓ::Vector,
    blk_sizes::AbstractVector{<:Integer}, 
    perm::AbstractVector{<:Integer}=1:sum(blk_sizes);
    θ, φ, ℓrange=1:length(φ)÷2+1
    )
    Γ, C   = CC.ΓCθ₁θ₂φ₁φ⃗_CMBpol(ℓ, eeℓ, bbℓ; ngrid=100_000)
    Σ_pre▫, P = spin2_az_bidiagΣ▫_P(Γ, C, blk_sizes, perm; θ, φ, ℓrange)
    blk_sizes′ = VF.blocksizes(Σ_pre▫[1],1) # for spin2 block sizes get doubled ...
    Σ▫ = map(Σ_pre▫) do Σ
        P' * VF.vecchia(Σ, blk_sizes′) * P
    end
    return Σ▫
end
function spin2_az_cov_vecchia_blks(
    Γ, C,
    blk_sizes::AbstractVector{<:Integer}, 
    perm::AbstractVector{<:Integer}=1:sum(blk_sizes);
    θ, φ, ℓrange=1:length(φ)÷2+1
    )
    Σ_pre▫, P = spin2_az_bidiagΣ▫_P(Γ, C, blk_sizes, perm; θ, φ, ℓrange)
    blk_sizes′ = VF.blocksizes(Σ_pre▫[1],1) # for spin2 block sizes get doubled ...
    Σ▫ = map(Σ_pre▫) do Σ
        P' * VF.vecchia(Σ, blk_sizes′) * P
    end
    return Σ▫
end




# az_cov½_vecchia_blks 
# ===============================================

# Spin0 preps the sqrt matrix
function spin0_az_cov½_vecchia_blks(
    ℓ::AbstractVector, ffℓ::Vector,
    blk_sizes::AbstractVector{<:Integer}, 
    perm::AbstractVector{<:Integer}=1:sum(blk_sizes);
    θ, φ, ℓrange=1:length(φ)÷2+1
    )
    Γ = CC.Γθ₁θ₂φ₁φ⃗_Iso(ℓ, ffℓ; ngrid=100_100) 
    Σ_pre▫, P = spin0_az_bidiagΣ▫_P(Γ, blk_sizes, perm; θ, φ, ℓrange)
    Σ▫ = map(Σ_pre▫) do Σ
        R, preM, = VF.R_M_P(Σ, blk_sizes)
        M½ = VF.Midiagonal(map(sqrt, preM.data))
        P' * inv(R) * M½ * P 
    end
    return Σ▫
end
function spin0_az_cov½_vecchia_blks(
    Γ,
    blk_sizes::AbstractVector{<:Integer}, 
    perm::AbstractVector{<:Integer}=1:sum(blk_sizes);
    θ, φ, ℓrange=1:length(φ)÷2+1
    )
    Σ_pre▫, P = spin0_az_bidiagΣ▫_P(Γ, blk_sizes, perm; θ, φ, ℓrange)
    Σ▫ = map(Σ_pre▫) do Σ
        R, preM, = VF.R_M_P(Σ, blk_sizes)
        M½ = VF.Midiagonal(map(sqrt, preM.data))
        P' * inv(R) * M½ * P 
    end
    return Σ▫
end


# Spin2
function spin2_az_cov½_vecchia_blks(
    ℓ::AbstractVector, eeℓ::Vector, bbℓ::Vector,
    blk_sizes::AbstractVector{<:Integer}, 
    perm::AbstractVector{<:Integer}=1:sum(blk_sizes);
    θ, φ, ℓrange=1:length(φ)÷2+1
    )
    Γ, C   = CC.ΓCθ₁θ₂φ₁φ⃗_CMBpol(ℓ, eeℓ, bbℓ; ngrid=100_000)
    Σ_pre▫, P = spin2_az_bidiagΣ▫_P(Γ, C, blk_sizes, perm; θ, φ, ℓrange)
    blk_sizes′ = VF.blocksizes(Σ_pre▫[1],1) # for spin2 block sizes get doubled ...
    Σ▫ = map(Σ_pre▫) do Σ
        R, preM, = VF.R_M_P(Σ, blk_sizes′)
        M½ = VF.Midiagonal(map(sqrt, preM.data))
        P' * inv(R) * M½ * P 
    end
    return Σ▫
end
function spin2_az_cov½_vecchia_blks(
    Γ, C,
    blk_sizes::AbstractVector{<:Integer}, 
    perm::AbstractVector{<:Integer}=1:sum(blk_sizes);
    θ, φ, ℓrange=1:length(φ)÷2+1
    ) 
    Σ_pre▫, P = spin2_az_bidiagΣ▫_P(Γ, C, blk_sizes, perm; θ, φ, ℓrange)
    blk_sizes′ = VF.blocksizes(Σ_pre▫[1],1) # for spin2 block sizes get doubled ...
    Σ▫ = map(Σ_pre▫) do Σ
        R, preM, = VF.R_M_P(Σ, blk_sizes′)
        M½ = VF.Midiagonal(map(sqrt, preM.data))
        P' * inv(R) * M½ * P 
    end
    return Σ▫
end



# Low level 
# az_bidiagΣ▫_P just computes the blocks of Σ▫ needed by Vecchia
# and also the permutation matrix that goes along with it.

# ------------------------------------------

# TODO: Harded coded Float64 in these cases. Fix it.

# Spin0
function spin0_az_bidiagΣ▫_P(
    Γ, 
    blk_sizes::AbstractVector{<:Integer}, 
    perm::AbstractVector{<:Integer}=1:sum(blk_sizes);
    θ, φ, ℓrange=1:length(φ)÷2+1
    )
    nθ, nφ = length(θ), length(φ)
    ptmW   = FFTW.plan_fft(Vector{ComplexF64}(undef, nφ))
    
    setΣ! = function (M▫,j,k)
        Mγⱼₖℓ⃗  = CC.γθ₁θ₂ℓ⃗(θ[j], θ[k], φ, Γ, ptmW)
        for (i,ℓ′) in enumerate(ℓrange)
            M▫[i][j,k] = real(Mγⱼₖℓ⃗[ℓ′])
        end
    end
    
    Σ▫     = [VF.initalize_bidiag_lblks(Float64, blk_sizes) for ℓ′ in ℓrange]
    
    blk_indices = blocks(PseudoBlockArray(perm, blk_sizes))
    N = length(blk_sizes)
    prgss  = Progress(N, dt=1, desc="Computing Block Diagonals")
    for ic in 1:N # loop over column block
        # start with diag block in ic's block column
        for k in blk_indices[ic], j in blk_indices[ic]
            setΣ!(Σ▫, j, k)
        end
        # then the lower diag in ic's block column
        if ic < N
            for k in blk_indices[ic], j in blk_indices[ic+1] 
                setΣ!(Σ▫, j, k)
            end
        end
        next!(prgss)
    end 

    P = VF.Piv(perm)

    return Σ▫, P
end


# Spin2
function spin2_az_bidiagΣ▫_P(
    Γ, C,
    blk_sizes::AbstractVector{<:Integer}, 
    perm::AbstractVector{<:Integer}=1:sum(blk_sizes);
    θ, φ, ℓrange=1:length(φ)÷2+1
    )
    
    nθ, nφ = length(θ), length(φ)
    ptmW   = FFTW.plan_fft(Vector{ComplexF64}(undef, nφ))
    
    setΣ! = function (Mγ▫,Mξ▫,cMγJ▫,cMξJ▫,j,k)
        Mγⱼₖℓ⃗, Mξⱼₖℓ⃗ = CC.γθ₁θ₂ℓ⃗_ξθ₁θ₂ℓ⃗(θ[j], θ[k], φ, Γ, C, ptmW)
        for (i,ℓ′) in enumerate(ℓrange)
            Jℓ′ = CC.Jperm(ℓ′, nφ)
            Mγ▫[i][j,k]   = Mγⱼₖℓ⃗[ℓ′]
            Mξ▫[i][j,k]   = Mξⱼₖℓ⃗[ℓ′]
            cMξJ▫[i][j,k] = conj(Mξⱼₖℓ⃗[Jℓ′])
            cMγJ▫[i][j,k] = conj(Mγⱼₖℓ⃗[Jℓ′])
        end
    end

    Mγ▫   = [VF.initalize_bidiag_lblks(ComplexF64, blk_sizes) for ℓ′ in ℓrange]
    Mξ▫   = [VF.initalize_bidiag_lblks(ComplexF64, blk_sizes) for ℓ′ in ℓrange]
    cMγJ▫ = [VF.initalize_bidiag_lblks(ComplexF64, blk_sizes) for ℓ′ in ℓrange]
    cMξJ▫ = [VF.initalize_bidiag_lblks(ComplexF64, blk_sizes) for ℓ′ in ℓrange]

    blk_indices = blocks(PseudoBlockArray(perm, blk_sizes))
    N = length(blk_sizes)
    prgss  = Progress(N, dt=1, desc="Computing Block Diagonals")
    for ic in 1:N # loop over column block
        # start with diag block in ic's block column
        for k in blk_indices[ic], j in blk_indices[ic]
            setΣ!(Mγ▫,Mξ▫,cMγJ▫,cMξJ▫, j, k) # this automatically sets
        end
        # then the lower diag in ic's block column
        if ic < N
            for k in blk_indices[ic], j in blk_indices[ic+1] 
                setΣ!(Mγ▫,Mξ▫,cMγJ▫,cMξJ▫, j, k)
            end
        end
        next!(prgss)
    end 

    # Put Mγ▫,Mξ▫,cMγJ▫,cMξJ▫  toghether for the full Spin2 operator
    Σ▫ = map(Mγ▫,Mξ▫,cMγJ▫,cMξJ▫) do Mγ,Mξ,cMγJ,cMξJ
        M = VF.initalize_bidiag_lblks(ComplexF64, 2 .* blk_sizes)
        for ic=1:N 
            M[Block(ic,ic)] = [ Mγ[Block(ic,ic)]   Mξ[Block(ic,ic)]
                              cMξJ[Block(ic,ic)] cMγJ[Block(ic,ic)] ]
            if ic < N
                M[Block(ic+1,ic)] = [ Mγ[Block(ic+1,ic)]   Mξ[Block(ic+1,ic)]
                                    cMξJ[Block(ic+1,ic)] cMγJ[Block(ic+1,ic)] ]
            end 
        end
        M
    end

    blk_sizes2 = 2 .* blk_sizes
    a1 = 1:2nθ |> x->reshape(x,nθ,2) # 2nθ indicies split in half and put in two columns
    a2 = a1[perm,:][:] # do a within θ perm of each block, i.e. perm the rows, re-stack into one column
    a3 = blocks(PseudoBlockArray(a2, vcat(blk_sizes, blk_sizes))) # divide into blocks
    perm2 = a3 |> x->reshape(x,N,2) |> x->permutedims(x) |> vec |> x->vcat(x...) # interlace the blocks
    P = VF.Piv(perm2)

    return Σ▫, P
end

