# Constructors for Block diagonals in AzEqui coordinates
# ====================================
# TODO: put these into CirculantCov

function az_cov_blks(ℓ, ffℓ::Vector{rT}; θ, φ, ngrid=100_000) where {rT}
    nθ, nφ = length(θ), length(φ)
    ptmW   = FFTW.plan_fft(Vector{complex(rT)}(undef, nφ))
    Γ      = CC.Γθ₁θ₂φ₁φ⃗_Iso(ℓ, ffℓ; ngrid)
    M▫     = Matrix{rT}[zeros(rT,nθ,nθ) for ℓ in 1:nφ÷2+1]
    prgss  = Progress(nθ, dt=1, desc="Computing Block Diagonals")
    for k = 1:nθ
        for j = 1:nθ
            Mγⱼₖℓ⃗  = CC.γθ₁θ₂ℓ⃗(θ[j], θ[k], φ, Γ,  ptmW)
            for ℓ in 1:nφ÷2+1
                M▫[ℓ][j,k] = real(Mγⱼₖℓ⃗[ℓ])
            end
        end
        next!(prgss)
    end
    return M▫
end


function az_cov_blks(
        ℓ, eeℓ::Vector{rT}, bbℓ::Vector{rT}; 
        θ, φ, ngrid=100_000
        ) where {rT}
    T      = complex(rT)
    nθ, nφ = length(θ), length(φ)
    ptmW   = FFTW.plan_fft(Vector{T}(undef, nφ))
    Γ, C   = CC.ΓCθ₁θ₂φ₁φ⃗_CMBpol(ℓ, eeℓ, bbℓ; ngrid)
    M▫     = Matrix{T}[zeros(T,2nθ,2nθ) for ℓ in 1:nφ÷2+1]
    prgss  = Progress(nθ, dt=1, desc="Computing Block Diagonals")
    for k = 1:nθ
        for j = 1:nθ
            Mγⱼₖℓ⃗, Mξⱼₖℓ⃗ = CC.γθ₁θ₂ℓ⃗_ξθ₁θ₂ℓ⃗(θ[j], θ[k], φ, Γ, C, ptmW)
            for ℓ in 1:nφ÷2+1
                Jℓ = CC.Jperm(ℓ, nφ)
                M▫[ℓ][j,   k   ] = Mγⱼₖℓ⃗[ℓ]
                M▫[ℓ][j,   k+nθ] = Mξⱼₖℓ⃗[ℓ]
                M▫[ℓ][j+nθ,k   ] = conj(Mξⱼₖℓ⃗[Jℓ])
                M▫[ℓ][j+nθ,k+nθ] = conj(Mγⱼₖℓ⃗[Jℓ])
            end
        end
        next!(prgss)
    end
    return M▫
end

## TODO: Can we make a Iso version of this and then just make diag blocks for Spin2 fields?
function az_beam4spin2_blks(
        ℓ, beamℓ::Vector{rT},
        θ, φ, Ω, ngrid=100_000
        ) where {rT<:Real}
    nθ, nφ = length(θ), length(φ)
    ptmW   = FFTW.plan_fft(Vector{complex(rT)}(undef, nφ))
    Γ      = CC.Γθ₁θ₂φ₁φ⃗_Iso(ℓ, beamℓ; ngrid)
    M▫     = Matrix{rT}[zeros(rT,2nθ,2nθ) for ℓ in 1:nφ÷2+1]
    prgss  = Progress(nθ, dt=1, desc="Computing Block Diagonals")
    for k = 1:nθ
        for j = 1:nθ
            Mγⱼₖℓ⃗  = CC.γθ₁θ₂ℓ⃗(θ[j], θ[k], φ, Γ, ptmW)
            for ℓ in 1:nφ÷2+1
                Jℓ = CC.Jperm(ℓ, nφ)
                M▫[ℓ][j, k   ]   = real(Mγⱼₖℓ⃗[ℓ])
                M▫[ℓ][j,   k+nθ] = 0
                M▫[ℓ][j+nθ,k   ] = 0
                M▫[ℓ][j+nθ,k+nθ] = real(Mγⱼₖℓ⃗[Jℓ])
            end
        end
        next!(prgss)
    end
    Ω2Diag = Diagonal(vcat(Ω,Ω))
    return map(M->M*Ω2Diag, M▫)
end

# Constructors for Block diagonals with 
# Vecchia approx in each block in AzEqui coordinates
# ====================================


function initalize_bidiag_lblks(::Type{T}, blk_sizes) where T
    N = length(blk_sizes)
    B = BlockArray{T}(undef_blocks, blk_sizes, blk_sizes)
    for ic=1:N
        B[Block(ic,ic)] = zeros(T, blk_sizes[ic], blk_sizes[ic])
        if ic < N 
            B[Block(ic+1,ic)] = zeros(T, blk_sizes[ic+1], blk_sizes[ic])
        end 
    end 
    B 
end 



function az_cov½_vecchia_blks(
    ℓ, ffℓ::Vector{rT}, 
    blk_sizes::AbstractVector{<:Integer}, 
    perm::AbstractVector{<:Integer}=1:sum(blk_sizes);
    θ, φ, ngrid=100_000
    ) where {rT}
    
    nθ, nφ = length(θ), length(φ)
    ptmW   = FFTW.plan_fft(Vector{complex(rT)}(undef, nφ))
    Γ      = CC.Γθ₁θ₂φ₁φ⃗_Iso(ℓ, ffℓ; ngrid)
    setΣ! = function (M▫,j,k)
        Mγⱼₖℓ⃗  = CC.γθ₁θ₂ℓ⃗(θ[j], θ[k], φ, Γ, ptmW)
        for ℓ in 1:nφ÷2+1
            M▫[ℓ][j,k] = real(Mγⱼₖℓ⃗[ℓ])
        end
    end
    
    N = length(blk_sizes)
    blk_indices = blocks(PseudoBlockArray(perm, blk_sizes))
    M▫     = [initalize_bidiag_lblks(rT, blk_sizes) for ℓ in 1:nφ÷2+1]

    prgss  = Progress(N, dt=1, desc="Computing Block Diagonals")
    for ic in 1:N # loop over column block
        # start with diag block in ic's block column
        for k in blk_indices[ic], j in blk_indices[ic]
            setΣ!(M▫, j, k)
        end
        # then the lower diag in ic's block column
        if ic < N
            for k in blk_indices[ic], j in blk_indices[ic+1] 
                setΣ!(M▫, j, k)
            end
        end
        next!(prgss)
    end 

    P = VF.Piv(perm)
    map(M▫) do M 
        R, preM, = VF.R_M_P(M, blk_sizes)
        M½ = VF.Midiagonal(map(sqrt, preM.data))
        P' * inv(R) * M½
    end
end



# Note: this inherits the permutation on θ given in perm
# and doubles blk_sizes
function az_cov½_vecchia_blks(
    ℓ, eeℓ::Vector{rT}, bbℓ::Vector{rT},
    blk_sizes::AbstractVector{<:Integer}, 
    perm::AbstractVector{<:Integer}=1:sum(blk_sizes);
    θ, φ, ngrid=100_000
    ) where {rT}
    
    T      = complex(rT)
    nθ, nφ = length(θ), length(φ)
    ptmW   = FFTW.plan_fft(Vector{T}(undef, nφ))
    Γ, C   = CC.ΓCθ₁θ₂φ₁φ⃗_CMBpol(ℓ, eeℓ, bbℓ; ngrid)
    setΣ! = function (Mγ▫,Mξ▫,cMγJ▫,cMξJ▫,j,k)
        Mγⱼₖℓ⃗, Mξⱼₖℓ⃗ = CC.γθ₁θ₂ℓ⃗_ξθ₁θ₂ℓ⃗(θ[j], θ[k], φ, Γ, C, ptmW)
        for ℓ in 1:nφ÷2+1
            Jℓ = CC.Jperm(ℓ, nφ)
            Mγ▫[ℓ][j,k]   = Mγⱼₖℓ⃗[ℓ]
            Mξ▫[ℓ][j,k]   = Mξⱼₖℓ⃗[ℓ]
            cMξJ▫[ℓ][j,k] = conj(Mξⱼₖℓ⃗[Jℓ])
            cMγJ▫[ℓ][j,k] = conj(Mγⱼₖℓ⃗[Jℓ])
        end
    end

    Mγ▫   = [initalize_bidiag_lblks(T, blk_sizes) for ℓ in 1:nφ÷2+1]
    Mξ▫   = [initalize_bidiag_lblks(T, blk_sizes) for ℓ in 1:nφ÷2+1]
    cMγJ▫ = [initalize_bidiag_lblks(T, blk_sizes) for ℓ in 1:nφ÷2+1]
    cMξJ▫ = [initalize_bidiag_lblks(T, blk_sizes) for ℓ in 1:nφ÷2+1]

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
    M▫ = map(Mγ▫,Mξ▫,cMγJ▫,cMξJ▫) do Mγ,Mξ,cMγJ,cMξJ
        M = initalize_bidiag_lblks(T, 2 .* blk_sizes)
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
    map(M▫) do M 
        R, preM, = VF.R_M_P(M, blk_sizes2)
        M½ = VF.Midiagonal(map(sqrt, preM.data))
        P' * inv(R) * M½
    end
end



# A one dimensional smooth mask
# ====================================


function pixweight(x::T; ▮l, ▯l, ▯r, ▮r) where T<:Number
    @assert ▮l ≤ ▯l ≤ ▯r ≤ ▮r
    if ▯l ≤ x ≤ ▯r
        return one(T)
    elseif (x ≤ ▮l) | (▮r ≤ x)
        return zero(T)
    elseif ▮l < x < ▯l
        return T((1-cos(π*(x-▮l)/(▯l-▮l))) / 2)
    else 
        @assert ▯r < x < ▮r 
        return T((1+cos(π*(x-▯r)/(▮r-▯r))) / 2)
    end
end



   

# custom pcg with function composition (Minv * A \approx I)
# =====================================
function pcg(Minv::Function, A::Function, b, x=0*b; nsteps::Int=75, rel_tol = 0)
    r       = b - A(x)
    z       = Minv(r)
    p       = deepcopy(z)
    res     = dot(r,z)
    reshist = Vector{typeof(res)}()
    for i = 1:nsteps
        Ap        = A(p)
        α         = res / dot(p,Ap)
        x         = x + α * p
        r         = r - α * Ap
        z         = Minv(r)
        res′      = dot(r,z)
        p         = z + (res′ / res) * p
        rel_error = XFields.nan2zero(sqrt(dot(r,r)/dot(b,b)))
        push!(reshist, rel_error)
        if rel_error < rel_tol
            return x, reshist
        end
        res = res′
    end
    return x, reshist
end


function pcg_coupled(;
        _Aᵍ::Function, # preconditioner 
        A::Function,   # operator we want to invert
        b_g, b_f,      # solution we want is A⁻¹*vcat(b_g, f_g)
        x_g, x_f,      # warm start for solution
        nsteps=30, rel_tol = 0.0,
        reshist=Vector{Float64}() 
    )
    Ax_g, Ax_f = A(x_g, x_f)
    r_g  = b_g - Ax_g
    r_f  = b_f - Ax_f
    z_g, z_f  =  _Aᵍ(r_g, r_f)
    p_g  = deepcopy(z_g)
    p_f  = deepcopy(z_f)

    res   = dot(r_g,z_g) + dot(r_f,z_f)

    for i = 1:nsteps
        p′_g, p′_f = A(p_g, p_f)
        α    = res / (dot(p_g,p′_g) + dot(p_f,p′_f))
        x_g  += α * p_g
        x_f  += α * p_f
        r_g  -= α * p′_g
        r_f  -= α * p′_f
        z_g, z_f = _Aᵍ(r_g, r_f)
        res′ = dot(r_g,z_g) + dot(r_f,z_f)
        p_g  = z_g + (res′ / res) * p_g
        p_f  = z_f + (res′ / res) * p_f
        rel_error = (dot(r_g,r_g) + dot(r_f,r_f)) / (dot(b_g,b_g) + dot(b_f,b_f))
        if rel_error < rel_tol
            break 
        end
        push!(reshist, rel_error)
        res = res′
    end
    return x_g, x_f, reshist
end



# WF pcg
# =====================================

function update_f(
    Łϕ, EB::CircOp; 
    data,
    Pr, Qr, Bm, No, Pc⁻¹,
    ginit=0*data,
    pcg_nsteps, pcg_rel_tol=1e-10,
    ds...
)
    Łϕᴴ = Łϕ'
    C1a = Pr * Bm * Łϕ * EB * Łϕᴴ * Bm'
    C1b = Pr * No
    C2b = Qr * No
    ## C2a = Qr * Bm * Łϕ * EB * Łϕᴴ * Bm' # this one or ....
    C2a = Qr * Bm * EB * Bm' # .... this one
    ## C2a and C2b can be combine into one op.

    A = function (g)
        Prᴴ_g = Pr' * g
        Qrᴴ_g = Qr' * g
        tmp1a = C1a * Prᴴ_g
        tmp1b = C1b * Prᴴ_g
        tmp2a = C2a * Qrᴴ_g
        tmp2b = C2b * Qrᴴ_g
        return tmp1a + tmp1b + tmp2a + tmp2b
    end

    gwf, hst = pcg(
        g -> Pc⁻¹ * g, A, 
        data, ginit,
        nsteps=pcg_nsteps, rel_tol=pcg_rel_tol,
    )
    fwf   = EB *  Łϕᴴ * Bm' * Pr' * gwf
    return  fwf, gwf, hst
end

