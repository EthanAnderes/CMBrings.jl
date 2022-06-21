

function quasi_bandpowers(f; θ, Δℓsph_bin = 15) # can we avoid passing θ??
    tm = fieldtransform(f)
    k    = FFTransforms.freq(tm)[2]
    ℓsph = k' ./ sin.(θ)

    ℓsph_bin∂  = 0:Δℓsph_bin:(maximum(ℓsph)+1)
    ℓsph_bin_mid = ℓsph_bin∂[1:end-1] .+ Δℓsph_bin ./ 2

    power_ℓsph_bin_mid = zeros(length(ℓsph_bin_mid))

    raw_power = abs2.(f[!])  

    for i in eachindex(power_ℓsph_bin_mid)
        ll = ℓsph_bin∂[i]
        lr = ℓsph_bin∂[i+1]
        idx  = ll .<= ℓsph .< lr
        nidx = sum(idx)
        power_ℓsph_bin_mid[i] = nidx > 0 ? sum(raw_power[idx]) / nidx : 0.0
    end
    ℓsph_bin_mid, power_ℓsph_bin_mid
end



function quasi_bandpowers(f, g; θ, Δℓsph_bin = 15) # can we avoid passing θ??
    tm = fieldtransform(f)
    k    = FFTransforms.freq(tm)[2]
    ℓsph = k' ./ sin.(θ)

    ℓsph_bin∂  = 0:Δℓsph_bin:(maximum(ℓsph)+1)
    ℓsph_bin_mid = ℓsph_bin∂[1:end-1] .+ Δℓsph_bin ./ 2

    raw_power = f[!] .* conj.(g[!])  
    power_ℓsph_bin_mid = zeros(eltype(raw_power), length(ℓsph_bin_mid))

    for i in eachindex(power_ℓsph_bin_mid)
        ll = ℓsph_bin∂[i]
        lr = ℓsph_bin∂[i+1]
        idx  = ll .<= ℓsph .< lr
        nidx = sum(idx)
        power_ℓsph_bin_mid[i] = nidx > 0 ? sum(raw_power[idx]) / nidx : 0.0
    end
    ℓsph_bin_mid, power_ℓsph_bin_mid
end




# Constructors for Block diagonals in AzEqui coordinates
# ====================================

# TODO: replace the body of az_cov_blks with the similar methods from CirculantCov

function az_cov_blks(
        ℓ, ffℓ::Vector{rT}; 
        θ, φ, ngrid=100_000, ℓrange=1:length(φ)÷2+1) where {rT}
    nθ, nφ = length(θ), length(φ)
    ptmW   = FFTW.plan_fft(Vector{complex(rT)}(undef, nφ))
    Γ      = CC.Γθ₁θ₂φ₁φ⃗_Iso(ℓ, ffℓ; ngrid)
    M▫     = Matrix{rT}[zeros(rT,nθ,nθ) for ℓ′ in ℓrange]
    prgss  = Progress(nθ, dt=1, desc="Computing Block Diagonals")
    for k = 1:nθ
        for j = 1:nθ
            Mγⱼₖℓ⃗  = CC.γθ₁θ₂ℓ⃗(θ[j], θ[k], φ, Γ,  ptmW)
            for (i,ℓ′) in enumerate(ℓrange)
                M▫[i][j,k] = real(Mγⱼₖℓ⃗[ℓ′])
            end
        end
        next!(prgss)
    end
    return M▫
end

function az_cov_blks(
        ℓ, eeℓ::Vector{rT}, bbℓ::Vector{rT}; 
        θ, φ, ngrid=100_000, ℓrange=1:length(φ)÷2+1) where {rT}
    T      = complex(rT)
    nθ, nφ = length(θ), length(φ)
    ptmW   = FFTW.plan_fft(Vector{T}(undef, nφ))
    Γ, C   = CC.ΓCθ₁θ₂φ₁φ⃗_CMBpol(ℓ, eeℓ, bbℓ; ngrid)
    M▫     = Matrix{T}[zeros(T,2nθ,2nθ) for ℓ′ in ℓrange]
    prgss  = Progress(nθ, dt=1, desc="Computing Block Diagonals")
    for k = 1:nθ
        for j = 1:nθ
            Mγⱼₖℓ⃗, Mξⱼₖℓ⃗ = CC.γθ₁θ₂ℓ⃗_ξθ₁θ₂ℓ⃗(θ[j], θ[k], φ, Γ, C, ptmW)
            for (i,ℓ′) in enumerate(ℓrange)
                Jℓ′ = CC.Jperm(ℓ′, nφ)
                M▫[i][j,   k   ] = Mγⱼₖℓ⃗[ℓ′]
                M▫[i][j,   k+nθ] = Mξⱼₖℓ⃗[ℓ′]
                M▫[i][j+nθ,k   ] = conj(Mξⱼₖℓ⃗[Jℓ′])
                M▫[i][j+nθ,k+nθ] = conj(Mγⱼₖℓ⃗[Jℓ′])
            end
        end
        next!(prgss)
    end
    return M▫
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
    blk_sizes = VF.blocksizes(Σ_pre▫[1],1)
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
    blk_sizes = VF.blocksizes(Σ_pre▫[1],1)
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
    blk_sizes = VF.blocksizes(Σ_pre▫[1],1)
    Σ▫ = map(Σ_pre▫) do Σ
        P' * VF.vecchia(Σ, blk_sizes) * P
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
    blk_sizes = VF.blocksizes(Σ_pre▫[1],1)
    Σ▫ = map(Σ_pre▫) do Σ
        P' * VF.vecchia(Σ, blk_sizes) * P
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
    blk_sizes = VF.blocksizes(Σ_pre▫[1],1)
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
    blk_sizes = VF.blocksizes(Σ_pre▫[1],1)
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
    blk_sizes = VF.blocksizes(Σ_pre▫[1],1)
    Σ▫ = map(Σ_pre▫) do Σ
        R, preM, = VF.R_M_P(Σ, blk_sizes)
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
    blk_sizes = VF.blocksizes(Σ_pre▫[1],1)
    Σ▫ = map(Σ_pre▫) do Σ
        R, preM, = VF.R_M_P(Σ, blk_sizes)
        M½ = VF.Midiagonal(map(sqrt, preM.data))
        P' * inv(R) * M½ * P 
    end
    return Σ▫
end



# Low level 
# az_bidiagΣ▫_P just computes the blocks of Σ▫ needed by Vecchia
# and also the permutation matrix that goes along with it.

# ------------------------------------------



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





# Pixel space non-stationary beams
# ====================================

fwhmrad2σ²(rad)    = rad^2 / 8 / log(2)
    
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

