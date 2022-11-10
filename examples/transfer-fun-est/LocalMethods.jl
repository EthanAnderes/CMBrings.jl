
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


## ----------------------------------------


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

## ----------------------------------------


function deproject_Xθm_eachθ_qr(f::AbstractMatrix, θ::AbstractVector, m::AbstractMatrix, Xfromθ::Function)
    # f is a matrix representing the CMB (θ, φ) pixel values. (θ, φ) <-> (row, col)
    # m is a matrix representing the (θ, φ) pixel mask (0 => not-observed)
    # X is a matrix with columns representing the unmasked modes to be removed from the rows of f.
    g        = similar(f)
    for (fr, θr, mr, gr) in zip(eachrow(f), θ, eachrow(m), eachrow(g))
        Xmr     = mr .* Xfromθ(θr) # mask the columns of X
        factXmr = qr!(copy(Xmr), ColumnNorm()) # modifies Xm_copy
        copyto!(gr, deproject_Xm(fr, Xmr, factXmr))
    end
    return g
end

struct EllDeprojector{T<:AbstractVector, U<:AbstractMatrix} <: AbstractLinearOp
    Xfromθ::Function
    θ::T
    m::U
    alg::Symbol 
    function EllDeprojector(Xfromθ::Function, θ::T, m::U; alg=:qr) where {T,U}
        # alg ∈ {:qr, :svg_divide_conquer, :svg_qr_iteration}
        @assert length(θ) == size(m,1)
        new{T,U}(Xfromθ,θ,m,alg)
    end
end

function *(D::EllDeprojector, f::T) where {T<:Xfield{<:EAZ0}}
    fmat = f[:]
    if D.alg == :qr
        Dfmat = deproject_Xθm_eachθ_qr(fmat, D.θ, D.m, D.Xfromθ)
    # elseif D.alg == :svg_divide_conquer
    #     Dfmat = deproject_Xθm_eachθ_svd(fmat, D.θ, D.m, D.Xfromθ; alg=LinearAlgebra.DivideAndConquer()) 
    # elseif  D.alg == :svg_qr_iteration
    #     Dfmat = deproject_Xθm_eachθ_svd(fmat, D.θ, D.m, D.Xfromθ; alg=LinearAlgebra.QRIteration()) 
    else 
        error("EllDeprojector.alg not a valid option")
    end
    Xmap(fieldtransform(f), Dfmat)
end

function *(D::EllDeprojector, f::T) where {T<:Xfield{<:EAZ2}}
    fmat = f[:]
    q, u = real(fmat), imag(fmat)
    if D.alg == :qr
        Dqmat = deproject_Xθm_eachθ_qr(q, D.θ, D.m, D.Xfromθ)  
        Dumat = deproject_Xθm_eachθ_qr(u, D.θ, D.m, D.Xfromθ)
    # elseif D.alg == :svg_divide_conquer
    #     Dqmat = deproject_Xθm_eachθ_svd(q, D.θ, D.m, D.Xfromθ; alg=LinearAlgebra.DivideAndConquer())   
    #     Dumat = deproject_Xθm_eachθ_svd(u, D.θ, D.m, D.Xfromθ; alg=LinearAlgebra.DivideAndConquer())   
    # elseif  D.alg == :svg_qr_iteration
    #     Dqmat = deproject_Xθm_eachθ_svd(q, D.θ, D.m, D.Xfromθ; alg=LinearAlgebra.QRIteration())   
    #     Dumat = deproject_Xθm_eachθ_svd(u, D.θ, D.m, D.Xfromθ; alg=LinearAlgebra.QRIteration())   
    else 
        error("EllDeprojector.alg not a valid option")
    end
    Xmap(fieldtransform(f), complex.(Dqmat, Dumat))
end

## ----------------------------------------










end

