
module LocalMethods
using XFields
using EAZTransforms
using LinearAlgebra
import LinearAlgebra: *

export RingDeprojector


function deproject_Xm(f::AbstractVector, Xm::AbstractMatrix, factXm::Factorization) 
    f - Xm * (factXm \ f)
end 

function deproject_Xm_iter!(f::AbstractVector, m::AbstractVector, X::AbstractMatrix) 
    vm = similar(f)
    for v in eachcol(X)
        vm .= m .* v
        f .-= v .* (vm \ f)
    end 
    return f
end 


####################################
#
# RingDeprojector (TODO add Ring Deprojector to CMBrings/src...)
#
####################################

struct RingDeprojector{T<:AbstractMatrix, U<:AbstractMatrix} <: AbstractLinearOp
    X::T
    m::U
    alg::Symbol 
    function RingDeprojector(X::T, m::U; alg=:iter) where {T,U}
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
    elseif  D.alg == :iter
        Dfmat = deproject_Xm_eachrow_iter(fmat, D.m, D.X)
    elseif  D.alg == :iter2
        Dfmat = deproject_Xm_eachrow_iter(fmat, D.m, D.X)  
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
    elseif  D.alg == :iter
        Dqmat = deproject_Xm_eachrow_iter(q, D.m, D.X)  
        Dumat = deproject_Xm_eachrow_iter(u, D.m, D.X)
    elseif  D.alg == :iter2
        Dqmat = deproject_Xm_eachrow_iter2(q, D.m, D.X)  
        Dumat = deproject_Xm_eachrow_iter2(u, D.m, D.X)    
    else 
        error("RingDeprojector.alg not a valid option")
    end
    Xmap(fieldtransform(f), complex.(Dqmat, Dumat))
end

function deproject_Xm_eachrow_iter(f::AbstractMatrix, m::AbstractMatrix, X::AbstractMatrix)
    # f is a matrix representing the CMB (θ, φ) pixel values. (θ, φ) <-> (row, col)
    # m is a matrix representing the (θ, φ) pixel mask (0 => not-observed)
    # X is a matrix with columns representing the unmasked modes to be removed from the rows of f.
    # g = copy(f)
    # for (gr, mr) in zip(eachrow(g), eachrow(m))
    #     deproject_Xm_iter!(gr, mr, X) 
    # end
    # return g
    # testing ...
    fᵀ   = Array(transpose(f))
    mᵀ   = Array(transpose(m))
    fᵀcs = eachcol(fᵀ)
    mᵀcs = eachcol(mᵀ)
    nθ   = length(mᵀcs)
    @assert length(fᵀcs) == length(mᵀcs) == nθ
    Threads.@threads for i=1:nθ
        deproject_Xm_iter!(fᵀcs[i], mᵀcs[i], X) 
    end
    return Array(transpose(fᵀ))
end

function deproject_Xm_eachrow_iter2(f::AbstractMatrix, m::AbstractMatrix, X::AbstractMatrix)
    # f is a matrix representing the CMB (θ, φ) pixel values. (θ, φ) <-> (row, col)
    # m is a matrix representing the (θ, φ) pixel mask (0 => not-observed)
    # X is a matrix with columns representing the unmasked modes to be removed from the rows of f.
    g  = similar(f)
    for Xi in eachcol(X)
        mXiᵀ = m .* Xi' 
        norm2_mXiᵀ = sum(abs2.(v) for v in eachcol(mXiᵀ))
        ci = sum(eachcol(mXiᵀ .* g)) ./ norm2_mXiᵀ
        g .-= ci .* Xi'
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


####################################
#
# EllDeprojector
#
####################################

struct EllDeprojector{T<:AbstractVector, U<:AbstractMatrix} <: AbstractLinearOp
    Xfromθ::Function
    θ::T
    m::U
    alg::Symbol 
    function EllDeprojector(Xfromθ::Function, θ::T, m::U; alg=:iter) where {T,U}
        # alg ∈ {:qr, :svg_divide_conquer, :svg_qr_iteration}
        @assert length(θ) == size(m,1)
        new{T,U}(Xfromθ,θ,m,alg)
    end
end

function *(D::EllDeprojector, f::T) where {T<:Xfield{<:EAZ0}}
    fmat = f[:]
    if D.alg == :qr
        Dfmat = deproject_Xθm_eachθ_qr(fmat, D.θ, D.m, D.Xfromθ)
    elseif D.alg == :iter
        Dfmat = deproject_Xθm_eachθ_iter(fmat, D.θ, D.m, D.Xfromθ)
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
    elseif D.alg == :iter
        Dqmat = deproject_Xθm_eachθ_iter(q, D.θ, D.m, D.Xfromθ)  
        Dumat = deproject_Xθm_eachθ_iter(u, D.θ, D.m, D.Xfromθ)
    else 
        error("EllDeprojector.alg not a valid option")
    end
    Xmap(fieldtransform(f), complex.(Dqmat, Dumat))
end

function deproject_Xθm_eachθ_qr(f::AbstractMatrix, θ::AbstractVector, m::AbstractMatrix, Xfromθ::Function)
    # f is a matrix representing the CMB (θ, φ) pixel values. (θ, φ) <-> (row, col)
    # m is a matrix representing the (θ, φ) pixel mask (0 => not-observed)
    # X is a matrix with columns representing the unmasked modes to be removed from the rows of f.
    g        = similar(f)
    for (fr, θr, mr, gr) in zip(eachrow(f), θ, eachrow(m), eachrow(g))
        X       = Xfromθ(θr)
        factXmr = qr!(mr.*X, ColumnNorm()) 
        copyto!(gr, deproject_Xm(fr, X, factXmr))
    end
    return g
end

function deproject_Xθm_eachθ_iter(f::AbstractMatrix, θ::AbstractVector, m::AbstractMatrix, Xfromθ::Function)
    # f is a matrix representing the CMB (θ, φ) pixel values. (θ, φ) <-> (row, col)
    # m is a matrix representing the (θ, φ) pixel mask (0 => not-observed)
    # X is a matrix with columns representing the unmasked modes to be removed from the rows of f.
    ##########
    # g  = copy(f)
    # for (gr, mr, θr) in zip(eachrow(g), eachrow(m), θ)
    #     deproject_Xm_iter!(gr, mr, Xfromθ(θr)) 
    # end
    # return g
    ##########
    # testing ...
    fᵀ   = Array(transpose(f))
    mᵀ   = Array(transpose(m))
    fᵀcs = eachcol(fᵀ)
    mᵀcs = eachcol(mᵀ)
    nθ   = length(mᵀcs)    
    @assert length(fᵀcs) == length(mᵀcs) == length(θ)
    Threads.@threads for i=1:nθ
        deproject_Xm_iter!(fᵀcs[i], mᵀcs[i], Xfromθ(θ[i])) 
    end
    return Array(transpose(fᵀ))
end


end

