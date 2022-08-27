
# CircOp struct
# =======================================
const MatrixOrFactorization{T} = Union{Factorization{T}, AbstractMatrix{T}}


"""
`CircOp{M<:MatrixOrFactorization{} <: XFields.AbstractLinearOp` holds the Diagonal blocks of a 
circulant field covariance op.

The storage format corresponds to the structure of the operator as it applies to the real and 
imag part of the pixel field as a function of (θ, φ).


General CircOp for complex fields (in map space)
[ Σ₁  Σ₃ ] [reP(θ,φ)]
[ Σ₂  Σ₄ ] [imP(θ,φ)]

Beam CircOp for complex fields (in map space)
[ Σ₁  0  ] [ Diagonal(Ω)  0           ] [reP(θ,φ)]
[ 0   Σ₁ ] [ 0            Diagonal(Ω) ] [imP(θ,φ)]


 CircOp for real fields (in map space)
[ Σ₁  0 ] [reP(θ,φ)]
[ 0  Σ₁ ] [0       ]
"""
# struct CircOp{M<:MatrixOrFactorization} <: XFields.AbstractLinearOp
struct CircOp{M} <: XFields.AbstractLinearOp
    Σ::Vector{M}
end

function CircOp(nrows::Int, ncols::Int, nblocks::Int, ::Type{M}) where {M<:MatrixOrFactorization}
    Σ = M[M(undef, nrows, ncols) for ℓ ∈ 1:nblocks]
    CircOp{M}(Σ)
end 

Base.parent(az::CircOp) = az.Σ

# AdjointCircOp
# =======================================

# struct AdjointCircOp{M<:MatrixOrFactorization} <: XFields.AbstractLinearOp
struct AdjointCircOp{M} <: XFields.AbstractLinearOp
    az::CircOp{M}
end

function Base.adjoint(az::CircOp{M}) where {M}
    return AdjointCircOp{M}(az)
end

Base.parent(az::AdjointCircOp) = az.az.Σ

# Preping 1-d FFT'd matrices for CircOp argument using the methods
# in CirculantCov.jl ℝfθk2▪ and ℂfθk2▪
# A bit higher level conversion from blk to the format accepted by CircOps
# =======================================

function field2▪(f::Xf) where {Tm,Ti<:Real,To,Xf<:Xfield{Tm,Ti,To,2}}
    CC.ℝfθk2▪(fielddata(FourierField(f)))
end

function field2▪(f::Xf) where {Tm,Ti<:Complex,To,Xf<:Xfield{Tm,Ti,To,2}}
    CC.ℂfθk2▪(fielddata(FourierField(f)))
end

# It would be nice to replace the else-if with dispatch
function ▪2field(tm::Transform, w::Vector{Vector{To}}) where {To} 
    if eltype_in(tm) <: Real 
        return Xfourier(tm, CC.▪2ℝfθk(w))
    elseif eltype_in(tm) <: Complex
        nφ = size_in(tm)[2]  
        return Xfourier(tm, CC.▪2ℂfθk(w,nφ))
    end
end

# Define map(fun::Function, az::CircOp, f::Xfield)
# where fun(Σℓ,vℓ) -> wℓ
# ==================================

function Base.map(fun::Function, az::Union{CircOp,AdjointCircOp}, f::XF)::XF where {Tm,Ti,To,XF<:Xfield{Tm,Ti,To,2}}
    Σf▪ = map(fun, az, field2▪(f))
    XF(▪2field(fieldtransform(f), Σf▪))
end 

# Define az * f and az \ f divide
# ==================================

# this avoids converting az * f to the basis f was stored in ...
function XFields._lmult(az::Union{CircOp, AdjointCircOp}, f::XF) where {Tm,Ti,To,XF<:Xfield{Tm,Ti,To,2}}
    Σf▪ = map(*, az, field2▪(f))
    ▪2field(fieldtransform(f), Σf▪)
end

function Base.:*(az::Union{CircOp, AdjointCircOp}, f::XF)::XF where {XF<:Xfield}
    XF(XFields._lmult(az, f))
end

function Base.:\(az::Union{CircOp,AdjointCircOp}, f::XF)::XF where {Tm,Ti,To,XF<:Xfield{Tm,Ti,To,2}}
    map(\, az, f)
end 


# Make CircOp an iterator
# =======================================

Base.length(az::Union{CircOp, AdjointCircOp})     = length(parent(az))
Base.eltype(::Type{CircOp{M}})        where {M}   = M 
Base.eltype(::Type{AdjointCircOp{M}}) where {A,M<:Matrix{A}} = Adjoint{A,M} 
Base.eltype(::Type{AdjointCircOp{M}}) where {M<:Symmetric} = M 
Base.eltype(::Type{AdjointCircOp{M}}) where {M<:Hermitian} = M 
Base.eltype(::Type{AdjointCircOp{M}}) where {M<:Diagonal}  = M 
Base.eltype(::Type{AdjointCircOp{M}}) where {A,B,M<:LowerTriangular{A,B}} = UpperTriangular{A,Adjoint{A,B}} 
Base.eltype(::Type{AdjointCircOp{M}}) where {A,B,M<:UpperTriangular{A,B}} = LowerTriangular{A,Adjoint{A,B}} 

Base.firstindex(az::Union{CircOp, AdjointCircOp}) = 1
Base.lastindex(az::Union{CircOp, AdjointCircOp})  = length(az)

Base.iterate(az::CircOp)        = (Σ=parent(az) ; isempty(Σ) ? nothing : (Σ[1],1))
Base.iterate(az::AdjointCircOp) = (Σ=parent(az) ; isempty(Σ) ? nothing : (Σ[1]',1))

Base.iterate(az::CircOp, st)        = st+1 > length(az) ? nothing : (parent(az)[st+1],  st+1)
Base.iterate(az::AdjointCircOp, st) = st+1 > length(az) ? nothing : (parent(az)[st+1]', st+1)

function Base.getindex(az::CircOp, i::Int) 
    1 <= i <= length(az) || throw(BoundsError(az, i))
    return parent(az)[i]
end

function Base.getindex(az::AdjointCircOp, i::Int) 
    1 <= i <= length(az) || throw(BoundsError(az, i))
    return parent(az)[i]'
end

function Base.setindex!(az::CircOp{M}, m::M, i::Int) where {M}
    1 <= i <= length(az) || throw(BoundsError(az, i))  
    setindex!(parent(az)[i], m)
end
