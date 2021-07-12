
# CircOp struct
# =======================================

"""
`CircOp{M<:AbstractMatrix} <: XFields.AbstractLinearOp` holds the Diagonal blocks of a 
circulant field covariance op.

The storage format corresponds to the structure of the operator as it applies to the real and 
imag part of the pixel field as a function of (θ, φ).
"""
struct CircOp{M<:AbstractMatrix} <: XFields.AbstractLinearOp
    Σ::Vector{M}
end

function CircOp(nrows::Int, ncols::Int, nblocks::Int, ::Type{M}) where {M<:AbstractMatrix}
    Σ = M[M(undef, nrows, ncols) for ℓ ∈ 1:nblocks]
    CircOp{M}(Σ)
end 

# AdjointCircOp
# =======================================

struct AdjointCircOp{M<:AbstractMatrix} <: XFields.AbstractLinearOp
    az::CircOp{M}
end

function Base.adjoint(az::CircOp{M}) where {M}
    return AdjointCircOp{M}(az)
end

# Preping 1-d FFT'd matrices for CircOp argument
# =======================================

"""
Real map fields have an implicit pairing with primal and dual frequency
so we instead construct nφ÷2+1 vectors of length nθ 
"""
function ℝfθk2▪(Uf::AbstractMatrix)
    return [v for v ∈ eachcol(Uf)]
end
 
function ▪2ℝfθk(w::Vector{Vector{To}}) where To 
    nθ, nφ½₊1 = length(w[1]), length(w)
    fθk = zeros(To, nθ, nφ½₊1)
    for i in 1:nφ½₊1 
        fθk[:,i] = w[i]
    end
    fθk
end

"""
Complex map fields get frequency paired with dual frequency ... to make nφ÷2+1 vectors of length 2nθ 
"""
function ℂfθk2▪(Up::AbstractMatrix{To}) where To
    nθ, nφ = size(Up)
    w  = Vector{To}[zeros(To,2nθ) for ℓ = Base.OneTo(nφ÷2+1)]
    Up_col = collect(eachcol(Up))
    for ℓ = 1:nφ÷2+1
        if (ℓ==1) | ((ℓ==nφ÷2+1) & iseven(nφ))
            w[ℓ][1:nθ]     .= Up_col[ℓ]
            w[ℓ][nθ+1:2nθ] .= conj.(Up_col[ℓ])
        else 
            Jℓ = nφ - ℓ + 2
            w[ℓ][1:nθ]     .= Up_col[ℓ]
            w[ℓ][nθ+1:2nθ] .= conj.(Up_col[Jℓ])
        end
    end
    w
end

function ▪2ℂfθk(w::Vector{Vector{To}}, nφ::Int) where To 
    nθₓ2, nφ½₊1   = length(w[1]), length(w)
    @assert nφ½₊1 == nφ÷2+1
    @assert iseven(nθₓ2)
    nθ  = nθₓ2÷2

    pθk = zeros(To, nθ, nφ)
    for ℓ = 1:nφ½₊1
        if (ℓ==1) | ((ℓ==nφ½₊1) & iseven(nφ))
            pθk[:,ℓ] .= w[ℓ][1:nθ] 
        else 
            Jℓ = nφ - ℓ + 2
            pθk[:,ℓ]  .= w[ℓ][1:nθ]      
            pθk[:,Jℓ] .= conj.(w[ℓ][nθ+1:2nθ])
        end
    end 
    pθk
end

# A bit higher level conversion from blk to the format accepted by CircOps
# =======================================

function field2▪(f::Xf) where {Tm,Ti<:Real,To,Xf<:Xfield{Tm,Ti,To,2}}
    # ℝfθk2▪(fielddata(FourierField(f)))
    ℝfθk2▪(f[!])
end

function field2▪(f::Xf) where {Tm,Ti<:Complex,To,Xf<:Xfield{Tm,Ti,To,2}}
    # ℂfθk2▪(fielddata(FourierField(f)))
    ℂfθk2▪(f[!])
end

function ▪2field(tm::Transform{Ti,2}, w::Vector{Vector{To}}) where {To, Ti<:Real} 
    Xfourier(tm, ▪2ℝfθk(w))
end

function ▪2field(tm::Transform{Ti,2}, w::Vector{Vector{To}}) where {To, Ti<:Complex} 
    nφ = size_in(tm)[2] 
    Xfourier(tm, ▪2ℂfθk(w,nφ))
end

# Define map(fun::Function, az::CircOp, f::Xfield)
# where fun(Σℓ,vℓ) -> wℓ
# ==================================
# Question: does fun::Tf where {Tf<:Function} overspecialize vrs fun::Function

function Base.map(fun::Tf, az::CircOp, f::XF)::XF where {Tf<:Function, Tm,Ti,To,XF<:Xfield{Tm,Ti,To,2}}
    Σf▪ = map(fun, az.Σ, field2▪(f))
    XF(▪2field(fieldtransform(f), Σf▪))
end 

function Base.map(fun::Tf, az::AdjointCircOp, f::XF)::XF where {Tf<:Function, Tm,Ti,To,XF<:Xfield{Tm,Ti,To,2}}
    Σf▪ = map((Σ,v)->fun(adjoint(Σ),v), az.az.Σ, field2▪(f))
    XF(▪2field(fieldtransform(f), Σf▪))
end 

# Define left divide (to be used sparingly) 
# ==================================

function Base.:\(az::Union{CircOp,AdjointCircOp}, f::XF)::XF where {Tm,Ti,To,XF<:Xfield{Tm,Ti,To,2}}
    map(\, az, f)
end 

function Base.:*(az::Union{CircOp,AdjointCircOp}, f::XF)::XF where {Tm,Ti,To,XF<:Xfield{Tm,Ti,To,2}}
    map(*, az, f)
end 

# this avoids converting back to the original basis ... not sure how much it helps
# ==================================

# General CircOp for complex fields (in map space)
# [ Σ₁  Σ₃ ] [reP(θ,φ)]
# [ Σ₂  Σ₄ ] [imP(θ,φ)]

#  CircOp for real fields (in map space)
# [ Σ₁  0 ] [reP(θ,φ)]
# [ 0  Σ₁ ] [0       ]

# function XFields._lmult(az::CircOp, f::XF) where {Tm,Ti,To,XF<:Xfield{Tm,Ti,To,2}}
#     Σf▪ = map(*, az.Σ, field2▪(f))
#     ▪2field(fieldtransform(f), Σf▪)
# end

# function XFields._lmult(az::AdjointCircOp, f::XF) where {Tm,Ti,To,XF<:Xfield{Tm,Ti,To,2}}
#     Σf▪ = map((A,b)->A'*b, az.az.Σ, field2▪(f))
#     ▪2field(fieldtransform(f), Σf▪)
# end

# function Base.:*(az::Union{CircOp, AdjointCircOp}, f::XF) where {XF<:Xfield}
#     XF(XFields._lmult(az, f))
# end

