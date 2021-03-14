


struct ComplexCircRings{M <: AbstractMatrix} <: XFields.AbstractLinearOp
    nblks::Int  # corresponds to nφ
    nside::Int  # corresponds to nθ
    Γdb::Vector{M}
    Cdb::Vector{M}
end

function ComplexCircRings(Γdb::Vector{M}, Cdb::Vector{M}) where {M <: AbstractMatrix}
    @assert length(Γdb) == length(Cdb)
    @assert size(Γdb[1],1) == size(Γdb[1],2) == size(Cdb[1],1) == size(Cdb[1],2)
    ComplexCircRings{M}(length(Γdb), size(Γdb[1],1), Γdb, Cdb)
end 


# Make ComplexCircRings an iterator
# =======================================

# ------- the following work with ((Γdb[ℓ], Cdb[ℓ]): ℓ=1:nblks)

Base.iterate(az::ComplexCircRings) = ((az.Γdb[1], az.Cdb[1]), 1)

function Base.iterate(az::ComplexCircRings, state) 
    if state < az.nblks
        return ((az.Γdb[state+1], az.Cdb[state+1]), state+1)
    else
        return nothing
    end
end

Base.length(az::ComplexCircRings) = az.nblks

Base.eltype(::Type{ComplexCircRings{M}}) where {M} = Tuple{M,M} 

Base.firstindex(az::ComplexCircRings) = 1

Base.lastindex(az::ComplexCircRings)  = az.nblks

function Base.setindex!(az::ComplexCircRings{M}, ΓdbℓCdbℓ::Tuple{M,M}, ℓ::Int) where {M}
    1 <= ℓ <= az.nblks || throw(BoundsError(az, ℓ))  
    az.Γdb[ℓ] .= ΓdbℓCdbℓ[1]
    az.Cdb[ℓ] .= ΓdbℓCdbℓ[2]
    return nothing
end

# ------- these on the other hand work with  
# Ωℓ = [ Γdb[ℓ]     Cdb[ℓ]
#        Cdb[Jℓ]^*  Γdb[Jℓ]^*  ]

# note that this sets at indices ℓ *and* Jℓ
function Base.setindex!(az::ComplexCircRings{M}, Ωℓ::AbstractMatrix, ℓ::Int) where {M}
    1 <= ℓ <= az.nblks || throw(BoundsError(az, ℓ))
    @assert size(Ωℓ,1) == size(Ωℓ,2) == 2az.nside
    Jℓ = ℓ==1 ? 1 : az.nblks - ℓ + 2
    nθ = az.nside
    az.Γdb[ℓ]  .= Ωℓ[1:nθ, 1:nθ] 
    az.Cdb[ℓ]  .= Ωℓ[1:nθ, nθ+1:end]
    az.Cdb[Jℓ] .= conj.(Ωℓ[nθ+1:end, 1:nθ])
    az.Γdb[Jℓ] .= conj.(Ωℓ[nθ+1:end, nθ+1:end]) 
    return nothing
end

function Base.getindex(az::ComplexCircRings, ℓ::Int) 
    1 <= ℓ <= az.nblks || throw(BoundsError(az, ℓ))
    Jℓ = ℓ==1 ? 1 : az.nblks - ℓ + 2
    Ωℓ = [ az.Γdb[ℓ]          az.Cdb[ℓ]
           conj.(az.Cdb[Jℓ])  conj.(az.Γdb[Jℓ]) ]
    return Ωℓ
end


# Interface methods for Abstract linear ops
# Matrix operations which propigate to the blocks 
# =======================================

##  actually anyfunction 
##  for op ∈ (:adjoint, :inv, :sqrt)
##      quote
##          function LinearAlgebra.$op(az::ComplexCircRings{M}) where {M}
##              Σ′ = $op.(az.Σ)
##              ComplexCircRings{eltype(Σ′)}(az.nblks, Σ′)
##          end
##      end |> eval
##  end


# Interface methods for Abstract linear ops
# Mult and div on the left of fields
# =======================================

## TODO: specify the conditions for XF<:Xfield to make it applicable...

## multiply 

function Base.:*(az::ComplexCircRings, f::XF) where {XF<:Xfield}
    XF(XFields._lmult(az, f))
end

## _lmult for ComplexCircRings on data storage 2 dim

function XFields._lmult(az::ComplexCircRings{M}, f::XF) where {M<:AbstractMatrix, TM, Ti, To, XF<:Xfield{TM,Ti,To,2}}
    v  = fielddata(FourierField(f))
    w  = similar(v)
    wc = collect(eachcol(w))
    vc = collect(eachcol(v))
    J  = Spectra.Jop(length(az))
    Threads.@threads for i ∈ axes(v, 2)
        ## mul!(wc[i], az.Γdb[i], vc[i])
        ## mul!(wc[i], az.Cdb[i], conj.(vc[J(i)]), true, true)
        wc[i]  .= az.Γdb[i] * vc[i] .+ az.Cdb[i] * conj.(vc[J(i)])
    end
    Xfourier(fieldtransform(f),w)
end

## div for ComplexCircRings on data storage 2 dim

## function Base.:\(az::ComplexCircRings{M}, f::XF)  where {M<:AbstractMatrix, TM, Ti, To, XF<:Xfield{TM,Ti,To,2}}
##     v  = fielddata(FourierField(f))
##     w  = similar(v)
##     Threads.@threads for i ∈ axes(v, 2)
##         ldiv!(view(w,:, i), factorize(az[i]), view(v,:, i))
##     end
##     Xfourier(fieldtransform(f),w)
## end

