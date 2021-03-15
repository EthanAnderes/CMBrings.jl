


struct ComplexCircRings{M<:AbstractMatrix, N<:AbstractMatrix} <: XFields.AbstractLinearOp
    nblks::Int  # corresponds to nφ
    nside::Int  # corresponds to nθ
    Γdb::Vector{M}
    Cdb::Vector{N}
end

function ComplexCircRings(Γdb::Vector{M}, Cdb::Vector{N}) where {M<:AbstractMatrix, N<:AbstractMatrix}
    @assert length(Γdb) == length(Cdb)
    @assert size(Γdb[1],1) == size(Γdb[1],2) == size(Cdb[1],1) == size(Cdb[1],2)
    ComplexCircRings{M,N}(length(Γdb), size(Γdb[1],1), Γdb, Cdb)
end 


# Make ComplexCircRings an iterator
# =======================================

# TODO: do we really want different indexing method for iterate???
# I'm considering removing it and working exclusively with 
# az[ℓ] -> Ωℓ
# ------- the following work with ((Γdb[ℓ], Cdb[ℓ]): ℓ=1:nblks)

Base.iterate(az::ComplexCircRings) = ((az.Γdb[1], az.Cdb[1]), 1)

function Base.iterate(az::ComplexCircRings, state) 
    if state < az.nblks
        return ((az.Γdb[state+1], az.Cdb[state+1]), state+1)
    else
        return nothing
    end
end

function Base.setindex!(az::ComplexCircRings{M,N}, ΓdbℓCdbℓ::Tuple{M,N}, ℓ::Int) where {M,N}
    1 <= ℓ <= az.nblks || throw(BoundsError(az, ℓ))  
    az.Γdb[ℓ] .= ΓdbℓCdbℓ[1]
    az.Cdb[ℓ] .= ΓdbℓCdbℓ[2]
    return nothing
end


# ------- these on the other hand work with  
# Ωℓ = [ Γdb[ℓ]     Cdb[ℓ]
#        Cdb[Jℓ]^*  Γdb[Jℓ]^*  ]

# note that this sets at indices ℓ *and* Jℓ
function Base.setindex!(az::ComplexCircRings{M,N}, Ωℓ::AbstractMatrix, ℓ::Int) where {M,N}
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


Base.length(az::ComplexCircRings) = az.nblks

Base.eltype(::Type{ComplexCircRings{M,N}}) where {M,N} = Tuple{M,N} 

Base.firstindex(az::ComplexCircRings) = 1

Base.lastindex(az::ComplexCircRings)  = az.nblks


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

function XFields._lmult(az::ComplexCircRings, f::XF) where {TM, Ti, To, XF<:Xfield{TM,Ti,To,2}}
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






# Test these methods 
# ===============================
# Note, the idea is that for CCR::ComplexCircRings
# CCR[ℓ] gives Ωℓ and to access the Γdb or Cdb one 
# does CCR.Γdb[ℓ] or CCR.Cdb[ℓ].


function Base.similar(az::ComplexCircRings)
    Γdb  = map(similar, az.Γdb)
    Cdb  = map(similar, az.Cdb)
    return ComplexCircRings(Γdb, Cdb)
end


function map_ring(fun::Function, az::ComplexCircRings, f::XF) where {TM, Ti, To, XF<:Xfield{TM,Ti,To,2}}
    fk  = fielddata(FourierField(f))
    gk  = similar(fk)
    fℓ  = collect(eachcol(fk))
    gℓ  = collect(eachcol(gk))
    J   = Spectra.Jop(az.nblks)
    Threads.@threads for ℓ = 1:J.n
        Ωℓ     = fun(az[ℓ]) 
        gℓ[ℓ] .= @view(Ωℓ[1:end÷2,:]) * vcat(fℓ[ℓ], conj.(fℓ[J(ℓ)]))
    end 
    Xfourier(fieldtransform(f), gk)
end;


function mod_ring!(az_new::ComplexCircRings, fun::Function, az::ComplexCircRings)

    Threads.@threads for ℓ = 1:az.blks÷2+1
        Ωℓ      = fun(az[ℓ])
        az_new[ℓ] = Ωℓ  
    end 

    return az_new
end

mod_ring!(fun::Function, az::ComplexCircRings) = mod_ring!(az, fun, az)


## function LinearAlgebra.mul!(C::CCR, A::CCR, B::CCR, α::Number, β::Number) where {CCR<:CMBrings.ComplexCircRings}
## 
##     Threads.@threads for ℓ = 1:C.nblks÷2+1
##         C[ℓ] = A[ℓ] * B[ℓ] * α + β * C[ℓ]
##     end 
## 
##     return C
## end
