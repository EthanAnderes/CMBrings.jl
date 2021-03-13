


struct ComplexCircRings{TΓ <: AbstractMatrix, TC <: AbstractMatrix} <: XFields.AbstractLinearOp
    nblks::Int  # corresponds to nφ
    nside::Int  # corresponds to nθ
    Γdb::Vector{TΓ}
    Cdb::Vector{TC}
end

function ComplexCircRings(Γdb::Vector{TΓ}, Cdb::Vector{TC}) where {TΓ <: AbstractMatrix, TC <: AbstractMatrix}
    @assert length(Γdb) == length(Cdb)
    @assert size(Γdb[1],1) == size(Γdb[1],2) == size(Cdb[1],1) == size(Cdb[1],2)
    ComplexCircRings{TΓ,TC}(length(Γdb), size(Γdb[1],1), Γdb, Cdb)
end 


# Make ComplexCircRings an iterator
# 
# =======================================

function Base.getindex(az::ComplexCircRings, ℓ::Int) 
    1 <= ℓ <= az.nblks || throw(BoundsError(az, ℓ))
    Jℓ = ℓ==1 ? 1 : az.nblks - ℓ + 2
    Ωℓ = [ az.Γdb[ℓ]          az.Cdb[ℓ]
           conj.(az.Cdb[Jℓ])  conj.(az.Γdb[Jℓ]) ]
    return Ωℓ
end

# note that this sets at only the ℓ indicies ...
function Base.setindex!(az::ComplexCircRings{TΓ,TC}, ΓdbℓCdbℓ::Tuple{TΓ, TC}, ℓ::Int) where {TΓ,TC}
    1 <= ℓ <= az.nblks || throw(BoundsError(az, ℓ))  
    az.Γdb[ℓ] .= ΓdbℓCdbℓ[1]
    az.Cdb[ℓ] .= ΓdbℓCdbℓ[2]
    return nothing
end

# note that this sets at indices ℓ *and* Jℓ
function Base.setindex!(az::ComplexCircRings{TΓ,TC}, Ωℓ::AbstractMatrix, ℓ::Int)
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

# Note, this returns (Γdbℓ, Cdbℓ)

Base.iterate(az::ComplexCircRings) = ((az.Γdb[1], az.Cdb[1]), 1)

function Base.iterate(az::ComplexCircRings, state) 
    if state > az.nblks
        return nothing
    else
        return ((az.Γdb[state+1], az.Cdb[state+1]), state+1)
    end
end

Base.length(az::ComplexCircRings) = az.nblks

Base.eltype(::Type{ComplexCircRings{TΓ,TC}}) where {TΓ,TC} = Tuple{TΓ, TC} 

Base.firstindex(az::ComplexCircRings) = 1

Base.lastindex(az::ComplexCircRings)  = az.nblks



# Interface methods for Abstract linear ops
# Matrix operations which propigate to the blocks 
# =======================================

# actually anyfunction 
for op ∈ (:adjoint, :inv, :sqrt)
    quote
        function LinearAlgebra.$op(az::ComplexCircRings{TΓ,TC}) where {TΓ,TC}
            Σ′ = $op.(az.Σ)
            ComplexCircRings{eltype(Σ′)}(az.nblks, Σ′)
        end
    end |> eval
end


# Interface methods for Abstract linear ops
# Mult and div on the left of fields
# =======================================

## TODO: specify the conditions for XF<:Xfield to make it applicable...

## multiply 

function Base.:*(az::ComplexCircRings, f::XF) where {XF<:Xfield}
    XF(XFields._lmult(az, f))
end

## _lmult for ComplexCircRings on data storage 2 dim

function XFields._lmult(az::ComplexCircRings{TΓ,TC}, f::XF) where {M<:AbstractMatrix, TM, Ti, To, XF<:Xfield{TM,Ti,To,2}}
    v  = fielddata(FourierField(f))
    w  = similar(v)
    Threads.@threads for i ∈ axes(v, 2)
        mul!(view(w,:, i), az[i], view(v,:, i))
    end
    Xfourier(fieldtransform(f),w)
end

## div for ComplexCircRings on data storage 2 dim

function Base.:\(az::ComplexCircRings{TΓ,TC}, f::XF)  where {M<:AbstractMatrix, TM, Ti, To, XF<:Xfield{TM,Ti,To,2}}
    v  = fielddata(FourierField(f))
    w  = similar(v)
    Threads.@threads for i ∈ axes(v, 2)
        ldiv!(view(w,:, i), factorize(az[i]), view(v,:, i))
    end
    Xfourier(fieldtransform(f),w)
end


# misc
# =======================================

function LinearAlgebra.pinv(M::Eigen)
    invM = deepcopy(M)
    invM.values .= pinv.(M.values)
    invM
end

check_factorization(az::ComplexCircRings) = all(map(issuccess, az))

function az_sim(tmU::Transform, az::ComplexCircRings{TΓ,TC}) where {M<:Eigen}
    vx  = randn(eltype_in(tmU), size_in(tmU))
    v   = Xmap(tmU, vx)[!]
    w   = similar(v)
    wk = collect(eachcol(w))
    vk = collect(eachcol(v))
    Threads.@threads for i ∈ eachindex(wk)
        vk[i] .*= sqrt.(az[i].values)
        mul!(wk[i], az[i].vectors, vk[i])
    end
    Xfourier(tmU, w) 
end


function az_sim(tmU::Transform, az::ComplexCircRings{TΓ,TC}) where {M<:Cholesky}
    wx  = randn(eltype_in(tmU), size_in(tmU))
    wk  = Xmap(tmU, wx)[!]
    wkc = collect(eachcol(wk))
    Threads.@threads for i ∈ eachindex(wkc)
        lmul!(az[i].L, wkc[i])
    end
    Xfourier(tmU, wk) 
end


function az_sim(tmU::Transform, az::ComplexCircRings{TΓ,TC}) where {M<:AbstractMatrix}
    wx  = randn(eltype_in(tmU), size_in(tmU))
    wk  = Xmap(tmU, wx)[!]
    wkc = collect(eachcol(wk))
    Threads.@threads for i ∈ eachindex(wkc)
        lmul!(cholesky(az[i], Val(false)).L, wkc[i])
    end
    Xfourier(tmU, wk) 
end



# function az_sim(tmU::Transform, az::ComplexCircRings{TΓ,TC}) where {M<:Cholesky}
#     wx = randn(eltype_in(tmU), size_in(tmU))
#     wk = Xmap(tmU, wx)[!]
#     for (Σ, wkc) ∈ zip(az, eachcol(wk)) 
#         lmul!(Σ.L, wkc)
#     end
#     Xfourier(tmU, wk) 
# end


# function az_sim(tmU::Transform, az::ComplexCircRings{TΓ,TC}) where {M<:AbstractMatrix}
#     wx = randn(eltype_in(tmU), size_in(tmU))
#     wk = Xmap(tmU, wx)[!]
#     for (Σ, wkc) ∈ zip(az, eachcol(wk)) 
#         lmul!(cholesky(Σ, Val(false)).L, wkc)
#     end
#     Xfourier(tmU, wk) 
# end

