
# ComplexCircRings struct
# =======================================

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

function ComplexCircRings(nblks::Int, nside::Int, ::Type{M}, ::Type{N}) where {M<:AbstractMatrix, N<:AbstractMatrix}
    Γdb = M[M(undef, nside, nside) for ℓ ∈ 1:nblks]
    Cdb = N[N(undef, nside, nside) for ℓ ∈ 1:nblks]
    ComplexCircRings{M,N}(nblks, nside, Γdb, Cdb)
end 

# AdjointCircRings
# =======================================

struct AdjointCircRings{M<:AbstractMatrix, N<:AbstractMatrix} <: XFields.AbstractLinearOp
    az::ComplexCircRings{M,N}
end

function Base.adjoint(az::ComplexCircRings{M,N}) where {M,N}
    return AdjointCircRings{M,N}(az)
end



# Define left mult 
# ==================================

# ComplexCircRings,  Complex in map space
function XFields._lmult(az::ComplexCircRings, f::XF) where {TM, Ti<:Complex, To, XF<:Xfield{TM,Ti,To,2}}
    v  = fielddata(FourierField(f))
    w  = similar(v)
    wc = collect(eachcol(w))
    vc = collect(eachcol(v))
    J  = Spectra.Jop(az.nblks)
    Threads.@threads for ℓ = 1:J.n÷2+1
        Jℓ = J(ℓ)
        vcℓ, vcJℓ = vc[ℓ], vc[Jℓ]
        wc[ℓ]    .= az.Γdb[ℓ]  * vcℓ        .+ az.Cdb[ℓ]  * conj.(vcJℓ)
        wc[Jℓ]   .= az.Cdb[Jℓ] * conj.(vcℓ) .+ az.Γdb[Jℓ] * vcJℓ
    end
    Xfourier(fieldtransform(f),w)
end

# ComplexCircRings, Real in map space
function XFields._lmult(az::ComplexCircRings, f::XF) where {TM, Ti<:Real, To, XF<:Xfield{TM,Ti,To,2}}
    v  = fielddata(FourierField(f))
    w  = similar(v)
    wc = collect(eachcol(w))
    vc = collect(eachcol(v))
    J  = Spectra.Jop(az.nblks)
    Threads.@threads for ℓ = 1:J.n÷2+1
        Jℓ     = J(ℓ)
        vcℓ    = vc[ℓ] # note conj.(vcJℓ) = vcℓ
        wc[ℓ] .= az.Γdb[ℓ]  * vcℓ .+ az.Cdb[ℓ]  * conj.(vcℓ)
    end
    Xfourier(fieldtransform(f),w)
end

# ----------------------------

# AdjointCircRings, Complex in pixel space
function XFields._lmult(az::AdjointCircRings, f::XF) where {TM, Ti<:Complex, To, XF<:Xfield{TM,Ti,To,2}}
    v  = fielddata(FourierField(f))
    w  = similar(v)
    wc = collect(eachcol(w))
    vc = collect(eachcol(v))
    J  = Spectra.Jop(az.az.nblks)
    Threads.@threads for ℓ = 1:J.n÷2+1
        Jℓ = J(ℓ)
        vcℓ, vcJℓ  = vc[ℓ], vc[Jℓ]
        wc[ℓ]     .= az.Γdb[ℓ]' * vcℓ        .+ az.Cdb[Jℓ]' * conj.(vcJℓ)
        wc[Jℓ]    .= az.Cdb[ℓ]' * conj.(vcℓ) .+ az.Γdb[Jℓ]' * vcJℓ
    end
    Xfourier(fieldtransform(f),w)
end

# AdjointCircRings, Real in pixel space
function XFields._lmult(az::AdjointCircRings, f::XF) where {TM, Ti<:Real, To, XF<:Xfield{TM,Ti,To,2}}
    v  = fielddata(FourierField(f))
    w  = similar(v)
    wc = collect(eachcol(w))
    vc = collect(eachcol(v))
    J  = Spectra.Jop(az.az.nblks)
    Threads.@threads for ℓ = 1:J.n÷2+1
        Jℓ     = J(ℓ)
        vcℓ    = vc[ℓ]
        wc[ℓ] .= az.Γdb[ℓ]' * vcℓ .+ az.Cdb[Jℓ]' * vcℓ
    end
    Xfourier(fieldtransform(f),w)
end

# * calls out to _lmult
# ----------------------------

function Base.:*(az::ComplexCircRings, f::XF) where {XF<:Xfield}
    XF(XFields._lmult(az, f))
end

function Base.:*(az::AdjointCircRings, f::XF) where {XF<:Xfield}
    XF(XFields._lmult(az, f))
end




# Define left divide
# ==================================

# ComplexCircRings,  Complex in map space
function _ldiv(az::ComplexCircRings, f::XF) where {TM, Ti<:Complex, To, XF<:Xfield{TM,Ti,To,2}}
    v  = fielddata(FourierField(f))
    w  = similar(v)
    wc = collect(eachcol(w))
    vc = collect(eachcol(v))
    J  = Spectra.Jop(az.nblks)
    Threads.@threads for ℓ = 1:J.n÷2+1
        Jℓ = J(ℓ)
        vcℓ, v̄cJℓ = vc[ℓ], conj.(vc[Jℓ])
        Γℓ, Γ̄Jℓ, Cℓ, C̄Jℓ = az.Γdb[ℓ], conj.(az.Γdb[Jℓ]), az.Cdb[ℓ], conj.(az.Cdb[Jℓ]) 
        Γℓ⁻¹vcℓ = Γℓ \ vcℓ
        w̄cJℓ    = (Γ̄Jℓ - C̄Jℓ / Γℓ * Cℓ) \ (v̄cJℓ - C̄Jℓ * Γℓ⁻¹vcℓ)
        wc[ℓ]  .= Γℓ⁻¹vcℓ - Γℓ \ (Cℓ * w̄cJℓ) 
        wc[Jℓ] .= conj.(w̄cJℓ)
    end
    Xfourier(fieldtransform(f),w)
end

# ComplexCircRings,  Real in map space
function _ldiv(az::ComplexCircRings, f::XF) where {TM, Ti<:Real, To, XF<:Xfield{TM,Ti,To,2}}
    v  = fielddata(FourierField(f))
    w  = similar(v)
    wc = collect(eachcol(w))
    vc = collect(eachcol(v))
    J  = Spectra.Jop(az.nblks)
    Threads.@threads for ℓ = 1:J.n÷2+1
        Jℓ = J(ℓ)
        # note conj.(vcJℓ) = vcℓ
        vcℓ = vc[ℓ]
        Γℓ, Γ̄Jℓ, Cℓ, C̄Jℓ = az.Γdb[ℓ], conj.(az.Γdb[Jℓ]), az.Cdb[ℓ], conj.(az.Cdb[Jℓ]) 
        Γℓ⁻¹vcℓ = Γℓ \ vcℓ
        w̄cJℓ    = (Γ̄Jℓ - C̄Jℓ / Γℓ * Cℓ) \ (vcℓ - C̄Jℓ * Γℓ⁻¹vcℓ)
        wc[ℓ]  .= Γℓ⁻¹vcℓ - Γℓ \ (Cℓ * w̄cJℓ) 
    end
    Xfourier(fieldtransform(f),w)
end



function Base.:\(az::CMBrings.ComplexCircRings, f::XF) where {TM, Ti, To, XF<:Xfield{TM,Ti,To,2}}
    ## CMBrings.map_ring((fℓ, Σℓ) -> factorize(Hermitian(Σℓ)) \ fℓ, f, az)
    XF(_ldiv(az, f))
end

## TODO: need to add adjoint ldiv direct eval at some point
function Base.:\(az::CMBrings.AdjointCircRings, f::XF) where {TM, Ti, To, XF<:Xfield{TM,Ti,To,2}}
    @warn "at the moment div(azᴴ,f0 assumes az is symmetric"
    CMBrings.map_ring((fℓ, Σℓ) -> factorize(Hermitian(Σℓ)) \ fℓ, f, az.az)
end



# Test these methods 
# ===============================
# Note, the idea is that for CCR::ComplexCircRings
# CCR[ℓ] gives Ωℓ and to access the Γdb or Cdb one 
# does CCR.Γdb[ℓ] or CCR.Cdb[ℓ].


# map_ring for Complex pixel fields
function map_ring(fun::Function, f::XF, azs::ComplexCircRings...) where {TM, Ti<:Complex, To, XF<:Xfield{TM,Ti,To,2}}
    fk  = fielddata(FourierField(f))
    gk  = similar(fk)
    fℓ  = collect(eachcol(fk))
    gℓ  = collect(eachcol(gk))
    J   = Spectra.Jop(azs[1].nblks)
    Threads.@threads for ℓ = 1:J.n÷2+1
        fℓf̄Jℓ     = vcat(fℓ[ℓ], conj.(fℓ[J(ℓ)]))
        gℓḡJℓ     = fun(fℓf̄Jℓ, map(az->az[ℓ], azs)...)
        gℓ[ℓ]    .= gℓḡJℓ[1:end÷2]
        gℓ[J(ℓ)] .= conj.(gℓḡJℓ[end÷2+1:end])
    end 
    XF(Xfourier(fieldtransform(f), gk))
end

# map_ring for Real pixel fields
function map_ring(fun::Function, f::XF, azs::ComplexCircRings...) where {TM, Ti<:Real, To, XF<:Xfield{TM,Ti,To,2}}
    fk  = fielddata(FourierField(f))
    gk  = similar(fk)
    fℓ  = collect(eachcol(fk))
    gℓ  = collect(eachcol(gk))
    J   = Spectra.Jop(azs[1].nblks)
    Threads.@threads for ℓ = 1:J.n÷2+1
        # note conj.(vcJℓ) = vcℓ
        fℓf̄Jℓ  = vcat(fℓ[ℓ], fℓ[ℓ])
        gℓḡJℓ  = fun(fℓf̄Jℓ, map(az->az[ℓ], azs)...)
        gℓ[ℓ] .= gℓḡJℓ[1:end÷2]
    end 
    XF(Xfourier(fieldtransform(f), gk))
end


function map_ring(fun::Function, azs::ComplexCircRings...) 
    az_new = Base.similar(azs[1])
    map_ring!(az_new, fun, azs...)
end

function map_ring!(az_new::ComplexCircRings, fun::Function, azs::ComplexCircRings...)
    Threads.@threads for ℓ = 1:az_new.nblks÷2+1
        Ωℓ      = fun(map(az->az[ℓ], azs)...)
        az_new[ℓ] = Ωℓ  
    end 
    return az_new
end


function Base.similar(az::ComplexCircRings)
    Γdb  = map(similar, az.Γdb)
    Cdb  = map(similar, az.Cdb)
    return ComplexCircRings(Γdb, Cdb)
end

# Make ComplexCircRings an iterator
# =======================================


# ------- these on the other hand work with  
# Ωℓ = [ Γdb[ℓ]     Cdb[ℓ]
#        Cdb[Jℓ]^*  Γdb[Jℓ]^*  ]
# which multiplies  on the left var(z[ℓ], conj.(z[Jℓ]))

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
function Base.setindex!(az::AdjointCircRings{M,N}, Ωℓ::AbstractMatrix, ℓ::Int) where {M,N}
    Base.setindex!(az.az, Ωℓ, ℓ)
end


function Base.getindex(az::ComplexCircRings, ℓ::Int) 
    1 <= ℓ <= az.nblks || throw(BoundsError(az, ℓ))
    Jℓ = ℓ==1 ? 1 : az.nblks - ℓ + 2
    Ωℓ = [ az.Γdb[ℓ]          az.Cdb[ℓ]
           conj.(az.Cdb[Jℓ])  conj.(az.Γdb[Jℓ]) ]
    return Ωℓ
end
function Base.getindex(az::AdjointCircRings, ℓ::Int) 
    return Base.getindex(az, ℓ)' 
end


Base.length(az::ComplexCircRings) = az.nblks
Base.length(az::AdjointCircRings) = az.az.nblks

Base.eltype(::Type{ComplexCircRings{M,N}}) where {M,N} = Tuple{M,N} 
Base.eltype(::Type{AdjointCircRings{M,N}}) where {M,N} = Tuple{M,N} 

Base.firstindex(az::ComplexCircRings) = 1
Base.firstindex(az::AdjointCircRings) = 1

Base.lastindex(az::ComplexCircRings)  = az.nblks
Base.lastindex(az::AdjointCircRings)  = az.az.nblks



# TODO: do we really want different indexing method for iterate???
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

