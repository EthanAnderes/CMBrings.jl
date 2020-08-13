

const MatrixOrFactorization{T} = Union{Factorization{T}, AbstractMatrix{T}}

struct AzBlock{M <: MatrixOrFactorization}
    nblks::Int 
    Σ::Vector{M}
end

function AzBlock(Σ::Vector{M}) where {M <: MatrixOrFactorization}
    AzBlock{M}(length(Σ),Σ)
end 


function AzBlock(f::Function, cov_θ1θ2Δφ::Function, θ, φ, tmW)
    nblks = length(φ)÷2+1
    @assert nblks     == size_out(tmW)[2]
    @assert length(φ) == size_in(tmW)[2]

    Ti    = eltype_out(tmW)
    szblk = (length(θ), length(θ))
    azΣ   = Matrix{Ti}[zeros(Ti, szblk) for k = 1:nblks]
    Σ_θi_θjcolon_φcolon = zeros(eltype_in(tmW),size_in(tmW))
    Σ_θi_θjcolon_kcolon = zeros(eltype_out(tmW),size_out(tmW))

    plan_tmW = plan(tmW)

    for i=1:length(θ)
        for j=1:length(θ)
            Σ_θi_θjcolon_φcolon[j,:] = cov_θ1θ2Δφ(θ[i], θ[j], φ .- φ[1])
        end
        mul!(Σ_θi_θjcolon_kcolon, plan_tmW, Σ_θi_θjcolon_φcolon)
        for k=1:nblks
            azΣ[k][i,:] = Σ_θi_θjcolon_kcolon[:,k]
        end
    end

    AzBlock(nblks, map(f, azΣ, 1:nblks))
end




# Make AzBlock an iterator
# =======================================

function Base.iterate(az::AzBlock, state=1) 
    if state > az.nblks
        return nothing
    else 
        return (az.Σ[state], state+1)
    end
end

Base.length(az::AzBlock) = az.nblks

Base.eltype(::Type{AzBlock{M}}) where {M} = M 

function Base.getindex(az::AzBlock, i::Int) 
    1 <= i <= az.nblks || throw(BoundsError(az, i))
    return az.Σ[i]
end

function Base.setindex!(az::AzBlock{M}, m::M, i::Int) where {M}
    1 <= i <= az.nblks || throw(BoundsError(az, i))  
    az.Σ[i] = m
end

Base.firstindex(az::AzBlock) = 1

Base.lastindex(az::AzBlock)  = az.nblks


# Matrix operations which propigate to the blocks 
# =======================================

for op ∈ (:adjoint, :transpose, :inv)
    quote
        function LinearAlgebra.$op(az::AzBlock{M}) where {M}
            Σ′ = $op.(az.Σ)
            AzBlock{eltype(Σ′)}(az.nblks, Σ′)
        end
    end |> eval
end


# Mult and div on the left of fields
# =======================================



function Base.:*(az::AzBlock{M}, f::XF) where {M<:AbstractMatrix, XF<:Xfield}
    v  = f[!]
    w  = similar(v)
    wk = collect(eachcol(w))
    vk = collect(eachcol(v))
    Threads.@threads for i ∈ eachindex(vk)
        mul!(wk[i], az[i], vk[i])
    end
    XF(Xfourier(fieldtransform(f),w))
end

function Base.:*(az::AzBlock{M}, f::XF) where {M<:Factorization, XF<:Xfield}
    v  = f[!]
    w  = similar(v)
    wk = collect(eachcol(w))
    vk = collect(eachcol(v))
    Threads.@threads for i ∈ eachindex(vk)
        mul!(wk[i], Matrix(az[i]), vk[i])
    end
    XF(Xfourier(fieldtransform(f),w))
end


function Base.:\(az::AzBlock{M}, f::XF)  where {M<:AbstractMatrix, XF<:Xfield}
    v  = f[!]
    w  = similar(v)
    wk = collect(eachcol(w))
    vk = collect(eachcol(v))
    Threads.@threads for i ∈ eachindex(vk)
        ldiv!(wk[i], factorize(az[i]), vk[i])
    end
    XF(Xfourier(fieldtransform(f),w))
end


function Base.:\(az::AzBlock{M}, f::XF)  where {M<:Factorization, XF<:Xfield}
    v  = f[!]
    w  = similar(v)
    wk = collect(eachcol(w))
    vk = collect(eachcol(v))
    Threads.@threads for i ∈ eachindex(vk)
        ldiv!(wk[i], az[i], vk[i])
    end
    XF(Xfourier(fieldtransform(f),w))
end



# non threaded versions 

# function Base.:*(az::AzBlock{M}, f::XF) where {M<:AbstractMatrix, XF<:Xfield}
#     v  = f[!]
#     w  = similar(v)
#     for (wk, Σk, vk) in zip(eachcol(w), az, eachcol(v))
#         mul!(wk, Σk, vk)
#     end
#     XF(Xfourier(fieldtransform(f),w))
# end

# function Base.:*(az::AzBlock{M}, f::XF) where {M<:Factorization, XF<:Xfield}
#     v  = f[!]
#     w  = similar(v)
#     for (wk, Σk, vk) in zip(eachcol(w), az, eachcol(v))
#         mul!(wk, Matrix(Σk), vk)
#     end
#     XF(Xfourier(fieldtransform(f),w))
# end


# function Base.:\(az::AzBlock{M}, f::XF)  where {M<:AbstractMatrix, XF<:Xfield}
#     v  = f[!]
#     w  = similar(v)
#     for (wk, Σk, vk) in zip(eachcol(w), az, eachcol(v))
#         ldiv!(wk,factorize(Σk),vk)
#     end
#     XF(Xfourier(fieldtransform(f),w))
# end


# function Base.:\(az::AzBlock{M}, f::XF)  where {M<:Factorization, XF<:Xfield}
#     v  = f[!]
#     w  = similar(v)
#     for (wk, Σk, vk) in zip(eachcol(w), az, eachcol(v))
#         ldiv!(wk,Σk,vk)
#     end
#     XF(Xfourier(fieldtransform(f),w))
# end



# misc
# =======================================

check_factorization(az::AzBlock) = all(map(issuccess, az))


function az_sim(tmU::Transform, az::AzBlock{M}) where {M<:Cholesky}
    wx  = randn(eltype_in(tmU), size_in(tmU))
    wk  = Xmap(tmU, wx)[!]
    wkc = collect(eachcol(wk))
    Threads.@threads for i ∈ eachindex(wkc)
        lmul!(az[i].L, wkc[i])
    end
    Xfourier(tmU, wk) 
end


function az_sim(tmU::Transform, az::AzBlock{M}) where {M<:AbstractMatrix}
    wx  = randn(eltype_in(tmU), size_in(tmU))
    wk  = Xmap(tmU, wx)[!]
    wkc = collect(eachcol(wk))
    Threads.@threads for i ∈ eachindex(wkc)
        lmul!(cholesky(az[i], Val(false)).L, wkc[i])
    end
    Xfourier(tmU, wk) 
end



# function az_sim(tmU::Transform, az::AzBlock{M}) where {M<:Cholesky}
#     wx = randn(eltype_in(tmU), size_in(tmU))
#     wk = Xmap(tmU, wx)[!]
#     for (Σ, wkc) ∈ zip(az, eachcol(wk)) 
#         lmul!(Σ.L, wkc)
#     end
#     Xfourier(tmU, wk) 
# end


# function az_sim(tmU::Transform, az::AzBlock{M}) where {M<:AbstractMatrix}
#     wx = randn(eltype_in(tmU), size_in(tmU))
#     wk = Xmap(tmU, wx)[!]
#     for (Σ, wkc) ∈ zip(az, eachcol(wk)) 
#         lmul!(cholesky(Σ, Val(false)).L, wkc)
#     end
#     Xfourier(tmU, wk) 
# end

