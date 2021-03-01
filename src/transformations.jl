

# Linear Algebra extensions general ring encoded as Xfield{<:𝕎} 
# =====================================


function LinearAlgebra.dot(f::Xfield{FT},g::Xfield{FT}) where FT<:𝕎 
    FFTransforms.sum_kbn(f[:].*g[:])
end



# Transform types Az𝕊0 and Az𝕊2 that are rings 
# but hold an extra 𝕊0 or 𝕊2 transform to allow 
# operations on the embedded sphere. 
# ==================================================================

struct Az𝕊0{Tf<:Number, Ts<:Number, Tp<:Number, C<:CartesianIndices} <: XFields.Transform{Tf,2} 
    tmAz::𝕎{Tf, 2, Ts, Tp}
    tm𝕊::𝕊0
    ringidx::C     
    function Az𝕊0(tmAz::𝕎{Tf, 2, Ts, Tp}, tm𝕊::𝕊0, ringidx::C) where {Tf, Ts, Tp, C}
        nθAz, nφAz = size_in(tmAz)
        nθ𝕊, nφ𝕊   = size_in(tm𝕊)
        @assert nθAz <= nθ𝕊
        @assert nφAz == nφ𝕊
        @assert isodd(nφ𝕊)
        @assert size(ringidx) == (nθAz, nφAz)
        ## ensure the transformation is unitary
        tmAz′ = unscale(tmAz) |> tm -> unitary_scale(tm)*tm
        rTf = real(Tf) 
        new{Tf, rTf, Tp, C}(tmAz′, tm𝕊, ringidx)
    end 
end 

@inline XFields.size_in(tm::Az𝕊0)   = XFields.size_in(tm.tmAz)
@inline XFields.size_out(tm::Az𝕊0)  = XFields.size_out(tm.tmAz)
@inline XFields.eltype_in(tm::Az𝕊0{Tf})  where {Tf}       = Tf
@inline XFields.eltype_out(tm::Az𝕊0{Tf}) where {Tf<:Real} = Complex{Tf}
@inline XFields.eltype_out(tm::Az𝕊0{Tf}) where {Tf<:Complex} = Tf
@inline XFields.plan(tm::Az𝕊0) = XFields.plan(tm.tmAz) 

struct Az𝕊2{Tf<:Number, Ts<:Number, Tp<:Number, C<:CartesianIndices} <: XFields.Transform{Tf,3} 
    tmAz::𝕎{Tf, 3, Ts, Tp}
    tm𝕊::𝕊2
    ringidx::C     
    function Az𝕊2(tmAz::𝕎{Tf, 2, Ts, Tp}, tm𝕊::𝕊2, ringidx::C) where {Tf, Ts, Tp, C}
        nθAz, nφAz, = size_in(tmAz)
        nθ𝕊, nφ𝕊,   = size_in(tm𝕊)
        @assert nθAz <= nθ𝕊
        @assert nφAz == nφ𝕊
        @assert isodd(nφ𝕊)
        @assert size(ringidx) == (nθAz, nφAz,2)
        ## ensure the transformation is unitary
        tmAz′ = unscale(tmAz) |> tm -> unitary_scale(tm)*tm 
        rTf = real(Tf) 
        new{Tf, rTf, Tp, C}(tmAz′, tm𝕊, ringidx)
    end 
end 

@inline XFields.size_in(tm::Az𝕊2)   = XFields.size_in(tm.tmAz)
@inline XFields.size_out(tm::Az𝕊2)  = XFields.size_out(tm.tmAz)
@inline XFields.eltype_in(tm::Az𝕊2{Tf})  where {Tf}       = Tf
@inline XFields.eltype_out(tm::Az𝕊2{Tf}) where {Tf<:Real} = Complex{Tf}
@inline XFields.eltype_out(tm::Az𝕊2{Tf}) where {Tf<:Complex} = Tf
@inline XFields.plan(tm::Az𝕊2) = XFields.plan(tm.tmAz) 

function SphereTransforms.Ωpix(tm::Union{Az𝕊0,Az𝕊2})
	SphereTransforms.Ωpix(tm.tm𝕊)[tm.ringidx[:,1,1]]
end

function SphereTransforms.pix(tm::Union{Az𝕊0,Az𝕊2})
    θ, φ = SphereTransforms.pix(tm.tm𝕊)
    return θ[tm.ringidx[:,1,1]], φ
end

# extras ========

function XFields.Xmap(tm::Az𝕊2{Tf}, x1, x2) where {Tf}
    mat = zeros(Tf, size_in(tm))
    mat[:,:,1] .= x1
    mat[:,:,2] .= x2
    return Xmap(tm, mat)
end

XFields.Xmap(tm::Az𝕊2, x::AbstractMatrix) = Xmap(tm, x, x)

function XFields.Xfourier(tm::Az𝕊2{Tf}, x1, x2) where {Tf}
    mat = zeros(Complex{Tf},size_out(tm))
    mat[:,:,1] .= x1
    mat[:,:,2] .= x2
    return Xfourier(tm, mat)
end

XFields.Xfourier(tm::Az𝕊2, x::AbstractMatrix) = Xfourier(tm, x, x)

function Base.getindex(f::Xfield{<:Az𝕊2}, sym::Symbol)
    (sym == :Qx) ? fielddata(MapField(f))[:,:,1] :
    (sym == :Ux) ? fielddata(MapField(f))[:,:,2] :
    (sym == :Qk) ? fielddata(FourierField(f))[:,:,1] :
    (sym == :Uk) ? fielddata(FourierField(f))[:,:,2] :
    error("index is not defined")
end


function LinearAlgebra.dot(f::Xfield{TM}, g::Xfield{TM}) where TM<:Union{Az𝕊0,Az𝕊2}
    FFTransforms.sum_kbn(f[:].*g[:])
end


# No need to teach AzBlocks how to multiply and divide Xfield{<:Az𝕊} ...
# it is already part of how AzBlocks operate on any Xfield

# We do however need to teach DiagOp{Xfields{<:𝕊0 or S2}} how to operate on 
# f::Xfield{<:Az𝕊}. The operation is defined by extending MapField(f) to pixels on the 
# full sphere, at which point DiagOp{Xfields{<:𝕊0 or S2}} activates, then 
# the map pixels are and again contracted back to the ring. 

function XFields._lmult(O::DiagOp{D1}, f::Xfield{T2}) where {T1<:Union{𝕊0,𝕊2}, T2<:Union{Az𝕊0,Az𝕊2}, D1<:Xfield{T1}}
    tmAzS, tmS = fieldtransform(f), fieldtransform(O.f)
    ## only allow 𝕊0 & Az𝕊0 or 𝕊2 & Az𝕊2
    @assert length(size_in(tmAzS)) == length(size_in(tmS))
    
    fx_on𝕊  = zeros(eltype_in(tmS), size_in(tmS))
    fx_on𝕊[tmAzS.ringidx] .= f[:]
    fmap_on𝕊 = O * Xmap(tmS, fx_on𝕊)
    
    return Xmap(tmAzS, fmap_on𝕊[:][tmAzS.ringidx])
end

function Base.:*(O::DiagOp{D1}, f::XT2) where {T1<:Union{𝕊0,𝕊2}, T2<:Union{Az𝕊0,Az𝕊2}, D1<:Xfield{T1}, XT2<:Xfield{T2}}
    return XT2(XFields._lmult(O, f))
end


function Base.:\(O::DiagOp{D1}, f::Xfield{T2}) where {T1<:Union{𝕊0,𝕊2}, T2<:Union{Az𝕊0,Az𝕊2}, D1<:Xfield{T1}}
    return inv(O) * f
end


# Simulation ======


# function simmap(Cl::DiagOp{Fi}) where {Fi<:Xfourier} 
#     tm  = fieldtransform(Cl.f)
#     √Cl * Xmap(tm, FFTransforms.randn_in(tm))
# end

# function simfourier(Cl::DiagOp{Fi}) where {Fi<:Xfourier} 
#     tm  = fieldtransform(Cl.f)
#     #√Cl * Xfourier(tm, FFTransforms.randn_out(tm))
#     # We need the following instead since we don't have randn for fft yet
#     Xfourier(simmap(Cl))
# end 
 
# function flatnoisemap(μK′n::Number, tm::Union{𝕎,QU𝕊2ring}) 
#     (μK′n * π / 60 / 180) * Xmap(tm, FFTransforms.randn_in(tm))
# end 

# function flatnoisefourier(μK′n::Number, tm::Union{𝕎,QU𝕊2ring}) 
#     # We need the following instead since we don't have randn for fft yet
#     Xfourier(flatnoisemap(μK′n, tm)) 
# end

