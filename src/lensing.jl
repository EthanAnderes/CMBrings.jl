# Specify how lensing will be generated ...


# 1. Specify how lensing lensing acts on the array storage fields
# 2. Define some gradient methods




# 1. Specify how lensing lensing acts on the array storage fields
# ===================================================

# 1. FieldLensing.flow_field(L::AbstractFlow, f::Field) = MapField(f)
# 2. FieldLensing.flow_data(L::AbstractFlow, ff::Field) = (fielddata(ff),)
# 3. FieldLensing.flow_reconstruct(L::AbstractFlow, ff::MF, ln_ffd::AbstractArray) 

AbstractArrayLense = Union{FieldLensing.ArrayLense, FieldLensing.ArrayLenseᴴ}

function FieldLensing.flow_data(L::AbstractArrayLense, ff::Xmap{<:Az𝕊2}) 
    fd = fielddata(ff)
    (fd[:,:,1], fd[:,:,2])
end


## for polarization fields stored as a complex field

function FieldLensing.flow_data(L::CMBrings.AbstractArrayLense, ff::Xmap{TM, TI, TO, d}) where {TM, TI<:Complex, TO, d}
    fd = fielddata(ff)
    (real.(fd), imag.(fd))
end


function FieldLensing.flow_reconstruct(L::AbstractFlow, ff::MF, ln_ffd::NTuple{2,<:AbstractArray}) where {n, TM, TI<:Complex, TO, d, MF<:Xfield{TM, TI, TO, d}}
    MF(fieldtransform(ff), complex.(ln_ffd[1], ln_ffd[2]))
end


# 2. Define some gradient methods
# ===============================================


# Sparse increments
# -----------------------------------------

struct Nabla!{Tθ,Tφ} <: FieldLensing.Gradient{2}
    ∂θ::Tθ
    ∂φᵀ::Tφ
end

function LinearAlgebra.adjoint(∇!::Nabla!)
    return Nabla!(
        ∇!.∂θ',
        ∇!.∂φᵀ',
    )
end

function (∇!::Nabla!{Tθ,Tφ})(des, y, ::Val{1}) where {Tθ,Tφ} 
    mul!(des, ∇!.∂θ, y)
end

function (∇!::Nabla!{Tθ,Tφ})(des, y, ::Val{2}) where {Tθ,Tφ}
    mul!(des, y, ∇!.∂φᵀ)
end 


# Same as PixFFTNabla but with 1-d gradients in second coordinates
# ---------------------------------------


struct Pix1dFFTNabla!{Tθ,TW,Tik,Tx} <: FieldLensing.Gradient{2}
    ∂θ::Tθ
    planW::TW
    ikφ::Tik
    sk::Tik
    sx::Tx
end

function LinearAlgebra.adjoint(∇!::Pix1dFFTNabla!{Tθ,TW,Tik,Tx}) where {Tθ,TW,Tik,Tx}
    return Pix1dFFTNabla!{Tθ,TW,Tik,Tx}(
        ∇!.∂θ',
        ∇!.planW, 
        .- ∇!.ikφ,
        similar(∇!.sk),
        similar(∇!.sx),
    )
end

function Pix1dFFTNabla!(∂θ, ::Type{Tf}, nφ, periodφ) where Tf
    wφ = FFTransforms.:⊗(FFTransforms.𝕀(size(∂θ,1)), FFTransforms.𝕎(Tf, nφ, periodφ))
    planW = plan(wφ)
    c_forFFTNabla = Tf(planW.scale_forward * planW.scale_inverse)

    ∇! = Pix1dFFTNabla!(
        ∂θ,
        planW, 
        im .* FFTransforms.fullfreq(wφ)[2] .* c_forFFTNabla,
        Array{eltype_out(wφ)}(undef,size_out(wφ)),
        Array{eltype_in(wφ)}(undef,size_in(wφ)),
    )

    return ∇!
end 

function (∇!::Pix1dFFTNabla!{Tθ,TW,Tik,Tx})(des, y, ::Val{1}) where {Tθ,TW,Tik,Tx}
    mul!(des, ∇!.∂θ, y)
end

function (∇!::Pix1dFFTNabla!{Tθ,TW,Tik,Tx})(des, y, ::Val{2}) where {Tθ,TW,Tik,Tx}
    @inbounds ∇!.sx .= y
    mul!(∇!.sk, ∇!.planW.unscaled_forward_transform, ∇!.sx)
    @inbounds ∇!.sk .*= ∇!.ikφ
    mul!(des, ∇!.planW.unscaled_inverse_transform, ∇!.sk)
end




# Fourier space increment gradients
# ---------------------------------------

struct FFTNabla!{TW,Tik,Tx} <: FieldLensing.Gradient{2}
    planW::TW
    ikθ::Tik
    ikφ::Tik
    sk::Tik
    sx::Tx
end

function LinearAlgebra.adjoint(∇!::FFTNabla!{TW,Tik,Tx}) where {TW,Tik,Tx}
    return FFTNabla!{TW,Tik,Tx}(
        ∇!.planW, 
        .- ∇!.ikθ,
        .- ∇!.ikφ,
        similar(∇!.sk),
        similar(∇!.sx),
    )
end

function FFTNabla!(::Type{Tf}, sz, period) where Tf
    w = 𝕎(Tf, sz, period)
    planW = plan(w)
    c_forFFTNabla = Tf(planW.scale_forward * planW.scale_inverse)

    ∇! = FFTNabla!(
        planW, 
        im .* fullfreq(w)[1] .* c_forFFTNabla, 
        im .* fullfreq(w)[2] .* c_forFFTNabla,
        Array{eltype_out(w)}(undef,size_out(w)),
        Array{eltype_in(w)}(undef,size_in(w)),
    )

    return ∇!
end 

function (∇!::FFTNabla!{TW,Tik,Tx})(des, y, ::Val{1}) where {TW,Tik,Tx}
    @inbounds ∇!.sx .= y
    mul!(∇!.sk, ∇!.planW.unscaled_forward_transform, ∇!.sx)
    @inbounds ∇!.sk .*= ∇!.ikθ
    mul!(des, ∇!.planW.unscaled_inverse_transform, ∇!.sk)
end

function (∇!::FFTNabla!{TW,Tik,Tx})(des, y, ::Val{2}) where {TW,Tik,Tx}
    @inbounds ∇!.sx .= y
    mul!(∇!.sk, ∇!.planW.unscaled_forward_transform, ∇!.sx)
    @inbounds ∇!.sk .*= ∇!.ikφ
    mul!(des, ∇!.planW.unscaled_inverse_transform, ∇!.sk)
end


