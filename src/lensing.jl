# Specify how lensing will be generated ...

# 1. First Define some gradient methods
# 2. Specify how lensing lensing acts on the array storage fields


# A gradient abstract type which does boiler plate extensions of the 
# basic mutating gradient.
# ===============================================


# Pixels space increment gradients
# ======================================

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

function Pix1dFFTNabla!(∂θ, w::𝕎{Tf}) where Tf
    wφ = 𝕀(w.sz[1]) ⊗ 𝕎(Tf, w.sz[2:2], w.period[2:2])
    planW = plan(wφ)
    c_forFFTNabla = Tf(planW.scale_forward * planW.scale_inverse)

    ∇! = Pix1dFFTNabla!(
        ∂θ,
        planW, 
        im .* fullfreq(wφ)[2] .* c_forFFTNabla,
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



# Now we specify  the details of how lensing distributes to array storage fields 
# in a particular corrdinate.
# ===================================================

# 1. FieldLensing.flow_field(L::AbstractFlow, f::Field) = MapField(f)
# 2. FieldLensing.flow_data(L::AbstractFlow, ff::Field) = (fielddata(ff),)
# 3. FieldLensing.flow_reconstruct(L::AbstractFlow, ff::MF, ln_ffd::AbstractArray) 

# AbstractArrayLense = Union{FieldLensing.ArrayLense, FieldLensing.ArrayLenseᴴ}

# function FieldLensing.flow_data(L::AbstractArrayLense, ff::Xmap{<:CMBflat.QU2EB}) 
#     fd = fielddata(ff)
#     (fd[:,:,1], fd[:,:,2])
# end


