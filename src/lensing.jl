# Specify how lensing will be generated ...


# 1. Specify how lensing lensing acts on the array storage fields
# 2. Define some gradient methods


# 1. Specify how lensing lensing acts on the array storage fields
# ===================================================

# 1. FieldLensing.flow_field(L::AbstractFlow, f::Field) = MapField(f)
# 2. FieldLensing.flow_data(L::AbstractFlow, ff::Field) = (fielddata(ff),)
# 3. FieldLensing.flow_reconstruct(L::AbstractFlow, ff::MF, ln_ffd::AbstractArray) 

AbstractArrayLense = Union{FieldLensing.ArrayLense, FieldLensing.ArrayLenseᴴ}

## for polarization fields stored as a complex field

function FieldLensing.flow_data(L::AbstractArrayLense, ff::Xmap{TM, TI, TO, d}) where {TM, TI<:Complex, TO, d}
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
    # wφ = FT.:⊗(FT.𝕀(size(∂θ,1)), FT.𝕎(Tf, nφ, periodφ))
    wφ = 𝕀(size(∂θ,1)) ⊗ 𝕎(Tf, nφ, periodφ)
    planW = plan(wφ)
    c_forFFTNabla = Tf(planW.scale_forward * planW.scale_inverse)

    ∇! = Pix1dFFTNabla!(
        ∂θ,
        planW, 
        im .* FT.fullfreq(wφ)[2] .* c_forFFTNabla,
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







# hastily put together generic lensing constructors
# that use sparse increments for gradients
# ===============================================



function generate_∇!∇!ϕ(θ::Vector{Tf}, φ::Vector{Tf}; uniformΔθ=true) where {Tf}

    if uniformΔθ
        Δθ = θ[2]-θ[1]
        ∂θ′ = spdiagm(
                -2 => fill( 1,length(θ)-2),
                -1 => fill(-8,length(θ)-1),
                 1 => fill( 8,length(θ)-1),
                 2 => fill(-1,length(θ)-2),
                )
        ∂θ′[1,end]   =  -8
        ∂θ′[1,end-1] =  1
        ∂θ′[2,end]   =  1
        ∂θ′[end,1]   =  8
        ∂θ′[end,2]   = -1
        ∂θ′[end-1,1] = -1
        ∂θ = (1 / (12Δθ)) * ∂θ′
    else
        Δθ = vcat(diff(θ), θ[end]-θ[end-1])
        ∂θ′ = spdiagm(
                0 => fill(-1,length(θ)), 
                1 => fill(1,length(θ)-1),
            )
        ∂θ′[end,1] =  1
        ∂θ = spdiagm(1 ./ Δθ) * ∂θ′
    end


    Δφ = φ[2] - φ[1]
    ∂φ  = spdiagm(
            -2 => fill( 1,length(φ)-2),
            -1 => fill(-8,length(φ)-1),
             1 => fill( 8,length(φ)-1),
             2 => fill(-1,length(φ)-2),
            )
    ∂φ[1,end]   =  -8
    ∂φ[1,end-1] =  1
    ∂φ[2,end]   =  1
    ∂φ[end,1]   =  8
    ∂φ[end,2]   =  -1
    ∂φ[end-1,1] =  -1
    ∂φᵀ = transpose((1 / (12Δφ)) * ∂φ)
    ## -------- or -------
    ## ∂φ  = spdiagm(
    ##     0 => fill(-1,length(φ)), 
    ##     1 => fill(1,length(φ)-1)
    ## )
    ## ∂φ[end,1] =  1
    ## ∂φᵀ = transpose(Tf(1 / (Δφ)) * ∂φ)

    ∇!   = Nabla!((∂θ - ∂θ')/2, (∂φᵀ - ∂φᵀ')/2)
    ∇!_ϕ = Nabla!(∂θ, ∂φᵀ)

    ## ∇!   = Nabla!(Matrix((∂θ - ∂θ')/2), Matrix((∂φᵀ - ∂φᵀ')/2))
    ## ∇!_ϕ = Nabla!(Matrix(∂θ), Matrix(∂φᵀ))

    ## ∇!   = Pix1dFFTNabla!((∂θ - ∂θ')/2, Tf, length(φ), Tf(2π))
    ## ∇!_ϕ = Pix1dFFTNabla!(∂θ, Tf, length(φ), Tf(2π))

    return ∇!, ∇!_ϕ
end  


function generate_lense(;
        θ, mv1x=1, mv2x=1, 
        ∇!,  ∇!_ϕ, ## subidx, sub_∇!, 
        nsteps_lensing=14
        ) 

    ## ∇!_ϕ used in ϕ2v! and ϕ2vᴴ!
    ## ∇! used in Ł
    
    sin⁻²θ = @. csc(θ)^2 
    mvx₁ = ones(size(θ)) .* mv1x
    mvx₂ = sin⁻²θ .* mv2x

    ϕ2v! = function (v::NTuple{2,Array}, ϕ::Array)
        ∇!_ϕ(v, ϕ)
        v[1] .*= mvx₁
        v[2] .*= mvx₂
        v
    end 

    ϕ2vᴴ! = function (ϕ::Array, v::NTuple{2,Array})
        mv = (similar(v[1]), similar(v[2]))
        ∇!_ϕ'(mv, (mvx₁.*v[1], mvx₂.*v[2]) )
        ϕ .= mv[1] .+ mv[2]
        ϕ 
    end 

    Ł = function (ϕ_az::Xfield)
        ϕ = ϕ_az[:]
        v = (similar(ϕ), similar(ϕ))
        ϕ2v!(v,ϕ)
        FieldLensing.ArrayLense(v, ∇!, 0, 1, nsteps_lensing)
    end

    Ł, ϕ2v!, ϕ2vᴴ!, ∇!
end



