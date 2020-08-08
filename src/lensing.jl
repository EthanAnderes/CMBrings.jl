# methods build on top of FieldLensing.jl


# Lensing 
# ======================================


struct Nabla!{Tθ,Tφ}
    ∂θ::Tθ
    ∂φᵀ::Tφ
end

function (∇!::Nabla!{Tθ,Tφ})(∇y::NTuple{2,A}, y::NTuple{2,B}) where {Tθ,Tφ,Tf,A<:AbstractMatrix{Tf}, B<:AbstractMatrix{Tf}}
    mul!(∇y[1], ∇!.∂θ, y[1])
    mul!(∇y[2], y[2], ∇!.∂φᵀ)
    ∇y
end
function (∇!::Nabla!{Tθ,Tφ})(y::NTuple{2,B}) where {Tθ,Tφ,Tf,B<:AbstractMatrix{Tf}}
    ∇y = (similar(y[1]), similar(y[2]))
    ∇!(∇y, (y[1],y[2]))
    ∇y
end


function (∇!::Nabla!{Tθ,Tφ})(∇y::NTuple{2,A}, y::B) where {Tθ,Tφ,Tf,A<:AbstractMatrix{Tf}, B<:AbstractMatrix{Tf}}
    ∇!(∇y, (y,y))
end
function (∇!::Nabla!{Tθ,Tφ})(y::B) where {Tθ,Tφ,Tf,B<:AbstractMatrix{Tf}}
    ∇y = (similar(y), similar(y))
    ∇!(∇y, (y,y))
    ∇y
end




