# # Construct three new transforms Spin0, Spin2 and Spin02
# # ======================================================
# 
# abstract type RingSpinTransform{Tf<:FFTR} <: Transform{Tf,2} end
# 
# struct RingS2Transform{Tf<:FFTR} <: RingSpinTransform{Tf}
#     szθ::Int    
#     szφ::Int    
#     θ::Vector{Tf}
#     function RingS2Transform{Tf}(szθ::Int, szφ::Int, θ::AbstractVector{Tp}) where {Tf<:FFTR, Tp<:Real}
#         @assert length(θ) == szθ
#         new{Tf}(szθ, szφ, θ)
#     end
# end 
# 
# # Now define the required methods to hook into XFields as a storage container
# # ===========================================================================
# 
# # note: the storage stacks Qx and Ux on top of eachother ... hence the 2*szθ below
# @inline XFields.size_in(rT::RingS2Transform)  = (2rT.szθ, rT.szφ)
# @inline XFields.size_out(rT::RingS2Transform) = (2rT.szθ, rT.szφ÷2 + 1)
# 
# XFields.eltype_in(rT::RingSpinTransform{Tf}) where {Tf<:FFTR} = Tf
# XFields.eltype_out(rT::RingSpinTransform{Tf}) where {Tf<:FFTR} = Complex{Tf}
# 
# function XFields.plan(rT::RingS2Transform{Tf}) where {Tf<:FFTR} 
#     w =  𝕀(2rT.szθ) ⊗ 𝕎(Tf, rT.szφ, 2π)
#     return plan(unitary_scale(w) * w)
# end
# 
# # extras
# # ===============================
# 
# FFTransforms.𝕎(rT::RingS2Transform{Tf}) where {Tf} = 𝕀(2rT.szθ) ⊗ 𝕎(Tf, rT.szφ, 2π)
# 
# 𝕎1field(rT::RingSpinTransform{Tf}) where {Tf} =  𝕀(rT.szθ) ⊗ 𝕎(Tf, rT.szφ, 2π)
# 
# 𝕎1d(rT::RingSpinTransform{Tf}) where {Tf} = 𝕎(Tf, rT.szφ, 2π)
# 
# function RingS2Transform(::Type{Tf}, θ::AbstractVector, φ::AbstractVector) where {Tf<:FFTR} 
#     return RingS2Transform{Tf}(length(θ), length(φ), θ)
# end
# 
