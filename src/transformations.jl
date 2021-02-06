


# CMBsphere hooking into SphereTransforms
# =================================================


# Simulation
# =====================================

# Extend this to AzBlocks once it is complete ...

# function simfourier(Cl::DiagOp{Fi}) where {Tm<:Abstract𝕊, Fi<:Xfourier{Tm}} 
#     tm  = fieldtransform(Cl.f)
#     wlm = SphereTransforms.randn_out(tm)
#     √Cl * Xfourier(tm, wlm)
# end 

# # function simmap(Cl::DiagOp{<:Xfourier}) 
# #     tm  = fieldtransform(Cl.f)
# #     wx = SphereTransforms.randn_in(tm)
# #     Xmap(√Cl * Xmap(tm, wx))
# # end 
# # ------ or here we make sure to bandpass the high frequency
# function simmap(Cl::DiagOp{Fi}) where {Tm<:Abstract𝕊, Fi<:Xfourier{Tm}} 
#     tm    = fieldtransform(Cl.f)
#     l,m,a = SphereTransforms.lma(tm)
#     A     = DiagOp(Xfourier(tm,a))
#     Xmap(simfourier(A*Cl)) 
# end

# # Fix me: Decide if your going to band limit high frequency
# function flatnoisemap(μK′n::Number, tm::Abstract𝕊) 
#     (μK′n * π / 60 / 180) * Xmap(tm, SphereTransforms.randn_in(tm))
# end 

# function flatnoisefourier(μK′n::Number, tm::Abstract𝕊) 
#     (μK′n * π / 60 / 180) * Xfourier(tm, SphereTransforms.randn_out(tm))
# end


# Linear Algebra extensions
# =====================================



function LinearAlgebra.pinv(M::Eigen)
    invM = deepcopy(M)
    invM.values .= pinv.(M.values)
    invM
end


function LinearAlgebra.dot(f::Xfield{FT},g::Xfield{FT}) where FT<:𝕎 
    FFTransforms.sum_kbn(f[:].*g[:])
end


# getindex and XFields stuff
# =====================================


# function XFields.Xmap(tm::𝕊2, x1, x2)
#     mat = zeros(eltype_in(tm),size_in(tm))
#     mat[:,:,1] .= x1
#     mat[:,:,2] .= x2
#     return Xmap(tm, mat)
# end

# function XFields.Xfourier(tm::𝕊2, x1, x2)
#     mat = zeros(eltype_out(tm),size_out(tm))
#     mat[:,:,1] .= x1
#     mat[:,:,2] .= x2
#     return Xfourier(tm, mat)
# end

# XFields.Xmap(tm::𝕊2, x::AbstractMatrix) = Xmap(tm, x, x)

# XFields.Xfourier(tm::𝕊2, x::AbstractMatrix) = Xfourier(tm, x, x)

# function Base.getindex(f::Xfield{<:𝕊2}, sym::Symbol)
#     if sym == :Qx
#         return fielddata(MapField(f))[:,:,1]
#     elseif sym == :Ux
#         return fielddata(MapField(f))[:,:,2]
#     elseif sym == :El 
#         return fielddata(FourierField(f))[:,:,1]
#     elseif sym == :Bl 
#         return fielddata(FourierField(f))[:,:,2]
#     elseif sym == :Ql
#         qx     = fielddata(MapField(f))[:,:,1]
#         tmS2  = fieldtransform(f)
#         tmS0c  = 𝕊(0, tmS2.nθ, tmS2.nφ)
#         return plan(tmS0c) * complex.(qx)
#     elseif sym == :Ul 
#         ux     = fielddata(MapField(f))[:,:,2]
#         tmS2  = fieldtransform(f)
#         tmS0c  = 𝕊(0, tmS2.nθ, tmS2.nφ)
#         return plan(tmS0c) * complex.(ux)
#     elseif sym == :Ex
#         elm   = fielddata(FourierField(f))[:,:,1]
#         tmS2  = fieldtransform(f)
#         tmS0c  = 𝕊(0, tmS2.nθ, tmS2.nφ)
#         return real.(plan(tmS0c) \ spin_s_to_0(elm; spin=2))
#     elseif sym == :Bx 
#         blm   = fielddata(FourierField(f))[:,:,2]
#         tmS2  = fieldtransform(f)
#         tmS0c  = 𝕊(0, tmS2.nθ, tmS2.nφ)
#         return real.(plan(tmS0c) \ spin_s_to_0(blm; spin=2))
#     else
#         error("index is not defined")
#     end
# end


