

# # Specialized constructors for Xmap and Xfourier with SpinT
# # ---------------------------------------------------------------

# import XFields: Xmap, Xfourier

# # Xmap(rT, <comma separated 2d maps or numbers>)

# function Xmap(rT::RingS2Transform{Tf}, x1, x2) where {Tf}
#     mat = zeros(Tf,size_in(rT))
#     mat[1:rT.szθ,:]       .= x1
#     mat[(rT.szθ+1):end,:] .= x2
#     return Xmap(rT, mat)
# end

# # Xforier(rT, <comma separated 2d maps or numbers>)

# function Xfourier(rT::RingS2Transform{Tf}, x1, x2) where {Tf}
#     mat = zeros(Complex{Tf},size_out(rT))
#     mat[1:rT.szθ,:]       .= x1
#     mat[(rT.szθ+1):end,:] .= x2
#     return Xfourier(rT, mat)
# end

# # add custom getindex. 
# # ---------------------------------------------------------------

# import Base: getindex

# function Base.getindex(f::Xfield{<:RingS2Transform}, sym::Symbol)
#     rT = fieldtransform(f)
#     (sym == :Qx) ? fielddata(MapField(f))[1:rT.szθ,:] :
#     (sym == :Ux) ? fielddata(MapField(f))[(rT.szθ+1):end,:] :
#     (sym == :Ql) ? fielddata(FourierField(f))[1:rT.szθ,:] :
#     (sym == :Ul) ? fielddata(FourierField(f))[(rT.szθ+1):end,:] :
#     error("index is not defined")
# end
