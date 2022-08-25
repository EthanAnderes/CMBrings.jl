

# Linear Algebra extensions general ring encoded as Xfield{<:𝕎} 
# =====================================

# TODO: check these ...

function LinearAlgebra.dot(f::Xfield{T},g::Xfield{T}) where T<:𝕎 
    real.(FT.sum_kbn(f[:] .* conj.(g[:])))
end

function LinearAlgebra.dot(f::Xfield{T},g::Xfield{T}) where T<:EAZ 
    real.(EZ.sum_kbn(f[:] .* conj.(g[:])))
end

