

# Linear Algebra extensions general ring encoded as Xfield{<:𝕎} 
# =====================================


function LinearAlgebra.dot(f::Xfield{T},g::Xfield{T}) where T<:𝕎 
    real.(FFTransforms.sum_kbn(f[:] .* conj.(g[:])))
end

