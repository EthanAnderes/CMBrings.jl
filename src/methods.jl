

# custom pcg with function composition (Minv * A \approx I)
# ---------------------------------------------------------
function pcg(Minv::Function, A::Function, b, x=0*b; nsteps::Int=75, rel_tol = 0)
    r       = b - A(x)
    z       = Minv(r)
    p       = deepcopy(z)
    res     = dot(r,z)
    reshist = Vector{typeof(res)}()
    for i = 1:nsteps
        Ap        = A(p)
        α         = res / dot(p,Ap)
        x         = x + α * p
        r         = r - α * Ap
        z         = Minv(r)
        res′      = dot(r,z)
        p         = z + (res′ / res) * p
        rel_error = XFields.nan2zero(sqrt(dot(r,r)/dot(b,b)))
        if rel_error < rel_tol
            return x, reshist
        end
        push!(reshist, rel_error)
        res = res′
    end
    return x, reshist
end





# # Bandpowers
# # ----------
# function power(f::Xfield{ST}, g::Xfield{ST}; bin::Int=2, kmax=Inf, mult=1) where {ST<:RingSpinTransform}
#     sT     = fieldtransform(f)
#     k      = wavenum(sT)
#     Δk     = Δfreq(sT)
#     pwr    = @. mult * real(f[!] * conj(g[!]) + conj(f[!]) * g[!]) / 2
#     k_left = 0
#     nfld   = size(pwr,3) 
#     while k_left < min(kmax, maximum(k))
#         k_right = k_left + bin
#         indx    = findall(k_left .< k .<= k_right)
#         for i=1:nfld
#         	pwr[indx,i] .= mean( pwr[indx,i])
#         end
#         k_left  = k_right
#     end
#     return pwr
# end

# function power(f::Xfield{ST}; bin::Int=2, kmax=Inf, mult=1) where {ST<:RingSpinTransform}
#     power(f, f; bin=bin, kmax=kmax, mult=mult)
# end


# function LinearAlgebra.dot(f::Xfield{ST},g::Xfield{ST}) where ST<:RingSpinTransform
#     return dot(f[:],g[:]) * Ωx(fieldtransform(f))
# end

### possibly use this for constructors with Δx in place of periods
# function _get_npd(;nᵢ, pᵢ=nothing, Δxᵢ=nothing)
#     @assert !(isnothing(pᵢ) & isnothing(Δxᵢ)) "either pᵢ or Δxᵢ needs to be specified (note: pᵢ = Δxᵢ .* nᵢ)"
#     d = length(nᵢ)
#     if isnothing(pᵢ)
#         @assert d == length(Δxᵢ) "Δxᵢ and nᵢ need to be tuples of the same length"
#         pᵢ = tuple((prod(xn) for xn in zip(Δxᵢ,nᵢ))...)
#     end
#     @assert d == length(pᵢ) "pᵢ and nᵢ need to be tuples of the same length"
#     nᵢ, pᵢ, d
# end

