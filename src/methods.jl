

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



# For simulating and working with healpix and fasttransforms
# =====================================

function ℍlm2𝕊lm(hlm, h0, s0)
    ls0′, ms0′ = SphereTransforms.lm(s0)
    lh0,  mh0  = HealpixTransforms.lm(h0)
    bew =  findall(ls0′ .<= maximum(lh0))
    ls0 = ls0′[bew]
    ms0 = ms0′[bew]

    idx_into_h0 = HealpixTransforms.lm2index(ls0, abs.(ms0), h0)
    mp = findall(ms0 .>= 0)
    mn = findall(ms0 .< 0)
    mp_bew = findall((ms0′ .>= 0) .& (ls0′ .<= maximum(lh0)))
    mn_bew = findall((ms0′ .< 0) .& (ls0′ .<= maximum(lh0)))


    slm = fill(0.0, size_out(s0))
    slm[mp_bew] = (-1).^(ms0[mp]) .* real.(hlm[idx_into_h0[mp]])
    slm[mn_bew] = (-1).^(ms0[mn].+1) .* imag.(hlm[idx_into_h0[mn]])
    slm 
end

function whitefourier(trn::𝕊)
    wlm  = SphereTransforms.white_fourier(trn)
    return Xfourier(trn, wlm)
end

function whitemap(trn::𝕊)
    wx   = SphereTransforms.white_map(trn) 
    return Xmap(trn, wx)
end

function whitefourier(trn::ℍ0)
    wlm    = randn(eltype_out(trn),size_out(trn))
    m      = HealpixTransforms.lm(trn)[2]
    m0Bool = findall(m .== 0)
    wlm[m0Bool] = randn(real(eltype_out(trn)), size(m0Bool))
    Xfourier(trn, wlm)
end

function whitemap(trn::ℍ0)
    wx  = randn(eltype_in(trn),size_in(trn)) ./ sqrt(HealpixTransforms.Ωpix(trn))
    Xmap(trn, wx)
end

flatnoisemap(μK′n::Number, trn)     = (μK′n * π / 60 / 180) * whitemap(trn)

flatnoisefourier(μK′n::Number, trn) = (μK′n * π / 60 / 180) * whitefourier(trn)

simfourier(Cl::DiagOp{<:Xfourier}) = √Cl * whitefourier(fieldtransform(Cl.f))



# LinearAlgebra Overload 
# ================================================

LinearAlgebra.adjoint(C::DiagOp) = C  


function LinearAlgebra.dot(f::Xmap{FT},g::Xmap{FT}) where FT<:𝕊
    trn  = fieldtransform(f)
    sqrtΩ = sqrt.(SphereTransforms.Ωpix(trn))
    return  dot(f[:].*sqrtΩ, g[:].*sqrtΩ)
end

function LinearAlgebra.dot(f::Xfield{FT},g::Xfield{FT}) where FT<:𝕊
    dot(f[!], g[!])
end

function LinearAlgebra.dot(f::Xfourier{FT},g::Xfourier{FT}) where FT<:ℍ0
    trn = fieldtransform(f)
    return HealpixTransforms.Ωpix(trn) * dot(f[:],g[:])
end

function LinearAlgebra.dot(f::Xfield{FT},g::Xfield{FT}) where FT<:ℍ0
    trn = fieldtransform(f)
    l,m   = HealpixTransforms.lm(trn)
    fl = f[!]
    gl = g[!]
    flm = fl[.!(m .== 0)]
    glm = gl[.!(m .== 0)]    
    fl0 = fl[m .== 0]
    gl0 = gl[m .== 0]
    rtn  = dot(real.(flm),real.(glm)) * 2  
    rtn += dot(imag.(flm),imag.(glm)) * 2
    rtn += dot(real.(fl0),real.(gl0)) 
    return rtn
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

