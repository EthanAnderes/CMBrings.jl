# =====================

function quasi_bandpowers(
        f; 
        θ=pix(fieldtransform(f))[1], 
        Δℓsph_bin = 15
    )
    tm   = fieldtransform(f)
    k    = freq(tm)[2]
    ℓsph = k' ./ sin.(θ)

    ℓsph_bin∂  = 0:Δℓsph_bin:(maximum(ℓsph)+1)
    ℓsph_bin_mid = ℓsph_bin∂[1:end-1] .+ Δℓsph_bin ./ 2

    power_ℓsph_bin_mid = zeros(length(ℓsph_bin_mid))

    raw_power = abs2.(f[!])  

    for i in eachindex(power_ℓsph_bin_mid)
        ll = ℓsph_bin∂[i]
        lr = ℓsph_bin∂[i+1]
        idx  = ll .<= ℓsph .< lr
        nidx = sum(idx)
        power_ℓsph_bin_mid[i] = nidx > 0 ? sum(raw_power[idx]) / nidx : 0.0
    end
    ℓsph_bin_mid, power_ℓsph_bin_mid
end



function quasi_bandpowers(
        f, g; 
        θ=pix(fieldtransform(f))[1], 
        Δℓsph_bin = 15
    ) 
    tm   = fieldtransform(f)
    k    = freq(tm)[2]
    ℓsph = k' ./ sin.(θ)

    ℓsph_bin∂  = 0:Δℓsph_bin:(maximum(ℓsph)+1)
    ℓsph_bin_mid = ℓsph_bin∂[1:end-1] .+ Δℓsph_bin ./ 2

    raw_power = f[!] .* conj.(g[!])  
    power_ℓsph_bin_mid = zeros(eltype(raw_power), length(ℓsph_bin_mid))

    for i in eachindex(power_ℓsph_bin_mid)
        ll = ℓsph_bin∂[i]
        lr = ℓsph_bin∂[i+1]
        idx  = ll .<= ℓsph .< lr
        nidx = sum(idx)
        power_ℓsph_bin_mid[i] = nidx > 0 ? sum(raw_power[idx]) / nidx : 0.0
    end
    ℓsph_bin_mid, power_ℓsph_bin_mid
end




# methods for kernel conv for Cl -> expected quasi-bandpowers
# ================================================

# the λlm_cache comes from 
# AssociatedLegendrePolynomials.jl

# perhaps ClassicalOrthogonalPolynomials.jl will work as well

function ifind2(m::Int,l₀::Int,l₁::Int, θ_vector) 
    aml₀ =  abs(m//l₀)
    aml₁ =  abs(m//l₁)
    if (aml₀ > 1) | (aml₁ > 1)
        return fill(false, length(θ_vector))
    else 
        return asin(aml₁) .<= (π .- θ_vector) .<= asin(aml₀)
    end 
end

function Ξ(l₀, l₁, ls_max, θ_vector, s, λlm_cache) 
    rtn = map(0:ls_max) do l
        ms       = - min(l,l₀,l₁) : min(l,l₀,l₁)
        sum( sum(abs2, index_λlm.(l, m, s; λlm_cache)[ifind2(m, l₀, l₁, θ_vector)] ) for m in ms)
    end
    # rtn_norm = map(1:ls_max) do l
    #     ms       = - min(l,l₀,l₁) : min(l,l₀,l₁)
    #     sum( sum(ifind2(m, l₀, l₁, θ_vector)) for m in ms)
    # end

    # rtn ./ maximum(rtn)
    # XFields.nan2zero.(rtn ./ rtn_norm)
    XFields.nan2zero.(rtn)
end


# λlm_cache  = λlm(0:lmax, 0:mmax, cos.(θ))
function index_λlm(l, m::Int, s; λlm_cache=nothing)
    lmax, mmax = size(λlm_cache,2)-1, size(λlm_cache,3)-1
    abs_m = abs(m)
    @assert abs_m <= mmax
    @assert all(abs_m .<= l .<= lmax)
    @assert s in (-2,0, 2)

    m_s_mult = m < 0 ? (-1)^(m+s) : 1
    sign_flip = m < 0 ? -1 : 1

    ## the symmetry needed, when m < 0, is: ₛλ_l^m = (-1)^(s+m) ₋ₛλ_l^(-m)
    if s == 0
        rtn  = λlm_cache[:, l.+1, abs_m + 1] 
        rtn *= m_s_mult
    elseif s in (-2, 2)
        rtn   = λlm_cache[:, l.+1, abs_m + 1] .* αθlm₊₋₂.(θ, (l.+1)', m * sign_flip, s * sign_flip)
        rtn .+= λlm_cache[:, l,    abs_m + 1] .* βθlm₊₋₂.(θ, l',      m * sign_flip, s * sign_flip)
        rtn  *= m_s_mult
    end 

    return rtn    
end


function αθlm₊₋₂(θ::T, l::Int, m::Int,  s::Int) where T <: Real
    @assert s in (-2,2)
    if (abs(m) > l) | (l < 2)
        return zero(T)
    end
    snθ, ctθ = sin(θ), cot(θ)
    t1 =  (2m^2-l*(l+1)) / snθ^2
    t2 = - sign(s)*2m*(l-1) * ctθ / snθ
    t3 =  (l*(l-1)) * ctθ^2
    dn = √(l+2) * √(l+1) * √(l) * √(l-1)
    return (t1 + t2 + t3) / dn 
end 

function βθlm₊₋₂(θ::T, l::Int, m::Int,  s::Int) where T <: Real
    @assert s in (-2,2)
    if (abs(m) > l) | (l < 2)
        return zero(T)
    end
    snθ, ctθ = sin(θ), cot(θ)
    m1 = 2 * √((l^2-m^2) * (2l+1) / (2l-1))
    m2 = sign(s) * m / snθ^2 + ctθ / snθ
    dn = √(l+2) * √(l+1) * √(l) * √(l-1)
    return m1 * m2 / dn 
end 

