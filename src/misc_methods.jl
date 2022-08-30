



# ================================================

rad2arcmin(x)      = 60*rad2deg(x) 
arcmin2rad(x)      = deg2rad(x/60) 

"""
`σpix(μk′, Ωrad²) -> σ`
μKarcmin noise level of `μk′` and a pixel of area ` Ωrad²` in square-radians, 
this function returns the correct pix standard deviation. 
"""
σpix(μk′, Ωrad²)   = arcmin2rad(μk′) / √(Ωrad²)

"""
`μKarcmin(σpix, Ωrad²) -> μk′`
Input is σpix (a pixel standard deviation) and Ωrad² (the corresponding pixel area in radians square).
Output is the corresponding `μKarcmin`. 
"""
μKarcmin(σpix, Ωrad²)   = rad2arcmin(σpix*√(Ωrad²))

"""
`μKrad(σpix, Ωrad²) -> μkrad`
Input is σpix (a pixel standard deviation) and Ωrad² (the corresponding pixel area in radians square).
Output is the corresponding `μKrad`. 
"""
μKrad(σpix, Ωrad²) = σpix*√(Ωrad²)


# Pixel space non-stationary beams
# ====================================

fwhmrad2σ²(rad)    = rad^2 / 8 / log(2)
    
function B̃eam1(θ₁, θ₂, σ²θ₁, σ²θ₂, Δφ)
    sinθ₁, cosθ₁ = sincos(θ₁)
    sinθ₂, cosθ₂ = sincos(θ₂)
    sinΔφ, cosΔφ = sincos(Δφ)
    Δx = sinθ₁ * cosθ₂ * cosΔφ - sinθ₂ * cosθ₁
    Δy = sinθ₁ * sinΔφ
    σ²θ₁θ₂ = (σ²θ₁ + σ²θ₂ ) / 2
    return exp( - (Δx^2 + Δy^2) / σ²θ₁θ₂ / 2 ) / σ²θ₁θ₂ / 2 / π
end 

function B̃eam2(θ₁, θ₂, σ²θ₁, σ²θ₂, Δφ)
    sinθ₁ = sin(θ₁)
    sinθ₂ = sin(θ₂)
    sinΔθ = sin((θ₁-θ₂)/2)
    sinΔφ = sin(Δφ/2)
    σ²θ₁θ₂ = (σ²θ₁ + σ²θ₂ ) / 2
    return exp( - 2 * (sinΔθ^2 + sinθ₁*sinθ₂*sinΔφ^2) / σ²θ₁θ₂) / σ²θ₁θ₂ / 2 / π
end 




# One dimensional smooth mask
# ====================================

function cosφ°Mask(φ°::T; lb, rb, Δl, Δr=Δl) where T<:Number
    ϕ° = rad2deg(CC.in_negπ_π(deg2rad(φ°)))
    l1, l2, r1, r2 = lb, lb+Δl, rb-Δr, rb
    @assert -180 <= l1 <= l2 <= r1 <= r2 ≤ 180

    if abs(Δl) <= 0
        return T(1)
    end

    if l1 < ϕ° < l2
        return (1+cospi((ϕ°-l2)/Δl))/2
    elseif r1 < ϕ° < r2
        return (1+cospi((ϕ°-r1)/Δr))/2
    elseif l2 <= ϕ° <= r1
        return T(1)
    else
        return T(0)
    end 
end


function pixweight(x::T; ▮l, ▯l, ▯r, ▮r) where T<:Number
    @assert ▮l ≤ ▯l ≤ ▯r ≤ ▮r
    if ▯l ≤ x ≤ ▯r
        return one(T)
    elseif (x ≤ ▮l) | (▮r ≤ x)
        return zero(T)
    elseif ▮l < x < ▯l
        return T((1-cos(π*(x-▮l)/(▯l-▮l))) / 2)
    else 
        @assert ▯r < x < ▮r 
        return T((1+cos(π*(x-▯r)/(▮r-▯r))) / 2)
    end
end



   