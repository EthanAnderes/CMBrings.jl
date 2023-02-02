# Right ascension RA <-> Azimuth φ ∈ [0,2π)

φ2RA(φ)   = rad2deg(CC.in_negπ_π(φ))
RA2φ(ra)  = CC.in_0_2π(deg2rad(ra))

# Declination Dec <-> Polar θ ∈ [0,π]

θ2Dec(θ)   = 90 - rad2deg(θ)
Dec2θ(dec) = deg2rad(90 - dec)

# Radian <-> Arcmin

rad2arcmin(x)      = 60*rad2deg(x) 
arcmin2rad(x)      = deg2rad(x/60) 

# Azimuth frequency m <-> Hz[deg/sec]

"""
`m2hz(m; scanᵒ_sec = 1) -> hz`
where `scanᵒ_sec` corresponds to scan_speed[deg/sec] and 
`m` corresponds to Azimuthal frequency
"""
m2hz(m; scanᵒ_sec = 1)  = m * scanᵒ_sec / 360

"""
`hz2m(hz; scanᵒ_sec = 1) -> m`
where `scanᵒ_sec` corresponds to scan_speed[deg/sec] and 
`m` corresponds to Azimuthal frequency
"""
hz2m(hz; scanᵒ_sec = 1) = hz * 360 / scanᵒ_sec

# Azimuth frequency m <-> ell

"""m2ℓ(m; θ)"""
m2ℓ(m; θ) = m / sin(θ)

"""ℓ2m(ℓ; θ)"""
ℓ2m(ℓ; θ) = ℓ * sin(θ)

#  σpix <-> μKarcmin <-> μKrad

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

fwhmrad2σ²(rad)    = rad^2 / 8 / log(2)


   