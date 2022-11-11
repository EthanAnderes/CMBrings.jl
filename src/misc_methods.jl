
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

fwhmrad2σ²(rad)    = rad^2 / 8 / log(2)


   