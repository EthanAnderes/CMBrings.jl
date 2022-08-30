# TOD like filters on the rows of EAZ fourier

# Mp:  construct point source pixel mask from pnt src list 
# ==========================================

function pix_point_src_mask(tm0::EAZ, point_src_file_) 

    hole_map = ones(size_in(tm0)) 

    point_src_list = readdlm(point_src_file_, '\t', skipstart=22)
    point_src_φ  = RA2φ.(point_src_list[:,2])
    point_src_θ  = Dec2θ.(point_src_list[:,3])
    point_src_Δβ = deg2rad.(point_src_list[:,4] / 60)

    θ, φ     = EZ.pix(tm0)
    for (θ₀, φ₀, Δβ₀) in zip(point_src_θ, point_src_φ, point_src_Δβ)
        radius_hole = Δβ₀
        radius_ramp = deg2rad(4/60) # Δβ₀/2 #!!! ad hoc choice here !!!
        radius_tot  = (radius_hole + radius_ramp) * 1.1 # expand this a bit

    
        θ_idx_cut = findall(@. abs(θ - θ₀) <= radius_tot)
        φ_idx_cut = findall(@. min(CC.counterclock_Δφ(φ₀, φ), CC.counterclock_Δφ(φ, φ₀)) <= (radius_tot/sin(θ₀+radius_tot)))
   
        if !isempty(θ_idx_cut) & !isempty(φ_idx_cut)
            θ_val_cut = θ[θ_idx_cut]
            φ_val_cut = φ[φ_idx_cut]

            hole_map_mini = hole_punch.(θ_val_cut, φ_val_cut'; θ₀, φ₀, radius_hole, radius_ramp)
            # @assert all(hole_map_mini[1,:]   .== 1)
            # @assert all(hole_map_mini[end,:] .== 1)
            # @assert all(hole_map_mini[:,end] .== 1)
            # @assert all(hole_map_mini[:,1]   .== 1)
            hole_map[θ_idx_cut, φ_idx_cut] .*= hole_map_mini
        end 
    end

    DiagOp(Xmap(tm0, hole_map))
end


θ2Dec(θ)   = 90 - rad2deg(θ)
Dec2θ(dec) = deg2rad(90 - dec)

φ2RA(φ)   = rad2deg(CC.in_negπ_π(φ))
RA2φ(ra)  = CC.in_0_2π(deg2rad(ra))

function hole_punch(θ, φ; θ₀, φ₀, radius_hole, radius_ramp)
    β      = CC.geoβ(θ, θ₀, φ, φ₀)
    r1, r2 = radius_hole, radius_hole + radius_ramp
    Δr     = r2 - r1

    if β <= r1 
        return zero(β)
    elseif r1 < β <= r2
        return (1+cospi((β-r2)/Δr))/2
    else
        return one(β)
    end
end

