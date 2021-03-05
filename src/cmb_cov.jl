# For computing pixel space covariance

# import Dierckx 
# using ApproxFun: Fun, Jacobi

## Intended usage example:
## 
## β              =  geoβ.(θ1, θgd, φ1, φgd) 
## covPP̄β, covPPβ = covP(β)   
## 
## covPP̄   = covPP̄β .* multPP̄.(θ1, θgd, φ1, φgd) 
## covPP   = covPPβ .* multPP.(θ1, θgd, φ1, φgd)
## covQ1Q2 = Q1Q2.(covPP̄, covPP)
## covU1U2 = U1U2.(covPP̄, covPP)
## covQ1U2 = Q1U2.(covPP̄, covPP)
## covU1Q2 = U1Q2.(covPP̄, covPP)



# Types that compute the isotropic part of 
# Spin2 and Spin0 CMBfields
# ==================================================


struct βcovSpin2
    covPP̄_premult_spln::Dierckx.Spline1D
    covPP_premult_spln::Dierckx.Spline1D
end

struct βcovSpin0 
    covII_premult_spln::Dierckx.Spline1D
end


# constructors
# ==================================================

function βcovSpin2(
        ℓ, eeℓ, bbℓ;
        n_grid::Int = 100_000, 
        β_grid = range(0, π^(1/3), length=n_grid).^3,
    )

    @assert ℓ[1] == 0
    @assert ℓ[2] == 1
    nℓ = @. (2ℓ+1)/(4π)
    ## ↓ starts at 2 since the Jacobi expansion goes like J^(a,b)_{ℓ-2}
    j2⁺2ℓ = (@. (eeℓ + bbℓ) * nℓ)[2:end]
    j2⁻2ℓ = (@. (eeℓ - bbℓ) * nℓ)[2:end]
    ## ↓  TODO: check the a,b swap
    f2⁺2  = ((a,b,jℓ)=(0,4,j2⁺2ℓ);  Fun(Jacobi(b,a),jℓ))
    f2⁻2  = ((a,b,jℓ)=(4,0,j2⁻2ℓ);  Fun(Jacobi(b,a),jℓ))
    # pre-canceled out cos β½ and sin β½ in the denom
    covPP̄ = x-> f2⁺2(cos(x))
    covPP = x-> f2⁻2(cos(x))
    β2covPP̄ = Dierckx.Spline1D(β_grid, covPP̄.(β_grid), k=3)
    β2covPP = Dierckx.Spline1D(β_grid, covPP.(β_grid), k=3)

    return βcovSpin2(β2covPP̄, β2covPP)

end 

function βcovSpin0(
        ℓ, ttℓ;
        n_grid::Int = 100_000, 
        β_grid = range(0, π^(1/3), length=n_grid).^3,
    )

    @assert ℓ[1] == 0
    @assert ℓ[2] == 1
    nℓ = @. (2ℓ+1)/(4π)
    ## ↓ starts at 2 since the Jacobi expansion goes like J^(a,b)_{ℓ-2}
    j0⁺0tℓ = @. ttℓ * nℓ
    ## ↓  TODO: check the a,b swap
    f0⁺0t = ((a,b,jℓ)=(0,0,j0⁺0tℓ); Fun(Jacobi(b,a),jℓ))
    ## leaving out the outer factors witch cancel with the sphere rotation
    covtt = x-> f0⁺0t(cos(x))
    β2covtt = Dierckx.Spline1D(β_grid, covtt.(β_grid), k=3)

    return βcovSpin0(β2covtt)

end 


# the types operate ... this is pre-vectorized since Spline1D is on vectors
# ==================================================

function (covP::βcovSpin2)(β::Matrix)
    rtnPP̄ = similar(β)
    rtnPP = similar(β)
    for (col, cβ) ∈ enumerate(eachcol(β))
	    rtnPP̄[:,col] = covP.covPP̄_premult_spln(cβ)
        rtnPP[:,col] = covP.covPP_premult_spln(cβ)
    end
    return complex.(rtnPP̄,0), complex.(rtnPP,0) 
end
function (covP::βcovSpin2)(β::Union{Vector, Number})
    rtnPP̄ = covP.covPP̄_premult_spln(β)
    rtnPP = covP.covPP_premult_spln(β)
    return complex.(rtnPP̄,0), complex.(rtnPP,0)     
end


function (covP::βcovSpin0)(β::Matrix)
    rtn = similar(β)
    for (col, cβ) ∈ enumerate(eachcol(β))
		rtn[:,col] = covP.covII_premult_spln(cβ)
    end
    return rtn  
end
function (covP::βcovSpin0)(β::Union{Vector, Number})
    return covP.covII_premult_spln(β)
end


# necessary geometric methods with angles and geodesics
# ==================================================


function sincosΔθpθΔφ(θ1, θ2, φ1, φ2)
    𝓅θ½ = (θ1 + θ2)/2
    Δθ½ = (θ1 - θ2)/2
    Δφ½ = (φ1 - φ2)/2
    s𝓅θ½, c𝓅θ½ = sincos(𝓅θ½)
    sΔθ½, cΔθ½ = sincos(Δθ½)
    sΔφ½, cΔφ½ = sincos(Δφ½)
    return sΔθ½, sΔφ½, s𝓅θ½, cΔθ½, cΔφ½, c𝓅θ½
end

function geoβ(θ1, θ2, φ1, φ2)
    sθ1, sθ2 = sin(θ1), sin(θ2)
    sΔθ½, sΔφ½, = sincosΔθpθΔφ(θ1, θ2, φ1, φ2)
    return 2asin(√(sΔθ½^2 + sθ1 * sθ2 * sΔφ½^2))    
end

# This one is left over from old code
# Repeats functionality of geoβ
# TODO: slated for removal but need to drop all instances of it
function geoθ1θ2Δφcol(θ1, θ2, Δφcol)
	@warn "Use CMBrings.geoβ(θ1, θ2, φ1, φ2) instead" maxlog=2
    sθ1, sθ2 = sin(θ1), sin(θ2)
    sΔθ½     = sin((θ1 - θ2)/2)
    sΔφ½     = @. sin(Δφcol / 2)
    β        = @. 2asin(√(sΔθ½^2 + sθ1 * sθ2 * sΔφ½^2))
    return β
end


# Multipliers needed to convert the isotropic parts to full polarization cov 
# =====================================================

function multPP̄(θ1, θ2, φ1, φ2)
    sΔθ½, sΔφ½, s𝓅θ½, cΔθ½, cΔφ½, c𝓅θ½ = sincosΔθpθΔφ(θ1, θ2, φ1, φ2)
    return complex(sΔφ½ * c𝓅θ½,   cΔφ½ * cΔθ½)^4
end

function multPP(θ1, θ2, φ1, φ2)
    sΔθ½, sΔφ½, s𝓅θ½, cΔθ½, cΔφ½, c𝓅θ½ = sincosΔθpθΔφ(θ1, θ2, φ1, φ2)
    return complex(sΔφ½ * s𝓅θ½, - cΔφ½ * sΔθ½)^4
end

## multII(θ1, θ2, φ1, φ2) = 1

Q1Q2(covPP̄, covPP) = ( real(covPP̄) + real(covPP) ) / 2

U1U2(covPP̄, covPP) = ( real(covPP̄) - real(covPP) ) / 2

Q1U2(covPP̄, covPP) = ( imag(covPP̄) + imag(covPP) ) / 2

U1Q2(covPP̄, covPP) = (- imag(covPP̄) + imag(covPP) ) / 2


