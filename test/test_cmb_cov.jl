
# Modules
# ==============================

using XFields
using CMBrings
using CMBsphere  # we will use CMBsphere to do the EBcovariance operator
using Spectra

import FFTransforms as FT
import SphereTransforms as ST

using LinearAlgebra
using SparseArrays
using DelimitedFiles
using LBblocks: @sblock
using PyPlot
using BenchmarkTools
using ProgressMeter

using Test 

#- 

if isdefined(Main,:PlutoRunner)
    import PlutoUI
    hide_plots = false
elseif isdefined(Main, :IJulia) && Main.IJulia.inited
    hide_plots = false
else 
    hide_plots = true
end


# Set ring transforms
# ==============================


tmAzS0, tmAzS2, θ, φ, Ω = @sblock let 

    ## size of the embedding full sphere
    𝕊nθ, 𝕊nφ = (2048, 2048-1)
    ## 𝕊nθ, 𝕊nφ = (2560, 2560-1)
    ## 𝕊nθ, 𝕊nφ = (3584, 2048-1)

    ## Spin ±2 transform
    tmS2 = ST.𝕊2(𝕊nθ, 𝕊nφ)
    tmS0 = ST.𝕊0(𝕊nθ, 𝕊nφ)

    ## grid coords on full sphere
    θ𝕊, φ𝕊 = ST.pix(tmS0) 

    ## north and southern boundaries and the corresponding indices
    θnorth∂ = 2.2 # 2.12
    θsouth∂ = 2.85
    θrng    = findall(θnorth∂ .<= θ𝕊 .<= θsouth∂)
    ringidxS0 = CartesianIndices((θrng[1]:θrng[end], 1:length(φ𝕊)))
    ringidxS2 = CartesianIndices((θrng[1]:θrng[end], 1:length(φ𝕊), 1:2))
    nθ, nφ  = size(ringidxS0)

    ## Spin 0 ring transform is just inherited from FFTransforms
    Tf = Float64
    tmW0  = FT.:⊗(FT.𝕀(nθ), FT.𝕎(Tf, nφ, 2π)) # 𝕀(nθ) ⊗ 𝕎(Tf, nφ, 2π)
    tmW2  = FT.:⊗(FT.𝕀(nθ), FT.𝕎(Tf, nφ, 2π), FT.𝕀(2)) 

    ## Spin 2 transform includes the ring embedding ...
    tmAzS0 = CMBrings.Az𝕊0(tmW0, tmS0, ringidxS0)
    tmAzS2 = CMBrings.Az𝕊2(tmW2, tmS2, ringidxS2)

    ## nθ, nφ,  = size_in(tmAzS2)
    θ, φ  = ST.pix(tmAzS2)
    Ω     = ST.Ωpix(tmAzS2)

    return tmAzS0, tmAzS2, θ, φ, Ω
end;


# Plot √Ωpix over ring θ's 

@sblock let θ, φ, Ω, hide_plots
    hide_plots && return
    fig,ax = subplots(1)
    ax.plot(θ, rad2deg.(sqrt.(Ω)).*60, label="sqrt pixel area (arcmin)")
    ax.plot(θ, zero(θ) .+ rad2deg.(θ[2] - θ[1]).*60, label="Δθ (arcmin)")
    ## ax.plot(θ, zero(θ) .+ rad2deg.(φ[2] - φ[1]).*60, label="Δφ (arcmin)")
    ax.set_xlabel(L"polar coordinate $\theta$")
    ax.legend()
    return fig
end;


# Spectral densities
# ==============================


eeℓ, bbℓ, ttℓ, ϕϕℓ, ℓ = @sblock let
    
    r  = 0.01

    lmax = 11000
    l = 0:lmax
    cld = Spectra.camb_cls(;lmax=lmax, r)
 
    ttsl = cld[:unlen_scalar] |> x->(x[:Ctt] ./ x[:factor_on_cl_cmb])
    tttl = cld[:unlen_tensor] |> x->(x[:Ctt] ./ x[:factor_on_cl_cmb])
    ttl  = ttsl .+ tttl
    ttl[1] = 0
   
    eesl = cld[:unlen_scalar] |> x->(x[:Cee] ./ x[:factor_on_cl_cmb])
    eetl = cld[:unlen_tensor] |> x->(x[:Cee] ./ x[:factor_on_cl_cmb])
    eel  = eesl .+ eetl
    eel[1] = 0

    bbsl = cld[:unlen_scalar] |> x->(x[:Cbb] ./ x[:factor_on_cl_cmb])
    bbtl = cld[:unlen_tensor] |> x->(x[:Cbb] ./ x[:factor_on_cl_cmb])
    ## note: bbsl == 0 
    bbl    = bbsl .+ bbtl
    bbl[1] = 0

    ϕϕl    = cld[:phi] |> x->(x[:Cϕϕ] ./ x[:factor_on_cl_phi])
    ϕϕl[1] =  0

    return eel, bbl, ttl, ϕϕl, l

end;


# Define the iso cov interpolators
# =================================================

# These two need more testing to check that the optional arguments work correctly

covPβ = Spectra.βcovSpin2(ℓ, eeℓ, bbℓ;
        ## n_grid::Int = 100_000, 
        ## β_grid = range(0, π^(1/3), length=n_grid).^3,
);

covTβ = Spectra.βcovSpin0(ℓ, ttℓ;
        ## n_grid::Int = 100_000, 
        ## β_grid = range(0, π^(1/3), length=n_grid).^3,
);




# Test (under construction): now construct ring Γ, C 
# .... be sure to use the CΛjkJ.
# =================================================


dΓΛjk, dCΛjk = @sblock let θ, φ, covPβ

    nθ, nφ  = length(θ), length(φ)
    ptmW    = plan(FT.𝕎(ComplexF64, nφ, 2π)) 

    # dΓΛ, dCΛ with `d` for diagonal
    dΓΛjk = Vector{ComplexF64}[zeros(ComplexF64, nφ) for j = 1:nθ, k = 1:nθ]
    dCΛjk = Vector{ComplexF64}[zeros(ComplexF64, nφ) for j = 1:nθ, k = 1:nθ]

    # ℓ indexes within ring. ℓ = 1 since we just compute 
    # first column of the ringj × ringk block
    ℓ = 1  

    @showprogress for j = 1:length(θ)
        for k = 1:length(θ)
            φ1 = φ[ℓ]
            θ1 = θ[j]
            θ2 = θ[k]
            β  =  Spectra.geoβ.(θ1, θ2, φ1, φ) 
            covPP̄, covPP = covPβ(β)  
            covPP̄ .*= Spectra.multPP̄.(θ1, θ2, φ1, φ) 
            covPP .*= Spectra.multPP.(θ1, θ2, φ1, φ)            
            mul!(dΓΛjk[j,k], ptmW, covPP̄)
            mul!(dCΛjk[j,k], ptmW, covPP)
        end
    end

    return dΓΛjk, dCΛjk
end;


# Reorganize dΓΛ, dCΛ by grouping by azimuth freq index ℓ

dΓRℓ, dCRℓ, J = @sblock let dΓΛjk, dCΛjk, nθ=length(θ), nφ=length(φ)

    J = Spectra.Jop(nφ)

    dΓRℓ  = Matrix{ComplexF64}[zeros(ComplexF64, nθ, nθ) for ℓ = 1:nφ]
    dCRℓ = Matrix{ComplexF64}[zeros(ComplexF64, nθ, nθ) for ℓ = 1:nφ]
    ## with 𝒰P[ℓ] := 𝒰_{ℓ,⋅} * P(θ,⋅)
    ## ΓΛ * 𝒰P       = sum(dΓRℓ[ℓ] * 𝒰P[ℓ] for ℓ=1:nφ)
    ## CΛ * conj(𝒰P) = sum(dCRℓ[ℓ] * conj(𝒰P[J(ℓ)]) for ℓ=1:nφ)

    @showprogress for ℓ = 1:nφ
        for k = 1:nθ
            for j = 1:nθ
                @inbounds dΓRℓ[ℓ][j,k] = dΓΛjk[j,k][ℓ]
                @inbounds dCRℓ[ℓ][j,k] = dCΛjk[j,k][ℓ]
            end
        end
    end
    
    return dΓRℓ, dCRℓ, J
end;



# TODO: write a function that computes the inverse ΓR and CR
# -----------------------------------


function inv_ΓC_ΓJCJ!(invΓℓ, invCℓ, invΓJℓ, invCJℓ, Γℓ, Cℓ, ΓJℓ, CJℓ, J) 

    Ωℓ  = [ 
        Γℓ          Cℓ
        conj.(CJℓ)  conj.(ΓJℓ)
    ] ## Note: Ωℓ acts on [𝒰P[ℓ]; 𝒰P[J(ℓ)]]

    ΩJℓ  = [ 
        ΓJℓ        CJℓ
        conj.(Cℓ)  conj.(Γℓ)
    ] ## Note: ΩJℓ acts on [𝒰P[J(ℓ)]; 𝒰P[ℓ]]

    Ωℓ⁻¹  = inv(Ωℓ)
    ΩJℓ⁻¹ = inv(ΩJℓ)

    invΓℓ  .=  Ωℓ⁻¹[  1:end÷2, 1:end÷2]
    invΓJℓ .=  ΩJℓ⁻¹[ 1:end÷2, 1:end÷2]
    invCℓ  .=  Ωℓ⁻¹[  1:end÷2, end÷2+1:end]
    invCJℓ .=  ΩJℓ⁻¹[ 1:end÷2, end÷2+1:end]

    invΓℓ, invCℓ, invΓJℓ, invCJℓ
end


function inv_ΓC_ΓJCJ(Γℓ, Cℓ, ΓJℓ, CJℓ, J) 

    invΓℓ  = similar(Γℓ)
    invCℓ  = similar(Cℓ)
    invΓJℓ = similar(ΓJℓ)
    invCJℓ = similar(CJℓ)

    inv_ΓC_ΓJCJ!(invΓℓ, invCℓ, invΓJℓ, invCJℓ, Γℓ, Cℓ, ΓJℓ, CJℓ, J) 
end



ℓₒ = 3 
𝕆 = zero(dΓRℓ[ℓₒ])

ΣRℓₒJℓₒ = [
    dΓRℓ[ℓₒ]            𝕆               𝕆               dCRℓ[ℓₒ]  
    𝕆                   dΓRℓ[J(ℓₒ)]     dCRℓ[J(ℓₒ)]     𝕆
    𝕆                   conj.(dCRℓ[ℓₒ]) conj.(dΓRℓ[ℓₒ]) 𝕆             
    conj.(dCRℓ[J(ℓₒ)])  𝕆               𝕆               conj.(dΓRℓ[J(ℓₒ)])   
]

dΓ⁻¹Rℓₒ, dC⁻¹Rℓₒ, dΓ⁻¹RJℓₒ, dC⁻¹RJℓₒ = inv_ΓC_ΓJCJ(dΓRℓ[ℓₒ], dCRℓ[ℓₒ], dΓRℓ[J(ℓₒ)], dCRℓ[J(ℓₒ)], J)       
ΣRℓₒJℓₒ⁻¹_test = [
    dΓ⁻¹Rℓₒ          𝕆               𝕆              dC⁻¹Rℓₒ  
    𝕆                dΓ⁻¹RJℓₒ        dC⁻¹RJℓₒ       𝕆
    𝕆                conj.(dC⁻¹Rℓₒ)  conj.(dΓ⁻¹Rℓₒ) 𝕆             
    conj.(dC⁻¹RJℓₒ)  𝕆               𝕆              conj.(dΓ⁻¹RJℓₒ)   
]

ΣRℓₒJℓₒ⁻¹_test * ΣRℓₒJℓₒ .|> abs |> matshow; colorbar()




### Also test the intermediate matrices in inv_ΓC_ΓJCJ

ΩRℓₒ  = [ 
    dΓRℓ[ℓₒ]            dCRℓ[ℓₒ]
    conj.(dCRℓ[J(ℓₒ)])  conj.(dΓRℓ[J(ℓₒ)])
] 
# inverse has first row [dΓ⁻¹Rℓₒ  dC⁻¹Rℓₒ]. why ??
## Note: ΩRℓₒ acts on [𝒰P[ℓₒ]; 𝒰P[J(ℓₒ)]]

ΩRJℓₒ = [ 
    dΓRℓ[J(ℓₒ)]            dCRℓ[J(ℓₒ)]
    conj.(dCRℓ[ℓₒ])  conj.(dΓRℓ[ℓₒ])
]
# inverse has first row [dΓ⁻¹Rℓₒ  dC⁻¹Rℓₒ]. 
## Note: ΩRJℓₒ acts on [𝒰P[J(ℓₒ)]; 𝒰P[ℓₒ]]


## these two are hermitian and positive definite
@test ΩRℓₒ  .- adjoint( ΩRℓₒ) |> x-> maximum(abs2.(x)) < 1e-10
@test ΩRJℓₒ .- adjoint( ΩRJℓₒ) |> x-> maximum(abs2.(x)) < 1e-10

@test all(eigen(Hermitian(ΩRℓₒ)).values .> 0)
@test all(eigen(Hermitian(ΩRJℓₒ)).values .> 0)


 










# Test case: plot radial profile of isotropic version
# =================================================


@sblock let covPβ, covTβ, hide_plots 
    hide_plots && return

    βs      = range(0,deg2rad(3),length=4000) |> collect
    covTTβs = covTβ(βs)
    covPP̄βs, covPPβs = covPβ(βs)
    
    fig,ax = subplots(2, figsize=(8,5))
    ax[1].plot(βs, covTTβs)
    ax[2].plot(βs, real.(covPP̄βs) .* cos.(βs./2).^4 )
    ax[2].plot(βs, real.(covPPβs) .* sin.(βs./2).^4 )

    fig 
end;



# Test: that the multipliers have the right conj symmetry
# =================================================

θ1, φ1 = π/2 + .01, π/8
θ2, φ2 = θ1 + .2, φ1 + .4
@test Spectra.multPP̄(θ1, θ2, φ1, φ2) == conj(Spectra.multPP̄(θ2, θ1, φ2, φ1))
@test Spectra.multPP(θ1, θ2, φ1, φ2) == Spectra.multPP(θ2, θ1, φ2, φ1)
## the above should be true for Γ and C

## ↓ these should be true via spin 2 to spin -2 conversion via conj I think
@test Spectra.multPP̄(θ1, θ2, φ1, φ2) == conj(Spectra.multPP̄(θ1, θ2, -φ1, -φ2))
@test Spectra.multPP̄(θ1, θ2, φ1, φ2) == conj(Spectra.multPP̄(θ1, θ2, φ2, φ1))
@test Spectra.multPP(θ1, θ2, φ1, φ2) == conj(Spectra.multPP(θ1, θ2, -φ1, -φ2))
@test Spectra.multPP(θ1, θ2, φ1, φ2) == conj(Spectra.multPP(θ1, θ2, φ2, φ1))

# test the non-sign symmetry of the cross correlations ...



# Test: view pixel space cov 
# =================================================


@time fig = @sblock let θ, φ, covPβ, hide_plots

    hide_plots && return
    
    r1, c1  = 100, 100 
    θ1, φ1  = θ[r1], φ[c1]

    nθ, nφ  = length(θ), length(φ)
    θgd     = θ  .+ zeros(nθ, nφ) 
    φgd     = φ' .+ zeros(nθ, nφ) 

    β              =  Spectra.geoβ.(θ1, θgd, φ1, φgd) 
    covPP̄, covPP = covPβ(β)   
    covPP̄ .*= Spectra.multPP̄.(θ1, θgd, φ1, φgd) 
    covPP .*= Spectra.multPP.(θ1, θgd, φ1, φgd)

    covQ1Q2 = Spectra.Q1Q2.(covPP̄, covPP)
    covU1U2 = Spectra.U1U2.(covPP̄, covPP)
    covQ1U2 = Spectra.Q1U2.(covPP̄, covPP)
    covU1Q2 = Spectra.U1Q2.(covPP̄, covPP)


    fig,ax = subplots(2,2,figsize=(7,5))
    ax[1,1].imshow(covQ1Q2[r1-50:r1+50, c1-50:c1+50])
    ax[1,2].imshow(covU1U2[r1-50:r1+50, c1-50:c1+50])
    ax[2,1].imshow(covQ1U2[r1-50:r1+50, c1-50:c1+50])
    ax[2,2].imshow(covU1Q2[r1-50:r1+50, c1-50:c1+50])


    fig
end;



# Test: Form the full covariance matrix for Q,U on a single ring
# =================================================


## Γjk, Cjk, jₒ, kₒ = @sblock let θ, φ, covPβ, jₒ = 100, kₒ = 150 
Γjk, Cjk, jₒ, kₒ = @sblock let θ, φ, covPβ, jₒ = 200, kₒ = 200 

    nθ, nφ  = length(θ), length(φ)

    Γ = zeros(ComplexF64, nφ, nφ)
    C = zeros(ComplexF64, nφ, nφ)

    θ1 = θ[jₒ]
    θ2 = θ[kₒ]
    @showprogress for c1 = 1:length(φ)

        φ1  = φ[c1]
        β   =  Spectra.geoβ.(θ1, θ2, φ1, φ) 
        covPP̄, covPP = covPβ(β)  
        covPP̄ .*= Spectra.multPP̄.(θ1, θ2, φ1, φ) 
        covPP .*= Spectra.multPP.(θ1, θ2, φ1, φ)
        
        Γ[:,c1] = covPP̄
        C[:,c1] = covPP

    end

    return Γ, C, jₒ, kₒ
end;


# Check Γjk, Cjk are circulant.
# ------------------------------------

@sblock let runit = jₒ == kₒ, Γjk, Cjk, nφ = length(φ)
    if runit
        j₁ = rand(1:nφ)
        @test maximum(abs2.(Γjk[:,j₁+1] .- circshift(Γjk[:,j₁],1))) < 1e-10
        @test maximum(abs2.(Cjk[:,j₁+1] .- circshift(Cjk[:,j₁],1))) < 1e-10
    end
end

# When j == k check Γjk is hermitian and Cjk is symmetric
# ------------------------------------

@sblock let runit = jₒ == kₒ, Γjk, Cjk
    if runit
        @test maximum(abs2.(Γjk - adjoint(Γjk))) < 1e-10
        @test maximum(abs2.(Cjk - transpose(Cjk))) < 1e-10
    end
end

# When j == k check Σ is positive definite (models the pixel cov of P(n̂) on right)
# ------------------------------------

Σ, dsΣ = @sblock let Γjk, Cjk

    Σ = [
        Γjk        Cjk
        conj.(Cjk) conj.(Γjk)
    ]

    dsΣ, = eigen(Hermitian(Σ))

    return Σ, dsΣ
end;

if jₒ == kₒ
    @test maximum(abs2.(Σ - adjoint(Σ))) < 1e-10
end

@test all(dsΣ .>= 0)

# Check dΓΛjk = eigen(ΓΛjk), dCΛjk = eigen(CΛjk) 
# ..and ΣΛ has eigen values the same as Σ 
# ------------------------------------

dΓΛjk = FT.fft(Γjk[:,1])
dCΛjk = FT.fft(Cjk[:,1])

ΓΛjk, CΛjkJ, ΣΛ = @sblock let dΓΛjk, dCΛjk
    ΓΛjk  = spdiagm(0 => dΓΛjk)
    
    CΛjkJ = spzeros(ComplexF64, length(dCΛjk), length(dCΛjk))
    CΛjkJ[1,1] = dCΛjk[1]
    for t = 0:length(dCΛjk)-2
        CΛjkJ[end-t,2+t] = dCΛjk[end-t]
    end

    ΣΛ = [
        ΓΛjk        CΛjkJ
        conj.(CΛjkJ) conj.(ΓΛjk)
    ]

    return ΓΛjk, CΛjkJ, ΣΛ
end;

@sblock let ΣΛ, runit = jₒ == kₒ
    !runit && return 
    I, J, V = findnz(ΣΛ - adjoint(ΣΛ))
    Vix, ix = findmax(abs.(V))
    @test Vix < 1e-10
    I[ix], J[ix], Vix
end;


# Test that diag(Uᴴ,U) * ΣΛ * diag(U,Uᴴ) == Σ 
# ( note that both operate on [P ; P̄] )
# ... and in particular they have the same eigen values
# ------------------------------------

## TODO diag(Uᴴ,U) * ΣΛ * diag(U,Uᴴ) == Σ


dsΣ′, = eigen(Hermitian(Matrix(ΣΛ)))
@test all(dsΣ′ .>= 0)
@test maximum(abs2.(dsΣ′ .- dsΣ)) < 1e-10




