
# Modules
# ==============================
# TODO: needs trimming

using XFields
using CMBrings
using CMBsphere  # we will use CMBsphere to do the EBcovariance operator
using Spectra

import FFTransforms as FT
import SphereTransforms as ST

using  LinearAlgebra
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
end


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
end


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


## Test case: first define the iso cov interpolators
## =================================================


## Recall at this point covP
covPβ = CMBrings.βcovSpin2(ℓ, eeℓ, bbℓ;
        ## n_grid::Int = 100_000, 
        ## β_grid = range(0, π^(1/3), length=n_grid).^3,
)

covTβ = CMBrings.βcovSpin0(ℓ, ttℓ;
        ## n_grid::Int = 100_000, 
        ## β_grid = range(0, π^(1/3), length=n_grid).^3,
)


@sblock let covPβ, covTβ 
    hide_plots && return

    βs      = range(0,deg2rad(3),length=4000) |> collect
    covTTβs = covTβ(βs)
    covPP̄βs, covPPβs = covPβ(βs)
    
    fig,ax = subplots(2, figsize=(8,5))
    ax[1].plot(βs, covTTβs)
    ax[2].plot(βs, real.(covPP̄βs) .* cos.(βs./2).^4 )
    ax[2].plot(βs, real.(covPPβs) .* sin.(βs./2).^4 )

    fig 
end


## Test that the multipliers have the right conj symmetry
## =================================================

θ1, φ1 = π/2 + .01, π/8
θ2, φ2 = θ1 + .2, φ1 + .4
@test CMBrings.multPP̄(θ1, θ2, φ1, φ2) == conj(CMBrings.multPP̄(θ2, θ1, φ2, φ1))
@test CMBrings.multPP(θ1, θ2, φ1, φ2) == CMBrings.multPP(θ2, θ1, φ2, φ1)
## the above should be true for Γ and C

## ↓ these should be true via spin 2 to spin -2 conversion via conj I think
@test CMBrings.multPP̄(θ1, θ2, φ1, φ2) == conj(CMBrings.multPP̄(θ1, θ2, -φ1, -φ2))
@test CMBrings.multPP̄(θ1, θ2, φ1, φ2) == conj(CMBrings.multPP̄(θ1, θ2, φ2, φ1))
@test CMBrings.multPP(θ1, θ2, φ1, φ2) == conj(CMBrings.multPP(θ1, θ2, -φ1, -φ2))
@test CMBrings.multPP(θ1, θ2, φ1, φ2) == conj(CMBrings.multPP(θ1, θ2, φ2, φ1))

@test CMBrings.multPP̄(θ1, θ2, 0, φ2) 
@test CMBrings.multPP̄(θ1, θ2, 0, -φ2)

@test CMBrings.multPP(θ1, θ2, 0, φ2) 
@test CMBrings.multPP(θ1, θ2, 0, -φ2)


## Test case: view pixel space cov 
## =================================================


@time fig = @sblock let θ, φ, covPβ

    hide_plots && return
    
    r1, c1  = 100, 100 
    θ1, φ1  = θ[r1], φ[c1]

    nθ, nφ  = length(θ), length(φ)
    θgd     = θ  .+ zeros(nθ, nφ) 
    φgd     = φ' .+ zeros(nθ, nφ) 

    β              =  CMBrings.geoβ.(θ1, θgd, φ1, φgd) 
    covPP̄, covPP = covPβ(β)   
    covPP̄ .*= CMBrings.multPP̄.(θ1, θgd, φ1, φgd) 
    covPP .*= CMBrings.multPP.(θ1, θgd, φ1, φgd)

    covQ1Q2 = CMBrings.Q1Q2.(covPP̄, covPP)
    covU1U2 = CMBrings.U1U2.(covPP̄, covPP)
    covQ1U2 = CMBrings.Q1U2.(covPP̄, covPP)
    covU1Q2 = CMBrings.U1Q2.(covPP̄, covPP)


    fig,ax = subplots(2,2,figsize=(7,5))
    ax[1,1].imshow(covQ1Q2[r1-50:r1+50, c1-50:c1+50])
    ax[1,2].imshow(covU1U2[r1-50:r1+50, c1-50:c1+50])
    ax[2,1].imshow(covQ1U2[r1-50:r1+50, c1-50:c1+50])
    ax[2,2].imshow(covU1Q2[r1-50:r1+50, c1-50:c1+50])


    fig
end



## Test case: Form the full covariance matrix for Q,U on a single ring
## =================================================

## testing without multipliers

ΓΛ, CΛ = @sblock let θ, φ, covPβ

    nθ, nφ  = length(θ), length(φ)

    ΓΛ = zeros(Complex{Float64}, nφ, nφ)
    CΛ = zeros(Complex{Float64}, nφ, nφ)

    θ1 = θ[100]
    @showprogress for c1 = 1:length(φ)

        φ1  = φ[c1]
        β            =  CMBrings.geoβ.(θ1, θ1, φ1, φ) 
        covPP̄, covPP = covPβ(β)  
        covPP̄ .*= CMBrings.multPP̄.(θ1, θ1, φ1, φ) 
        covPP .*= CMBrings.multPP.(θ1, θ1, φ1, φ)
        
        ΓΛ[:,c1] = covPP̄
        CΛ[:,c1] = covPP

    end

    return ΓΛ, CΛ
end



Σ = [
    ΓΛ        CΛ
    conj.(CΛ) conj.(ΓΛ)
]


covQ1Q2 = CMBrings.Q1Q2.(ΓΛ, CΛ)
covU1U2 = CMBrings.U1U2.(ΓΛ, CΛ)
covQ1U2 = CMBrings.Q1U2.(ΓΛ, CΛ)
covU1Q2 = CMBrings.U1Q2.(ΓΛ, CΛ)


fk =   fft( exp.(-im .* φ ./ 2) .* covQ1U2[:,1])
fk′, Uk′ = eigen(covQ1U2)

imag.(fk′)
fk′ .|> abs |> sort |> plot
fk  .|> real |> plot
covQ1U2 |> matshow


Σ .|> real |> matshow; colorbar()
Σ .|> imag |> matshow; colorbar()
Σ .- adjoint(Σ) .|> abs |> matshow; colorbar()


CΛ[200,:] .|> real |> plot
CΛ[200,:] .|> imag |> plot

ΓΛ[end÷2,:] .|> real |> plot
ΓΛ[end÷2,:] .|> imag |> plot

##

d,U =  eigen(Hermitian(Σ))
d′ = FT.fft(Σ[:,1])
g′ = FT.fft(ΓΛ[:,1])


plot(d)
sort(vcat(real.(d′[1:2:end]), imag.(d′[2:2:end]))) |> plot

real.(d′[1:2:end]) |> plot
imag.(d′[2:2:end]) |> plot

imag.(d′[1:2:end]) |> plot
real.(d′[2:2:end]) |> plot




plot(sort(real.(d′[1:2:end])))

# check Σ is Hermitian.

# perhaps Σ is diagonalized by fourier transform ... so 


## Test case: now construct ring Γ, C
## =================================================


nθ, nφ  = length(θ), length(φ)
tmW  = FT.:⊗(FT.𝕀(nθ), FT.𝕎(Complex{Float64}, nφ, 2π)) #  |> x -> FT.unitary_scale(x)*x
ptmW = plan(tmW)
# We want complex fft here since covPP̄ and covPP will be complex

lengthθ, nblks = size_out(tmW)
Tb = Complex{Float64}
azΓ = Matrix{Tb}[zeros(Tb, lengthθ, lengthθ) for k = 1:nblks]
azC = Matrix{Tb}[zeros(Tb, lengthθ, lengthθ) for k = 1:nblks]


@sblock let covPβ, azΓ, azC, ptmW, θ, φ      
    nθ, nφ  = length(θ), length(φ)
    nblks   = nφ 
    θgd     = θ  .+ zeros(nθ, nφ) 
    φgd     = φ' .+ zeros(nθ, nφ) 
    c1  = 1
    @showprogress for r1 = 1:length(θ)

        θ1, φ1  = θ[r1], φ[c1]
        β            =  CMBrings.geoβ.(θ1, θgd, φ1, φgd) 
        covPP̄, covPP = covPβ(β)  
        # testing without multipliers 
        # covPP̄ .*= CMBrings.multPP̄.(θ1, θgd, φ1, φgd) 
        # covPP .*= CMBrings.multPP.(θ1, θgd, φ1, φgd)
        
        ΓΛ = ptmW * covPP̄
        CΛ = ptmW * covPP

        ## Threads.@threads for k = 1:nblks
        for k = 1:nblks
            azΓ[k][:,r1] .= ΓΛ[:,k]
            azC[k][:,r1] .= CΛ[:,k]
        end
    end
end
k = 4

azΓ[k] .|> real |> matshow; colorbar()
azΓ[k] .|> imag |> matshow; colorbar()
azΓ[k] - adjoint(azΓ[k]) .|> real |> matshow; colorbar()
azΓ[k] - adjoint(azΓ[k]) .|> imag |> matshow; colorbar()

azC[k] .|> real |> matshow; colorbar()
azC[k] .|> imag |> matshow; colorbar()
azC[k] - transpose(azC[k]) .|> real |> matshow; colorbar()
azC[k] - transpose(azC[k]) .|> imag |> matshow; colorbar()


k = 15
M = [
     azΓ[k]        azC[k]
     conj.(azC[k]) conj.(azΓ[k])
]
M  .|> real |> matshow; colorbar()
M  .|> imag |> matshow; colorbar()
M - adjoint(M) .|> abs |> matshow; colorbar()



va, Ve = Symmetric( M, :U ) |> eigen
## va, Ve = M |> eigen

plot(va)

plot(Ve[:,end-15])
plot(Ve[:,end-5])
plot(Ve[:,end])
plot(Ve[:,1])

# Base.summarysize(azΣ) * 1e-9 #-> gigabites

k = 10
azΓ[k] .- azΓ[k]' |> matshow; colorbar() 
azΓ[k]  |> matshow; colorbar() 

azC[k] .- azC[k]' |> matshow; colorbar() 
azC[k] |> matshow; colorbar() 










### old stuff. slated for removal








#%% Compute the cross covariance of the fourier modes 
#%% ----------------------------------------------------

#nsd = Nside(512)
#nsd = Nside(1024)
nsd = Nside(2048)

Q₁Q₂_U₁U₂_Q₁U₂_U₁Q₂, P̄P_PP, θ, φ, Δφk = @sblock let nsd, β2covϕϕ, β2covP̄P, β2covPP
    θ,φ   = HH.θφ_eqbelt_align(nsd) # .|> x -> T.(x)
    θ,φ = T.(θ),T.(φ) 
    # ---------
    # θ₀    = 1.0 
    # θ₁    = 2.0
    # θ₀idx = findmin(abs2.(θ .- θ₀))[2]
    # θ₁idx = findmin(abs2.(θ .- θ₁))[2]
    # #θ     = θ[θ₀idx:θ₁idx]
    # θ     = θ[θ₀idx:2:θ₁idx]
    # ---------
    # θ = θ[1:2:end]
    # θ = θ[1:3:end]
    # ---------
    # θ₀    = 0.25 
    # θ₁    = 1.0
    # θ   = range(θ₀, θ₁, length = 700)

    #--------
    #θ = π/2 .+ φ[1:512] .- mean(φ[1:512]) 
    θ = π/2 .+ φ[1:1024] .- mean(φ[1:1024]) 

    φcol = φ[:]

    Δφk    = collect((sin.( ((1:length(φ)) .- 1) .* (π/length(φ)) ).^2)')[1:1,1:(length(φ)÷2+1)]
    #𝒲col  = plan_rfft(similar(φcol))
    
    P̄P_PP = function (θ1, θ2)
        sθ1, sθ2 = sin(θ1), sin(θ2)
        𝓅θ½ = (θ1 + θ2)/2
        Δθ½ = (θ1 - θ2)/2
        Δφ½ = φcol ./ 2
        s𝓅θ½, c𝓅θ½ = sincos(𝓅θ½)
        sΔθ½, cΔθ½ = sincos(Δθ½)
        sΔφ½ = sin.(Δφ½)
        cΔφ½ = cos.(Δφ½)

        β = @. 2asin(√(sΔθ½^2 + sθ1 * sθ2 * sΔφ½^2))
        ξ⁺β, ξ⁻β  = T.(β2covP̄P(β)), T.(β2covPP(β))
    
        # pre-cancel out cos β½ and sin β½ in the denom
        P̄P½ = @. complex(sΔφ½ * c𝓅θ½, - cΔφ½ * cΔθ½)^4 / 2 
        PP½ = @. complex(sΔφ½ * s𝓅θ½, - cΔφ½ * sΔθ½)^4 / 2
        P̄P½ .*= ξ⁺β
        PP½ .*= ξ⁻β

        reP̄P½, imP̄P½ = real.(P̄P½), imag.(P̄P½)
        rePP½, imPP½ = real.(PP½), imag.(PP½)
        return reP̄P½, imP̄P½, rePP½, imPP½
    end

    Q₁Q₂_U₁U₂_Q₁U₂_U₁Q₂ = function (θ1, θ2, 𝒲col)
        reP̄P½, imP̄P½, rePP½, imPP½ = P̄P_PP(θ1, θ2)        
        Q₁Q₂ = 𝒲col * (  reP̄P½ .+ rePP½)
        U₁U₂ = 𝒲col * (  reP̄P½ .- rePP½)
        Q₁U₂ = 𝒲col * (  imP̄P½ .+ imPP½)
        U₁Q₂ = 𝒲col * (.-imP̄P½ .+ imPP½)
        # try replacing the rfft plan with a direct call 
        # ... to see if this is where the parallel issues is
        #
        # indeed, it appears that plan_rfft appearing in a closure 
        # gives problems when running in parallel!
        #
        # Q₁Q₂ = rfft(  reP̄P½ .+ rePP½)
        # U₁U₂ = rfft(  reP̄P½ .- rePP½)
        # Q₁U₂ = rfft(  imP̄P½ .+ imPP½)
        # U₁Q₂ = rfft(.-imP̄P½ .+ imPP½)

        return Q₁Q₂, U₁U₂, Q₁U₂, U₁Q₂
    end

    return Q₁Q₂_U₁U₂_Q₁U₂_U₁Q₂, P̄P_PP, θ, φ, Δφk

end


#=

rad2deg(φ[2] - φ[1])*60
rad2deg(θ[2] - θ[1])*60

1.0 .+ φ[1:1024]

ki = fftfreq(length(φ), length(φ)/2π)[1:end÷2+1]

θ1, θ2 = θ[1],θ[2]
#reP̄P½, imP̄P½, rePP½, imPP½ = P̄P_PP(θ1, θ2)
Q₁Q₂,U₁U₂,Q₁U₂,U₁Q₂ = Q₁Q₂_U₁U₂_Q₁U₂_U₁Q₂(θ1, θ2)

@sblock let  qq=Q₁Q₂,qu=Q₁U₂,uu=U₁U₂,uq=U₁Q₂, ki

    fig,ax = subplots(2,2, figsize=(14,10))
    abs.(ki) .* qq .|> real |>  ax[1,1].plot; ax[1,1].set_title("qq")
    abs.(ki) .* qu .|> real |>  ax[1,2].plot; ax[1,2].set_title("qu")
    abs.(ki) .* uu .|> real |>  ax[2,1].plot; ax[2,1].set_title("uu")
    abs.(ki) .* uq .|> real |>  ax[2,2].plot; ax[2,2].set_title("uq")

    abs.(ki) .* qq .|> imag |>  ax[1,1].plot; ax[1,1].set_title("qq")
    abs.(ki) .* qu .|> imag |>  ax[1,2].plot; ax[1,2].set_title("qu")
    abs.(ki) .* uu .|> imag |>  ax[2,1].plot; ax[2,1].set_title("uu")
    abs.(ki) .* uq .|> imag |>  ax[2,2].plot; ax[2,2].set_title("uq")

end

=#







#%% construct ΣQQ, ΣUU, ΣQU in blocks of frequency sheets 
#%% ----------------------------------------------------




Σsheets_k = @sblock let T, Q₁Q₂_U₁U₂_Q₁U₂_U₁Q₂, φ

    Σsheets_k = function (θ, idxk)
        nθx = length(θ)
        lowrΣQ₁Q₂_upperΣU₁U₂  = zeros(T,length(idxk), nθx,nθx)
        ΣQ₁U₂    = zeros(Complex{T},length(idxk), nθx,nθx)

        𝒲col  = plan_rfft(similar(φ[:]))

        @showprogress for j=1:nθx, i=j+1:nθx 
            Q₁Q₂,U₁U₂,Q₁U₂,U₁Q₂ = Q₁Q₂_U₁U₂_Q₁U₂_U₁Q₂(θ[i],θ[j], 𝒲col)
            lowrΣQ₁Q₂_upperΣU₁U₂[:,i,j] = real.(Q₁Q₂[idxk])
            lowrΣQ₁Q₂_upperΣU₁U₂[:,j,i] = real.(U₁U₂[idxk])
            ΣQ₁U₂[:,i,j]   = Q₁U₂[idxk]
            ΣQ₁U₂[:,j,i]   = conj.(U₁Q₂[idxk])
        end

        diagΣQ₁Q₂ = zeros(T,length(idxk), nθx)
        diagΣU₁U₂ = zeros(T,length(idxk), nθx)
        diagΣQ₁U₂ = zeros(Complex{T},length(idxk), nθx)
        for j=1:nθx
            Q₁Q₂,U₁U₂,Q₁U₂,U₁Q₂ = Q₁Q₂_U₁U₂_Q₁U₂_U₁Q₂(θ[j],θ[j], 𝒲col)
            diagΣQ₁Q₂[:,j] = real.(Q₁Q₂[idxk])
            diagΣU₁U₂[:,j] = real.(U₁U₂[idxk])
            diagΣQ₁U₂[:,j] = Q₁U₂[idxk]
        end

        rtΣQQ = map(1:length(idxk)) do k 
            Symmetric(lowrΣQ₁Q₂_upperΣU₁U₂[k,:,:], :L) + Diagonal(diagΣQ₁Q₂[k,:])
        end 
        rtΣUU = map(1:length(idxk)) do k 
            Symmetric(lowrΣQ₁Q₂_upperΣU₁U₂[k,:,:], :U) + Diagonal(diagΣU₁U₂[k,:])
        end 

        rtΣQU = map(1:length(idxk)) do k 
            ΣQ₁U₂[k,:,:] +  Diagonal(diagΣQ₁U₂[k,:])
        end 

        return rtΣQQ, rtΣUU, rtΣQU
    end

    return Σsheets_k
end


# ----- here is a parallel version 

@everywhere function shared_chunck!(lowrΣQ₁Q₂_upperΣU₁U₂, ΣQ₁U₂, jrange, Q₁Q₂_U₁U₂_Q₁U₂_U₁Q₂, θ, φ, idxk)
    nθx = length(θ)
    𝒲col  = plan_rfft(similar(φ[:]))
    for j=jrange, i=j+1:nθx 
        Q₁Q₂,U₁U₂,Q₁U₂,U₁Q₂ = Q₁Q₂_U₁U₂_Q₁U₂_U₁Q₂(θ[i],θ[j],𝒲col)
        lowrΣQ₁Q₂_upperΣU₁U₂[:,i,j] = real.(Q₁Q₂[idxk])
        lowrΣQ₁Q₂_upperΣU₁U₂[:,j,i] = real.(U₁U₂[idxk])
        ΣQ₁U₂[:,i,j]   = Q₁U₂[idxk]
        ΣQ₁U₂[:,j,i]   = conj.(U₁Q₂[idxk])
    end
end

@everywhere function split_col_ranges(ncols,nwrks)
    tot = 0
    breaks = Int[0]
    num_ind = (ncols*(ncols-1)/2)÷nwrks
    for c = 1:ncols,r=c+1:ncols
            tot += 1
            if tot > num_ind
                push!(breaks,c)
                tot = 0
            end 
    end
    push!(breaks,ncols)

    jranges = UnitRange{Int64}[]
    for i = 1:length(breaks)-1
        push!(jranges, breaks[i]+1:breaks[i+1])
    end

    jranges
end

parallel_Σsheets_k = @sblock let T, Q₁Q₂_U₁U₂_Q₁U₂_U₁Q₂, φ

    parallel_Σsheets_k = function (θ, idxk)

        nθx = length(θ)
        
        lowrΣQ₁Q₂_upperΣU₁U₂ = SharedArray{T,3}(
            (length(idxk), nθx,nθx), 
            init = S -> S[localindices(S)] = repeat([T(0)], length(localindices(S))),
            # pids = workers(),
        ) 
        
        ΣQ₁U₂ = SharedArray{Complex{T},3}(
            (length(idxk), nθx,nθx), 
            init = S -> S[localindices(S)] = repeat([Complex{T}(0)], length(localindices(S))),
            # pids = workers(),
        )

        jranges = split_col_ranges(nθx,nworkers())
        
        @sync begin
            for p in workers()
                @async remotecall_wait(
                    shared_chunck!, p, lowrΣQ₁Q₂_upperΣU₁U₂, ΣQ₁U₂, jranges[p-1], Q₁Q₂_U₁U₂_Q₁U₂_U₁Q₂, θ, φ, idxk
                )
            end
        end

        diagΣQ₁Q₂ = zeros(T,length(idxk), nθx)
        diagΣU₁U₂ = zeros(T,length(idxk), nθx)
        diagΣQ₁U₂ = zeros(Complex{T},length(idxk), nθx)
        𝒲col  = plan_rfft(similar(φ[:]))

        for j=1:nθx
            Q₁Q₂,U₁U₂,Q₁U₂,U₁Q₂ = Q₁Q₂_U₁U₂_Q₁U₂_U₁Q₂(θ[j],θ[j],𝒲col)
            diagΣQ₁Q₂[:,j] = real.(Q₁Q₂[idxk])
            diagΣU₁U₂[:,j] = real.(U₁U₂[idxk])
            diagΣQ₁U₂[:,j] = Q₁U₂[idxk]
        end

        rtΣQQ = map(1:length(idxk)) do k 
            Symmetric(lowrΣQ₁Q₂_upperΣU₁U₂[k,:,:], :L) + Diagonal(diagΣQ₁Q₂[k,:])
        end 
        rtΣUU = map(1:length(idxk)) do k 
            Symmetric(lowrΣQ₁Q₂_upperΣU₁U₂[k,:,:], :U) + Diagonal(diagΣU₁U₂[k,:])
        end 

        rtΣQU = map(1:length(idxk)) do k 
            ΣQ₁U₂[k,:,:] +  Diagonal(diagΣQ₁U₂[k,:])
        end 

        return rtΣQQ, rtΣUU, rtΣQU
    end

    return parallel_Σsheets_k
end







#%% Construct the cholesky (in θ) for each frequency, in blocks.
#%% Save them in a jld2 file
#%% ----------------------------------------------------


kidx = 1:4:(length(φ)÷2+1)
#kidx = 1:8:(length(φ)÷2+1)

kidx_blk = [
    kidx[1:end÷4],
    kidx[(1(end÷4)+1):(2(end÷4))],
    kidx[(2(end÷4)+1):(3(end÷4))],
    kidx[(3(end÷4)+1):end],
]


# kidx_blk = [
#     kidx[1:end÷6],
#     kidx[(1(end÷6)+1):(2(end÷6))],
#     kidx[(2(end÷6)+1):(3(end÷6))],
#     kidx[(3(end÷6)+1):(4(end÷6))],
#     kidx[(4(end÷6)+1):(5(end÷6))],
#     kidx[(5(end÷6)+1):end],
# ]

#=
@time ΣQQ, ΣUU, ΣQU = Σsheets_k(θ, kidx_blk[1]);
@time pΣQQ, pΣUU, pΣQU =parallel_Σsheets_k(θ, kidx_blk[1]);
ΣQQ[2]
pΣQQ[2]
=#


@time filenm = @sblock let θ, φ, kidx_blk, 
                     #Σsheets_k = Σsheets_k, 
                     Σsheets_k = parallel_Σsheets_k, 
                     filenm = normpath(joinpath(HH.module_dir,"..","notebooks","L_kblock.jld2"))
    
    jld2file = jldopen(filenm, "w")
    write(jld2file, "θ", θ)
    write(jld2file, "φ", φ)
    write(jld2file, "kidx_blk", kidx_blk)
    
    nθx = length(θ)
    lower_tri_Idx = [CartesianIndex(r,c) for r=1:2nθx for c=1:2nθx if r>=c]
    write(jld2file, "lower_tri_Idx", lower_tri_Idx)

    Lsheet_names = String[]
    
    @showprogress for (i,k) ∈ enumerate(kidx_blk)
        ΣQQ, ΣUU, ΣQU = Σsheets_k(θ, k)

        #L = progress_map(ΣQQ, ΣQU, ΣUU) do mqq, mqu, muu 
        L = map(ΣQQ, ΣQU, ΣUU) do mqq, mqu, muu 
            m = Hermitian(
                [ mqq  mqu
                  mqu' muu ]
            )
            C = cholesky(m, Val(false), check=false)
            Lcol = C.L[lower_tri_Idx]
            if !issuccess(C) 
                Lcol[1] = NaN
            end
            Lcol
        end

        write(jld2file, "L$i", L)
        push!(Lsheet_names, "L$i")  
    end
    write(jld2file, "Lsheet_names", Lsheet_names)
    close(jld2file)

    return filenm 
end;



#%% use the saved cholesky decomps (in θ) for each frequency, in blocks, ...
#%% to simulate the field
#%% ----------------------------------------------------



simQx, simUx = @sblock let filenm, T

    jld2file = jldopen(filenm, "r")
    θ             = read(jld2file, "θ")
    φ             = read(jld2file, "φ")
    kidx_blk      = read(jld2file, "kidx_blk")
    Lsheet_names  = read(jld2file, "Lsheet_names")
    lower_tri_Idx = read(jld2file, "lower_tri_Idx")
    
    nθx = length(θ)
    nφk = length(φ)÷2+1
    𝒰row  = T(1/√(length(φ))) * plan_rfft(zeros(T,2nθx,length(φ)),2) 
    simQUk  = zeros(Complex{T}, 2nθx, nφk)

    Lstorage = LowerTriangular(zeros(Complex{T},2nθx,2nθx)) 
    @showprogress for blk_id ∈ 1:length(kidx_blk)
        L    = read(jld2file, Lsheet_names[blk_id])
        kidx = kidx_blk[blk_id]
        for (indx,k) in enumerate(kidx)
            if isfinite(L[indx][1])
                Lstorage[lower_tri_Idx] = L[indx]
                if (k==1) | (k==nφk)
                    simQUk[:,k] = Lstorage * complex.(randn(T,2nθx),0)
                else
                    simQUk[:,k] = Lstorage * complex.(randn(T,2nθx)./√2, randn(T,2nθx)./√2)
                end
            else 
                println("NaN at (indx, k) ", (indx,k))
            end 
        end
    end
    close(jld2file)
    simQUx  = 𝒰row \ simQUk

    simQx = simQUx[1:end÷2,:]
    simUx = simQUx[(end÷2+1):end,:]
    return simQx, simUx

end




fig,ax = subplots(2,1,figsize=(25/2,2*7))
ax[1].pcolormesh(φ[:,1:end÷4],θ,simQx[:,1:end÷4])
ax[2].pcolormesh(φ[:,1:end÷4],θ,simUx[:,1:end÷4])
fig.tight_layout()


fig,ax = subplots(1,2,figsize=(12,7),subplot_kw=Dict("projection"=>"polar"))
ax[1].pcolormesh(φ[:,1:end÷4],π .- θ,simQx[:,1:end÷4])
ax[2].pcolormesh(φ[:,1:end÷4],π .- θ,simUx[:,1:end÷4])
fig.tight_layout()


rad2deg.(diff(φ[1:end÷4])).*60 |> plot
rad2deg.(diff(θ)).*60 |> plot



#=

Δθ = 3 arcmins


jld2file = jldopen("example.jld2", "w") # open read-only (default)
L = read(jld2file, "L_kblock1") 
close(jld2file)



ΣQQ[10] |> matshow
ΣQQ[10] |> x->inv(x .+ 0.001 * minimum(diag(x)) * I(size(x,1))) |>  matshow
ΣQQ[10] |> x->inv(x .+ 0.001 * minimum(diag(x)) * I(size(x,1))) |>  x->plot(abs.(x[:,end÷2]))
ΣQQ[10] |> x->plot(abs.(x[:,end÷2]))

ΣQQ[50] |> eigvals |> plot

ΣQQ[200][:,end÷2] |> plot
inv(ΣQQ[10])[:,end÷2] .|> abs |> plot

plot(θ, inv(ΣQQ[1])[:,end÷2] .|> abs)
=#






#%% test the bandpowers of these maps
#%% ----------------------------------------------------





P, g = let T=T, Θpix = rad2deg(φ[2] - φ[1])*60, nside=1024
    P     = FieldFlows.Flat{Θpix,nside}
    g     = FieldFlows.r𝔽(P,T)
    P, g
end

#let 
    ell = collect(2:10_000)
    cl = FieldFlows.Cl{P,T}(
        ell,
        tt=cTTℓ.(ell), 
        ee=cEEℓ.(ell), 
        bb=cBBℓ.(ell),  
        te=cTEℓ.(ell), 
        ϕϕ=cϕϕℓ.(ell), 
    )

    qx = simQx[1:1024,1:1024]
    ux = simUx[1:1024,1:1024]

    # --------  pixel masking    
    αt = 1
    αd = 20
    dist_to_mid = (abs.(g.x[1]).^αd .+ abs.(g.x[2]).^αd).^(1/αd) # smoother boundary
    taper(x::A) where A<:Number =  ((x < 0) | !isfinite(x)) ? A(1) : (x < 1) ? A((cos(π*x)+1)/2) : A(0)
    a = (pixel_mask_lower=.9; pixel_mask_lower*g.period/2)
    b = (pixel_mask_upper=.95; pixel_mask_upper*g.period/2)
    Δab = b - a
    𝓂sK = taper.((dist_to_mid .- a) ./ Δab).^αt |> fftshift

    a = (pixel_mask_lower=.85; pixel_mask_lower*g.period/2)
    b = (pixel_mask_upper=.9; pixel_mask_upper*g.period/2)
    Δab = b - a
    𝓂sK2 = taper.((dist_to_mid .- a) ./ Δab).^αt |> fftshift

    #𝕄  = T.(𝓂sK) |> x->FieldFlows.QUmap{P,T}(x, x) |> FieldFlows.DiagOp


    p    = FieldFlows.QUmap{P,T}(fftshift(𝓂sK.* qx), .- fftshift(𝓂sK.*ux))
    pebmask    = FieldFlows.EBmap{P,T}(fftshift(𝓂sK2).*p[:ex],fftshift(𝓂sK2).*p[:bx])
    opp  = FieldFlows.DiagOp(FieldFlows.EBfourier{P,T}(cl.kmag, cl.kmag))
    pwr  = FieldFlows.bandpowers(g, p=opp * pebmask, Δl = 1)

    # g.cos2ϕk

    # qk = g * (𝓂sK.*qx)
    # uk = g * (𝓂sK.*ux)
    # ek = @.   qk * g.cos2ϕk + uk * g.sin2ϕk
    # # bk = @. - qk * g.sin2ϕk + uk * g.cos2ϕk
    # bk = @. qk * g.sin2ϕk + uk * g.cos2ϕk
    # bx = g \ (bk)
    # matshow(bx)
#end



fig,ax=subplots(1,1)
ax.loglog(ell, ell.^2 .* cEEℓ.(ell))
ax.loglog(ell, ell.^2 .* cBBℓ.(ell))
ax.loglog(pwr.kmag[:,1], pwr.Σee[:,1])
ax.loglog(pwr.kmag[:,1], pwr.Σbb[:,1])
fig.tight_layout()

fig,ax=subplots(2,2)
ax[1,1].pcolormesh(fftshift(p[:qx])); ax[1,1].set_title("Qx")
ax[1,2].pcolormesh(fftshift(p[:ux])); ax[1,2].set_title("Ux")
ax[2,1].pcolormesh(fftshift(pebmask[:ex])); ax[2,1].set_title("Ex")
ax[2,2].pcolormesh(fftshift(pebmask[:bx])); ax[2,2].set_title("Bx")
fig.tight_layout()



