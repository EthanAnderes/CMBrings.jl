module CMBrings

using XFields
using FFTransforms
using FieldLensing
using SphereTransforms  
using Spectra # needed for complex_circ_rings

using LinearAlgebra
using Statistics 
using SharedArrays
using Distributed
using JLD2
using FFTW
using ProgressMeter
using PyPlot
using PyCall
using NLopt

const module_dir  = joinpath(@__DIR__, "..") |> normpath

# Extras on SphereTransforms like simulation, getindex etc. 
include("transformations.jl")

include("complex_circ_rings.jl")

# the latest prototype for covariance type, spin0 only
include("az_blocks.jl")

# RingBeam struct for beaming an azimuthal strip on the full sky
include("ring_beam.jl")

include("lensing.jl")

include("likelihoods.jl")

include("methods.jl")

include("plot.jl")


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


# moved to likelihoods.jl, methods.jl or transofrmations.jl
# =====================================
#=
function update_ϕ(ϕ, lnf, data; Pr, NΦNaz, Σaz_fctr, Naz_fctr, Φaz_fctr, Baz, ϕ2v, ϕ2vᴴ, Ł, ∇!, tmU, grad_nsteps, linesearch_time_max, solver = :LN_COBYLA,  ds...)
    # here are a couple other solvers :LN_SBPLX :LN_NELDERMEAD, :LN_COBYLA

    gradϕ   = CMBrings.∇ϕ(ϕ, lnf, data; Pr, Σaz_fctr, Naz_fctr, Baz, ϕ2v, ϕ2vᴴ, Ł, ∇!, tmU, grad_nsteps)
    # inHgrad = NΦNaz * ((Φaz_fctr * gradϕ) - ϕ) 
    inHgrad = NΦNaz * gradϕ - NΦNaz * (Φaz_fctr \ ϕ) 
    ## Note that ∇ϕ skips the Φ⁻¹⋅ϕ term ... so it is added to inHgrad. 
    ## With the approx inverse Hessian of the form (Φ⁻¹ + N⁻¹)⁻¹ = N(Φ + N)⁻¹Φ 
    ## we get to cancel it out so that (Φ⁻¹ + N⁻¹)⁻¹⋅Φ⁻¹⋅ϕ == N(Φ + N)⁻¹⋅ϕ

    T   = eltype_in(tmU)
    opt = NLopt.Opt(solver, 1)
    opt.maxtime      = linesearch_time_max
    opt.upper_bounds = T[1.0]
    opt.lower_bounds = T[0]
    opt.max_objective = function (β, grad)
        ϕβ = ϕ + β[1] * inHgrad
        lllnf(ϕβ, lnf, Ł, Σaz_fctr) + llϕ(ϕβ, Φaz_fctr) 
    end

    ll_opt, β_opt, = NLopt.optimize(opt,  T[0])
    @show ll_opt, β_opt
    
    return inHgrad, β_opt[1]
end


function update_ϕ(gradϕ, ϕ, lnf, data; Pr, NΦNaz, Σaz_fctr,  Φaz_fctr, Ł, ∇!, tmU, linesearch_time_max, solver = :LN_COBYLA,  ds...)
    # here are a couple other solvers :LN_SBPLX :LN_NELDERMEAD, :LN_COBYLA
    inHgrad = NΦNaz * gradϕ - NΦNaz * (Φaz_fctr \ ϕ) 

    T   = eltype_in(tmU)
    opt = NLopt.Opt(solver, 1)
    opt.maxtime      = linesearch_time_max
    opt.upper_bounds = T[1.0]
    opt.lower_bounds = T[0]
    opt.max_objective = function (β, grad)
        ϕβ = ϕ + β[1] * inHgrad
        lllnf(ϕβ, lnf, Ł, Σaz_fctr) + llϕ(ϕβ, Φaz_fctr) 
    end

    ll_opt, β_opt, = NLopt.optimize(opt,  T[0])
    @show ll_opt, β_opt
    
    return inHgrad, β_opt[1]
end


function ∇ϕ(ϕ, lnf, data; Pr, Σaz_fctr, Naz_fctr, Baz, ϕ2v, ϕ2vᴴ, Ł, ∇!, tmU, grad_nsteps, ds...)
    ## Remark: for the next line to be correct Naz_fctr must be diagonal in pixel space
    Ma        = DiagOp(Xmap(tmU, abs.(Pr[:]).>0))
    dΔlnf     = Baz' * (Ma * (Naz_fctr \ (Pr \ (data - Pr * (Baz * lnf)))))
    v         = ϕ2v(ϕ)
    f         = Ł(ϕ) \ lnf 
    τŁ₀₁      = CMBrings.FieldLensing.τArrayLense(v, (f[:],), ∇!, 0, 1, grad_nsteps)
    τŁ₁₀      = CMBrings.FieldLensing.τArrayLense(v, (lnf[:],), ∇!, 1, 0, grad_nsteps)        
    τv₀, τf   = τŁ₁₀(map(zero,v),  (dΔlnf[:],))
    ∇f        = Xmap(tmU, τf[1]) - Σaz_fctr \ f
    τv₁, τlnf = τŁ₀₁(τv₀,  (∇f[:],))
    return ϕ2vᴴ(τv₁) #  - Φaz_fctr \ ϕ # this last term is added later
end



function llϕ(ϕ,  Φaz_fctr)
    w  = llfield(ϕ, Φaz_fctr)
    wx = w[:] 
    - dot(wx,wx) / 2 
end


function lllnf(ϕ, lnf, Ł, Σaz_fctr)
    f  =  Ł(ϕ) \ lnf
    w  = llfield(f, Σaz_fctr)
    wx = w[:] 
    - dot(wx,wx) / 2 
end


function llfield(f, Σaz_fctr::AzBlock{M}) where {M<:Eigen}
    v  = deepcopy(f[!])
    w  = similar(v)
    wk = collect(eachcol(w))
    vk = collect(eachcol(v))
    Threads.@threads for i ∈ eachindex(vk)
        mul!(wk[i], Σaz_fctr[i].vectors', vk[i])
        wk[i] .*= pinv.(sqrt.(Σaz_fctr[i].values))
        ## mul!(wk[i], Σaz_fctr[i].vectors, vk[i])
    end
    Xfourier(fieldtransform(f), w)
end


function llfield(f, Σaz_fctr::AzBlock{M}) where {M<:Cholesky}
    w  = deepcopy(f[!])
    wk = collect(eachcol(w))
    Threads.@threads for i ∈ eachindex(wk)
        lmul!(Σaz_fctr[i].L, wk[i])
    end
    Xfourier(fieldtransform(f), w)
end



=#

end
