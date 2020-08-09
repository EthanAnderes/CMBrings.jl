module CMBrings

using XFields
using FFTransforms
using FieldLensing
using SphereTransforms  # specify what we need from this
using HealpixTransforms # do we need this?

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

include("az_blocks.jl")

include("az_cov.jl")

include("lensing.jl")

include("methods.jl")

include("plot.jl")

LinearAlgebra.dot(f::Xfield{FT},g::Xfield{FT}) where FT<:𝕎 = dot(f[:], g[:])




function update_lnf_f(ϕ, data; Pr, Qr, Ł, tmU, Σaz_fctr, Naz_fctr, Baz, Precon_fctr, pcg_nsteps, ds...)

    Ln    = Ł(ϕ)
    Lnᴴ   = Ln'
    f′    = az_sim(tmU, Σaz_fctr)
    data′ = Pr * (Baz * (Ln * f′) + az_sim(tmU, Naz_fctr))
    
    # these make the multiplications faster ...
    Baz′ = Baz'
    mΣaz = map(Matrix, Σaz_fctr) |> AzBlock
    mNaz = map(Matrix, Naz_fctr) |> AzBlock
    mPrecon = map(Matrix, Precon_fctr) |> AzBlock

    A = function (g)
        tmp0  = Pr * (Baz * (Ln * (mΣaz * (Lnᴴ * (Baz′ * (Pr' * g))))))
        tmp1  = Pr * (mNaz * (Pr' * g))
        tmp2  = Qr * (mPrecon * (Qr' * g))   
        return tmp0 + tmp1 + tmp2
    end 

    gwf, hst = pcg(
        g -> Precon_fctr \ g, A, 
        data + data′, 
        nsteps=pcg_nsteps, rel_tol=1e-3,
    )

    fsim    = mΣaz * ( Lnᴴ * (Baz′ * (Pr' * gwf)))
    fsim   -= f′
    lnfsim  = Ln * fsim

    return  lnfsim, fsim, hst
end

 


function ll(ϕ, lnf, data; Ł, Σaz_fctr, Φaz_fctr, ds...)
    f    =  Ł(ϕ) \ lnf 
    rtn  = - ( dot(f[:], (Σaz_fctr \ f)[:])  + dot(ϕ[:], (Φaz_fctr \ ϕ)[:]) ) / 2 
    ## Δdlnf = data - Baz * lnf
    ## rtn += -dot(Δdlnf, Naz \ Δdlnf) / 2 
    rtn 
end

function ∇ϕ(ϕ, lnf, data; Pr, Σaz_fctr, Naz_fctr, Φaz_fctr, Baz, ϕ2v, ϕ2vᴴ, Ł, ∇!, tmU, grad_nsteps, ds...)
    
    dΔlnf     = Baz' * (Pr' * (Naz_fctr \ (data - Pr * (Baz * lnf))))
    v         = ϕ2v(ϕ)
    f         = Ł(ϕ) \ lnf 
    τŁ₀₁      = CMBrings.FieldLensing.τArrayLense(v, (f[:],), ∇!, 0, 1, grad_nsteps)
    τŁ₁₀      = CMBrings.FieldLensing.τArrayLense(v, (lnf[:],), ∇!, 1, 0, grad_nsteps)        
    τv₀, τf   = τŁ₁₀(map(zero,v),  (dΔlnf[:],))
    ∇f        = Xmap(tmU, τf[1]) - Σaz_fctr \ f
    τv₁, τlnf = τŁ₀₁(τv₀,  (∇f[:],))
    return ϕ2vᴴ(τv₁) - Φaz_fctr \ ϕ
end

function update_ϕ(ϕ, lnf, data; Pr, bHϕaz, Σaz_fctr, Naz_fctr, Φaz_fctr, Baz, ϕ2v, ϕ2vᴴ, Ł, ∇!, tmU, grad_nsteps, linesearch_time_max,  ds...)

    nH⁻¹∇ϕ = bHϕaz * ∇ϕ(ϕ, lnf, data; Pr, Σaz_fctr, Naz_fctr, Φaz_fctr, Baz, ϕ2v, ϕ2vᴴ, Ł, ∇!, tmU, grad_nsteps)
    ## nH⁻¹∇ϕ = 0.01 * (Φaz * ∇ϕ(ϕ, lnf, f, data))

    ## solver = :LN_SBPLX 
    solver = :LN_COBYLA
    ## solver = :LN_NELDERMEAD
    T   = eltype_in(tmU)
    opt = NLopt.Opt(solver, 1)
    opt.maxtime      = linesearch_time_max
    opt.upper_bounds = T[1.0]
    opt.lower_bounds = T[0]
    ## opt.initial_step = T[0.00001]
    opt.max_objective = function (β, grad)
        ll(ϕ + β[1] * nH⁻¹∇ϕ, lnf, data; Ł, Σaz_fctr, Φaz_fctr)
    end

    ll_opt, β_opt, = NLopt.optimize(opt,  T[0])
    @show ll_opt, β_opt
    
    return ϕ + β_opt[1] * nH⁻¹∇ϕ
end




end
