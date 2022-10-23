
# lensing-spin2
# ====================================

function ∇ll_ϕf′_usingf(
    ϕ, f, Φ½::CircOp, EB½::CircOp; 
    data, Ł, Ð⁻¹, M, B, N⁻¹, 
    ϕ2v!, ϕ2vᴴ!, ∇!, grad_nsteps=14, ds...)

    L    = Ł(ϕ)
    Lᴴ   = Ł(ϕ)'
    lnf  = L * f
    tmf    = fieldtransform(f)
    tmϕ    = fieldtransform(ϕ)
    Ma     = DiagOp(Xmap(tmf, abs.(M[:]).>0))    
    dΔlnf  = B' * Ma * (N⁻¹ * (M \ (data - M * (B * lnf))))
    # dΔlnf  = B' * Ma * (Ncov \ (M \ (data - M * (B * lnf))))

    # --------------------------
    ϕx  = ϕ[:]
    vx  = (similar(ϕx), similar(ϕx)) # can you do this without hardcoding it?
    ϕ2v!(vx, ϕx)

    g1x = dΔlnf[L] 
    f1x = lnf[L]   
    τvx₀ = FieldLensing.ᴴ∂Łfx_∂vx(g1x, f1x, vx, ∇!, grad_nsteps)
    
    g0x  = (Ð⁻¹' * (Lᴴ*dΔlnf - EB½'\(EB½\f)))[L]
    
    f0x  = f[L]
    τvx₁ = FieldLensing.ᴴ∂Ł⁻¹fx_∂vx(g0x, f0x, vx, ∇!, grad_nsteps)

    τvx  = τvx₀ .+ τvx₁

    τϕx = similar(τvx[1])
    ϕ2vᴴ!(τϕx, τvx)

    τϕ = Xmap(tmϕ, τϕx) - Φ½'\(Φ½\ϕ)
    # --------------------------

    return τϕ
end





# log likelihood and quasi-gibbs and optimization updates
function ll_ϕf′(
    ϕ, f′, Φ½::CircOp, EB½::CircOp; 
    data, Ł, Ð⁻¹, M, B, N⁻¹, ds...)

    L    = Ł(ϕ)
    f    = Ð⁻¹ * (L \ f′)
    lnf  = L * f
    estn = data - M * B * lnf
    z1   = M \ (N⁻¹ * (M' \ estn))
    # This needs checking ...
    # !!!!!!!!
    # z1   = (M' * Noise_ring * M) \ estn
    z2   = Φ½ \ ϕ
    z3   = EB½ \ f
    rtn =  - EZ.sum_kbn([dot(estn,z1), dot(z2,z2), dot(z3,z3)]) / 2
    return isnan(rtn) ? -inv(zero(rtn)) : rtn
end


      
#  linesearch updates for ϕ
function linesearch_ϕf′(
        inHgrad, ϕ, f′, Φ½::CircOp, EB½::CircOp; 
        data, Ł, Ð⁻¹, M, B, N⁻¹,
        seconds_max = 0, # seconds 
        eval_max    = 0,
        upper_bound = 2, 
        startval = 0,
        stopval  = Inf, #  stop when an objective value of at least stopval is found. 
        ftol_rel = 0,    #  relative tolerance on function value. 
        ftol_abs = 0,    #  absolute tolerance on function value. 
        xtol_rel = 0,    #  relative tolerance on arg value. 
        xtol_abs = 0,    #  absolute tolerance on arg value.
        solver = :LN_COBYLA,  
        ds...)

    # solvers :LN_SBPLX :LN_NELDERMEAD, :LN_COBYLA
    opt = NLopt.Opt(solver, 1)
    opt.maxtime      = seconds_max
    opt.maxeval      = eval_max
    opt.upper_bounds = [upper_bound]
    opt.lower_bounds = [0]
    opt.stopval  = stopval
    opt.ftol_rel = ftol_rel
    opt.ftol_abs = ftol_abs
    opt.xtol_rel = xtol_rel
    opt.xtol_abs = xtol_abs


    ϕₒ, inHgradₒ = promote(ϕ, inHgrad)
    
    opt.max_objective = function (β, grad)
        ϕβ = ϕₒ + β[1] * inHgradₒ       
        return ll_ϕf′(ϕβ, f′, Φ½, EB½; data, Ł, Ð⁻¹, M, B, N⁻¹)
    end
    
    ll_opt, β_opt, = NLopt.optimize(opt,  [startval])
    
    return β_opt[1]
end

   




# This is currently used in lensing-spin0 ... 
# TODO: merge with lensing-spin2 version 
# =====================================

# gradient of log likelihood of f′ w.r.t ϕ
function ∇ll_ϕf′(ϕ, f′; data, Ł, Ð, Pr, Bm, CMBcov, Φcov, Ncov, ϕ2v!, ϕ2vᴴ!, ∇!, grad_nsteps=14, ds...)
    L    = Ł(ϕ)
    Lᴴ   = Ł(ϕ)'
    D    = Ð
    Dᴴ   = map(B->adjoint(Matrix(B)), Ð) |> AzBlock

    f    = D \ (L \ f′)
    lnf  = L * f

    tmf    = fieldtransform(f′)
    tmϕ    = fieldtransform(ϕ)

    Ma     = DiagOp(Xmap(tmf, abs.(Pr[:]).>0))
    dΔlnf  = Bm' * Ma * (Ncov \ (Pr \ (data - Pr * (Bm * lnf))))

    # --------------------------
    ϕx  = ϕ[:]
    vx  = (similar(ϕx), similar(ϕx)) # can you do this without hardcoding it?
    ϕ2v!(vx, ϕx)

    g1x = dΔlnf[L] 
    f1x = lnf[L]   
    τvx₀ = FieldLensing.ᴴ∂Łfx_∂vx(g1x, f1x, vx, ∇!, grad_nsteps)
    
    g0x  = (Dᴴ\(Lᴴ*dΔlnf - CMBcov\f))[L]
    f0x  = f[L]
    τvx₁ = FieldLensing.ᴴ∂Ł⁻¹fx_∂vx(g0x, f0x, vx, ∇!, grad_nsteps)

    τvx  = τvx₀ .+ τvx₁

    τϕx = similar(τvx[1])
    ϕ2vᴴ!(τϕx, τvx)

    τϕ = Xmap(tmϕ, τϕx) - Φcov \ ϕ
    # --------------------------

    return τϕ
end


# log likelihood and quasi-gibbs and optimization updates
function ll_ϕf′(ϕ, f′; data, Ł, Ð, CMBcov, Φcov, Ncov, Pr, Bm, ds...)
    L    = Ł(ϕ)
    f    = Ð \ (Ł(ϕ) \ f′)
    lnf  = Ł(ϕ) * f
    estn = data - Pr * Bm * lnf
    z1   = (Pr' * Ncov * Pr) \ estn
    z2   = Φcov \ ϕ
    z3   = CMBcov \ f
    return - EZ.sum_kbn([dot(estn,z1), dot(ϕ,z2), dot(f,z3)]) / 2
end

#  linesearch updates for ϕ
function linesearch_ϕf′(inHgrad, ϕ, f′; 
        data, Ł, Ð, CMBcov, Φcov, Ncov, Pr, Bm, 
        seconds_max = 0, # seconds 
        eval_max    = 0,
        upper_bound = 2, 
        stopval  = Inf, #  stop when an objective value of at least stopval is found. 
        ftol_rel = 0,    #  relative tolerance on function value. 
        ftol_abs = 0,    #  absolute tolerance on function value. 
        xtol_rel = 0,    #  relative tolerance on arg value. 
        xtol_abs = 0,    #  absolute tolerance on arg value.
        solver = :LN_COBYLA,  
        ds...)

    # solvers :LN_SBPLX :LN_NELDERMEAD, :LN_COBYLA
    opt = NLopt.Opt(solver, 1)
    opt.maxtime      = seconds_max
    opt.maxeval      = eval_max
    opt.upper_bounds = [upper_bound]
    opt.lower_bounds = [0]
    opt.stopval  = stopval
    opt.ftol_rel = ftol_rel
    opt.ftol_abs = ftol_abs
    opt.xtol_rel = xtol_rel
    opt.xtol_abs = xtol_abs

    ϕₒ, inHgradₒ = promote(ϕ, inHgrad)
    
    opt.max_objective = function (β, grad)
        ϕβ = ϕₒ + β[1] * inHgradₒ       
        return ll_ϕf′(ϕβ, f′; data, Ł, Ð, CMBcov, Φcov, Ncov, Pr, Bm)
    end
    
    ll_opt, β_opt, = NLopt.optimize(opt,  [0])
    
    return β_opt[1]
end






