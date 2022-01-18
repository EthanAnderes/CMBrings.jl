# A one dimensional smooth mask
# ====================================


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



   

# custom pcg with function composition (Minv * A \approx I)
# =====================================
function pcg(Minv::Function, A::Function, b, x=0*b; nsteps::Int=75, rel_tol = 0)
    r       = b - A(x)
    z       = Minv(r)
    p       = deepcopy(z)
    res     = dot(r,z)
    reshist = Vector{typeof(res)}()
    for i = 1:nsteps
        Ap        = A(p)
        α         = res / dot(p,Ap)
        x         = x + α * p
        r         = r - α * Ap
        z         = Minv(r)
        res′      = dot(r,z)
        p         = z + (res′ / res) * p
        rel_error = XFields.nan2zero(sqrt(dot(r,r)/dot(b,b)))
        push!(reshist, rel_error)
        if rel_error < rel_tol
            return x, reshist
        end
        res = res′
    end
    return x, reshist
end


function pcg_coupled(;
        _Aᵍ::Function, # preconditioner 
        A::Function,   # operator we want to invert
        b_g, b_f,      # solution we want is A⁻¹*vcat(b_g, f_g)
        x_g, x_f,      # warm start for solution
        nsteps=30, rel_tol = 0.0,
        reshist=Vector{Float64}() 
    )
    Ax_g, Ax_f = A(x_g, x_f)
    r_g  = b_g - Ax_g
    r_f  = b_f - Ax_f
    z_g, z_f  =  _Aᵍ(r_g, r_f)
    p_g  = deepcopy(z_g)
    p_f  = deepcopy(z_f)

    res   = dot(r_g,z_g) + dot(r_f,z_f)

    for i = 1:nsteps
        p′_g, p′_f = A(p_g, p_f)
        α    = res / (dot(p_g,p′_g) + dot(p_f,p′_f))
        x_g  += α * p_g
        x_f  += α * p_f
        r_g  -= α * p′_g
        r_f  -= α * p′_f
        z_g, z_f = _Aᵍ(r_g, r_f)
        res′ = dot(r_g,z_g) + dot(r_f,z_f)
        p_g  = z_g + (res′ / res) * p_g
        p_f  = z_f + (res′ / res) * p_f
        rel_error = (dot(r_g,r_g) + dot(r_f,r_f)) / (dot(b_g,b_g) + dot(b_f,b_f))
        if rel_error < rel_tol
            break 
        end
        push!(reshist, rel_error)
        res = res′
    end
    return x_g, x_f, reshist
end



# WF pcg
# =====================================

function update_f(
    Łϕ, EB::CircOp; 
    data,
    Pr, Qr, Bm, No, Pc⁻¹,
    ginit=0*data,
    pcg_nsteps, pcg_rel_tol=1e-10,
    ds...
)
    Łϕᴴ = Łϕ'
    C1a = Pr * Bm * Łϕ * EB * Łϕᴴ * Bm'
    C1b = Pr * No
    C2b = Qr * No
    ## C2a = Qr * Bm * Łϕ * EB * Łϕᴴ * Bm' # this one or ....
    C2a = Qr * Bm * EB * Bm' # .... this one
    ## C2a and C2b can be combine into one op.

    A = function (g)
        Prᴴ_g = Pr' * g
        Qrᴴ_g = Qr' * g
        tmp1a = C1a * Prᴴ_g
        tmp1b = C1b * Prᴴ_g
        tmp2a = C2a * Qrᴴ_g
        tmp2b = C2b * Qrᴴ_g
        return tmp1a + tmp1b + tmp2a + tmp2b
    end

    gwf, hst = pcg(
        g -> Pc⁻¹ * g, A, 
        data, ginit,
        nsteps=pcg_nsteps, rel_tol=pcg_rel_tol,
    )
    fwf   = EB *  Łϕᴴ * Bm' * Pr' * gwf
    return  fwf, gwf, hst
end

