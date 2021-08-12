   

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

