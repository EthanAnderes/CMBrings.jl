

# custom pcg with function composition (Minv * A \approx I)
# ---------------------------------------------------------
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
    Ln; 
    data,
    Pr, Qr, CMBcov, Ncov, Bm, Precon,
    ginit=0*data,
    pcg_nsteps, pcg_rel_tol=1e-10,
    simdata=0*data, simf=0*data, 
    ds...
)
    Lnᴴ = Ln'
   
    # these make the multiplications faster ...
    mCMBcov = map(Matrix, CMBcov) |> AzBlock
    mNcov   = map(Matrix, Ncov)   |> AzBlock
    mPrecon = map(Matrix, Precon) |> AzBlock
    C0 = Pr * Bm * Ln * mCMBcov * Lnᴴ * Bm'
    C1 = Pr * mNcov
    C2 = Qr * mPrecon * Qr'

    A = function (g)
        Prᴴ_g = Pr' * g
        tmp0  = C0 * Prᴴ_g
        tmp1  = C1 * Prᴴ_g
        tmp2  = C2 * g   
        return tmp0 + tmp1 + tmp2
    end 

    gwf, hst = pcg(
        g -> Precon \ g, 
        A, 
        data + simdata,
        ginit,
        nsteps=pcg_nsteps, rel_tol=pcg_rel_tol,
    )

    fsim      = mCMBcov *  Lnᴴ * Bm' * Pr' * gwf
    fsim_out  = fsim - simf

    return  fsim_out, gwf, hst
end
