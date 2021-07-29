# Useful grid generation
# =====================================

function θ_healpix_j_Nside(j_Nside) 
    0 < j_Nside < 1  ? acos(1-abs2(j_Nside)/3)      :
    1 ≤ j_Nside ≤ 3  ? acos(2*(2-j_Nside)/3)        :
    3 < j_Nside < 4  ? acos(-(1-abs2(4-j_Nside)/3)) : 
    error("argument ∉ (0,4)")
end

θ_healpix(Nside) = θ_healpix_j_Nside.((1:4Nside-1)/Nside)

θ_equicosθ(N)    = acos.( ((N-1):-1:-(N-1))/N )

θ_equiθ(N)       = π*(1:N-1)/N

function θ_grid(;θspan::Tuple{<:Real,<:Real}, N::Int, type=:equiθ)
    @assert N > 0
    @assert 0 <= θspan[1] < θspan[2] <= π

    # θgrid′ is the full grid from 0 to π
    if type==:equiθ
        θgrid′ = θ_equiθ(N)
    elseif type==:equicosθ
        θgrid′ = θ_equicosθ(N)
    elseif type==:healpix
        θgrid′ = θ_healpix(N)
    else
        error("`type` is not valid. Options include `:equiθ`, `:equicosθ` or `:healpix`")
    end 

    # θgrid′′ subsets θgrid′ to be within θspan
    # δ½south′′ and δ½north′′ are the arclength midpoints to the adjacent pixel
    θgrid′′   = θgrid′[θspan[1] .≤ θgrid′ .≤ θspan[2]]
    δ½south′′ = (circshift(θgrid′′,-1)  .- θgrid′′) ./ 2
    δ½north′′ = (θgrid′′ .- circshift(θgrid′′,1)) ./ 2   
    
    # now restrict to the interior of the range of θgrid′′
    θ       = θgrid′′[2:end-1]
    δ½south = δ½south′′[2:end-1]
    δ½north = δ½north′′[2:end-1]
    # Δθ      = @. δ½south + δ½north
    # Δz      = @. cos(θ - δ½north) - cos(θ + δ½south)

    # These are the pixel boundaries along polar
    # so length(θ∂) == length(θ)+1
    θ∂ = vcat(θ[1] .- δ½north[1], θ .+ δ½south)

    θ, θ∂, type 
end 


function φ_grid(;φspan::Tuple{T,T}, N::Int) where T<:Real

    @assert N > 0
    # TODO: relax this condition ...
    @assert 0 <= φspan[1] < φspan[2] <= 2π 

    φ∂    = collect(φspan[1] .+ (φspan[2] - φspan[1])*(0:N)/N)
    Δφ    = φ∂[2] - φ∂[1]
    φ     = φ∂[1:end-1] .+ Δφ/2
    
    φ, φ∂
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

