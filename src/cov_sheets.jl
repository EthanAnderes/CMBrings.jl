# still workong on the format here ...



# Computing the covariance matrix of the Fourier coefficient at fixed frequency k 
# For rings as a function of θcol
# ==============================================


function Σsheets_k(θcol, φcol::Vector{T}, idxk, covf) where T<:Real
    nθx         = length(θcol)
    lowrΣT₁T₂   = zeros(T, length(idxk), nθx, nθx)

    Σ_chunck!(lowrΣT₁T₂, θcol, φcol, 1:nθx, idxk, covf)
    
    rtΣTT = map(1:length(idxk)) do k 
        Symmetric(lowrΣT₁T₂[k,:,:], :L)
    end 
    
    return rtΣTT 
end


function shared_Σsheets_k(θcol, φcol::Vector{T}, idxk, covf) where T<:Real
    nθx = length(θcol)
    lowrΣT₁T₂ = SharedArray{T,3}(
        (length(idxk), nθx,nθx), 
        init = S -> S[localindices(S)] = repeat([T(0)], length(localindices(S))),
        # pids = workers(),
    ) 

    jranges = split_col_ranges(nθx, nworkers())
    @sync begin
        for p in workers()
            @async remotecall_wait(
                Σ_chunck!, p, lowrΣT₁T₂, θcol, φcol, jranges[p-1], idxk, covf 
            )
        end
    end

    rtΣTT = map(1:length(idxk)) do k 
        Symmetric(lowrΣT₁T₂[k,:,:], :L)
    end 

    return rtΣTT
end

function Σ_chunck!(lowrΣT₁T₂, θcol, φcol, jrange, idxk, covf)
    nθx = length(θcol)
    𝒲col  = plan_rfft(similar(φcol))
    for j=jrange, i=j:nθx 
        T₁T₂ = 𝒲col * colΣ(θcol[i], θcol[j], φcol, covf)
        lowrΣT₁T₂[:,i,j] = real.(T₁T₂[idxk])
    end
end


# covf should be of the form covf(θ1::Number, θ2::Number, Δφcol::Vector) 
colΣ(θ1, θ2, φcol, covf) = covf(θ1, θ2, φcol .- φcol[1])    


# misc 
# --------------------------------


function split_col_ranges(ncols,nwrks)
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


# TODO: set for removal
# function pixunitary(::Type{T}, nθx, nφx) where {T<:Number} 
#     randn(T, nθx, nφx)
# end


# This can be used to help define the pixel space covariance function
function geoθ1θ2Δφcol(θ1, θ2, Δφcol)
    sθ1, sθ2 = sin(θ1), sin(θ2)
    sΔθ½     = sin((θ1 - θ2)/2)
    sΔφ½     = @. sin(Δφcol / 2)
    β        = @. 2asin(√(sΔθ½^2 + sθ1 * sθ2 * sΔφ½^2))
    return β
end




# Storage container for memory mapped object holding covariance sheets
# ==============================================



mutable struct AzCov{T, d, spin, Fplan<:AbstractFFTs.ScaledPlan, U}
    filenm::String
    jld2file::JLD2.JLDFile{JLD2.MmapIO}
    Urow::Fplan
    nθx::Int
    nφx::Int
    nφk::Int
    θcol::Array{T,1}
    φcol::Array{T,1} 
    kidx_blk::U
    Lsheet_names::Array{String,1}
    lower_tri_Idx::Array{CartesianIndex{2},1}

    function AzCov(::Type{T}, filenm::String; d::Int=2, spin::Int=0) where T<:Real
        jld2file = jldopen(filenm, "r")
        θcol     = read(jld2file, "θcol")
        φcol     = read(jld2file, "φcol")
        nθx      = length(θcol)
        nφx      = length(φcol)
        nφk      = length(φcol)÷2+1
        Urow     = T(1/√(nφx)) * plan_rfft(zeros(T, nθx, nφx),2) 

        kidx_blk      = read(jld2file, "kidx_blk")
        Lsheet_names  = read(jld2file, "Lsheet_names")
        lower_tri_Idx = read(jld2file, "lower_tri_Idx")

        cs = new{T, d, spin, typeof(Urow), typeof(kidx_blk)}(filenm, jld2file, Urow, nθx, nφx, nφk, T.(θcol), T.(φcol),  kidx_blk, Lsheet_names, lower_tri_Idx)       
        finalizer(c->close(c.jld2file), cs)
        return cs 
    end
end    


function AzCov(covf, θcol::AA, φcol::AA, kidx_blk; d::Int=2, spin::Int=0) where {T, AA<:Array{T,1}}

    dirsave  = mktempdir()
    filenm   = joinpath(dirsave,"L_kblock.jld2")
    jld2file = jldopen(filenm, "w")
    @show filenm
    @show jld2file

    nθx = length(θcol)
    Lsheet_names = String[]
    lower_tri_Idx = [CartesianIndex(r,c) for r=1:nθx for c=1:nθx if r>=c]
    ## lower_tri_Idx = [CartesianIndex(r,c) for r=1:2nθx for c=1:2nθx if r>=c]

    @showprogress for (i,ki) ∈ enumerate(kidx_blk)
        ## ΣTT = CMBrings.Σsheets_k(θcol, φcol, ki, covf)
        ΣTT = CMBrings.shared_Σsheets_k(θcol, φcol, ki, covf)

        L = map(ΣTT) do mtt
            C = cholesky(mtt, Val(false), check=false)
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
    write(jld2file, "θcol", θcol)
    write(jld2file, "φcol", φcol)
    write(jld2file, "kidx_blk", kidx_blk)
    write(jld2file, "lower_tri_Idx", lower_tri_Idx)

    close(jld2file)

    return AzCov(T, filenm; d=d, spin=spin)
end



function Base.:*(cs::AzCov{T,2,0}, fx::Array{T,2}) where T<:Real
    ifk  = cs.Urow * fx
    ofk  = zero(ifk)
    
    Lstorage = LowerTriangular(zeros(Complex{T},cs.nθx, cs.nθx))     
    for blk_id ∈ 1:length(cs.kidx_blk)
        L    = read(cs.jld2file, cs.Lsheet_names[blk_id])
        kidx = cs.kidx_blk[blk_id]
        for (indx,k) in enumerate(kidx)
            Lstorage[cs.lower_tri_Idx] = L[indx]
            mul!(view(ofk,:,k), Lstorage', view(ifk,:,k))
            lmul!(Lstorage, view(ofk,:,k))
            #ofk[:,k] = Lstorage * (Lstorage' * ifk[:,k])
            if !isfinite(L[indx][1])
                println("NaN at (indx, k) ", (indx,k))
            end 
        end
    end
    ofx = cs.Urow \ ofk
    ofx
end 


function Base.:\(cs::AzCov{T,2,0}, fx::Array{T,2}) where T<:Real
    ifk  = cs.Urow * fx
    ofk  = zero(ifk)
    
    Lstorage = LowerTriangular(zeros(Complex{T},cs.nθx, cs.nθx))     
    for blk_id ∈ 1:length(cs.kidx_blk)
        L    = read(cs.jld2file, cs.Lsheet_names[blk_id])
        kidx = cs.kidx_blk[blk_id]
        for (indx,k) in enumerate(kidx)
            Lstorage[cs.lower_tri_Idx] = L[indx]
            ldiv!(view(ofk,:,k), Lstorage, view(ifk,:,k))
            ldiv!(Lstorage', view(ofk,:,k))
            #ofk[:,k] = Lstorage' \ (Lstorage \ ifk[:,k])
            if !isfinite(L[indx][1])
                println("NaN at (indx, k) ", (indx,k))
            end 
        end
    end
    ofx = cs.Urow \ ofk
    ofx
end 

function ksupport(cs::AzCov{T,2,0}, fx::Array{T,2}) where T<:Real
    ifk  = cs.Urow * fx
    ofk  = zero(ifk)
    
    for blk_id ∈ 1:length(cs.kidx_blk)
        kidx = cs.kidx_blk[blk_id]
        for (indx,k) in enumerate(kidx)
            ofk[:,k] = ifk[:,k]
        end
    end
    
    ofx = cs.Urow \ ofk
    ofx
end

function cholmul(cs::AzCov{T,2,0}, fx::Array{T,2}) where T<:Real
    ifk  = cs.Urow * fx
    ofk  = zero(ifk)
    
    Lstorage = LowerTriangular(zeros(Complex{T},cs.nθx, cs.nθx))     
    for blk_id ∈ 1:length(cs.kidx_blk)
        L    = read(cs.jld2file, cs.Lsheet_names[blk_id])
        kidx = cs.kidx_blk[blk_id]
        for (indx,k) in enumerate(kidx)
            Lstorage[cs.lower_tri_Idx] = L[indx]
            mul!(view(ofk,:,k), Lstorage, view(ifk,:,k))
            if !isfinite(L[indx][1])
                println("NaN at (indx, k) ", (indx,k))
            end 
        end
    end
    ofx = cs.Urow \ ofk
    ofx
end 



function choldiv(cs::AzCov{T,2,0}, fx::Array{T,2}) where T<:Real
    ifk  = cs.Urow * fx
    ofk  = zero(ifk)
    
    Lstorage = LowerTriangular(zeros(Complex{T},cs.nθx, cs.nθx))     
    for blk_id ∈ 1:length(cs.kidx_blk)
        L    = read(cs.jld2file, cs.Lsheet_names[blk_id])
        kidx = cs.kidx_blk[blk_id]
        for (indx,k) in enumerate(kidx)
            Lstorage[cs.lower_tri_Idx] = L[indx]
            ldiv!(view(ofk,:,k), Lstorage, view(ifk,:,k))
            if !isfinite(L[indx][1])
                println("NaN at (indx, k) ", (indx,k))
            end 
        end
    end
    ofx = cs.Urow \ ofk
    ofx
end 

