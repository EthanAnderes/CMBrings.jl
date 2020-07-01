
# still workong on the format here ...

function Σsheets_k(θ, idxk, covf, φcol::Vector{T}) where T<:Real
    nθx         = length(θ)
    lowrΣT₁T₂   = zeros(T, length(idxk), nθx, nθx)

    Σ_chunck!(lowrΣT₁T₂, 1:nθx, θ, idxk, covf, φcol)
    
    rtΣTT = map(1:length(idxk)) do k 
        Symmetric(lowrΣT₁T₂[k,:,:], :L)
    end 
    
    return rtΣTT 
end


function shared_Σsheets_k(θ, idxk, covf, φcol::Vector{T}) where T<:Real
    nθx = length(θ)
    lowrΣT₁T₂ = SharedArray{T,3}(
        (length(idxk), nθx,nθx), 
        init = S -> S[localindices(S)] = repeat([T(0)], length(localindices(S))),
        # pids = workers(),
    ) 

    jranges = split_col_ranges(nθx,nworkers())
    @sync begin
        for p in workers()
            @async remotecall_wait(
                Σ_chunck!, p, lowrΣT₁T₂, jranges[p-1], θ, idxk, covf, φcol
            )
        end
    end

    rtΣTT = map(1:length(idxk)) do k 
        Symmetric(lowrΣT₁T₂[k,:,:], :L)
    end 

    return rtΣTT
end

function Σ_chunck!(lowrΣT₁T₂, jrange, θ, idxk, covf, φcol::Vector{T}) where T<:Real
    nθx = length(θ)
    𝒲col  = plan_rfft(similar(φcol))
    for j=jrange, i=j:nθx 
        T₁T₂ = colΣ(θ[i],θ[j], covf, 𝒲col, φcol)
        lowrΣT₁T₂[:,i,j] = real.(T₁T₂[idxk])
    end
end


function colΣ(θ1, θ2, covf, 𝒲col, φcol)
    θv = colθ1θ2(θ1, θ2, φcol)     
    T₁T₂ = 𝒲col * covf(θv)
    return T₁T₂
end

function colθ1θ2(θ1, θ2, φcol)
    sθ1, sθ2 = sin(θ1), sin(θ2)
    sΔθ½     = sin((θ1 - θ2)/2)
    sΔφ½     = @. sin(φcol / 2)
    β        = @. 2asin(√(sΔθ½^2 + sθ1 * sθ2 * sΔφ½^2))
    return β
end


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




