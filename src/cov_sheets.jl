# still workong on the format here ...




# Storage container for memory mapped object holding covariance sheets
# ==============================================



mutable struct AzCov{Tf, szf, spin, Fplan<:AbstractFFTs.ScaledPlan}
    filenm::String
    jld2file::JLD2.JLDFile{JLD2.MmapIO}
    Urow::Fplan
    ks_Σs_sheet_names::Array{String,1}
    function AzCov{Tf, szf, spin}(filenm::String, ks_Σs_sheet_names::Array{String,1}) where {Tf<:Number, szf, spin}
        jld2file = jldopen(filenm, "r")
        Urow     = Tf(1/√(szf[end])) * plan_rfft(zeros(Tf, szf), length(szf)) 
        cs = new{Tf, szf, spin, typeof(Urow)}(filenm, jld2file, Urow, ks_Σs_sheet_names)       
        finalizer(c->close(c.jld2file), cs)
        return cs 
    end
end    

# e.g. mapΣf = Σ -> cholesky(Σ, Val(false), check=false)
function AzCov(mapΣf, covf, θcol::AA, φcol::AA, kidx_blk; Σsymmetric=true, filename="L_kblock.jld2", dirsave=mktempdir(), spin=0) where {Tf, AA<:Array{Tf,1}}
    
    filenm   = joinpath(dirsave, filename)
    jld2file = jldopen(filenm, "w")
    @show filenm
    @show jld2file

    szf = (length(θcol), length(φcol))
    ks_Σs_sheet_names = String[]

    @showprogress for (i,ki) ∈ enumerate(kidx_blk)
        if Σsymmetric
            ΣTT = CMBrings.shared_Σsheets_k(θcol, φcol, ki, covf)
        else
            ΣTT = CMBrings.nonsym_shared_Σsheets_k(θcol, φcol, ki, covf)
        end
        L = map(ΣTT) do mtt
            mapΣf(mtt)
        end
        write(jld2file, "k_Σ$i", (ki, L))
        push!(ks_Σs_sheet_names, "k_Σ$i")  
    end

    write(jld2file, "ks_Σs_sheet_names", ks_Σs_sheet_names)
    write(jld2file, "Tf", Tf)
    write(jld2file, "szf", szf)
    write(jld2file, "spin", spin)
    close(jld2file)

    return AzCov{Tf, szf, spin}(filenm, ks_Σs_sheet_names)
end



# f(k::Number, Σ::AbstractMatrix...) -> AbstractMatrix
function azmap(f, azc::AZ...; filename="L_kblock.jld2", dirsave=mktempdir()) where {Tf, szf, spin, AZ<:AzCov{Tf, szf, spin}}
        
    filenm   = joinpath(dirsave, filename)
    jld2file = jldopen(filenm, "w")
    @show filenm
    @show jld2file

    # Fixme: this is pretty janky .. it would be nice to make it more general
    azc_jld2files   = getfield.(azc, :jld2file)
    azc_sheet_names = getfield.(azc, :ks_Σs_sheet_names)
    # set the sheet names and k ranges of the new azcov to the first entry 

    for nm ∈ zip(azc_sheet_names...)
        k  = read(azc_jld2files[1],nm[1])[1]
        Σs = map(azc_jld2files, nm) do j, s 
            read(j,s)[2]
        end
        L = map(f, k, Σs...)
        write(jld2file, nm[1], (k, L))
    end

    write(jld2file, "ks_Σs_sheet_names", azc_sheet_names[1])
    write(jld2file, "Tf", Tf)
    write(jld2file, "szf", szf)
    write(jld2file, "spin", 0)
    close(jld2file)

    AzCov{Tf, szf, spin}(filenm, azc[1].ks_Σs_sheet_names)
end


# Computing the covariance matrix of the Fourier coefficient at fixed frequency k 
# For rings as a function of θcol
# ==============================================


# ------------------------


function shared_Σsheets_k(θcol, φcol::Vector{T}, idxk, covf) where T<:Real
    nθx = length(θcol)
    lowrΣT₁T₂ = SharedArray{T,3}(
        (length(idxk), nθx,nθx), 
        init = S -> S[localindices(S)] = repeat([T(0)], length(localindices(S))),
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


function nonsym_shared_Σsheets_k(θcol, φcol::Vector{T}, idxk, covf) where T<:Real
    nθx = length(θcol)
    ΣT₁T₂ = SharedArray{T,3}(
        (length(idxk), nθx,nθx), 
        init = S -> S[localindices(S)] = repeat([T(0)], length(localindices(S))),
    ) 
    jranges = split_col_ranges(nθx, nworkers())
    @sync begin
        for p in workers()
            @async remotecall_wait(
                nonsym_Σ_chunck!, p, ΣT₁T₂, θcol, φcol, jranges[p-1], idxk, covf 
            )
        end
    end
    rtΣTT = map(1:length(idxk)) do k 
        ΣT₁T₂[k,:,:]
    end 
    return rtΣTT
end


function Σsheets_k(θcol, φcol::Vector{T}, idxk, covf) where T<:Real
    nθx         = length(θcol)
    lowrΣT₁T₂   = zeros(T, length(idxk), nθx, nθx)
    Σ_chunck!(lowrΣT₁T₂, θcol, φcol, 1:nθx, idxk, covf)
    rtΣTT = map(1:length(idxk)) do k 
        Symmetric(lowrΣT₁T₂[k,:,:], :L)
    end 
    return rtΣTT 
end


# ------------------------

function Σ_chunck!(lowrΣT₁T₂, θcol, φcol, jrange, idxk, covf)
    nθx = length(θcol)
    𝒲col  = plan_rfft(similar(φcol))
    for j=jrange, i=j:nθx 
        T₁T₂ = 𝒲col * colΣ(θcol[i], θcol[j], φcol, covf)
        lowrΣT₁T₂[:,i,j] = real.(T₁T₂[idxk])
    end
end


function nonsym_Σ_chunck!(rΣT₁T₂, θcol, φcol, jrange, idxk, covf)
    nθx = length(θcol)
    𝒲col  = plan_rfft(similar(φcol))
    for j=jrange, i=1:nθx 
        T₁T₂ = 𝒲col * colΣ(θcol[i], θcol[j], φcol, covf)
        rΣT₁T₂[:,i,j] = real.(T₁T₂[idxk])
    end
end


# covf should be of the form covf(θ1::Number, θ2::Number, Δφcol::Vector) 
colΣ(θ1, θ2, φcol, covf) = covf(θ1, θ2, φcol .- φcol[1])    


# AzCov's operating on fields
# =================================


# f(Σ, ifk[:,k]) -> AbstractArray which can multiply m(θvec, k)
function az2op(f, cs::AzCov{Tf,sz,0}, fx::Array{Tf,2}) where {Tf<:Real,sz}
    ifk  = cs.Urow * fx
    ofk  = zero(ifk)
    for nm ∈ cs.ks_Σs_sheet_names
        ks, Σs  = read(cs.jld2file, nm)
        for (k, Σ) ∈ zip(ks, Σs)
            ofk[:,k] = f(Σ, ifk[:,k])
        end
    end
    ofx = cs.Urow \ ofk
    ofx
end 

# f!(view(ofk,:,k), Σ, ifk[:,k])
function az3op(f!, cs::AzCov{Tf,sz,0}, fx::Array{Tf,2}) where {Tf<:Real,sz}
    ifk  = cs.Urow * fx
    ofk  = zero(ifk)
    ofki = ofk[:,1]
    for nm ∈ cs.ks_Σs_sheet_names
        ks, Σs  = read(cs.jld2file, nm)
        for (k, Σ) ∈ zip(ks, Σs)
            f!(ofki, Σ, ifk[:,k])
            ofk[:,k] = ofki
        end
    end
    ofx = cs.Urow \ ofk
    ofx
end 


# TODO: all the following can be obtained from az3op
# ===========================

# f(Σ) -> AbstractArray which can multiply m(θvec, k)
function azmul(f, cs::AzCov{Tf,sz,0}, fx::Array{Tf,2}) where {Tf<:Real,sz}
    ifk  = cs.Urow * fx
    ofk  = zero(ifk)
    ofki = ofk[:,1]
    for nm ∈ cs.ks_Σs_sheet_names
        ks, Σs  = read(cs.jld2file, nm)
        for (k, Σ) ∈ zip(ks, Σs)
            mul!(ofki, f(Σ), ifk[:,k])
            ofk[:,k] = ofki
        end
    end
    ofx = cs.Urow \ ofk
    ofx
end 


function azdiv(f, cs::AzCov{Tf,sz,0}, fx::Array{Tf,2}) where {Tf<:Real,sz}
    ifk  = cs.Urow * fx
    ofk  = zero(ifk)
    ofki = ofk[:,1]
    for nm ∈ cs.ks_Σs_sheet_names
        ks, Σs  = read(cs.jld2file, nm)
        for (k, Σ) ∈ zip(ks, Σs)
            ldiv!(ofki, f(Σ), ifk[:,k])
            ofk[:,k] = ofki
        end
    end
    ofx = cs.Urow \ ofk
    ofx
end 




function Base.:*(cs::AzCov{Tf,sz,0}, fx::Array{Tf,2}) where {Tf<:Real,sz}
    ifk  = cs.Urow * fx
    ofk  = zero(ifk)  
    for ksΣs_nm ∈ cs.ks_Σs_sheet_names
        ks, Σs  = read(cs.jld2file, ksΣs_nm)
        for (k, Σ) ∈ zip(ks, Σs)
            ΣL = Σ.L
            mul!(view(ofk,:,k), ΣL', ifk[:,k])
            lmul!(ΣL, view(ofk,:,k))
            if !issuccess(Σ)
                println("warning, cholesky failed at k index ", k)
            end 
        end
    end
    ofx = cs.Urow \ ofk
    ofx
end 


function Base.:\(cs::AzCov{Tf,sz,0}, fx::Array{Tf,2}) where {Tf<:Real,sz}
    ifk  = cs.Urow * fx
    ofk  = zero(ifk)
    for ksΣs_nm ∈ cs.ks_Σs_sheet_names
        ks, Σs  = read(cs.jld2file, ksΣs_nm)
        for (k, Σ) ∈ zip(ks, Σs)
            ofk[:,k] = Σ \ ifk[:,k]
            if !issuccess(Σ)
                println("warning, cholesky failed at k index ", k)
            end 
        end
    end
    ofx = cs.Urow \ ofk
    ofx
end 


function ksupport(cs::AzCov{Tf,sz,0}, fx::Array{Tf,2}) where {Tf<:Real,sz}
    ifk  = cs.Urow * fx
    ofk  = zero(ifk)
    for ksΣs_nm ∈ cs.ks_Σs_sheet_names
        ks  = read(cs.jld2file, ksΣs_nm)[1]
        for k ∈ ks
            ofk[:,k] = ifk[:,k]
        end
    end
    ofx = cs.Urow \ ofk
    ofx
end






# misc 
# =================================


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



# This can be used to help define the pixel space covariance function
function geoθ1θ2Δφcol(θ1, θ2, Δφcol)
    sθ1, sθ2 = sin(θ1), sin(θ2)
    sΔθ½     = sin((θ1 - θ2)/2)
    sΔφ½     = @. sin(Δφcol / 2)
    β        = @. 2asin(√(sΔθ½^2 + sθ1 * sθ2 * sΔφ½^2))
    return β
end


size_arg(cs::AzCov{Tf,sz,0}) where {Tf<:Real,sz} = sz

