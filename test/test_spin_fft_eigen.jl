using LinearAlgebra
using SparseArrays
using PyPlot 
using FFTW

N = 25
R = spdiagm(-1 => fill(1,N-1))
P = spdiagm(-1 => fill(1,N-1))
R[1,end] = -1
P[1,end] = 1

ω = exp(- im * 2 * π / N)
ε = exp(- im * π / N)

U  = [ ω^((x-1)*(k-1)) / √N for k ∈ 1:N, x ∈ 1:N ]
Λᵣ = diagm([ε*ω^(k-1) for k ∈ 1:N ])
Λₚ = diagm([  ω^(k-1) for k ∈ 1:N ])
D  = diagm([  ε^(x-1) for x ∈ 1:N ])

P_aprx  =      U' * Λₚ * U
R_aprx  = D' * U' * Λᵣ * U * D
R_aprx2 =  ε * D' * P * D

abs.(P .- P_aprx) |> sum
abs.(R .- R_aprx) |> sum
abs.(R .- R_aprx2) |> sum

##

f = 1:N |> collect
g = randn(N)
p = randn(ComplexF64,N)

abs.(U^2 * f - vcat(f[1], f[end:-1:2])) |> sum
abs.(U^2 * g - vcat(g[1], g[end:-1:2])) |> sum
abs.(U^2 * p - vcat(p[1], p[end:-1:2])) |> sum

U^2 * (U * g) - conj.(U * g) .|> abs2 |> sum
U^2 * conj.(U * g) - (U * g) .|> abs2 |> sum


#### 

J = real.(U^2)
Γ = diagm(randn(ComplexF64,N))
C = diagm(randn(ComplexF64,N))
C̄ = conj.(C)
Γ̄ = conj.(Γ)

C̄ * J * inv(Γ) * C * J .|> abs |> matshow # a diagonal matrix ...

C * J .|> abs |> matshow
C̄ * J * inv(Γ) .|> abs |> matshow

Σ = [ Γ    C*J
	  C̄*J  Γ̄   ] # an X

inv(Σ) .|> abs |> matshow # an X
sqrt(inv(Σ)) .|> abs |> matshow