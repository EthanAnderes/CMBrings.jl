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
