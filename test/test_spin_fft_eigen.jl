using LinearAlgebra
using SparseArrays
using PyPlot 
using FFTW


N = 25
P̃ = spdiagm(-1 => fill(1,N-1))
P = spdiagm(-1 => fill(1,N-1))
P̃[1,end] = -1
P[1,end] = 1

ω = exp(- im * 2 * π / N)
ε = exp(- im * π / N)

U = [ ω^( (x-1)*(k-1) ) / √(N) for k ∈ 1:N, x ∈ 1:N ]
Λ = diagm([ ω^( k-1 )  for k ∈ 1:N  ])
D = diagm([ ω^( -(x-1)/2 )  for x ∈ 1:N  ])

Papprox  =  U' * Λ * U
P̃approx  =  (1/ε) * D' * U' * Λ * U * D
P̃approx2 =  (1/ε) * D' * P * D

abs.(P .- Papprox) |> sum
abs.(P̃ .- P̃approx) |> sum
abs.(P̃ .- P̃approx2) |> sum
