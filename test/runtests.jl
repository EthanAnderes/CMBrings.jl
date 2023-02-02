using CMBrings
using Test

@testset "conversions.jl" begin

    scanᵒ_sec = 1.1     # traverse 1.1 deg each sec
    hz = 1.0            # 1 full wavelength each sec
                        # so wavelength[deg] = 1.1
                        # i.e wavelength[rad] = deg2rad(1.1)
    m = 2π/deg2rad(1.1) # so m = 2π / wavelength[rad] = 2π / deg2rad(1.1)
    @test CMBrings.hz2m(hz; scanᵒ_sec) ≈ m
    @test CMBrings.m2hz(CMBrings.hz2m(hz; scanᵒ_sec); scanᵒ_sec) ≈ hz
    @test CMBrings.hz2m(CMBrings.m2hz(m; scanᵒ_sec); scanᵒ_sec) ≈ m

    #scanᵒ_sec = 1      # (default) 1 sec ≡ 1 deg
    hz = 0.5            # 1/2 * wavelength each 1 sec
                        # 1/2 * wavelength in 1 deg
                        # wavelength[deg] = 2 deg
                        # wavelength[rad] = deg2rad(2)
    m = 2π/deg2rad(2)   # so m = 2π / deg2rad(2)
    @test CMBrings.hz2m(hz) ≈ m
    @test CMBrings.m2hz(CMBrings.hz2m(hz)) ≈ hz
    @test CMBrings.m2hz(CMBrings.hz2m(m))  ≈ m

    #scanᵒ_sec = 1      # (default) 1 sec ≡ 1 deg
    m = 2               # wavelength[rad]=2π/2 = π
                        # wavelength[deg]=180
                        # so each second covers 1/180 of a wavelength
    hz = 1/180          # i.e. hz = 1/180
    @test CMBrings.m2hz(m) ≈ hz
    @test CMBrings.hz2m(CMBrings.m2hz(m)) ≈ m
    @test CMBrings.hz2m(CMBrings.m2hz(hz)) ≈ hz


    # TODO: add tests for RA and Dec ...
    # make sure the orientation is correct ...

end