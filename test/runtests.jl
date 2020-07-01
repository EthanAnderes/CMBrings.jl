using CMBrings

using Test

@testset "CMBrings.jl" begin

	# let 
	# 	Tf = Float32
	# 	nsd = CMBrings.HH.Nside(512)
	# 	θ,φ = CMBrings.HH.θφ_eqbelt_align(nsd) .|> x -> Tf.(x)
	# 	rT  = RingS2Transform(Tf,θ,φ[:]) 

 #    	qu1 = Xmap(rT, rand(Tf, 2rT.szθ, rT.szφ))
 #    	qu2 = Xmap(rT, rand(Tf, 2rT.szθ, rT.szφ)) |> Xfourier

 #    	2qu1 - 3qu2

 #    	qu1[:Qx]
 #    	qu1[:Ux]
 #    	qu1[:Ql]
 #    	qu1[:Ul]

 #    	qu2[:Qx]
 #    	qu2[:Ux]
 #    	qu2[:Ql]
 #    	qu2[:Ul]

	# end

end
