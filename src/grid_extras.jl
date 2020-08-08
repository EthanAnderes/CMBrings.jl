# 
# for op ∈ (:Δpix, :Δfreq, :nyq, :Ωx, :Ωk, :inv_scale, :unitary_scale, :ordinary_scale)
#     @eval FFTransforms.$op(st::RingSpinTransform) = $op(𝕎1d(st))
# end
# 
# for op ∈ (:wavenum, :pix, :fullpix, :freq, :fullfreq)
#     @eval FFTransforms.$op(st::RingSpinTransform) = $op(𝕎1d(st))
# end
# 
# 