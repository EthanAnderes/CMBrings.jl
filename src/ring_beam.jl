# RingBeam <: AbstractLinearOp
# ======================================

struct RingBeam{M<:CMBrings.MatrixOrFactorization, TM<:𝕎} <: AbstractLinearOp
    Baz::CMBrings.AzBlock{M}
    tmAZ::TM
    θrange::UnitRange{Int64}
end


# Interface methods for RingBeam
# ---------------------------------------
# Note: we are not returning the input field in its original basis.
# Check to make sure this will not be a problem ...

for op ∈ (:*, :\)
    quote
		function Base.$op(rb::RingBeam{M,TW}, f::Xfield{TS}) where {M,TW,TS<:𝕊0} 
			fx  = deepcopy(f[:])
			faz = Xmap(rb.tmAZ, fx[rb.θrange,:])
			fx[rb.θrange,:] = Base.$op(rb.Baz, faz)[:]
			Xmap(fieldtransform(f), fx)
		end
		function Base.$op(rb::RingBeam{M,TW}, f::Xfield{TS}) where {M,TW,TS<:𝕊2} 
			fx  = deepcopy(f[:])
			fazQ = Xmap(rb.tmAZ, fx[rb.θrange,:,1])
			fazU = Xmap(rb.tmAZ, fx[rb.θrange,:,2])
			fx[rb.θrange,:,1] = Base.$op(rb.Baz, fazQ)[:]
			fx[rb.θrange,:,2] = Base.$op(rb.Baz, fazU)[:]
			Xmap(fieldtransform(f), fx)
		end
    end |> eval
end


for op ∈ (:adjoint, :transpose, :inv, :sqrt, :-)
    quote
        function LinearAlgebra.$op(rb::RingBeam{M,TM}) where {M,TM}
        	RingBeam(LinearAlgebra.$op(rb.Baz), rb.tmAZ, rb.θrange)
        end
    end |> eval
end



# Operations op(DiagOp, Number) and  op(Number, DiagOp)
# ------------------------------------------

# op(DiagOp, Number) and op(Number, DiagOp)
Base.:*(O::RingBeam, a::Number) = RingBeam(a * O.Baz, O.tmAZ, O.θrange)
Base.:*(a::Number, O::RingBeam) = RingBeam(a * O.Baz, O.tmAZ, O.θrange)

Base.:\(O::RingBeam, a::Number)  = RingBeam(O.Baz \ a, O.tmAZ, O.θrange)
Base.:\(a::Number, O::RingBeam)  = RingBeam(a \ O.Baz, O.tmAZ, O.θrange)

Base.:/(O::RingBeam, a::Number)  = RingBeam(O.Baz / a, O.tmAZ, O.θrange)
Base.:/(a::Number, O::RingBeam)  = RingBeam(a / O.Baz, O.tmAZ, O.θrange)

Base.:^(O::RingBeam, a::Number)  = RingBeam(O.Baz^a, O.tmAZ, O.θrange) 
Base.:^(O::RingBeam, a::Integer) = RingBeam(O.Baz^a, O.tmAZ, O.θrange)


# Operations between DiagOp and UniformScaling
# ------------------------------------------


LinearAlgebra.:*(J::UniformScaling, O::RingBeam) = RingBeam(J.λ * O.Baz, O.tmAZ, O.θrange)
LinearAlgebra.:*(O::RingBeam, J::UniformScaling) = RingBeam(J.λ * O.Baz, O.tmAZ, O.θrange)
LinearAlgebra.:+(O::RingBeam, J::UniformScaling) = RingBeam(J.λ + O.Baz, O.tmAZ, O.θrange)
LinearAlgebra.:+(J::UniformScaling, O::RingBeam) = RingBeam(J.λ + O.Baz, O.tmAZ, O.θrange)
LinearAlgebra.:-(O::RingBeam, J::UniformScaling) = RingBeam(O.Baz - J.λ, O.tmAZ, O.θrange)
LinearAlgebra.:-(J::UniformScaling, O::RingBeam) = RingBeam(J.λ - O.Baz, O.tmAZ, O.θrange)


# Operations op(DiagOp, DiagOp)
# ------------------------------------------


Base.:+(O1::RingBeam{X}, O2::RingBeam{X}) where X<:Field = RingBeam(O1.Baz + O2.Baz, O.tmAZ, O.θrange)
Base.:-(O1::RingBeam{X}, O2::RingBeam{X}) where X<:Field = RingBeam(O1.Baz - O2.Baz, O.tmAZ, O.θrange)
Base.:*(O1::RingBeam{X}, O2::RingBeam{X}) where X<:Field = RingBeam(O1.Baz * O2.Baz, O.tmAZ, O.θrange)
Base.:\(O1::RingBeam{X}, O2::RingBeam{X}) where X<:Field = RingBeam(O1.Baz \ O2.Baz, O.tmAZ, O.θrange)
Base.:/(O1::RingBeam{X}, O2::RingBeam{X}) where X<:Field = RingBeam(O1.Baz / O2.Baz, O.tmAZ, O.θrange)
