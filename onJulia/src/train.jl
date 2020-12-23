using TensorBoardLogger, Logging
using Flux: gradient, update!, WeightDecay
using Juno
using Profile
using Statistics: norm
using LinearAlgebra: rmul!
import Flux.Optimise: apply!

lg=TBLogger("./TesorBoardLog/run", min_level=Logging.Info)

## ---

# python impl uses clip_grad_norm_ to clip the norms.  It calculates the total norm of all gradients then does:
# clip_coef = max_norm / (total_norm + 1e-6) to clip all the gradients
#  See https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
function clipGlobalNorm!(gs, ps, max=1.0)
	sumnorm = sum(norm(gs[p])  for p in ps)
	if sumnorm > max
		coef = max/sumnorm
		for p in ps
				rmul!(gs[p], coef)
		end
	end
end



## ---
mutable struct SelectiveWeightDecay
	wd::WeightDecay
	noDecayList::IdDict
end




function apply!(o::SelectiveWeightDecay, x, Δ)
	length(o.noDecayList)
	if haskey(o.noDecayList, x)
		@show "in no decaylist"
		return Δ
	end
	apply!(o.wd, x,  Δ)
end

## ---

defaultValidate()=0
function train!(loss, ps, data, opt, validate=defaultValidate, logger=lg)
	  # training_loss is declared local so it will be available for logging outside the gradient calculation.
	  local training_loss
	  ctr=0

	  @progress for d in data
		@show ctr
	    gs = gradient(ps) do
	      training_loss = loss(d...)
	      # Code inserted here will be differentiated, unless you need that gradient information
	      # it is better to do the work outside this block.
	      return training_loss
	    end
		normBeforeAdj=sum(norm(gs[p])  for p in ps)
		clipGlobalNorm!(gs, ps)
		normAfterAdj=sum(norm(gs[p])  for p in ps)
		@show normBeforeAdj, normAfterAdj

	    # Insert whatever code you want here that needs training_loss, e.g. logging.
	    # logging_callback(training_loss)
	    # Insert what ever code you want here that needs gradient.
	    # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge.

		if ctr % 10 == 0
			@show ctr, training_loss
			@show training_accuracy = validate()
			with_logger(logger) do
			   @info "train" loss=training_loss
			   @info "train" accuracy=training_accuracy
			end
		end
		ctr+=1
	    update!(opt, ps, gs)
	    # Here you might like to check validation set accuracy, and break out to do early stopping.
	  end
end
