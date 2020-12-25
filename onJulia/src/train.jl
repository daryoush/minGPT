using TensorBoardLogger, Logging
using Flux: gradient, update!, WeightDecay, gpu
using Juno
using Profile
using Statistics: norm
using LinearAlgebra: rmul!
import Flux.Optimise: apply!
using Statistics: norm, mean, std
using Flux: ADAMW, ADAM, RADAM, Momentum, Optimiser
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
		return Δ
	end
	apply!(o.wd, x,  Δ)
end

## ---

function cosRate( lr=6e-4, warmuptoken=512*20, len_train_dataset=1115394, blocksize=128, repeat=1)
    finalTok=repeat*len_train_dataset*blocksize

	function cosRateFun(token)
		if token < warmuptoken
			lr_mult = token / max(1, warmuptoken)
		else
	    	progress = (token - warmuptoken) / max(1, finalTok - warmuptoken)
	    	lr_mult = max(0.1, 0.5 * (1.0 + cos(pi * progress)))
		end
    	lr_mult*lr
	end
	cosRateFun
end

updateLR(opt::ADAM, lr::Float64) = opt.eta=lr
updateLR(opt::Optimiser, lr::Float64) = for o in opt.os  updateLR(o, lr) end
updateLR(o, ::Float64)=nothing



defaultValidate()=0
function train!(loss, ps, data, opt, validate=defaultValidate, logger=lg;rateDecay=cosRate(), repeat=1)
	tokenCnt = 0
	for batchCnt in 1:repeat
	  # training_loss is declared local so it will be available for logging outside the gradient calculation.
	  local training_loss
	  ctr=0
	  normList=[]
	  tokensInBatch(b)=size(b[1])[1] * size(b[1])[2]   #number of char in sentences, and number of sentences
	  numberOfBatches=size(data)
	  @progress for next in data
		tokenCnt = tokenCnt + tokensInBatch(next)
		lr=rateDecay(tokenCnt)
		updateLR(opt, lr)
		d = next |> gpu
		@show ctr, numberOfBatches
	    gs = gradient(ps) do
	      training_loss = loss(d...)
	      # Code inserted here will be differentiated, unless you need that gradient information
	      # it is better to do the work outside this block.
	      return training_loss
	    end

		normBeforeAdj=sum(norm(gs[p])  for p in ps)
		push!(normList, normBeforeAdj)
		clipGlobalNorm!(gs, ps)
		#normAfterAdj=sum(norm(gs[p])  for p in ps)


	    # Insert whatever code you want here that needs training_loss, e.g. logging.
	    # logging_callback(training_loss)
	    # Insert what ever code you want here that needs gradient.
	    # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge.

		if ctr % 10 == 0

			##TODO:::: SHOW AVE gradient norm over the last 10 runs
			@show mean(normList), std(normList)
			normList=[]
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
end
