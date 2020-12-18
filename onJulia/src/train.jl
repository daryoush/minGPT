using TensorBoardLogger, Logging
using Flux: gradient, update!
using Juno
using Profile

lg=TBLogger("./TesorBoardLog/run", min_level=Logging.Info)

defaultValidate()=0
function train!(loss, ps, data, opt , validate=defaultValidate, logger=lg)
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
