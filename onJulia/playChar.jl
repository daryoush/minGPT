## ---
	using Flux: onehot
	using Transformers: Vocabulary, decode
	using Random
	using Flux: throttle, @epochs, ADAMW, ADAM, RADAM, Momentum, Optimiser, logitcrossentropy, params
	using Flux: softmax, onecold
	using DataStructures: Queue
	Random.seed!(42)

## ---
module Datasets include("src/datasets.jl") end
dataSetlength, vocab, sample = Datasets.openDataset("input.txt")
batch(dataLength,batchLength ) = hcat((sample(dataLength) for x in 1:batchLength)...)

## ---
module Model include("src/model.jl") end

lengthOfSentences=520
model=Model.minGPT(maxBlockSize=lengthOfSentences, vocabSize=length(vocab))
@show NumberOfParameters=sum(length(p) for p in params(model))

## ---


# for a batch of sentences, extract 1st to one before last character as input
# and 2nd to last char of each sentence as prediction.  Shape the predictred
# values as a long one dim series of values, one hot encode them using the vocab

trainingDataPair(b)=(b[1:end-1, :], onehot(vocab, reshape(b[2:end,:], :)))
trainingData(dataLength,batchLength)= [trainingDataPair(batch(dataLength,batchLength))
		for i in 1:dataSetlength/(dataLength*batchLength)]

function loss(x,y)
	yhat=model(x)
	logitcrossentropy(reshape(yhat,size(yhat)[1], :), y)
end


function validate()
	s=collect("Wher")
	for i in 1:20
		out=model(vocab(s))
		push!(s, decode(vocab,onecold(softmax(out)[:,end])))
	end
	join(s)
end
opt = ADAM()

## ---
module Train include("src/train.jl") end


## ---

decay = 0.1

@show opt=Optimiser(opt,
			Train.SelectiveWeightDecay(WeightDecay(decay), Model.noDecayList(model)))

numberOfSentencesInBatch=15
for i in 1:5
	Train.train!(loss,
 				params(model),
				trainingData(lengthOfSentences, numberOfSentencesInBatch),
				opt,
				validate)
			end


## ---

# pass a string to the model and see the top prediction for the next char in sequence
out=model(vocab(collect("Where ")))
@show sort(collect(zip(softmax(out[:,end]), decode(vocab, collect(1:size(out)[1])))), rev=true)
@show decode(vocab,onecold(softmax(out)[:,end]))

## ---
print(validate())

## ---




#trying to track allocated
d = [trainingData(lengthOfSentences, numberOfSentencesInBatch)[1]]
@timed Train.train!(loss,
 				params(model),
				d,
				opt)
## ---
using Profile

Profile.clear()
d = [trainingData(lengthOfSentences, numberOfSentencesInBatch)[1]]

@profile Train.train!(loss,
 				params(model),
				d,
				opt)


# (Since Julia's garbage collector is written in C, such events can be detected using the C=true
# output mode described below, or by using ProfileView.jl.)
#https://docs.julialang.org/en/v1/manual/profile/#Memory-allocation-analysis-1

# format – Introduced above, determines whether backtraces are printed with (default, :tree) or without (:flat) indentation indicating tree structure.
# C – If true, backtraces from C and Fortran code are shown (normally they are excluded). Try running the introductory example with Profile.print(C = true). This can be extremely helpful in deciding whether it's Julia code or C code that is causing a bottleneck; setting C = true also improves the interpretability of the nesting, at the cost of longer profile dumps.
# combine – Some lines of code contain multiple operations; for example, s += A[i] contains both an array reference (A[i]) and a sum operation. These correspond to different lines in the generated machine code, and hence there may be two or more different addresses captured during backtraces on this line. combine = true lumps them together, and is probably what you typically want, but you can generate an output separately for each unique instruction pointer with combine = false.
# maxdepth – Limits frames at a depth higher than maxdepth in the :tree format.
# sortedby – Controls the order in :flat format. :filefuncline (default) sorts by the source line, whereas :count sorts in order of number of collected samples.
# noisefloor – Limits frames that are below the heuristic noise floor of the sample (only applies to format :tree). A suggested value to try for this is 2.0 (the default is 0). This parameter hides samples for which n <= noisefloor * √N, where n is the number of samples on this line, and N is the number of samples for the callee.
# mincount – Limits frames with less than mincount occurrences.
#


Juno.profiler(;C = false)


####### TODO:  Try https://github.com/astrieanna/TypeCheck.jl   to make sure types are stable in the network

## ---

function sampleFromModel(s="Where ", minprob=.05)
	function explore(x, minprob)
		x[1] < minprob && return currentRes
		out=model(vocab(collect(x[2])))
		choices=sort(collect(zip(softmax(out[:,end]), decode(vocab, collect(1:size(out)[1])))), rev=true)
		options = [(x[1]*np, x[2]*nc) for (np,nc) in choices if (x[1]*np) > minprob ]
	end
	toexplore=Queue{Tuple{Float32,String}}()
	enqueue!(toexplore,(1, s))  #start with the initial string and prob 1
	results=[]
	for i in toexplore
		nxt=dequeue!(toexplore)
		res=explore(nxt, minprob)
		if isempty(res)  # go as long as you can't add any more character and satisfy the prob
			push!(results, nxt)
			continue #continue emptying the queue
		end
		for r in res
			enqueue!(toexplore, r)  ## add elements for further exploration
		end
	end
    #elements of the results are (prob, string), sort from high to low of the
	#longest match with highest prob
	sort(results, by=x->(length(x[2]), x[1]), rev=true)
end

@show res = sampleFromModel("VINCENTIO: Oh God, My lord!", 0.01)

## ---
# Try a random model with the data and see what the losses would look like
using Statistics: norm, mean, std
function justRandomData()
	function readData(f)
		s = open(f) do file
			read(file, String)
		end
	end
	function generateRandomY()

		randomExpectedString = rand(s, lengthOfSentences-1, numberOfSentencesInBatch)
		randomExpectedy= reshape(onehot(vocab, vocab.(randomExpectedString)), length(vocab), :)
	end

	# let model output  be a complete random numbers, => uniform distribution
	#randModel(x,y)=rand(Float32, length(vocab), lengthOfSentences-1, numberOfSentencesInBatch), y

	# shuffle the model output, so it is not uniform, but  has all the distributions of the model, but
	# result is all in wrong place (this would be a case where model outout is not properly wired!)
	#randModel(x,y)=shuffle(model(x)), y

	# let the output be a random sample of the text.  So we respect the distribution of the input
	# so the output is wrong but it is wrong with the same distribution as the data
	# result is uniform+entropy of the system
	s=readData("input.txt")
	randModel(x,y)=model(x), generateRandomY()
	losses=[]
	ctr=0
	for (x,y) in trainingData(lengthOfSentences, numberOfSentencesInBatch)
		yhat, y =randModel(x,y)
		@show l=logitcrossentropy(reshape(yhat,size(yhat)[1], :), y)
		push!(losses, l)
		ctr +=1
		if  ctr > 10 break end
	end
	@show mean(losses), std(losses)
end

justRandomData()

## ---
using StatsBase
using Plots
using LinearAlgebra
function getOutputResults(m, data, limit=5)
	output=[]
	ctr=0
	for (x,y) in data
		out=softmax(m(x))
		out=reshape(out,size(out)[1], :)
		push!(output, onecold(out)...)  ## get the encoded vocab using straight onecold, not onecold(vocab, ..) version

		ctr +=1
		if  ctr > limit break end
	end
	convert(Array{Int}, output)
end
function OutputHistogram(m)
	outChars=getOutputResults(m, d)
	fit(Histogram, outChars, 0:1:length(vocab))
end

#Try a untrained model
raw=OutputHistogram(Model.minGPT(maxBlockSize=lengthOfSentences, vocabSize=length(vocab)))

#... and trained model
trained=OutputHistogram(model)


inputData=fit(Histogram, vocab(collect(s)), 0:1:length(vocab))
plot(normalize(raw, mode=:pdf))
plot!(normalize(trained, mode=:pdf))
plot!(normalize(inputData, mode=:pdf))
