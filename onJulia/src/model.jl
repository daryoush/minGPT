"""
From Andrej Karpathy's implementation:

- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

"""
for the Julia implementation
    Use Transformer.jl's Transformer.
    For some reason PositionEmbedding was not learning, defined my own
    Didn't find a simple linear layer, defined my own (TODO look for one in FLUX)
"""


import Flux: Dense
using Transformers: Transformer, PositionEmbedding, Embed
using Transformers.Basic: Positionwise, Dropout
using Flux:params, @functor, LayerNorm, Chain
using NNlib: gelu
using Random
using Distributions: Normal, UnivariateDistribution
using BSON: @save, @load

smallNormalInit(out, in) = Float32.(rand(Normal(0.0, 0.02), out, in))
zeroBias(x) = Float32.(zeros(x))
Dense(in::Integer, out::Integer)= Dense(in,out, initW=smallNormalInit,initb=zeroBias)
mingptTransformer(hiddenSize=512, attentionHeads=4, pdrop=.1) = Transformer(hiddenSize,attentionHeads, 4*hiddenSize, future = false, act = gelu, pdrop = 0.1)
gptEmbed(d::UnivariateDistribution, size::Int, vocab_size::Int; scale = one(Float32)) = Embed(Float32(scale), Float32.(rand(d, size, vocab_size)))

# Trying to see if a linear layer works better than dense layer for the last layer in gpt
struct Linear{S<:AbstractArray}
  W::S
end

Linear(W) = Linear(W)

function Linear(in::Integer, out::Integer)
  return Linear(smallNormalInit(out, in))
end

@functor Linear

function (a::Linear)(x::AbstractArray{T}) where T
  a.W*x
end

struct GPTPositionEmbedding{F, W <: AbstractArray{F}}
	    embedding::W
	end

@functor GPTPositionEmbedding

function GPTPositionEmbedding(size::Int, max_len::Int = 1024)
    GPTPositionEmbedding( zeros(Float32,size, max_len))
end

function (pe::GPTPositionEmbedding{F})(x::AbstractArray{F}) where F
    pe.embedding[:, 1:size(x,2)]  # just return the needed position embedding
end

struct minGPT{ EM<:Embed,
		PM<:GPTPositionEmbedding,
		C<:Chain,DP<:Dropout,
		LN<:LayerNorm,
		PW<:Positionwise }
    em::EM
    pe::PM
	drop::DP
	ln::LN
    c::C
	pw::PW
end

@functor minGPT

function (g::minGPT)(x::A) where {T, N, A<:AbstractArray{T, N}}
	x=g.em(x)
	x=g.drop(x .+g.pe(x))
	x = g.ln(x)
	x=g.c(x)
	g.pw(x)
end

function minGPT(;numberAttnLayers=2, hiddenSize=512, attentionHeads=4, vocabSize=65, maxBlockSize=50, pdrop=.1)
    embed = gptEmbed(Normal(0.0, 0.02), hiddenSize, vocabSize)
	posembd = GPTPositionEmbedding(hiddenSize, maxBlockSize)
	ln = LayerNorm(hiddenSize)
	pw = Positionwise(Dense(hiddenSize, vocabSize))   #Linear(hiddenSize, vocabSize)) #Logits.
    c=Chain(
        [mingptTransformer(hiddenSize,attentionHeads, pdrop) for i in 1:numberAttnLayers]...,
    )

    minGPT(embed,posembd, Dropout(pdrop), ln, c, pw)
end

## ---
using Zygote
using Flux
using Transformers
function functionDummy( m)
	## Don't call the above function, just put here to show how to save and restore model
	weights = params(model);
	@save "mymodel.bson" weights

	#model needs same parameters
	newModel=minGPT(maxBlockSize=lengthOfSentences, vocabSize=length(vocab))
	 @load "mymodel.bson"  weights   ## this must be the same name as the one used in the save!!!!
	Flux.loadparams!(newModel, weights)
end


function functionDummy2()
	WWW## Don't call the above function, just put here to show how to save and restore model
	##assuming mytestModel is set  flux, zygote and transformers must be included
	@save "testmodel.bson" mytestmodel

	@load "testmodel.bson"  mytestmodel   ## this must be the same name as the one used in the save!!!!
	mytestmodel(vocab(collect("Where ")))
end
