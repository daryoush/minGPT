## ---
using Transformers: Vocabulary, encode
using DataStructures: SortedSet

function openDataset(f)
    s = open(f) do file
        read(file, String)
    end
    asDataset(s)
end

function asDataset(s)
    @show datasize=length(s)
    chars=[SortedSet(s)...]
    unksym = '□'
    vocab = Vocabulary(chars, unksym)

    getBlockFrom(start, size)  =vocab(collect(s[start:start+size-1]))
    function sampledata(blocksize=100)
        #TODO if blocksize is greater that the data, then throw exception
        lastbyte = datasize-blocksize   # last byte that can be the start for this sample
        getBlockFrom(rand(1:lastbyte), blocksize)  #and encoded block of data
    end
    datasize, vocab, sampledata
end
