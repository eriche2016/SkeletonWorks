-- generic stuff
require 'torch'
require 'nn'


torch.setdefaulttensortype('torch.FloatTensor')

local classes  
if opt.classes then 
    classes = opt.classes 
else 
    classes = {'skeleton', 'background'}
end 

print '=> construct model'

local function add_block(model,n_conv,sizes,wid,str,pad)
    local wid = wid or 3
    local str = str or 1
    local pad = pad or 1
    for i=1,n_conv do
        model:add(cudnn.SpatialConvolution(sizes[i],sizes[i+1],wid,wid,str,str,pad,pad))
        model:add(nn.SpatialBatchNormalization(sizes[i+1]))
        model:add(cudnn.ReLU())
    end
    return model
end

conv_sizes = {3,32,32,64,64,128,128,128,256,256,256,256,256,256}

-- encoder network 
encoder = nn.Sequential() 

pool = {} 

counter = 1
for i=1,2 do
	-- i = 1: {3, 32, 32}
	-- i = 2:  {32,64,64}
    sizes = {conv_sizes[counter],conv_sizes[counter+1],conv_sizes[counter+2]}
    -- 2 convolution layers 
    encoder = add_block(encoder,2,sizes)
    counter = counter + 2
    pool[i] = nn.SpatialMaxPooling(2,2,2,2)
    encoder:add(pool[i])
end

for i=3,5 do
	-- i = 3: {64,128,128,128}
	-- i = 4: {128,256,256, 256}
	-- i = 5: {256,256,256,256}
    sizes = {conv_sizes[counter],conv_sizes[counter+1],conv_sizes[counter+2],conv_sizes[counter+3]}
    encoder = add_block(encoder,3,sizes)
    counter = counter + 3
    pool[i] = nn.SpatialMaxPooling(2,2,2,2)
    encoder:add(pool[i])
end

-- decoder network 
decoder = nn.Sequential()

counter = #conv_sizes
for i=5,3,-1 do
    decoder:add(nn.SpatialMaxUnpooling(pool[i]))
    sizes = {conv_sizes[counter],conv_sizes[counter-1],conv_sizes[counter-2],conv_sizes[counter-3]}
    decoder = add_block(decoder,3,sizes)
    counter = counter - 3
end
for i=2,1,-1 do
    decoder:add(nn.SpatialMaxUnpooling(pool[i]))
    sizes = {conv_sizes[counter],conv_sizes[counter-1],conv_sizes[counter-2]}
    counter = counter - 2
    decoder = add_block(decoder,i,sizes)
end

decoder:add(nn.SpatialConvolution(conv_sizes[2],#classes,3,3,1,1,1))
print(" | ==> Last layer:"..conv_sizes[2].." --> "..#classes)

net = nn.Sequential()
net:add(encoder)
net:add(decoder)

net:cuda()

-- optimize model memory usage
print(' | ==> optnet optimization...')
local optnet = require 'optnet'
local sampleInput = torch.zeros(2,3, 224, 224):cuda()
optnet.optimizeMemory(net, sampleInput, {inplace = true, mode = 'training'})

-- set loss function
local loss

loss = cudnn.SpatialCrossEntropyCriterion()

loss:cuda()


---------------------------------------------------------------------
-- sample model run
print(" | ==> sample model run")
local rnd_input = torch.rand(4, 3, 56, 180):cuda()
local output = net:forward(rnd_input)
print(" | | ==> SegNet model output size (masks will be resized to these values) -- ")
print(" | | | ==> width: ".. output:size(4))
print(" | | | ==> height: ".. output:size(3))

assert(56 == output:size(3))
assert(180 == output:size(4))

return { model = net, loss = loss}
