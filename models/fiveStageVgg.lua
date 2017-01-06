require 'torch'
require 'nn'
require 'nngraph'


torch.setdefaulttensortype('torch.FloatTensor')

-- comming from 5 stage vgg 
-- 1st stage vgg
local function make_stage1(input)
	local stage1 = nn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1)(input) -- conv1_1
	stage1 = nn.ReLU(true)(stage1)									-- relu1_1
	stage1 = nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)(stage1) -- conv1_2
	stage1 = nn.ReLU(true)(stage1) 									-- relu1_2
	return stage1 
end 


-- 2nd stage
local pool = {} 
pool[1] = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0)
local function make_stage2(stage1_output)
	local stage2 = pool[1](stage1_output)    
	stage2 = nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1)(stage2) -- conv2_1
	stage2 = nn.ReLU(true)(stage2)									-- relu2_1
	stage2 = nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1)(stage2)-- conv2_2
	stage2 = nn.ReLU(true)(stage2)									-- relu2_2
	return stage2 
end 

-- 3rd stage
pool[2] = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0) 
local function make_stage3(stage2_output)  
	local stage3 = pool[2](stage2_output)  	-- pool2
	stage3 = nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1)(stage3)-- conv3_1
	stage3 = nn.ReLU(true)(stage3)									-- relu3_1
	stage3 = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(stage3)-- conv3_2
	stage3 = nn.ReLU(true)(stage3)									-- relu3_2
	stage3 = nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1)(stage3)-- conv3_3
	stage3 = nn.ReLU(true)(stage3)									-- relu3_3
	return stage3  
end 
-- 4th stage 
pool[3] = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0)
local function make_stage4(stage3_output)
	local stage4 = pool[3](stage3_output)	-- pool3
	stage4 = nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1)(stage4)-- conv4_1
	stage4 = nn.ReLU(true)(stage4)									-- relu4_1
	stage4 = nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(stage4)-- conv4_2
	stage4 = nn.ReLU(true)(stage4)									-- relu4_2
	stage4 = nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(stage4)-- conv4_3
	stage4 = nn.ReLU(true)(stage4)								-- relu4_3
	return stage4 
end 

-- 5th stage 
pool[4] = nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0)
local function make_stage5(stage4_output) 
	local stage5 = pool[4](stage4_output) 	 -- pool4
	stage5 = nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(stage5) -- conv5_1
	stage5 = nn.ReLU(true)(stage5)								 -- relu5_1
	stage5 = nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(stage5) -- conv5_2
	stage5 = nn.ReLU(true)(stage5)									 -- relu5_2
	stage5 = nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1)(stage5) -- conv5_3
	stage5 = nn.ReLU(true)(stage5)									 -- relu5_3 
	return stage5 
end 

local input = nn.Identity()() -- input 
local stage1_out = make_stage1(input)

-- DSN2 for conv2 
stage2_out = make_stage2(stage1_out) 
local DSN2  = nn.SpatialConvolution(128, 64, 1, 1, 1, 1, 0, 0)(stage2_out)
DSN2 = nn.SpatialMaxUnpooling(pool[1])(DSN2)
DSN2  = nn.SpatialConvolution(64, 2, 1, 1, 1, 1, 0, 0)(DSN2)


-- DSN3 for conv3 
local stage3_out  = make_stage3(stage2_out) 
local DSN3 = nn.SpatialConvolution(256, 128, 1, 1, 1, 1, 0, 0)(stage3_out)
-- upsampling
DSN3 = nn.SpatialMaxUnpooling(pool[2])(DSN3)
DSN3 = nn.SpatialConvolution(128, 64, 1, 1, 1, 1, 0, 0)(DSN3)
DSN3 = nn.SpatialMaxUnpooling(pool[1])(DSN3)
DSN3  = nn.SpatialConvolution(64, 3, 1, 1, 1, 1, 0, 0)(DSN3)


-- DSN4  for conv4 
local stage4_out = make_stage4(stage3_out)
local DSN4 = nn.SpatialConvolution(512, 256, 1, 1, 1, 1, 0, 0)(stage4_out)
DSN4 = nn.SpatialMaxUnpooling(pool[3])(DSN4)
DSN4 = nn.SpatialConvolution(256, 128, 1, 1, 1, 1, 0, 0)(DSN4)
DSN4 = nn.SpatialMaxUnpooling(pool[2])(DSN4)
DSN4 = nn.SpatialConvolution(128, 64, 1, 1, 1, 1, 0, 0)(DSN4)
DSN4 = nn.SpatialMaxUnpooling(pool[1])(DSN4)
DSN4 = nn.SpatialConvolution(64, 4, 1, 1, 1, 1, 0, 0)(DSN4)

-- DSN5  for conv5
local stage5_out = make_stage5(stage4_out)
local DSN5 = nn.SpatialConvolution(512, 512, 1, 1, 1, 1, 0, 0)(stage5_out)
-- upsampling stage 
DSN5 = nn.SpatialMaxUnpooling(pool[4])(DSN5)
DSN5 = nn.SpatialConvolution(512, 256, 1, 1, 1, 1, 0, 0)(DSN5)
DSN5 = nn.SpatialMaxUnpooling(pool[3])(DSN5)
DSN5 = nn.SpatialConvolution(256, 128, 1, 1, 1, 1, 0, 0)(DSN5)
DSN5 = nn.SpatialMaxUnpooling(pool[2])(DSN5)
DSN5 = nn.SpatialConvolution(128, 64, 1, 1, 1, 1, 0, 0)(DSN5)
DSN5 = nn.SpatialMaxUnpooling(pool[1])(DSN5)
DSN5 = nn.SpatialConvolution(64, 5, 1, 1, 1, 1, 0, 0)(DSN5)


local model = nn.gModule({input}, {DSN2, DSN3, DSN4, DSN5})
-- scatter grad_DSN_fused to DSN2, DSN3, DSN4, DSN5

-- model:cuda()

--[[
-- optimize model memory usage
print(' | ==> optnet optimization...')
local optnet = require 'optnet'
local sampleInput = torch.zeros(2,3, 224, 224):cuda()
optnet.optimizeMemory(model, sampleInput, {inplace = true, mode = 'training'})
--]] 

-- output = {DSN2, DSN3, DSN4, DSN5}
-- label = {gt2, gt3, gt4, gt5}
local loss = nn.ParallelCriterion(false) -- flase means not repeate target 
--[[
   It does a LogSoftMax on the input (over the channel dimension),
   so no LogSoftMax is needed in the network at the end
--]]

loss:add(cudnn.SpatialCrossEntropyCriterion()) -- for DSN2
loss:add(cudnn.SpatialCrossEntropyCriterion()) -- for DSN3
loss:add(cudnn.SpatialCrossEntropyCriterion()) -- for DSN4
loss:add(cudnn.SpatialCrossEntropyCriterion()) -- for DSN5

--[[
loss:cuda()

-- sample model run
print(" | ==> sample model run")
local rnd_input = torch.rand(4, 3, 56, 180):cuda()
local output = model:forward(rnd_input)
print(output) -- output is a table of 4 tensors 
print(" | | ==> SegNet model output size (masks will be resized to these values) -- ")
print(" | | | ==> width: ".. output[1]:size(4))
print(" | | | ==> height: ".. output[1]:size(3))
--]]

return {model = model, loss = loss}
