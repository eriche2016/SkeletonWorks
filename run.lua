----------------------------------------------------------------------
-- Main script for training a model for semantic segmentation
--
-- Abhishek Chaurasia, Eugenio Culurciello
-- Sangpil Kim, Adam Paszke
-- Edited by Eren Golge
----------------------------------------------------------------------
require 'pl'
require 'nn'
require 'optim'

local opts = require 'opts'
local DataLoader = require 'data/dataloader'

torch.setdefaulttensortype('torch.FloatTensor')
----------------------------------------------------------------------
-- Get the input arguments parsed and stored in opt
-- global variable 
opt = opts.parse(arg)  
print(opt)

if opt.gpuid >=0 then 
    require 'cutorch'
    require 'cunn'
    if opt.backend == 'cudnn' then require 'cudnn' end 
    cutorch.manualSeed(opt.seed)
    cutorch.setDevice(opt.gpuid + 1)
end 

print("Folder created at " .. opt.save)
os.execute('mkdir -p ' .. opt.save) -- make directonary if opt.save not exists

-------------------------------------------------------------
print '==> load modules'

local data, chunks, ft

-- data loading
local trainLoader, testLoader = DataLoader.create(opt)
opt.classes =  {'skeleton', 'background'}

-- save opt to file
print 'saving opt as txt and t7'
local filename = paths.concat(opt.save,'opt.txt')
local file = io.open(filename, 'w')
for i,v in pairs(opt) do
    file:write(tostring(i)..' : '..tostring(v)..'\n')
end
file:close()
torch.save(path.join(opt.save,'opt.t7'),opt)

----------------------------------------------------------------------
print '==> training!'

-- optimization parameters
-- set as global table 
print '==> configuring optimizer'
-- training network with different optimization method 
if opt.optimization == 'SGD' then
	optimState = {
	        learningRate = opt.learningRate,
	        learningRateDecay = 0.0,
	        momentum = opt.momentum,
	        nesterov = true,
	        dampening = 0.0,
	        weightDecay = opt.weightDecay,
	 }
   optimMethod = optim.sgd
else
   error('unknown optimization method')
end

local epoch = 1
-- create model
-- t is global variable 
t = paths.dofile("models/"..opt.model..".lua")

local train = require 'train'
local test  = require 'test'
local besterr = 9999999999

-- test
opt.maxepoch=-1
while epoch < opt.maxepoch do
    print("----- epoch # " .. epoch)
    -- trainConf: train confusion table
    -- model: trained model is returned 
    -- loss: loss is returned 

  
   local trainConf, model, loss = train(trainLoader, epoch)
   besterr = test(testLoader, epoch, trainConf, model, loss, besterr) 

   -- trainConf = nil
   collectgarbage()

   epoch = epoch + 1
end
