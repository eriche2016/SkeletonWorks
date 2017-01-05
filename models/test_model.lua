require 'pl'
require 'nn'
require 'totem'
require 'nngraph'

local opts = require '../opts'

local test = {} 
local tester = totem.Tester() 

opt = opts.parse(arg)  
local function checkGradients(...)
   totem.nn.checkGradients(tester, ...)
end

print(opt)

if opt.gpuid >=0 then 
    require 'cutorch'
    require 'cunn'
    if opt.backend == 'cudnn' then require 'cudnn' end 
    cutorch.manualSeed(opt.seed)
    cutorch.setDevice(opt.gpuid + 1)
end 

opt.model = "fiveStageVgg" --"SkeletonNet"
t = paths.dofile(opt.model..".lua")


function test.test_model()
      local x = torch.rand(1, 3, 16, 16)

      t.model:forward(x) 
      local gradInput4 = t.model:backward(x, {torch.rand(1, 2, 16, 16), torch.rand(1, 3, 16, 16), 
      										  torch.rand(1, 4, 16, 16),torch.rand(1, 5, 16, 16)})
      checkGradients(t.model, x)
end

tester:add(test):run()


