require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

torch.setdefaulttensortype('torch.FloatTensor')

local loss = t.loss
print '==> Creating Train function ...'
local model = t.model

print ' | ==> flattening model parameters'

local w,dE_dw = model:getParameters()
print ' | ==> defining training procedure'

local confusion = optim.ConfusionMatrix(opt.classes)

-- optimization parameters 
local optimState = optimState or {
        learningRate = opt.learningRate,
        learningRateDecay = 0.0,
        momentum = opt.momentum,
        nesterov = true,
        dampening = 0.0,
        weightDecay = opt.weightDecay,
    }

print ' | ==> allocating minibatch memory'

local x
local yt

local function train(dataloader, epoch)
    local classes = opt.classes
    local dataSize = dataloader:size()

    -- print epoch info
    print('==> Training:')

    local function copyInputs(sample)
        -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
        -- if using DataParallelTable. The target is always copied to a CUDA tensor
        x =  x or torch.CudaTensor()
        yt = yt or torch.CudaTensor()

        x:resize(sample.input:size()):copy(sample.input)
        yt:resize(sample.target:size(1), sample.target:size(2), sample.target:size(3)):copy(sample.target)
        -- yt:resize(sample.target:size()):copy(sample.target)
    end

    -- set learning rate
    local decay = math.floor((epoch - 1) / opt.learningRateDecaySteps)
    optimState.learningRate = opt.learningRate *  math.pow(0.1, decay)
    print("==> Learning rate: ".. optimState.learningRate)

    -- total loss error
    local err
    local totalerr = 0
    local batchSize
    -- start training
    model:training()

    -- do one epoch
    for n, sample in dataloader:run() do
        copyInputs(sample)
        batchSize = yt:size(1)
        local eval_E = function(w)
            model:zeroGradParameters()
            local y = model:forward(x)

            err = loss:forward(y,yt)            -- updateOutput

            local dE_dy = loss:backward(y,yt)   -- updateGradInput
            model:backward(x,dE_dy)
            return err, dE_dw
        end

        -- finish training 
        local _, errt = optimMethod(eval_E, w, optimState)

        local norm = opt.learningRate * dE_dw:norm() / w:norm()

        print(string.format('train err: %f, norm : %f epoch: %d   lr: %f  ', err, norm, epoch, opt.learningRate))
        
        -- should we construct confusion table for 
        -- image training dataset
        if opt.noConfusion == 'all' then 
            model:evaluate()
            local y = model:forward(x):transpose(2, 4):transpose(2, 3) -- bz x H x W x 2(the last dimenstion[Prob0,Prob1])
            y = y:reshape(y:numel()/y:size(4), #classes)
            local _, predictions = y:max(2)
            predictions = predictions:view(-1)
            local k = yt:view(-1)  
            confusion:batchAdd(predictions, k)
            model:training()
        end
        totalerr = totalerr + err*batchSize

        if n % 10 == 0 then collectgarbage() end 
        
        xlua.progress(n, dataSize)
    end

    totalerr = totalerr / dataSize
    print(' Train Error: ', totalerr )
    trainError = totalerr
    collectgarbage()

    return confusion, model, loss
end

return train
