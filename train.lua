require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

torch.setdefaulttensortype('torch.FloatTensor')


-- DSN_fused
-- fusion DSN2, DSN3, DSN4, DSN5 after apply spatialsoftmax,
-- output: bz x 5 x H x W 
-- then feed into logsoftmax, spatial cross entropy   
-- DSN2: bz x 2 x H x W 
-- DSN3:  bz x 3 x H x W 
-- DSN5:  bz x 4 x H x W 
-- DSN5:  bz x 5 x H x W
-- scatter gradDSN_Fused to DSN2, DSN3, DSN4, DSN5



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

local yt_stage2, yt_stage3, yt_stage4, yt_stage5
local function Yt2Table(yt)
    yt_stage2 = yt_stage2 or torch.CudaTensor()
    yt_stage3 = yt_stage3 or torch.CudaTensor()
    yt_stage4 = yt_stage4 or torch.CudaTensor()
    yt_stage5 = yt_stage5 or torch.CudaTensor()
    
    -- stage 2 
    yt_stage2:resizeAs(yt)
    yt_stage2:copy(yt[torch.le(yt, 2)]) -- {1,2}
    -- stage 3 
    yt_stage3:resizeAs(yt)
    yt_stage3:copy(yt[torch.le(yt, 3)]) -- {1, 2, 3}

    -- stage 4 
    yt_stage4:resizeAs(yt) -- {1, 2, 3}
    yt_stage4:copy(yt[torch.le(yt, 4)]) -- {1, 2, 3, 4}

    -- stage 5 
    yt_stage5:resizeAs(yt)
    yt_stage5:copy(yt[torch.le(yt, 5)]) -- {1, 2, 3, 4}

    return {yt_stage2, yt_stage3, yt_stage4, yt_stage5}
end 


local x
local yt
local YTable

local function train(dataloader, epoch)
    -- 4 stages 
    local classes = {2, 3, 4, 5}

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
        YTable = Yt2Table(yt)

        local eval_E = function(w)
            model:zeroGradParameters()
            local y = model:forward(x) 

            --[[
            -- if five stage vgg use the follow code 
            -- to transform yt to a table of label tensor
            -- to guide the 5 stage repectively 
            --]] 
            err = loss:forward(y,YTable)            -- updateOutput

            local dE_dy = loss:backward(y,YTable)   -- updateGradInput
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

            local y = model:forward(x) -- y is a table of 4 tensors: {bz X H x W x 2, bz X H X W x 3, 
                                       -- bz x H x W x 4, bz x H x W x 5}

            for i, stage_y in ipairs(y) 
                stage_y:transpose(2, 4):transpose(2, 3) -- bz x H x W x 2(the last dimenstion[Prob0,Prob1])
                stage_y = stage_y:reshape(stage_y:numel()/stage_y:size(4), classes[i])
                local _, predictions = stage_y:max(2)
                predictions = predictions:view(-1)

                local k = YTable[1]:view(-1)

                confusion:batchAdd(predictions, k)

            end 

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
