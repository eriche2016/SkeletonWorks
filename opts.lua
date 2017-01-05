local opts = {}

lapp = require 'pl.lapp'
function opts.parse(arg)
   local opt = lapp [[
   Command line options:
   General options:
   --seed            (default 123),          Manually set RNG seed
   --backend         (default cudnn)         cudnn backend 

   Training Related:
   -l, --learningRate     (default 1e-3)    learning rate
   -d,--learningRateDecaySteps  (default 5)   number of epochs to reduce LR by 0.1
   -w,--weightDecay        (default 2e-4)        L2 penalty on the weights
   -m,--momentum           (default 0.9)         momentum
   -b,--batchSize          (default 1)          batch size, currently, batchSize must be 1 to avoid different size of images
   --maxepoch              (default 300)         maximum number of training epochs
   --plot                  (default true)                plot training/testing error in real-time
   --checkpoint            (default false)       Save model per epoch
   --optimization          (default SGD)         optimization methods, currently, just use SGD  
   --noConfusion           (default all)         add confution table, can be test, or all 

   Device Related:
   -t,--nThreads           (default 2)           number of threads for data loader
   --gpuid              (default 0)           device ID (if using CUDA)
   --nGPU                  (default 1)           number of GPUs you want to train on
   --save                  (default ./checkpoints/)     save trained model here

   Dataset Related:
   --channels              (default 3)
   --datapath              (default ./data/SK506_32)
                           dataset location
   --dataset               (default SK506)         dataset type
   --cachepath             (default gen)     cache directory to save the loaded dataset
   --classes                                 classes of the dataset 
   
   Model Related:
   --model                 (default SkeletonNet)     model name 
 ]]

   return opt
end

return opts
