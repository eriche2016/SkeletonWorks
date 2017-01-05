--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--

local datasets = require 'data/init'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('SkeletonNet.DataLoader', M)

function DataLoader.create(opt)
   -- The train and test loader
   local loaders = {}

   for i, split in ipairs{'train', 'test'} do
      local dataset = datasets.create(opt, split) -- train dataset
      loaders[i] = M.DataLoader(dataset, opt, split)
   end

   return table.unpack(loaders)
end

function DataLoader:__init(dataset, opt, split)
   local manualSeed = opt.seed
   local function init()
      require('data/' .. opt.dataset)
   end
   local function main(idx) -- idx: the id of each thread 
      if manualSeed ~= 0 then
         torch.manualSeed(manualSeed + idx) -- seed for each thread is thread id dependent
      end

      torch.setnumthreads(1)
      -- lua keeps all its global variables in a regular table named _G
      -- _G: the lua environment itself 
      _G.dataset = dataset
      _G.preprocess = dataset:preprocess()

      return dataset:size() -- return the number of images, 300. 206  
   end
   
   local threads, sizes = Threads(opt.nThreads, init, main) 
   -- if opt.tenCrop is set to be true, then we will average output during testing
   self.nCrops = (split == 'test' and opt.tenCrop) and 10 or 1
   
   self.threads = threads
   self.__size = sizes[1][1]
   self.batchSize = math.floor(opt.batchSize / self.nCrops)
end

function DataLoader:size()
   return math.ceil(self.__size / self.batchSize)
end

function DataLoader:run()
   local threads = self.threads
   local size, batchSize = self.__size, self.batchSize
   local perm = torch.randperm(size)

   local idx, sample = 1, nil
   local function enqueue()
      while idx <= size and threads:acceptsjob() do
         local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))

         threads:addjob(
            function(indices, nCrops)
               local sz = indices:size(1) -- batchSize or idx->size-idx+1 
               local batch, imageSize

               for i, idx in ipairs(indices:totable()) do 
                  local sample = _G.dataset:get(idx)
                  -- sample.target: 1 x H x W 
                  local input, target = _G.preprocess(sample.input, sample.target)
                  -- print('target size')
                  -- print(target) -- 1 x H x W
                  target = target:squeeze()  -- H x W
                  -- io.read()
                  -- print(target)
                 
                  -- recompute target so that target only contains zero or one values 
                  -- target = torch.eq(target, 1):double()  
                  -- target map will be value of 255(may ranging from 1 to 255) for displaying
                  -- so here we transform it to label: 1 or 2 
                  -- local save = true 
                  -- if save then
                  -- image.save('../image.jpg', sample.input)
                  -- image.save('../target.png', target*255) 
                  -- end
                  -- To do: compute merged labels to guide different stages 
                  -- receptive field: 5 14 40 92 196 -- H x W 
                  -- s < 1: 1, means background 
                  -- s >= 150: 1, means background 
                  local mask_lt1_gt150 = torch.add(torch.lt(target, 1.0), torch.ge(target, 150.0)) 
                  mask_lt1_gt150[mask_lt1_gt150:ge(1)] = 1 -- remove double counting 
                  -- 1=< < 10: 2
                  local mask_ge1_lt10 = torch.cmul(torch.ge(target, 1.0), torch.lt(target, 10.0)) 
                  -- 10=< < 26: 3
                  local mask_ge10_lt26 = torch.cmul(torch.ge(target, 10.0), torch.lt(target, 26.0)) 
                  -- 26=< < 60: 4 
                  local mask_ge26_lt60 = torch.cmul(torch.ge(target, 26.0), torch.lt(target, 60.0)) 
                  -- 60=< < 150: 5 
                  local mask_ge60_lt50 = torch.cmul(torch.ge(target, 60.0), torch.lt(target, 150.0)) 
                  

                  target[mask_lt1_gt150] = 1 -- background 
                  target[mask_ge1_lt10] = 2 
                  target[mask_ge10_lt26] = 3
                  target[mask_ge26_lt60] = 4
                  target[mask_ge60_lt50] = 5

                  if not batch then
                     imageSize = input:size():totable()
                     targetSize = target:size():totable() -- H x W

                     batch = torch.FloatTensor(sz, table.unpack(imageSize))
                     -- print('batch')
                     -- print(batch:size()) -- bz x 1 x 3 x H x W 
                     target_batch = torch.FloatTensor(sz, nCrops, table.unpack(targetSize))
                  end
                  batch[i]:copy(input)

                  target_batch[i]:copy(target)

               end
               collectgarbage()
               return {
                  input = batch:view(sz, table.unpack(imageSize)),
                  target = target_batch:view(sz, table.unpack(targetSize)), --bz x H x W
               }
            end,
            function(_sample_)
               sample = _sample_
            end,
            indices,
            self.nCrops
         )
         idx = idx + batchSize
      end
   end

   local n = 0
   local function loop()
      enqueue()
      if not threads:hasjob() then
         return nil
      end
      threads:dojob()
      if threads:haserror() then
         threads:synchronize()
      end
      enqueue()
      n = n + 1
      return n, sample
   end

   return loop
end

return M.DataLoader
