--
--  Copyright (c) 2016, Facebook, Inc.
--  Copyright (c) 2016, Fedor Chervinskii
--

local M = {}

local function isvalid(opt, cachePath)
    local imageInfo = torch.load(cachePath)
    if imageInfo.basedir and imageInfo.basedir ~= opt.data then
        return false
    end
    return true
end

function M.create(opt, split)
    -- create the data split (train or testing dataset)
    -- ./SK506.t7 
    local cachePath = paths.concat(opt.cachepath, opt.dataset .. '.t7')

    if not paths.filep(cachePath) or not isvalid(opt, cachePath) then
        paths.mkdir('gen')
        -- SK506-gen.lua 
        local script = paths.dofile(opt.dataset .. '-gen.lua')
        script.exec(opt, cachePath)
    end

    local imageInfo = torch.load(cachePath)

    -- make accesible to other functions by opt closure
    if split == 'train' then
        opt.mean = imageInfo.mean
        opt.std = imageInfo.std
    end

    local Dataset = require('data/' .. opt.dataset)
    return Dataset(imageInfo, opt, split)
end

return M
