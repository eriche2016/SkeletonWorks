local image = require 'image'
local paths = require 'paths'
local t = require 'data/transforms'
local ffi = require 'ffi'

local M = {}
local SK506Dataset = torch.class('SkletonNet.SK506Dataset', M)

function SK506Dataset:__init(imageInfo, opt, split)
    self.imageInfo = imageInfo[split]
    self.opt = opt
    self.split = split
    -- self.dir = opt.datapath
    -- assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
    collectgarbage()
end

function SK506Dataset:get(i)
    -- get image one by one 
    -- return a new input, target couple
    local image_path = ffi.string(self.imageInfo.imagePath[i]:data())
    local label_path = ffi.string(self.imageInfo.labelPath[i]:data())
    -- testing 
    -- local index1 = string.find(image_path, '/[^/]*$') 
    -- local index2 = string.find(label_path, '/[^/]*$')
    -- assert(string.sub(image_path, index1, -5) == string.sub(label_path, index2, -5)) 
    
    -- local image = self:_loadImage(paths.concat(self.dir, image_path), 3)
    -- local label = self:_loadImage(paths.concat(self.dir, label_path), 1)
    
    local image = self:_loadImage(image_path, 3)
    local label = self:_loadImage(label_path, 1) -- .mat file 

    return {
        input = image,
        target = label,
        input_path = image_path,
        target_path = label_path
    }
end

function SK506Dataset:_loadImage(path, channels)
    local ok, input = pcall(function()
        if channels == 1 then
            return image.load(path, channels, 'byte') -- for testting image, the value will be 255
        else
            return image.load(path, channels, 'float')
        end
    end)


    -- Sometimes image.load fails because the file extension does not match the
    -- image format. In that case, use image.decompress on a ByteTensor.
    if not ok then
        local f = io.open(path, 'r')
        assert(f, 'Error reading: ' .. tostring(path))
        local data = f:read('*a')
        f:close()

        local b = torch.ByteTensor(string.len(data))
        ffi.copy(b:data(), data, b:size(1))

        input = image.decompress(b, channels, 'float')
    end

    return input
end

function SK506Dataset:size()
    return self.imageInfo.imagePath:size(1)
end

function SK506Dataset:preprocess()
    if self.split == 'train' then
        return t.Compose{
            t.ColorNormalize(self.opt.mean, self.opt.std),
            t.Scale(360),
            t.HorizontalFlip(0.5)
        }
    elseif self.split == 'test' then
        return t.Compose{
            t.ColorNormalize(self.opt.mean, self.opt.std),
            t.Scale(360)
        }
    else
        error('invalid split: ' .. self.split)
    end
end

return M.SK506Dataset
