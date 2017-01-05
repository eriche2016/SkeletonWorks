-- Todo: add sobel edge label information 

require 'xlua'
require 'image'
local sys = require 'sys'
local ffi = require 'ffi'

local M = {}


local maxLength = 55 -- upvalue  

-- given dir, will find the all the images in all the dir or subdirs
local function findImages(dir)
    local imagePath = torch.CharTensor()
    local imageClass = torch.LongTensor()

    ----------------------------------------------------------------------
    -- Options for the GNU and BSD find command
    -- note that .mat file are the corresponding skeleton label files 
    local extensionList = {'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP', 'tif', 'mat'}
    local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
    for i=2,#extensionList do
        findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
    end

    -- Find all the images using the find command
    local f = io.popen('find -L ' .. dir .. findOptions .. '| sort')

    -- local maxLength = -1
    local imagePaths = {}
    local imageClasses = {}
    local counter = 0

    -- Generate a list of all the images and their class
    while true do
        local line = f:read('*line')
        if not line then break end
        counter = counter + 1
        if counter % 100 == 0 then print(counter) end

        table.insert(imagePaths, line)

        -- maxLength = math.max(maxLength, #line + 1)
    end
    print(imagePaths)

    f:close()

    -- Convert the generated list to a tensor for faster loading
    local nImages = #imagePaths
    local imagePath = torch.CharTensor(nImages, maxLength):zero()

    for i, path in ipairs(imagePaths) do
        ffi.copy(imagePath[i]:data(), path)
    end

    return imagePath, imagePaths
end

local function setDatasetStats(imagePaths, maskPaths)

    -- compute image mean and std
    print(' | Estimating mean, std...')
    local meanEstimate = {0,0,0}
    local stdEstimate = {0,0,0}

    for i=1,#maskPaths do
        -- compute mean std
        local imagePath = imagePaths[i]
        local img = image.load(imagePath, 3, 'float')
        for j=1,3 do
            meanEstimate[j] = meanEstimate[j] + img[j]:mean()
            stdEstimate[j] = stdEstimate[j] + img[j]:std()
        end
        xlua.progress(i, #maskPaths)
    end

    for j=1,3 do
        meanEstimate[j] = meanEstimate[j] / #maskPaths
        stdEstimate[j] = stdEstimate[j] / #maskPaths
    end

    local mean = meanEstimate
    local std = stdEstimate
    local meanstd = {["mean"]=mean, ["std"]=std}
    print(' | mean, std:')
    print(meanstd)

    collectgarbage()
    return mean, std
end

local function table_merge(t1, t2)
   for k,v in ipairs(t2) do
      table.insert(t1, v)
   end 
 
   return t1
end

function M.exec(opt, cacheFile)
    -- define class maps

    -- find the image path names
    print(" | finding all images")

    -- image path opt.datapath/train/im_scale1/o0/f0
    -- data augmentation images for training images  
    local scales = {'scale0.8', 'scale1', 'scale1.2'}
    local orients = {'o0', 'o90', 'o180', 'o270'}
    local flips = {'f0', 'f1', 'f2'}
    local sub_dir = {} 
    for _, s in ipairs(scales) do 
        for _, o in ipairs(orients) do 
            for _, f in ipairs(flips) do 
                local temp_dir = s .. '/' .. o .. '/' .. f 
                table.insert(sub_dir, temp_dir)
            end  
        end 
    end  

    -- stores path 
    -- trainImagePaths: torch.CharTensor(nImages, maxLength)
    -- trainImagePathsList: a table 
    local trainImagePaths = torch.CharTensor()
    local trainImagePathsList = {} 
    local trainMaskPaths = torch.CharTensor()
    local trainMaskPathsList = {} 

    for _, sub_path in ipairs(sub_dir) do  
        local trainImagePaths_sub, trainImagePathsList_sub = findImages(opt.datapath ..'/train'..'/'.. 'im_' .. sub_path)
        if trainImagePaths:dim() == 0 then 
            trainImagePaths:resizeAs(trainImagePaths_sub):copy(trainImagePaths_sub) 
        else 
            trainImagePaths = torch.cat(trainImagePaths, trainImagePaths_sub, 1)
        end 

        trainImagePathsList  = table_merge(trainImagePathsList, trainImagePathsList_sub)

        local trainMaskPaths_sub, trainMaskPathsList_sub = findImages(opt.datapath ..'/train' .. '/'..'gt_' ..sub_path)
        
        if trainMaskPaths:dim() == 0 then 
            trainMaskPaths:resizeAs(trainMaskPaths_sub):copy(trainMaskPaths_sub) 
        else 
            trainMaskPaths = torch.cat(trainMaskPaths, trainMaskPaths_sub, 1)
        end 
        trainMaskPathsList  = table_merge(trainMaskPathsList, trainMaskPathsList_sub)
    end 

    -- item format: ./data/SK506/images/test/2219.jpg
    print(trainMaskPaths)

    local testImagePaths, tmp1 = findImages(opt.datapath .. '/test/images')
    -- labels 
    local testMaskPaths, tmp2 = findImages(opt.datapath .. '/test/groundTruth/symmetry')
    
    local mean, std = setDatasetStats(trainImagePathsList, trainMaskPathsList)

    -- sanity check
    assert(trainImagePaths:size(1) == trainMaskPaths:size(1))
    assert(testMaskPaths:size(1) == testImagePaths:size(1))

    if trainImagePaths:nElement() == 0 then
        error(" !!! Check datapath, it might be wrong !!! ")
    end

    local info = {
        basedir = opt.dataPath, 
        -- classes = classes,
        -- classWeights = classWeights,
        mean = mean,
        std = std,
        train = {
            imagePath = trainImagePaths,
            labelPath = trainMaskPaths,
        },
        test = {
            imagePath = testImagePaths,
            labelPath = testMaskPaths,
        },
    }

    print(" | saving list of images to " .. cacheFile)
    print(" | number of traning images: ".. trainImagePaths:size(1))
    print(" | number of validation images: ".. testImagePaths:size(1))
    torch.save(cacheFile, info)
    
    return info
end

return M
