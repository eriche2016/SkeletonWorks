require 'pl'

local opts = require 'opts'
local DataLoader = require 'data/dataloader'

torch.setdefaulttensortype('torch.FloatTensor')

opt = opts.parse() 
print(opt)

-- data loading 
trainLoader, testLoader = DataLoader.create(opt)

for n, sample in trainLoader:run() do 
	--[[sample will be of type 
							{
								input: 
								target: 
							}
	--]]
    print(sample.target)
    io.read() 	
	-- print(sample.target) 
end 
