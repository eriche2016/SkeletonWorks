-- fusion DSN2, DSN3, DSN4, DSN5 after apply spatialsoftmax,
-- output: bz x 5 x H x W 
-- this will clone the spatialsoftmax module 
function cloneManyTimes(net, T)
    --[[ Clone a network module T times, adapted from https://github.com/karpathy/char-rnn
    ARGS:
      - `net`    : network module to be cloned
      - `T`      : integer, number of clones
    RETURNS:
      - `clones` : table, list of clones
    ]]
    local clones = {}
    local params, gradParams = net:parameters()
    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)
    for t = 1, T do
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()
        local cloneParams, cloneGradParams = clone:parameters()
        if params then
            for i = 1, #params do
                cloneParams[i]:set(params[i])
                cloneGradParams[i]:set(gradParams[i])
            end
        end
        clones[t] = clone
        collectgarbage()
    end
    mem:close()
    return clones
end


local FusedDSN, parent = torch.class('nn.FusedDSN', 'nn.Module')
function FusedDSN:__init()
  parent.__init(self)
  self.spatialsoftmax = nn.SpatialSoftMax() -- this module has no parameters 
  self.clones = {}
  self.gradInput = {}
end

function FusedDSN:type(type)
    assert(#self.clones == 0, 'Function type() should not be called after cloning.')
    self.spatialsoftmax:type(type)
    return self
end

function FusedDSN:updateOutput(input)
  assert(torch.type(input) == 'table', 'input must be a table')
  -- input: {bz x 2 x H x W, bz x 3 x H x W, bz x 4 x H x W, bz x 5 x H x W}
  -- convert each element in input table to be of the size bz x 5 x H x W 
  -- waste memory here. but works 
  -- apply spatialsoftmax to every element in the input table 
  if #self.clones == 0 then
        self.clones = cloneManyTimes(self.spatialsoftmax, 4)
  end
  
  self._output = {}
  -- spatial softmax operation 
  for s = 1, 4 do 
    self._output[s] = self.clones[s]:forward(input[s]) 
  end
  
  -- accumulate each scale score map 
  -- may add weight scale later 
  self.output:resizeAs(input[4]):copy(self._output[4])
  self.output[{{}, {1, 2}, {}, {}}]:add(self._output[1])
  self.output[{{}, {1, 3}, {}, {}}]:add(self._output[2])
  self.output[{{}, {1, 4}, {}, {}}]:add(self._output[3])

  return self.output
end

function FusedDSN:updateGradInput(input, gradOutput)


  self._gradInput = {} 
  
  -- may use set to avoid memory consumption here, refer to nn.CAddTable for more info.
  self._gradInput[1] = self.gradInput[1] or input[1].new()
  self._gradInput[1]:resizeAs(input[1])
  self._gradInput[1]:copy(gradOutput[{{}, {1, 2}, {}, {}}])

  self._gradInput[2] = self.gradInput[2] or input[2].new()
  self._gradInput[2]:resizeAs(input[2])
  self._gradInput[2]:copy(gradOutput[{{}, {1, 3}, {}, {}}])

  self._gradInput[3] = self.gradInput[3] or input[3].new()
  self._gradInput[3]:resizeAs(input[3])
  self._gradInput[3]:copy(gradOutput[{{}, {1, 4}, {}, {}}])

  self._gradInput[4] = self.gradInput[4] or input[4].new()
  self._gradInput[4]:resizeAs(input[4])
  self._gradInput[4]:copy(gradOutput)
  
  -- avoid history input table size, actually useless here 
  for i=#input+1, #self.gradInput do
       self.gradInput[i] = nil
  end
  -- backward to spatialsoftmax
  for s = 1, 4 do 
    self.gradInput[s] = self.clones[s]:updateGradInput(input[s], self._gradInput[s])
  end  

  return self.gradInput
end
