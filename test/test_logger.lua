require 'optim'
logger = optim.Logger('accuracy.log')

logger:setNames{'Training acc.', 'Test acc.'}

for i = 1, 10 do
   trainAcc = math.random(0, 100)
   testAcc = math.random(0, 100)
   logger:add{trainAcc, testAcc}
end

logger:display(true)
logger:style{'+-', '+-'}
logger:plot()

