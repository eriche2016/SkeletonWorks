input = torch.rand(10, 10)
print(input)

-- error code, not work as expected 
input[torch.lt(input, 0.2) or torch.ge(input, 0.8)] = 1 
input[torch.ge(input, 0.2) and torch.lt(input, 0.6)] = 2
input[torch.ge(input, 0.6) and torch.lt(input, 0.8)] = 3
print('this will give wrong answer, so there is not and or or for torch.Tensor')
-- 1 or 0  is 1 
-- 0 or 1 is 0
-- 0 and 1 is 1  
print(input)