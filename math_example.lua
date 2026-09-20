local MLP=require("mlp")
local brain=MLP.new({1,8,1},{learning_rate=0.05,activation="tanh",output_activation="tanh"})
local data={}
for x=-1,1,0.1 do data[#data+1]={{x},{x}} end
for epoch=1,5000 do
 local loss=brain:train_batch(data,1)
 if epoch%500==0 then print("epoch "..epoch.." loss "..string.format("%.6f",loss)) end
end
for x=-1,1,0.25 do print(string.format("x=%+.2f -> %.4f",x,brain:predict({x})[1])) end
