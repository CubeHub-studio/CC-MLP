local MLP=require("mlp")
local brain=MLP.new({2,4,4,1},{learning_rate=0.7,activation="tanh",output_activation="sigmoid"})
local data={{{0,0},{0}},{{0,1},{1}},{{1,0},{1}},{{1,1},{0}}}
for epoch=1,10000 do
 local loss=brain:train_batch(data,1)
 if epoch%1000==0 then print("epoch "..epoch.." loss "..string.format("%.6f",loss)) end
end
print("Predictions:")
for _,sample in ipairs(data) do print(sample[1][1].." XOR "..sample[1][2].." = "..string.format("%.4f",brain:predict(sample[1])[1])) end
brain:save("xor.mlp")
print("Saved to xor.mlp")
