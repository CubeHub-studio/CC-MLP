-- CC-MLP: Multilayer Perceptron for CC:Tweaked
local MLP={}
MLP.__index=MLP
local function sigmoid(x) if x < -60 then return 0 elseif x > 60 then return 1 else return 1/(1+math.exp(-x)) end end
local function sigmoid_derivative(y) return y*(1-y) end
local function relu(x) return x>0 and x or 0 end
local function relu_derivative(x) return x>0 and 1 or 0 end
local function tanh_derivative(y) return 1-y*y end
local function activation(name,x)
 if name=="relu" then return relu(x) elseif name=="tanh" then return math.tanh(x) else return sigmoid(x) end
end
local function derivative(name,pre,out)
 if name=="relu" then return relu_derivative(pre) elseif name=="tanh" then return tanh_derivative(out) else return sigmoid_derivative(out) end
end
local function copy(a) local r={} for i=1,#a do r[i]=a[i] end return r end
local function rand() return (math.random()*2-1)*0.5 end
function MLP.new(sizes,options)
 assert(type(sizes)=="table" and #sizes>=2,"sizes needs at least input and output")
 options=options or {}
 local self=setmetatable({},MLP)
 self.sizes=copy(sizes)
 self.learning_rate=options.learning_rate or 0.1
 self.activation=options.activation or "sigmoid"
 self.output_activation=options.output_activation or self.activation
 self.layers={}
 for l=1,#sizes-1 do
  self.layers[l]={weights={},bias={}}
  for j=1,sizes[l+1] do
   self.layers[l].weights[j]={}
   for i=1,sizes[l] do self.layers[l].weights[j][i]=rand() end
   self.layers[l].bias[j]=rand()
  end
 end
 return self
end
function MLP:predict(input)
 assert(#input==self.sizes[1],"input size mismatch")
 local values=copy(input)
 for l=1,#self.layers do
  local layer=self.layers[l]; local out={}
  local act=l==#self.layers and self.output_activation or self.activation
  for j=1,#layer.weights do
   local sum=layer.bias[j]
   for i=1,#values do sum=sum+values[i]*layer.weights[j][i] end
   out[j]=activation(act,sum)
  end
  values=out
 end
 return values
end
function MLP:train(input,target)
 assert(#input==self.sizes[1],"input size mismatch")
 assert(#target==self.sizes[#self.sizes],"target size mismatch")
 local values={copy(input)}; local preacts={}
 for l=1,#self.layers do
  local layer=self.layers[l]; local prev=values[l]; local out={}
  preacts[l]={}
  local act=l==#self.layers and self.output_activation or self.activation
  for j=1,#layer.weights do
   local sum=layer.bias[j]
   for i=1,#prev do sum=sum+prev[i]*layer.weights[j][i] end
   preacts[l][j]=sum; out[j]=activation(act,sum)
  end
  values[l+1]=out
 end
 local deltas={}; local last=#self.layers; deltas[last]={}; local loss=0
 for j=1,#values[last+1] do
  local e=target[j]-values[last+1][j]; loss=loss+e*e
  deltas[last][j]=e*derivative(self.output_activation,preacts[last][j],values[last+1][j])
 end
 for l=last-1,1,-1 do
  deltas[l]={}
  for i=1,self.sizes[l+1] do
   local e=0
   for j=1,self.sizes[l+2] do e=e+self.layers[l+1].weights[j][i]*deltas[l+1][j] end
   deltas[l][i]=e*derivative(self.activation,preacts[l][i],values[l+1][i])
  end
 end
 for l=1,last do
  local prev=values[l]; local layer=self.layers[l]
  for j=1,#layer.weights do
   for i=1,#prev do layer.weights[j][i]=layer.weights[j][i]+self.learning_rate*deltas[l][j]*prev[i] end
   layer.bias[j]=layer.bias[j]+self.learning_rate*deltas[l][j]
  end
 end
 return loss/#target
end
function MLP:train_batch(samples,epochs)
 epochs=epochs or 1; local loss=0
 for _=1,epochs do
  loss=0
  for _,s in ipairs(samples) do loss=loss+self:train(s[1],s[2]) end
  loss=loss/#samples
 end
 return loss
end
function MLP:save(path)
 local f=fs.open(path,"w"); assert(f,"cannot open "..path)
 f.writeLine("return "..textutils.serialize({sizes=self.sizes,learning_rate=self.learning_rate,activation=self.activation,output_activation=self.output_activation,layers=self.layers}))
 f.close()
end
function MLP.load(path)
 local f=fs.open(path,"r"); assert(f,"cannot open "..path)
 local data=load(f.readAll())(); f.close()
 return setmetatable(data,MLP)
end
function MLP:accuracy(samples,threshold)
 threshold=threshold or 0.5; local correct=0
 for _,s in ipairs(samples) do
  local out=self:predict(s[1]); local ok=true
  for i=1,#out do if (out[i]>=threshold and 1 or 0)~=s[2][i] then ok=false break end end
  if ok then correct=correct+1 end
 end
 return correct/#samples
end
return MLP
