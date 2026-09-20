local GPT=require("tinygpt")
local tok=require("tokenizer")

local args={...}
local input=args[1] or "training.txt"
local epochs=tonumber(args[2]) or 3
local seq=tonumber(args[3]) or 32
local stride=tonumber(args[4]) or seq

if not fs.exists(input) then
 print("Missing "..input)
 print("Create training.txt, then run: train_gpt.lua training.txt 10 32")
 return
end

local tokens=tok.from_file(input)
if #tokens<seq+1 then error("training text is too short") end

math.randomseed(os.epoch("utc"))

-- Smaller model: much more practical on CC:Tweaked computers.
local model=GPT.new({
 d_model=8,
 ff_dim=16,
 max_seq=seq,
 learning_rate=0.01
})

local samples=math.floor((#tokens-seq-1)/stride)+1

print("CC-GPT training")
print("bytes: "..#tokens)
print("epochs: "..epochs)
print("context: "..seq)
print("samples/epoch: "..samples)
print("stride: "..stride)
print("")

for e=1,epochs do
 local total=0
 local n=0

 for i=1,#tokens-seq,stride do
  local x={}
  local y={}

  for j=1,seq do
   x[j]=tokens[i+j-1]
   y[j]=tokens[i+j]
  end

  total=total+model:train(x,y)
  n=n+1

  if n==1 or n%1==0 then
   local loss=total/n
   print("epoch "..e.."/"..epochs.."  sample "..n.."/"..samples.."  loss "..string.format("%.4f",loss))
  end

  os.sleep(0)
 end
end

model:save("tiny-gpt.model")
print("")
print("Saved tiny-gpt.model")
