local GPT=require("tinygpt")
local tok=require("tokenizer")
local args={...}; local input=args[1] or "training.txt"; local epochs=tonumber(args[2]) or 3; local seq=tonumber(args[3]) or 32
if not fs.exists(input) then print("Missing "..input); print("Create training.txt, then run: train_gpt.lua training.txt 10 32"); return end
local tokens=tok.from_file(input); if #tokens<seq+1 then error("training text is too short") end
math.randomseed(os.epoch("utc")); local model=GPT.new({d_model=16,ff_dim=32,max_seq=seq,learning_rate=0.01})
print("CC-GPT training"); print("bytes: "..#tokens.."  epochs: "..epochs.."  context: "..seq)
for e=1,epochs do local loss=model:train_text(tokens,1,seq); print("epoch "..e.." loss "..string.format("%.4f",loss)) end
model:save("tiny-gpt.model"); print("Saved tiny-gpt.model")
