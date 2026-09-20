local GPT=require("tinygpt")
local tok=require("tokenizer")
if not fs.exists("tiny-gpt.model") then print("No tiny-gpt.model found."); print("Train with: train_gpt.lua training.txt 10 32"); return end
local model=GPT.load("tiny-gpt.model"); print("CC-GPT ready. Type 'exit' to quit.")
while true do
 write("> "); local prompt=read(); if prompt=="exit" then break end
 local generated=model:generate(tok.encode(prompt),64,0.8); print(tok.decode(generated))
end
