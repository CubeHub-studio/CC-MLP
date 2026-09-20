local GPT=require("tinygpt")
local tok=require("tokenizer")

if not fs.exists("tiny-gpt.model") then
 print("No tiny-gpt.model found.")
 print("Train with: train_gpt.lua training.txt 10 32")
 return
end

local model=GPT.load("tiny-gpt.model")

-- Solve basic arithmetic safely.
local function solveMath(text)
 local s=text:lower()
 s=s:gsub("what is",""):gsub("calculate",""):gsub("solve","")
 s=s:gsub("plus","+"):gsub("minus","-"):gsub("times","*"):gsub("multiplied by","*")
 s=s:gsub("divided by","/"):gsub("x","*"):gsub("%?","")
 s=s:gsub("%s+","")

 local a,op,b=s:match("^([%-]?%d+%.?%d*)([%+%-%*/])([%-]?%d+%.?%d*)$")
 if not a then return nil end

 a=tonumber(a)
 b=tonumber(b)
 if not a or not b then return nil end

 if op=="+" then return a+b end
 if op=="-" then return a-b end
 if op=="*" then return a*b end
 if op=="/" then
  if b==0 then return "cannot divide by zero" end
  return a/b
 end
end

local function formatAnswer(value)
 if type(value)=="number" and value%1==0 then
  return tostring(math.floor(value))
 end
 return tostring(value)
end

print("CC-GPT ready!")
print("I can chat and solve basic arithmetic.")
print("Examples: 2+2, 10*5, what is 20 divided by 4")
print("Type 'exit' to quit.")

while true do
 write("> ")
 local prompt=read()

 if prompt=="exit" then
  break
 end

 if prompt=="" then
  -- Ignore empty messages.
 else
  local answer=solveMath(prompt)

  if answer~=nil then
   print("Math: "..formatAnswer(answer))
  else
   local generated=model:generate(tok.encode(prompt),64,0.8)
   local decoded=tok.decode(generated)
   print(decoded)
  end
 end
end
