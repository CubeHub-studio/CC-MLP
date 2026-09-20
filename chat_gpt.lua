local GPT=require("tinygpt")
local tok=require("tokenizer")
local web=require("web_search")

local model=nil
if fs.exists("tiny-gpt.model") then
 model=GPT.load("tiny-gpt.model")
end

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

local function looksLikeSearch(text)
 local s=text:lower()
 return s:match("^search ") or
        s:match("^look up ") or
        s:match("^lookup ") or
        s:match("^web ") or
        s:match("^internet ")
end

local function searchQuery(text)
 return text:gsub("^[Ss][Ee][Aa][Rr][Cc][Hh]%s+","")
            :gsub("^[Ll][Oo][Oo][Kk]%s+[Uu][Pp]%s+","")
            :gsub("^[Ll][Oo][Oo][Kk][Uu][Pp]%s+","")
            :gsub("^[Ww][Ee][Bb]%s+","")
            :gsub("^[Ii][Nn][Tt][Ee][Rr][Nn][Ee][Tt]%s+","")
end

print("CC-GPT ready!")
print("Math + chat + internet search enabled.")
if model then
 print("GPT model loaded.")
else
 print("No tiny-gpt.model found. Chat is unavailable until you train one.")
end
print("Search: search Minecraft 1.21")
print("Math: 25+17")
print("Type 'exit' to quit.")

while true do
 write("> ")
 local prompt=read()

 if prompt=="exit" then
  break
 elseif prompt=="" then
 else
  local answer=solveMath(prompt)

  if answer~=nil then
   print("Math: "..formatAnswer(answer))
  elseif looksLikeSearch(prompt) then
   local query=searchQuery(prompt)
   print("Searching the internet for: "..query)
   local results,err=web.search(query)

   if not results then
    printError(err)
   elseif #results==0 then
    print("No results found.")
   else
    for i,result in ipairs(results) do
     print("")
     print("["..i.."] "..result.title)
     print(result.text)
     if result.url~="" then
      print(result.url)
     end
     if i>=8 then break end
    end
   end
  elseif model then
   local generated=model:generate(tok.encode(prompt),64,0.8)
   local decoded=tok.decode(generated)
   print(decoded)
  else
   print("I don't have a trained GPT model yet.")
   print("Train one with: train_gpt.lua training.txt 10 32")
  end
 end
end
