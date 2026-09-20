local GPT=require("tinygpt")
local tok=require("tokenizer")
local web=require("web_search")

local model=nil
if fs.exists("tiny-gpt.model") then
 model=GPT.load("tiny-gpt.model")
end

local memoryFile="gpt_memory.txt"
local memory=""

if fs.exists(memoryFile) then
 local f=fs.open(memoryFile,"r")
 if f then memory=f.readAll() or ""; f.close() end
end

local function saveMemory()
 local f=fs.open(memoryFile,"w")
 if f then
  f.write(memory)
  f.close()
 end
end

local function remember(role,text)
 memory=memory.."\n"..role..": "..text
 -- Keep the persistent memory file from growing forever.
 if #memory>12000 then
  memory=memory:sub(#memory-11999)
 end
 saveMemory()
end

local function solveMath(text)
 local s=text:lower()
 s=s:gsub("what is",""):gsub("calculate",""):gsub("solve","")
 s=s:gsub("plus","+"):gsub("minus","-"):gsub("times","*"):gsub("multiplied by","*")
 s=s:gsub("divided by","/"):gsub("x","*"):gsub("%?","")
 s=s:gsub("%s+","")
 local a,op,b=s:match("^([%-]?%d+%.?%d*)([%+%-%*/])([%-]?%d+%.?%d*)$")
 if not a then return nil end
 a=tonumber(a); b=tonumber(b)
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
 if type(value)=="number" and value%1==0 then return tostring(math.floor(value)) end
 return tostring(value)
end

local function looksLikeSearch(text)
 local s=text:lower()
 return s:match("^search ") or s:match("^look up ") or s:match("^lookup ") or
        s:match("^web ") or s:match("^internet ")
end

local function searchQuery(text)
 return text:gsub("^[Ss][Ee][Aa][Rr][Cc][Hh]%s+","")
  :gsub("^[Ll][Oo][Oo][Kk]%s+[Uu][Pp]%s+","")
  :gsub("^[Ll][Oo][Oo][Kk][Uu][Pp]%s+","")
  :gsub("^[Ww][Ee][Bb]%s+","")
  :gsub("^[Ii][Nn][Tt][Ee][Rr][Nn][Ee][Tt]%s+","")
end

local function contextForModel(prompt)
 local context=memory.."\nUser: "..prompt.."\nAssistant:"
 local tokens=tok.encode(context)
 local max=32
 if #tokens>max then
  local start=#tokens-max+1
  local trimmed={}
  for i=start,#tokens do trimmed[#trimmed+1]=tokens[i] end
  tokens=trimmed
 end
 return tokens
end

print("CC-GPT ready!")
print("Chat + persistent memory + math + internet search enabled.")
if model then print("GPT model loaded.") else print("No tiny-gpt.model found. Math and search still work.") end
print("Memory: "..memoryFile)
print("Type 'exit' to quit. Type 'clear memory' to erase memory.")

while true do
 write("> ")
 local prompt=read()

 if prompt=="exit" then
  break
 elseif prompt=="" then
 else
  if prompt:lower()=="clear memory" then
   memory=""
   saveMemory()
   print("Memory cleared.")
  else
   -- Every user input is added to the conversation memory first.
   remember("User",prompt)

   local answer=solveMath(prompt)

   if answer~=nil then
    local response="Math: "..formatAnswer(answer)
    print(response)
    remember("Assistant",response)

   elseif looksLikeSearch(prompt) then
    local query=searchQuery(prompt)
    print("Searching the internet for: "..query)
    local results,err=web.search(query)

    if not results then
     printError(err)
     remember("Assistant","Web search failed: "..tostring(err))
    elseif #results==0 then
     print("No results found.")
     remember("Assistant","No web results found for "..query)
    else
     local responseParts={}
     for i,result in ipairs(results) do
      local line="["..i.."] "..result.title.." - "..result.text
      print("")
      print(line)
      if result.url~="" then print(result.url) end
      responseParts[#responseParts+1]=line
      if i>=8 then break end
     end
     remember("Web",table.concat(responseParts,"\n"))
    end

   elseif model then
    -- The model receives the current input together with recent conversation memory.
    local generated=model:generate(contextForModel(prompt),64,0.8)
    local decoded=tok.decode(generated)
    -- generate() returns the prompt too, so display only the generated continuation when possible.
    print(decoded)
    remember("Assistant",decoded)
   else
    local response="I don't have a trained GPT model yet. Train one with: train_gpt.lua training.txt 10 32"
    print(response)
    remember("Assistant",response)
   end
  end
 end
end

saveMemory()
