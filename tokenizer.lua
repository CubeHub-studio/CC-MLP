-- CC-GPT tokenizer: simple byte-level tokenizer for CC:Tweaked
local T={}
T.__index=T
T.vocab_size=128
function T.encode(text)
 local out={}
 for i=1,#text do out[#out+1]=string.byte(text,i) end
 return out
end
function T.decode(tokens)
 local out={}
 for i,t in ipairs(tokens) do out[i]=(t>=0 and t<=127) and string.char(t) or "?" end
 return table.concat(out)
end
function T.from_file(path)
 local f=fs.open(path,"r"); assert(f,"cannot open "..path)
 local s=f.readAll(); f.close(); return T.encode(s)
end
return T
