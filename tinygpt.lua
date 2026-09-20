-- CC-GPT: tiny decoder-only Transformer for CC:Tweaked
local GPT={}; GPT.__index=GPT

-- CC:Tweaked has a watchdog that stops programs which do not yield.
-- Keep large matrix operations cooperative without changing the math.
local _work=0
local function yieldIfNeeded(amount)
 _work=_work+(amount or 1)
 if _work>=400 then
  _work=0
  os.sleep(0)
 end
end
local function zeros(n) local a={} for i=1,n do a[i]=0 end return a end
local function rand() return (math.random()*2-1)*0.08 end
local function mat(r,c) local a={} for i=1,r do a[i]={}; for j=1,c do a[i][j]=rand(); yieldIfNeeded() end end return a end
local function vec(n) local a={}; for i=1,n do a[i]=rand(); yieldIfNeeded() end return a end
local function dot(a,b) local s=0 for i=1,#a do s=s+a[i]*b[i]; yieldIfNeeded() end return s end
local function softmax(a)
 local m=-math.huge; for i=1,#a do if a[i]>m then m=a[i] end; yieldIfNeeded() end
 local e,sum={},0
 for i=1,#a do e[i]=math.exp(math.max(-60,math.min(60,a[i]-m))); sum=sum+e[i]; yieldIfNeeded() end
 for i=1,#a do e[i]=e[i]/sum; yieldIfNeeded() end
 return e
end
local function outer_add(M,a,b,scale)
 for i=1,#a do for j=1,#b do M[i][j]=M[i][j]+scale*a[i]*b[j]; yieldIfNeeded() end end
end
local function zeros_like(M) local r={}; for i=1,#M do r[i]=zeros(#M[i]); yieldIfNeeded(#M[i]) end return r end
local function addv(a,b) local r={}; for i=1,#a do r[i]=a[i]+b[i]; yieldIfNeeded() end return r end
local function mv(M,x,b)
 local y={}
 for i=1,#M do local s=b and b[i] or 0; for j=1,#x do s=s+M[i][j]*x[j]; yieldIfNeeded() end y[i]=s end
 return y
end
local function mtv(M,x)
 local y=zeros(#M[1]); for i=1,#M do for j=1,#M[i] do y[j]=y[j]+M[i][j]*x[i]; yieldIfNeeded() end end return y
end
function GPT.new(options)
 options=options or {}; local self=setmetatable({},GPT)
 self.vocab=128; self.d=options.d_model or 16; self.ff=options.ff_dim or 32
 self.max_seq=options.max_seq or 32; self.learning_rate=options.learning_rate or 0.01
 self.E=mat(self.vocab,self.d); self.Wq=mat(self.d,self.d); self.Wk=mat(self.d,self.d); self.Wv=mat(self.d,self.d)
 self.W1=mat(self.ff,self.d); self.b1=vec(self.ff); self.W2=mat(self.d,self.ff); self.b2=vec(self.d)
 self.O=mat(self.vocab,self.d); self.bo=vec(self.vocab); return self
end
function GPT:forward(tokens)
 assert(#tokens>=1 and #tokens<=self.max_seq,"sequence length out of range")
 local L=#tokens; local x,q,k,v,a,h={}, {}, {}, {}, {}, {}
 for t=1,L do x[t]=self.E[tokens[t]+1]; q[t]=mv(self.Wq,x[t]); k[t]=mv(self.Wk,x[t]); v[t]=mv(self.Wv,x[t]); yieldIfNeeded() end
 local scale=math.sqrt(self.d)
 for t=1,L do
  local scores={}; for j=1,t do scores[j]=dot(q[t],k[j])/scale; yieldIfNeeded() end; a[t]=softmax(scores)
  local z=zeros(self.d); for j=1,t do for c=1,self.d do z[c]=z[c]+a[t][j]*v[j][c]; yieldIfNeeded() end end
  h[t]=addv(x[t],z)
 end
 local fpre,ffout,y={},{},{}
 for t=1,L do
  fpre[t]=mv(self.W1,h[t],self.b1); ffout[t]=zeros(self.ff); yieldIfNeeded()
  for j=1,self.ff do ffout[t][j]=math.max(0,fpre[t][j]); yieldIfNeeded() end
  y[t]=addv(h[t],mv(self.W2,ffout[t],self.b2))
 end
 local probs={}; for t=1,L do probs[t]=softmax(mv(self.O,y[t],self.bo)); yieldIfNeeded() end
 return {x=x,q=q,k=k,v=v,a=a,h=h,fpre=fpre,ff=ffout,y=y,probs=probs}
end
function GPT:predict(tokens) local c=self:forward(tokens); return c.probs[#tokens],c end
function GPT:train(tokens,targets)
 local c=self:forward(tokens); local L=#tokens; local d=self.d; local ff=self.ff
 local g={E=zeros_like(self.E),Wq=zeros_like(self.Wq),Wk=zeros_like(self.Wk),Wv=zeros_like(self.Wv),
  W1=zeros_like(self.W1),b1=zeros(ff),W2=zeros_like(self.W2),b2=zeros(d),O=zeros_like(self.O),bo=zeros(self.vocab)}
 local loss=0; local gy={}
 for t=1,L do
  local p=c.probs[t]; local target=targets[t]+1; yieldIfNeeded() loss=loss-math.log(math.max(p[target],1e-12))
  local go=zeros(self.vocab); for vtx=1,self.vocab do go[vtx]=p[vtx]-(vtx==target and 1 or 0); yieldIfNeeded() end
  outer_add(g.O,go,c.y[t],1); for vtx=1,self.vocab do g.bo[vtx]=g.bo[vtx]+go[vtx]; yieldIfNeeded() end
  gy[t]=mtv(self.O,go)
 end
 local gh={}
 for t=1,L do
  gh[t]=addv(gy[t],gy[t]); yieldIfNeeded() local gf=mtv(self.W2,gy[t])
  for j=1,ff do if c.fpre[t][j]<=0 then gf[j]=0 end; g.b1[j]=g.b1[j]+gf[j]; yieldIfNeeded() end
  outer_add(g.W2,gy[t],c.ff[t],1); gh[t]=addv(gh[t],mtv(self.W1,gf))
  for j=1,d do g.b2[j]=g.b2[j]+gy[t][j]; yieldIfNeeded() end; outer_add(g.W1,gf,c.h[t],1)
 end
 local gx={}; for t=1,L do gx[t]=zeros(d) end; local scale=math.sqrt(d)
 for t=1,L do
  local gz=gy[t]; local ga={}
  for j=1,t do
   local w=c.a[t][j]; yieldIfNeeded()
   for r=1,d do
    local gv=w*gz[r]; yieldIfNeeded()
    for cc=1,d do g.Wv[r][cc]=g.Wv[r][cc]+gv*c.x[j][cc]; yieldIfNeeded() end
    for cc=1,d do gx[j][cc]=gx[j][cc]+gv*self.Wv[r][cc]; yieldIfNeeded() end
   end
   ga[j]=dot(gz,c.v[j])
  end
  local da=0; for j=1,t do da=da+ga[j]*c.a[t][j] end
  for j=1,t do
   local gs=c.a[t][j]*(ga[j]-da)/scale
   local gq=zeros(d); local gk=zeros(d)
   for r=1,d do gq[r]=gs*c.k[j][r]; gk[r]=gs*c.q[t][r]; yieldIfNeeded() end
   outer_add(g.Wq,gq,c.x[t],1); outer_add(g.Wk,gk,c.x[j],1)
   gx[t]=addv(gx[t],mtv(self.Wq,gq)); gx[j]=addv(gx[j],mtv(self.Wk,gk))
  end
 end
 for t=1,L do
  gx[t]=addv(gx[t],gh[t]); local token=tokens[t]+1
  for r=1,d do g.E[token][r]=g.E[token][r]+gx[t][r] end
 end
 local lr=self.learning_rate
 local function apply(M,G) for i=1,#M do for j=1,#M[i] do M[i][j]=M[i][j]-lr*math.max(-5,math.min(5,G[i][j])); yieldIfNeeded() end end end
 local function applyv(M,G) for i=1,#M do M[i]=M[i]-lr*math.max(-5,math.min(5,G[i])); yieldIfNeeded() end end
 apply(self.E,g.E); apply(self.Wq,g.Wq); apply(self.Wk,g.Wk); apply(self.Wv,g.Wv)
 apply(self.W1,g.W1); applyv(self.b1,g.b1); apply(self.W2,g.W2); applyv(self.b2,g.b2)
 apply(self.O,g.O); applyv(self.bo,g.bo); return loss/L
end
function GPT:train_text(tokens,epochs,seq_len)
 seq_len=seq_len or self.max_seq; epochs=epochs or 1; local final=0
 for ep=1,epochs do
  local total,n=0,0
  for i=1,#tokens-seq_len do
   local x,y={},{}; yieldIfNeeded() for j=1,seq_len do x[j]=tokens[i+j-1]; y[j]=tokens[i+j]; yieldIfNeeded() end
   total=total+self:train(x,y); n=n+1
  end
  final=total/math.max(n,1)
 end
 return final
end
function GPT:generate(prompt_tokens,count,temperature)
 temperature=temperature or 1; count=count or 32; local out={}
 for i=1,#prompt_tokens do out[i]=prompt_tokens[i] end
 for _=1,count do
  local start=math.max(1,#out-self.max_seq+1); local ctx={}; for i=start,#out do ctx[#ctx+1]=out[i] end
  local p=self:predict(ctx); local scaled={}; local sum=0
  for i=1,#p do scaled[i]=p[i]^(1/math.max(temperature,0.05)); sum=sum+scaled[i] end
  local r=math.random()*sum; local acc=0; local next=1
  for i=1,#scaled do acc=acc+scaled[i]; if r<=acc then next=i-1; break end end
  out[#out+1]=next
 end
 return out
end
function GPT:save(path)
 local f=fs.open(path,"w"); assert(f,"cannot open "..path)
 f.writeLine("return "..textutils.serialize({d=self.d,ff=self.ff,max_seq=self.max_seq,learning_rate=self.learning_rate,E=self.E,Wq=self.Wq,Wk=self.Wk,Wv=self.Wv,W1=self.W1,b1=self.b1,W2=self.W2,b2=self.b2,O=self.O,bo=self.bo}))
 f.close()
end
function GPT.load(path)
 local f=fs.open(path,"r"); assert(f,"cannot open "..path); local data=load(f.readAll())(); f.close(); return setmetatable(data,GPT)
end
return GPT
