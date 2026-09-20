-- CC-MLP web search helper
-- Uses DuckDuckGo's public Instant Answer endpoint.

local M={}

local function urlEncode(s)
 s=s:gsub("([^%w%-_%.~])",function(c)
  return string.format("%%%02X",string.byte(c))
 end)
 return s
end

local function jsonDecode(s)
 if textutils and textutils.unserializeJSON then
  return textutils.unserializeJSON(s)
 end
 return nil
end

function M.search(query)
 local url="https://api.duckduckgo.com/?q="..urlEncode(query).."&format=json&no_html=1&skip_disambig=1"
 local response=http.get(url,{["User-Agent"]="CC-MLP/1.0"})
 if not response then
  return nil,"Internet access is unavailable or the request was blocked."
 end

 local body=response.readAll()
 response.close()

 local data=jsonDecode(body)
 if not data then
  return nil,"Could not read the search response."
 end

 local results={}

 if data.AbstractText and data.AbstractText~="" then
  table.insert(results,{
   title=data.Heading or "Answer",
   text=data.AbstractText,
   url=data.AbstractURL or ""
  })
 end

 local function collect(items)
  for _,item in ipairs(items or {}) do
   if item.Text and item.Text~="" then
    table.insert(results,{
     title=item.Text,
     text=item.Text,
     url=item.FirstURL or ""
    })
   end
   collect(item.Topics)
   if #results>=8 then return end
  end
 end

 collect(data.RelatedTopics)
 return results
end

return M
