function createToolTypes(modelResponse) {
  const children = [];
  for (const tool of modelResponse.tools) {
    const toolSpan = {
      "type": "Span",
      "namespace": "dash_html_components",
      "props": {
        "children": [
          {
            "type": "B",
            "namespace": "dash_html_components",
            "props": {"children": `${tool.type}: `}
          },
          tool.name
        ]
      }
    };
    children.push({
      "type": "Div",
      "namespace": "dash_html_components",
      "props": {"children": toolSpan}
    });
  }
  return children;
}


function addChatbotData(chatbotProvidedDataList) {
  const triggeredId = dash_clientside.callback_context.triggered_id;
  if (triggeredId !== "chatbot-provided") {
    return window.dash_clientside.no_update;
  }
  const out = [];
  console.log(chatbotProvidedDataList);
  for (const chatbotData of chatbotProvidedDataList) {
    const label = {
        "type": "B",
        "namespace": "dash_html_components",
        "props": {
            "children": "Chatbot: "
        }
    };
    const childrenDiv = {
      "type": "Div",
      "namespace": "dash_html_components",
      "props": {
        "children": [
            {"type": "Span", 
            "namespace": "dash_html_components", 
            "props": {"children": [label, chatbotData["species"]]}},
            {"type": "Div",
            "namespace": "dash_html_components",
            "props": {"children": createToolTypes(chatbotData)}}
          ]
        }
    };
    out.push({
      "type": "ListGroupItem", 
      "namespace": "dash_bootstrap_components", 
      "props": {
        "children": childrenDiv, 
        "action": true, 
        "id": {"type": "chatbot-values", 
        "index": out.length}
      }
    });
  }
  return out;
}


window.dash_clientside = Object.assign({}, window.dash_clientside, {
  chatbot_vis: {
    add_chatbot_data: addChatbotData 
  }
});
