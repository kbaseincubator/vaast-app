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
  for (const chatbotProvidedData of chatbotProvidedDataList) {
    const chatbotData = chatbotProvidedData.results;
    const label = {
        "type": "B",
        "namespace": "dash_html_components",
        "props": {
            "children": "Chatbot: "
        }
    };
    for (const modelResponse of chatbotData) {
      const childrenDiv = {
        "type": "Div",
        "namespace": "dash_html_components",
        "props": {
          "children": [
              {"type": "Span", 
              "namespace": "dash_html_components", 
              "props": {"children": [label, modelResponse["species"]]}},
              {"type": "Div",
              "namespace": "dash_html_components",
              "props": {"children": createToolTypes(modelResponse)}}
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
  }
  return out;
}


window.dash_clientside = Object.assign({}, window.dash_clientside, {
  chatbot_vis: {
    add_chatbot_data: addChatbotData 
  }
});
