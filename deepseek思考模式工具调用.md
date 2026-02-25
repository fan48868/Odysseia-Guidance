# 非思考模式
样例代码
这里以获取用户当前位置的天气信息为例，展示了使用 Tool Calls 的完整 Python 代码。

Tool Calls 的具体 API 格式请参考对话补全文档。

from openai import OpenAI

def send_messages(messages):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools
    )
    return response.choices[0].message

client = OpenAI(
    api_key="<your api key>",
    base_url="https://api.deepseek.com",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather of a location, the user should supply a location first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"]
            },
        }
    },
]

messages = [{"role": "user", "content": "How's the weather in Hangzhou, Zhejiang?"}]
message = send_messages(messages)
print(f"User>\t {messages[0]['content']}")

tool = message.tool_calls[0]
messages.append(message)

messages.append({"role": "tool", "tool_call_id": tool.id, "content": "24℃"})
message = send_messages(messages)
print(f"Model>\t {message.content}")


这个例子的执行流程如下：

用户：询问现在的天气
模型：返回 function get_weather({location: 'Hangzhou'})
用户：调用 function get_weather({location: 'Hangzhou'})，并传给模型。
模型：返回自然语言，"The current temperature in Hangzhou is 24°C."
注：上述代码中 get_weather 函数功能需由用户提供，模型本身不执行具体函数。

# 思考模式
从 DeepSeek-V3.2 开始，API 支持了思考模式下的工具调用能力，详见思考模式。

回答问题 1 过程中（请求 1.1 - 1.3），模型进行了多次思考 + 工具调用后给出答案。在这个过程中，用户需回传思维链内容（reasoning_content）给 API，以让模型继续思考。
在下一个用户问题开始时（请求 2.1），需删除之前的 reasoning_content，并保留其它内容发送给 API。如果保留了 reasoning_content 并发送给 API，API 将会忽略它们。
兼容性提示
因思考模式下的工具调用过程中要求用户回传 reasoning_content 给 API，若您的代码中未正确回传 reasoning_content，API 会返回 400 报错。正确回传方法请您参考下面的样例代码。

样例代码
下面是一个简单的在思考模式下进行工具调用的样例代码：

import os
import json
from openai import OpenAI

# The definition of the tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_date",
            "description": "Get the current date",
            "parameters": { "type": "object", "properties": {} },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather of a location, the user should supply the location and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": { "type": "string", "description": "The city name" },
                    "date": { "type": "string", "description": "The date in format YYYY-mm-dd" },
                },
                "required": ["location", "date"]
            },
        }
    },
]

# The mocked version of the tool calls
def get_date_mock():
    return "2025-12-01"

def get_weather_mock(location, date):
    return "Cloudy 7~13°C"

TOOL_CALL_MAP = {
    "get_date": get_date_mock,
    "get_weather": get_weather_mock
}

def clear_reasoning_content(messages):
    for message in messages:
        if hasattr(message, 'reasoning_content'):
            message.reasoning_content = None

def run_turn(turn, messages):
    sub_turn = 1
    while True:
        response = client.chat.completions.create(
            model='deepseek-chat',
            messages=messages,
            tools=tools,
            extra_body={ "thinking": { "type": "enabled" } }
        )
        messages.append(response.choices[0].message)
        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        tool_calls = response.choices[0].message.tool_calls
        print(f"Turn {turn}.{sub_turn}\n{reasoning_content=}\n{content=}\n{tool_calls=}")
        # If there is no tool calls, then the model should get a final answer and we need to stop the loop
        if tool_calls is None:
            break
        for tool in tool_calls:
            tool_function = TOOL_CALL_MAP[tool.function.name]
            tool_result = tool_function(**json.loads(tool.function.arguments))
            print(f"tool result for {tool.function.name}: {tool_result}\n")
            messages.append({
                "role": "tool",
                "tool_call_id": tool.id,
                "content": tool_result,
            })
        sub_turn += 1

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url=os.environ.get('DEEPSEEK_BASE_URL'),
)

# The user starts a question
turn = 1
messages = [{
    "role": "user",
    "content": "How's the weather in Hangzhou Tomorrow"
}]
run_turn(turn, messages)

# The user starts a new question
turn = 2
messages.append({
    "role": "user",
    "content": "How's the weather in Hangzhou Tomorrow"
})
# We recommended to clear the reasoning_content in history messages so as to save network bandwidth
clear_reasoning_content(messages)
run_turn(turn, messages)


在 Turn 1 的每个子请求中，都携带了该 Turn 下产生的 reasoning_content 给 API，从而让模型继续之前的思考。response.choices[0].message 携带了 assistant 消息的所有必要字段，包括 content、reasoning_content、tool_calls。简单起见，可以直接用如下代码将消息 append 到 messages 结尾：

messages.append(response.choices[0].message)

这行代码等价于：

messages.append({
    'role': 'assistant',
    'content': response.choices[0].message.content,
    'reasoning_content': response.choices[0].message.reasoning_content,
    'tool_calls': response.choices[0].message.tool_calls,
})

在 Turn 2 开始时，我们建议丢弃掉之前 Turn 中的 reasoning_content 来节省网络带宽：

clear_reasoning_content(messages)

该代码的样例输出如下：

Turn 1.1
reasoning_content="The user is asking about the weather in Hangzhou tomorrow. I need to get the current date first, then calculate tomorrow's date, and then call the weather API. Let me start by getting the current date."
content=''
tool_calls=[ChatCompletionMessageToolCall(id='call_00_Tcek83ZQ4fFb1RfPQnsPEE5w', function=Function(arguments='{}', name='get_date'), type='function', index=0)]
tool_result(get_date): 2025-12-01

Turn 1.2
reasoning_content='Today is December 1, 2025. Tomorrow is December 2, 2025. I need to format the date as YYYY-mm-dd: "2025-12-02". Now I can call get_weather with location Hangzhou and date 2025-12-02.'
content=''
tool_calls=[ChatCompletionMessageToolCall(id='call_00_V0Uwt4i63m5QnWRS1q1AO1tP', function=Function(arguments='{"location": "Hangzhou", "date": "2025-12-02"}', name='get_weather'), type='function', index=0)]
tool_result(get_weather): Cloudy 7~13°C

Turn 1.3
reasoning_content="I have the weather information: Cloudy with temperatures between 7 and 13°C. I should respond in a friendly, helpful manner. I'll mention that it's for tomorrow (December 2, 2025) and give the details. I can also ask if they need any other information. Let's craft the response."
content="Tomorrow (Tuesday, December 2, 2025) in Hangzhou will be **cloudy** with temperatures ranging from **7°C to 13°C**.  \n\nIt might be a good idea to bring a light jacket if you're heading out. Is there anything else you'd like to know about the weather?"
tool_calls=None

Turn 2.1
reasoning_content="The user wants clothing advice for tomorrow based on the weather in Hangzhou. I know tomorrow's weather: cloudy, 7-13°C. That's cool but not freezing. I should suggest layered clothing, maybe a jacket, long pants, etc. I can also mention that since it's cloudy, an umbrella might not be needed unless there's rain chance, but the forecast didn't mention rain. I should be helpful and give specific suggestions. I can also ask if they have any specific activities planned to tailor the advice. Let me respond."
content="Based on tomorrow's forecast of **cloudy weather with temperatures between 7°C and 13°C** in Hangzhou, here are some clothing suggestions:\n\n**Recommended outfit:**\n- **Upper body:** A long-sleeve shirt or sweater, plus a light to medium jacket (like a fleece, windbreaker, or light coat)\n- **Lower body:** Long pants or jeans\n- **Footwear:** Closed-toe shoes or sneakers\n- **Optional:** A scarf or light hat for extra warmth, especially in the morning and evening\n\n**Why this works:**\n- The temperature range is cool but not freezing, so layering is key\n- Since it's cloudy but no rain mentioned, you likely won't need an umbrella\n- The jacket will help with the morning chill (7°C) and can be removed if you warm up during the day\n\n**If you have specific plans:**\n- For outdoor activities: Consider adding an extra layer\n- For indoor/office settings: The layered approach allows you to adjust comfortably\n\nWould you like more specific advice based on your planned activities?"
tool_calls=None
