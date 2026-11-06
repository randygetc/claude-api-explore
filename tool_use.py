import anthropic
from pydantic import BaseModel, ConfigDict
import json

client = anthropic.Anthropic()

#define the tools
tools = [
    {
        "name": "get_invoice",
        "description": (
            "Retrieve an invoice by its invoice number. "
            "Returns vendor, amount, currency, and status."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "invoice_number": {
                    "type": "string",
                    "description": "Invoice identifier"
                }
            },
            "required": ["invoice_number"],
            "additionalProperties": False
        }
    }
]

# implement the python function

INVOICES = {
    "INV-9981": {
        "invoice_number": "INV-9981",
        "vendor": "Acme Corp",
        "total": 1250.00,
        "currency": "USD",
        "status": "unpaid"
    }
}

def get_invoice(invoice_number: str) -> dict:
    return INVOICES.get(
        invoice_number,
        {"error": "Invoice not found"}
    )

# ask claude to use the tool, but do not execute yet


messages = [
    {
        "role": "user",
        "content": "What is the status of invoice INV-9981?"
    }
]

response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=1024,
    tools=tools,
    messages=messages
)

print("Stop reason:", response.stop_reason)

for block in response.content:
    if block.type == "tool_use":
        print("Tool:", block.name)
        print("Arguments:", block.input)
        print("Tool use ID:", block.id)

# execute the tool claude has identified



if response.stop_reason == "tool_use":
    messages.append({
        "role": "assistant",
        "content": response.content
    })

    results = []

    for block in response.content:
        if block.type != "tool_use":
            continue

        if block.name == "get_invoice":
            result = get_invoice(**block.input)
        else:
            result = {"error": "Unknown tool"}

        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": json.dumps(result)
        })

    messages.append({
        "role": "user",
        "content": results
    })

    final_response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )

    for block in final_response.content:
        if block.type == "text":
            print(block.text)

print("FORCE PARTICULAR TOOL")
# force claude to use get_invoce tool instead of searching for tool
response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=1024,
    tools=tools,
    tool_choice={
        "type": "tool",
        "name": "get_invoice"
    },
    messages=messages
)

print(messages)