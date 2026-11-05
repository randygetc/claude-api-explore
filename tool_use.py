import anthropic
from pydantic import BaseModel, ConfigDict

client = anthropic.Anthropic()

class InvoiceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    invoice_number: str
    vendor_name: str
    total_amount: float
    is_paid: bool
    

response = client.messages.parse(
    model="claude-haiku-5-5",
    max_tokens=1024,
    output_config= {
        "format": {
            "type": "json_schema",
            "schema": InvoiceSummary.model_json_schema()
        }
    },
    messages = [{
        "role": "user",
        "content": "Extract data from this invoice: Inv-9981, Acme Corp, Total; $1,250.00, Status: unpaid"
    }]
)

parsed_data = response.parsed_output
print(response.content[0].text)