import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def llm_ask(prompt: str) -> str:
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )
    return response.text.strip()

def lead_generation_agent(product: str, target_industry: str) -> str:
    prompt = f"""You are an Expert Lead Generation Agent.
Product/Service: {product}
Target Industry: {target_industry}

Generate 3 realistic company profiles that could be prospective customers.
For each company, output:
- Company Name
- Industry Sub-segment
- Estimated Company Size
- Key Pain Point / Business Need"""
    return llm_ask(prompt)

def qualification_agent(leads: str, criteria: str) -> str:
    prompt = f"""You are a Lead Qualification Agent.
Evaluate the following generated leads based on these criteria: {criteria}

Leads:
{leads}

For each lead:
1. Assign a Qualification Score (1-10).
2. Mark as QUALIFIED (Score >= 7) or NOT QUALIFIED (Score < 7).
3. Provide brief reasoning."""
    return llm_ask(prompt)

def email_agent(qualified_leads: str, value_prop: str) -> str:
    prompt = f"""You are an SDR Outreach Email Agent.
For each QUALIFIED lead in the list below, write a short, highly personalized cold outreach email (under 120 words) with a clear Call to Action (CTA).

Value Proposition: {value_prop}
Qualified Leads:
{qualified_leads}"""
    return llm_ask(prompt)

def run_sdr_workflow(product: str, industry: str):
    print("\n==========================================")
    print("      MULTI-AGENT SDR SYSTEM WORKFLOW")
    print("==========================================")
    print(f"Product: {product}")
    print(f"Target Industry: {industry}\n")
    
    # 1. Lead Generation
    print("--- [Agent 1] Generating Target Leads ---")
    leads = lead_generation_agent(product, industry)
    print(leads)
    
    # 2. Lead Qualification
    print("\n--- [Agent 2] Qualifying Leads ---")
    criteria = "B2B companies with over 50 employees looking to automate workflows"
    qual_results = qualification_agent(leads, criteria)
    print(qual_results)
    
    # 3. Email Generation
    print("\n--- [Agent 3] Drafting Outreach Emails ---")
    value_prop = "Cut response times by 70% and reduce operational costs with intelligent AI agents."
    emails = email_agent(qual_results, value_prop)
    print(emails)

if __name__ == "__main__":
    run_sdr_workflow(
        product="Agentic Workflow Automation Platform",
        industry="Financial Services & Fintech"
    )
