def security_agent(logs, llm):

    prompt = f"""
You are a cybersecurity log analysis agent.

Analyze the following security logs.

LOGS:
{logs}

Perform the following tasks:

1. Identify suspicious activities.
2. Identify potential attacks.
3. Classify severity as:
   LOW, MEDIUM, HIGH, or CRITICAL.
4. Explain why the activity is suspicious.
5. Suggest mitigation steps.
6. Provide indicators of compromise if present.

Return the result in this format:

Threat:
Evidence:
Attack Type:
Severity:
Reason:
Indicators:
Recommended Mitigation:

Do not invent information that is not present in the logs.
"""

    response = llm.invoke(prompt)

    return response.content