import requests
from bs4 import BeautifulSoup


def search_web(query):

    url = "https://www.google.com/search"

    params = {
        "q": query
    }

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=10
    )

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    results = []

    for result in soup.select("div.MjjYud")[:5]:

        title = result.select_one("h3")
        link = result.select_one("a")

        if title and link:

            results.append({
                "title": title.get_text(),
                "url": link.get("href")
            })

    return results

def research_agent(query, llm):

    results = search_web(query)

    sources = "\n".join(
        f"- {r['title']} : {r['url']}"
        for r in results
    )

    prompt = f"""
You are a Research Agent.

Research the following topic using the sources
provided below.

TOPIC:
{query}

SOURCES:
{sources}

Analyze the available information and produce:

1. Key findings
2. Important facts
3. Possible limitations
4. Source references

Do not invent information.
"""

    response = llm.invoke(prompt)

    return {
        "findings": response.content,
        "sources": results
    }