import wikipediaapi

wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="AgenticAI/1.0"
)

def search_wikipedia(query):

    # Try the complete query first
    page = wiki.page(query)

    if page.exists():
        return page.summary[:1000]

    # Remove last word repeatedly until a page is found
    words = query.split()

    while len(words) > 1:

        words.pop()

        new_query = " ".join(words)

        page = wiki.page(new_query)

        if page.exists():
            return page.summary[:1000]

    return "No information found on Wikipedia."


def calculator(expression):
    try:
        return eval(expression, {"__builtins__": {}}, {})
    except Exception:
        return "Invalid mathematical expression."