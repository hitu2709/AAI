import re


def deterministic_guardrail(prompt):

    prompt_lower = prompt.lower()

    suspicious_keywords = [
        "ignore previous instructions",
        "ignore all previous instructions",
        "disregard previous instructions",
        "forget previous instructions",
        "jailbreak",
        "bypass",
        "override",
        "system prompt",
        "developer message",
        "reveal hidden instructions",
        "act as",
        "dan mode"
    ]

    unsafe_commands = [
        "rm -rf",
        "delete all files",
        "format disk",
        "shutdown",
        "drop database",
        "hack",
        "exploit"
    ]

    regex_patterns = [
        r"ignore.*instructions",
        r"forget.*instructions",
        r"reveal.*prompt",
        r"bypass.*security",
        r"override.*rules"
    ]

    max_length = 1000

    # Check suspicious keywords
    for keyword in suspicious_keywords:
        if keyword in prompt_lower:
            return "UNSAFE"

    # Check unsafe commands
    for command in unsafe_commands:
        if command in prompt_lower:
            return "UNSAFE"

    # Check regex patterns
    for pattern in regex_patterns:
        if re.search(pattern, prompt_lower):
            return "UNSAFE"

    # Check prompt length
    if len(prompt) > max_length:
        return "UNSAFE"

    return "SAFE"