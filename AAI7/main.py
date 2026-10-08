import os

from dotenv import load_dotenv

from crewai import Crew, Process

from agents.agents import create_agents
from tasks.tasks import create_tasks


# ==================================================
# LOAD ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()


# ==================================================
# CHECK API KEY
# ==================================================

if not os.getenv("GROQ_API_KEY"):

    raise ValueError(
        "GROQ_API_KEY is missing.\n"
        "Add it to your .env file."
    )


# ==================================================
# GET TOPIC
# ==================================================

topic = input(
    "Enter the research topic: "
).strip()


if not topic:

    raise ValueError(
        "Research topic cannot be empty."
    )


# ==================================================
# CREATE AGENTS
# ==================================================

(
    research_agent,
    writer_agent,
    reviewer_agent
) = create_agents()


# ==================================================
# CREATE TASKS
# ==================================================

(
    research_task,
    writing_task,
    review_task
) = create_tasks(

    topic,

    research_agent,
    writer_agent,
    reviewer_agent
)


# ==================================================
# CREATE CREW
# ==================================================

crew = Crew(

    agents=[
        research_agent,
        writer_agent,
        reviewer_agent
    ],

    tasks=[
        research_task,
        writing_task,
        review_task
    ],

    process=Process.sequential,

    verbose=True
)


# ==================================================
# START WORKFLOW
# ==================================================

print("\n")
print("=" * 70)
print("CREWAI MULTI-AGENT RESEARCH SYSTEM")
print("=" * 70)

print("\nStarting workflow...\n")


result = crew.kickoff()


# ==================================================
# DISPLAY FINAL REPORT
# ==================================================

print("\n")
print("=" * 70)
print("FINAL REVIEWED REPORT")
print("=" * 70)

print("\n")

print(result)


# ==================================================
# SAVE FINAL REPORT
# ==================================================

os.makedirs(
    "output",
    exist_ok=True
)

output_file = (
    "output/final_report.md"
)


with open(
    output_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        str(result)
    )


print("\n")
print("=" * 70)

print(
    f"Final report saved to: {output_file}"
)

print("=" * 70)