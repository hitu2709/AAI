from crewai import Task


def create_tasks(
    topic,
    research_agent,
    writer_agent,
    reviewer_agent
):

    # =============================================
    # Research Task
    # =============================================

    research_task = Task(

        description=f"""
        Research the following topic:

        {topic}

        Collect and organize information about:

        1. Introduction
        2. Important concepts
        3. Working or principle
        4. Applications
        5. Advantages
        6. Limitations
        7. Real-world examples
        8. Important facts

        Provide accurate and well-organized research
        findings that can be used by the Content Writer.
        """,

        expected_output="""
        Detailed and organized research findings about
        the given topic.
        """,

        agent=research_agent
    )

    # =============================================
    # Writing Task
    # =============================================

    writing_task = Task(

        description=f"""
        Create a structured research report about:

        {topic}

        Use the research findings produced by the
        Research Agent.

        Include:

        1. Title
        2. Introduction
        3. Main Concepts
        4. Working / Explanation
        5. Applications
        6. Advantages
        7. Limitations
        8. Real-world Examples
        9. Conclusion

        Use simple and clear technical language.
        Do not unnecessarily repeat information.
        """,

        expected_output="""
        A complete, structured and informative
        research report.
        """,

        agent=writer_agent,

        context=[research_task]
    )

    # =============================================
    # Review Task
    # =============================================

    review_task = Task(

        description=f"""
        Review the generated research report about:

        {topic}

        Check the report for:

        1. Accuracy
        2. Completeness
        3. Clarity
        4. Consistency
        5. Missing information
        6. Repetition
        7. Technical correctness

        Correct any problems you find.

        Return only the final reviewed report.
        """,

        expected_output="""
        A corrected, accurate, complete and
        well-structured final research report.
        """,

        agent=reviewer_agent,

        context=[writing_task]
    )

    return (
        research_task,
        writing_task,
        review_task
    )