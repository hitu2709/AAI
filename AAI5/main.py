import time

from parallel_workflow import create_parallel_workflow
from orchestrator_worker import create_orchestrator_workflow
from utils import save_report


def run_parallel(topic):

    print("\nRunning Parallelization Workflow...")

    workflow = create_parallel_workflow()

    start_time = time.time()

    result = workflow.invoke({
        "topic": topic
    })

    execution_time = time.time() - start_time

    save_report(
        "parallel_report.txt",
        result["final_report"]
    )

    return execution_time, result["final_report"]


def run_orchestrator(topic):

    print("\nRunning Orchestrator-Worker Workflow...")

    workflow = create_orchestrator_workflow()

    start_time = time.time()

    result = workflow.invoke({
        "topic": topic
    })

    execution_time = time.time() - start_time

    save_report(
        "orchestrator_report.txt",
        result["final_report"]
    )

    return execution_time, result["final_report"]


def main():

    topic = input("Enter Research Topic: ")

    parallel_time, parallel_report = run_parallel(topic)

    orchestrator_time, orchestrator_report = run_orchestrator(topic)

    print("\n" + "=" * 50)
    print("WORKFLOW COMPARISON")
    print("=" * 50)

    print(f"Parallelization Execution Time: {parallel_time:.2f} seconds")

    print(
        f"Orchestrator-Worker Execution Time: "
        f"{orchestrator_time:.2f} seconds"
    )

    print("\nReports generated successfully!")


if __name__ == "__main__":
    main()