import dspy

class CustomerSupport(dspy.Signature):

    query=dspy.InputField()

    answer=dspy.OutputField()