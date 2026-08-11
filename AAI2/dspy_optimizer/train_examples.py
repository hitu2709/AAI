import dspy

trainset=[

dspy.Example(

query="Wrong product received",

answer="""

Apologize.

Ask Order ID.

Ask Product Photo.

Arrange replacement.

"""

).with_inputs("query")

]