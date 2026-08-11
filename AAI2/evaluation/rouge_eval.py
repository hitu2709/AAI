from rouge_score import rouge_scorer

reference = """
We apologize for the inconvenience.
Please share your order ID,
product image,
and product received.
We will arrange replacement.
"""

def rouge(response):

    scorer = rouge_scorer.RougeScorer(['rouge1','rougeL'],use_stemmer=True)

    score=scorer.score(reference,response)

    return score["rouge1"].fmeasure