from bert_score import score

reference = """
We apologize for the inconvenience.
Please share order ID and product image.
"""

def bert(response):

    P,R,F1 = score([response],[reference],lang="en")

    return float(F1.mean())