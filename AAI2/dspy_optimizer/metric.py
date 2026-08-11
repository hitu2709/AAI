from evaluation.rouge_eval import rouge

def metric(*args):
    example = args[0]
    pred = args[1]
    return rouge(pred.answer)