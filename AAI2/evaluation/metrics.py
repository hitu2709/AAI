import pandas as pd

from rouge_eval import rouge
from bertscore_eval import bert
from cosine_eval import cosine
from llm_judge import judge

df=pd.read_csv("outputs/responses.csv")

rouges=[]
berts=[]
cosines=[]
judges=[]

for response in df["Response"]:

    rouges.append(rouge(response))

    berts.append(bert(response))

    cosines.append(cosine(response))

    judges.append(judge(response))

df["ROUGE"]=rouges
df["BERTScore"]=berts
df["Cosine"]=cosines
df["Judge"]=judges

df.to_csv("outputs/evaluation.csv",index=False)

print(df)