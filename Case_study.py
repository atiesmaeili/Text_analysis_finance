

import pandas as pd
import numpy as np

df = pd.read_csv("/content/JNJ_8k_filings_raw.csv")
form_8k_filing = df[df.form == "8-K"]
items_sep =form_8k_filing["items"].apply(lambda x: x.split(",")).tolist()

items_all = []
for item in items_sep:
    for i in item:
        items_all.append(i)

item_type, num_filings = np.unique(np.array(items_all), return_counts=True)
jnj_8k_items = pd.DataFrame(dict(item_type = item_type, num_filings = num_filings))
jnj_8k_items.sort_values(by = "num_filings", ascending=False).set_index("item_type")

del df

import pandas as pd

jnj_8ks = pd.read_csv("/content/JNJ_8k_extension.csv",index_col=0)
jnj_8ks.head()

jnj_8ks = jnj_8ks[(jnj_8ks.filingDate > "2019-01-1") & (jnj_8ks.filingDate < "2023-01-1")]

from transformers import BertTokenizer
import matplotlib.pylab as plt


tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
reports = [report for report in jnj_8ks.report if not(pd.isnull(report))]
tokenized_8k_reports = tokenizer(reports)
extensions = [extension for extension in jnj_8ks.extension991 if not(pd.isnull(extension))]
tokenized_extensions = tokenizer(extensions)
report_lengths = [sum(att_mask) for att_mask in tokenized_8k_reports.attention_mask]
extension_lengths = [sum(att_mask) for att_mask in tokenized_extensions.attention_mask]

fig, axs = plt.subplots(1, 2, figsize = (12, 6))
axs[0].hist(report_lengths)
axs[0].set_title("Number of tokens per report")
axs[1].hist(extension_lengths)
axs[1].set_title("Number of tokens per extension (if present)")
plt.show()

from google.colab import drive
drive.mount('/content/drive')

import nltk
nltk.download('punkt')

from transformers import pipeline
import nltk.data
from tqdm.auto import tqdm
import numpy as np
import os
import matplotlib.pylab as plt


# a function to filter out sentences with less than five white space separated tokens
def nbr_ws_tokens(sentence):
    if len(sentence.split(" ")) > 5:
        return True
    else:
        return False

# a function to summarize financial sentiments of all sentences found in a report
def finbert_polarity(sentiment_labels):
    finbert_polarity = {}
    labels, counts = np.unique(sentiment_labels, return_counts = True)

    for label, count in zip(labels, counts):
        finbert_polarity[label] = count

    finbert_keys = list(finbert_polarity.keys())

    if "Positive" in finbert_keys:
        num_positives = finbert_polarity["Positive"]
    else:
        num_positives = 0

    if "Negative" in finbert_keys:
        num_negatives = finbert_polarity["Negative"]
    else:
        num_negatives = 0

    if (num_positives == 0) and (num_negatives == 0):
        final_score = 0
    else:
        final_score = (num_positives - num_negatives) / (num_positives + num_negatives)

    return final_score

# define the Bert model for labeling the sentences
fin_sentiment = pipeline("text-classification", tokenizer = "yiyanghkust/finbert-tone", model = "yiyanghkust/finbert-tone")
# the sentence split model
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

jnj_8ks.loc[:, "report_polarities"] = pd.NA
jnj_8ks.loc[:, "extension_polarities"] = pd.NA


for idx, row in tqdm(jnj_8ks.iterrows()):
    try:
        if not(pd.isnull(row.report)):
            report_sentences = sent_detector.tokenize(row.report)
            report_sentences = list(filter(nbr_ws_tokens, report_sentences))
            max_length = min(max([len(sentence.split(" ")) for sentence in report_sentences]), 512)
            sentiments = fin_sentiment(report_sentences, padding = "max_length", max_length = max_length, truncation = True)
            sentiment_labels = [sentiment["label"] for sentiment in sentiments]
            sentiment_scores = [sentiment["score"] for sentiment in sentiments]
            jnj_8ks.loc[idx, "report_polarities"] = finbert_polarity(sentiment_labels)
        if not(pd.isnull(row.extension991)):
            report_sentences = sent_detector.tokenize(row.extension991)
            report_sentences = list(filter(nbr_ws_tokens, report_sentences))
            max_length = min(max([len(sentence.split(" ")) for sentence in report_sentences]), 512)
            sentiments = fin_sentiment(report_sentences, padding = "max_length", max_length = max_length, truncation = True)
            sentiment_labels = [sentiment["label"] for sentiment in sentiments]
            sentiment_scores = [sentiment["score"] for sentiment in sentiments]
            jnj_8ks.loc[idx, "extension_polarities"] = finbert_polarity(sentiment_labels)
    except:
        print(idx)
jnj_8ks.to_csv("/content/drive/MyDrive/DLTA/WS23 24/Data/jnj_8k_reports_and_extensions_w_polarities.csv", index = False)


fig, axs = plt.subplots(1, 2, figsize = (12, 6))

axs[0].hist([pol for pol in jnj_8ks.report_polarities if not(pol is pd.NA)])
axs[0].set_title("Report polarities")
axs[1].hist([pol for pol in jnj_8ks.extension_polarities if not(pol is pd.NA)])
axs[1].set_title("Extension polarities (if present)")
plt.show()

!pip install pysentiment2

import pysentiment2 as ps

lm = ps.LM()

report_polarities = []
extension_polarities = []

for idx, row in jnj_8ks.iterrows():
    if not(pd.isnull(row.report)):
        lm_tokens_report = lm.tokenize(row.report)
        lm_score_report = lm.get_score(lm_tokens_report)
        report_polarities.append(lm_score_report["Polarity"])
    else:
        report_polarities.append(pd.NA)
    if not(pd.isnull(row.extension991)):
        lm_tokens_extension = lm.tokenize(row.extension991)
        lm_score_extension = lm.get_score(lm_tokens_extension)
        extension_polarities.append(lm_score_extension["Polarity"])
    else:
        extension_polarities.append(pd.NA)

jnj_8ks.loc[:, "report_polarities_lm"] = report_polarities
jnj_8ks.loc[:, "extension_polarities_lm"] = extension_polarities

jnj_8ks["report_polarities"] = pd.to_numeric(jnj_8ks["report_polarities"])
jnj_8ks["extension_polarities"] = pd.to_numeric(jnj_8ks["extension_polarities"])
jnj_8ks["extension_polarities_lm"] = pd.to_numeric(jnj_8ks["extension_polarities_lm"])

jnj_8ks.loc[:, ["report_polarities", "report_polarities_lm"]].dropna().corr()

jnj_8ks.loc[:, ["extension_polarities", "extension_polarities_lm"]].dropna().corr()

import matplotlib.pylab as plt

fig, axs = plt.subplots(1, 2, figsize = (12, 6))
jnj_8ks.loc[: , "report_polarities"].hist(ax = axs[0], color = "green", alpha = 0.75, label = "FinBert")
jnj_8ks.loc[: , "report_polarities_lm"].hist(ax = axs[0], color = "orange", alpha = 0.75, label = "LMcD")
jnj_8ks.loc[: , "extension_polarities"].hist(ax = axs[1], color = "green", alpha = 0.75)
jnj_8ks.loc[: , "extension_polarities_lm"].hist(ax = axs[1], color = "orange", alpha = 0.75)
axs[0].legend()

ff3 = pd.read_csv("/content/FF3.csv")
Stock = pd.read_csv("/content/JNJ_stock_data.csv", parse_dates= ["Date"])

Stock

Stock["Date"] = pd.to_datetime(Stock["Date"],utc=True)
Stock['Date'] = Stock['Date'].dt.strftime('%Y-%m-%d')

Stock

Stock.set_index(pd.to_datetime(Stock.Date),inplace=True)
Stock = Stock.drop("Date",axis=1)

ff3.loc[:,"Date"] = [str(d) for d in ff3.Date]
ff3.set_index(pd.to_datetime([d[:4]+"-"+d[4:6]+"-"+d[6:]for d in ff3.Date]), inplace=True)
ff3.drop(["Date"], axis=1, inplace =True)
ff3 = ff3/100

ff3

Stock

Ret = Stock.pct_change()[["Close","Open"]].dropna()

Ret

df_ff3 = Ret.merge(ff3, left_index=True, right_index=True)

df_ff3

import statsmodels.api as sm

CARS = []
for Date in jnj_8ks.filingDate:
  start_reg = pd.to_datetime(Date) - pd.offsets.Day(200)
  end_reg = pd.to_datetime(Date) - pd.offsets.Day(10)
  X = df_ff3.loc[start_reg:end_reg][["Mkt-RF","SMB","HML"]]
  X = sm.add_constant(X)
  Y = df_ff3.loc[start_reg:end_reg]["Close"] - df_ff3.RF.loc[start_reg:end_reg]
  model = sm.OLS(Y,X).fit()
  betas = model.params.values
  Event_ret = df_ff3.loc[pd.to_datetime(Date) - pd.offsets.Day(1) : pd.to_datetime(Date) + pd.offsets.Day(4)]
  abnormal_ret = (Event_ret.Close - Event_ret.RF) - (betas[0] + Event_ret["Mkt-RF"]*betas[1]+Event_ret["SMB"]*betas[2]+Event_ret["HML"]*betas[3])
  CAR= (abnormal_ret.apply(lambda x: 1 + x).cumprod()-1)[2]
  CARS.append(CAR)

jnj_8ks["CAR"] = CARS

jnj_8ks

jnj_8ks.loc[:, ["extension_polarities", "extension_polarities_lm","CAR"]].dropna().corr()

jnj_8ks.loc[:, ["report_polarities", "report_polarities_lm","CAR"]].dropna().corr()

jnj_8ks.reset_index(inplace=True,drop=True)

jnj_8ks_ = jnj_8ks.dropna()

jnj_8ks_.reset_index(inplace=True,drop=True)

jnj_8ks_.report_polarities_lm[1] < 0

report_polarities = []
report_polarities_lm = []
extension_polarities = []
extension_polarities_lm = []
CAR_list = []
for i in range(len(jnj_8ks_)):
  if jnj_8ks_.report_polarities[i] > 0.2:
    report_polarities.append(1)
  elif jnj_8ks_.report_polarities[i] <-0.2:
    report_polarities.append(-1)
  else:
    report_polarities.append(0)

  if jnj_8ks_.report_polarities_lm[i] > 0.2:
    report_polarities_lm.append(1)
  elif jnj_8ks_.report_polarities_lm[i] <-0.2:
    report_polarities_lm.append(-1)
  else:
    report_polarities_lm.append(0)

  if jnj_8ks_.extension_polarities[i] > 0.2:
    extension_polarities.append(1)
  elif jnj_8ks_.extension_polarities[i] <-0.2:
    extension_polarities.append(-1)
  else:
    extension_polarities.append(0)

  if jnj_8ks_.extension_polarities_lm[i] > 0.2:
    extension_polarities_lm.append(1)
  elif jnj_8ks_.extension_polarities_lm[i] <-0.2:
    extension_polarities_lm.append(-1)
  else:
    extension_polarities_lm.append(0)

  if jnj_8ks_.CAR[i] > 0.01:
    CAR_list.append(1)
  elif jnj_8ks_.CAR[i] <-0.01:
    CAR_list.append(-1)
  else:
    CAR_list.append(0)

match_df = pd.DataFrame()
match_df["report_polarities"] = report_polarities
match_df["report_polarities_lm"] = report_polarities_lm
match_df["extension_polarities"] = extension_polarities
match_df["extension_polarities_lm"] = extension_polarities_lm
match_df["CAR_list"] = CAR_list

match_df.corr()

match_df.extension_polarities.value_counts()

match_df.CAR_list.value_counts()

11/26

