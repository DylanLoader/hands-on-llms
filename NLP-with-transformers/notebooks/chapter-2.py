#%%[markdown]

## Chapter 2: Text Classification
#%% Package imports
import pandas as pd
from matplotlib import pyplot as plt
import umap.umap_ as umap
from sklearn.preprocessing import MinMaxScaler
#%%
# building an emotion classifier with HF transformers
from datasets import load_dataset
emotions = load_dataset("emotion")

#%%
print(emotions)


# %%
# split out the train-test-val datasets in apache arrow 
train_ds = emotions["train"]
val_ds = emotions["validation"]
test_ds = emotions["test"]
# %%
print(train_ds.features)
# %%
# Look at a few values with the corresponding labels
train_ds[:5]
# %%
# Converting a hf dataset to pandas 
emotions.set_format(type="pandas")
df = emotions["train"][:]
df.head()
# %%
def convert_label_to_str(row):
    return emotions["train"].features["label"].int2str(row)

df["label_name"] = df["label"].apply(convert_label_to_str)
df.head()
# converting a ds column from int to string
# df["label_name"] = df["label"].astype(str)
# %%
df.dtypes
# %%
# Check class distribution
df['label_name'].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
plt.show()
# %%
# Find words per tweet
df["words_per_tweet"] = df["text"].str.split().apply(len)
df.boxplot("words_per_tweet", by="label_name",grid=False,showfliers=False, color="black")
plt.suptitle("")
plt.xlabel("")
plt.show()

# %%
# reset the format of the dataset from pandas to arrow
emotions.reset_format()
# %%
# tokenization

# The most simple character-level tokenization
text = "Tokenizing text is a core task of NLP."
tokenized_text = list(text)
print(tokenized_text)
# %%
# Numericalize the tokens
token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
# %%
input_ids = [token2idx[token] for token in tokenized_text]
print(input_ids)
# %%
# Convert the integer mapping to one-hot encoded vectors
categorical_df = pd.DataFrame(
    {"Name": ["Bumblebee", "Optimus Prime", "Megatron"], "Label ID": [0,1,2]}
)

categorical_df
# %%
import torch
import torch.nn.functional as F

input_ids = torch.tensor(input_ids)
one_hot_encoding = F.one_hot(input_ids, num_classes = len(token2idx))
one_hot_encoding.shape
# %%[markdown]

### Word Tokenization
input_ids

# %%
tokenized_text = text.split()
# %%
tokenized_text
print(tokenized_text)
# %%
# Subword tokenization with WordPiece
from transformers import AutoTokenizer

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
# %%
# We can also load the tokenizer for the BERT specific model
from transformers import DistilBertTokenizer
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)

# %%
assert(tokenizer(text)==distilbert_tokenizer(text))

#%%
# The tokenizers return a dict containing the token id mappings and the attention masks
tokenizer(text)

#%% 
# We can also convert the token ids back to text
tokens = tokenizer.convert_ids_to_tokens(tokenizer(text)['input_ids'])
print(tokens)


# %%
# Or converting the tokens back to a string
tokenizer.convert_tokens_to_string(tokens)
# %%
# For each pretrained tokenizer we can check the following attributes
# vocab size 
tokenizer.vocab_size
#%%
# Maximum context len
tokenizer.model_max_length
 
# %%
# Model fields required in the forward pass
tokenizer.model_input_names
# %%
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)
# %%
print(tokenize(emotions["train"][:2]))
#%%[markdown]
#The attention mask allows us to tokenize text of differing lengths, but allows the model to ignore padded sections. 

# %%
# Applying the tokenization approach across the corpus
emotions.map(tokenize, batched=True, batch_size=None)

# %%
# Load the model using AutoModel
from transformers import AutoModel

model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)
#%%
# Extracting the last hidden states for additional modelling 
text = "this is a text"
inputs = tokenizer(text, return_tensors="pt")
print(f"Input tensor shape: {inputs['input_ids'].size()}")
# The shape of the input tensor i[batch_size, n_tokens]

#%% 
# We take the encodings as a tensor and move them to the device we want to use
inputs = {k:v.to(device) for k, v in inputs.items()}

# no_grad reduces the memory footprint for inference as it does not automatically compute gradients
with torch.no_grad():
    outputs = model(**inputs)
print(outputs)
# %%
outputs.last_hidden_state.size()
# %%
# We often use the state for the [CLS] token as it occurs in every sequence
outputs.last_hidden_state[:, 0].size()
# %%
# We now want to get the hidden state for the whole dataset
tokenizer.model_input_names

#%%

def extract_hidden_states(batch):
    # Place model inputs on the GPU
    inputs = {k:v.to(device) for k,v in batch.items() 
              if k in tokenizer.model_input_names}
    # Extract last hidden states
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    # Return vector for [CLS] token
    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}
# %%
emotions = load_dataset("emotion")
emotions
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
# Convert input ids and mask columns to torch
emotions_encoded.set_format("torch", columns = ["input_ids", "attention_mask", "label"])

# %%
#extract the hidden states across all splits
emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)
# %%
# Check that the hidden states have been extracted for the default batch_size=1000
emotions_hidden["train"].column_names

# Creating a feature matrix
import numpy as np
X_train = np.array(emotions_hidden["train"]["hidden_state"])
X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
y_train = np.array(emotions_hidden["train"]["label"])
y_valid = np.array(emotions_hidden["validation"]["label"])
X_train.shape, X_valid.shape

# %%
# Scaled features to [0,1] range 
X_scaled = MinMaxScaler().fit_transform(X_train)
# initialize and fit umap
mapper = umap.UMAP(n_components=2, metric="cosine").fit(X_scaled)
# %%
df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
df_emb["label"] = y_train
df_emb.head()
# %%
fig, axes = plt.subplots(2, 3, figsize=(7,5))
axes = axes.flatten()
cmaps = ["Greys", "Blues", "Oranges", "Reds", "Purples", "Greens"]
labels = emotions["train"].features["label"].names
for i, (label, cmap) in enumerate(zip(labels, cmaps)): 
    df_emb_sub = df_emb.query(f"label == {i}") 
    axes[i].hexbin(
        df_emb_sub["X"], 
        df_emb_sub["Y"], 
        cmap=cmap,
        gridsize=20, 
        linewidths=(0,)
        )
    axes[i].set_title(label)
    axes[i].set_xticks([]), axes[i].set_yticks([])
plt.tight_layout()
plt.show()


# %%
# Use the embeddings to create a logistic classifier
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression(max_iter=3000)
lr_clf.fit(X_train, y_train)
lr_clf.score(X_valid, y_valid)

# %%

# Create a dummy classifier for reference
from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
dummy_clf.score(X_valid, y_valid)

# %%
# Create a confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plot_confusion_matrix(y_preds:np.array, y_true:np.array, labels=np.array):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6,6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalizes Confusion Matrix")
    plt.show()
    
y_preds = lr_clf.predict(X_valid)
plot_confusion_matrix(y_preds, y_valid, labels)
    
# %%
# Fine tuning transformers
####

# Load the pretrained distilbert model 
from transformers import AutoModelForSequenceClassification
num_labels =6
model = (AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to(device))

# Define the performance metrics
from sklearn.metrics import accuracy_score, f1_score
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1":f1}

    
# %%
# training the model
from huggingface_hub import notebook_login

#%%
notebook_login()

# %%
from transformers import Trainer, TrainingArguments
batch_size=64
logging_steps = len(emotions_encoded["train"]) // batch_size
model_name = f"{model_ckpt}-finetuned-emotion"
training_args = TrainingArguments(
    output_dir =model_name, 
    num_train_epochs=2,
    
)
# %%
trainer = Trainer(
    model=model, 
    compute_metrics=compute_metrics,
    train_dataset=emotions_encoded["train"],
    eval_dataset=emotions_encoded["validation"],
    tokenizer=tokenizer
                  )
trainer.train()
# %%
preds_output = trainer.predict(emotions_encoded["validation"])

# %%
preds_output.metrics
# %%
