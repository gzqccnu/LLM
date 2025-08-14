# Embedding

> PRE
>
> Must: [tokenizer](../tokenizer/tokenizer.ipynb)
>
> Optional: [Dataset](../basic_pytorch/data/Dataset/Dataset.ipynb) and [DataLoader](../basic_pytorch/data/DataLoader/DataLoader.ipynb)

Here you will learn what is **Embedding**.
<br>
The subsection first talk about **token embedding**, then with **pos embedding** full name is **position embedding**, in which we will mainly deep into three **pos embedding** methods:
- Learned Position Embedding
- Sinusoidal Position Embedding
- RoPE(Rotart Position Embedding)

We will first the math mechanism then implement it in code.

Finally we will try **pre-trained embedding**.