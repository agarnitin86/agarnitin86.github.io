---
layout: post
title:  "Anaphora resolution for binary entities"
date:   2019-11-23 17:49:34
categories: nlp, coreference resolution
---

**Anaphora resolution** is the task of finding all expressions that refer to the same entity in a text. It is an important step for a lot of higher level NLP tasks that involve natural language understanding such as document summarization, question answering, and information extraction. Putting in simple words, it can be defined as the task of resolving pronouns like *he, she, you, me, I, we, us, this, them, that, etc* to their matching entity in given context. Context here can be same sentence, paragraph or a larger piece of text.
Entity can be a named entity like name of person, place, organization, etc, or in more complex settings it can be noun phrase, verb phrase, sentence, or even whole paragraph.
Presence of 

Primary types: 

- Pronominal: This is the most common type where a referent is referred by a pronoun.
Example: "John found the love of his life" where 'his' refers to 'John'.
- Definite noun phrase: The antecedent is referred by a phrase of the form "\<the\> \<noun phrase\>".
Continued example: "The relationship did not last long", where 'The relationship' refers
to 'the love' in the preceding sentence.
- Quantifier/Ordinal: The anphor is a quantifier such as 'one' or an ordinal such as 'first'.
Continued Example: "He started a new one" where 'one' refers to 'The relationship'
(effectively meaning 'a relationship').

**Cataphora** : when anaphora *precedes* the antecedent 

For eg.,
Because ***she*** was going to the departmental store, ***Mary*** was asked to pick up the vegetables.

**What has already been done**

**BERT (Bidirectional Encoder Representations from Transformers)** is one of latest developments in the field of NLP by Google.
The [paper](https://arxiv.org/pdf/1810.04805.pdf) presents two model sizes for BERT:

- [BERT-Large, Uncased (Whole Word Masking)][BERT-Large-Uncased]: 24-layer, 1024-hidden, 16-heads, 340M parameters
- [BERT-Large, Cased (Whole Word Masking)][BERT-Large-Cased]: 24-layer, 1024-hidden, 16-heads, 340M parameters

**Problem statement**

**Solution approach**

**References**
[]https://nlp.stanford.edu/courses/cs224n/2003/fp/iqsayed/project_report.pdf
[BERT-Large-Uncased]:https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
[BERT-Large-Cased]:https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip
