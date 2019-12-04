---
layout: post
title:  "Anaphora resolution for binary entities using BERT"
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

**BERT (Bidirectional Encoder Representations from Transformers)** is one of latest developments in the field of NLP by Google.
The [paper](https://arxiv.org/pdf/1810.04805.pdf) presents two model sizes for BERT:

- [BERT-Large, Uncased (Whole Word Masking)][BERT_Large_Uncased]: 24-layer, 1024-hidden, 16-heads, 340M parameters
- [BERT-Large, Cased (Whole Word Masking)][BERT_Large_Cased]: 24-layer, 1024-hidden, 16-heads, 340M parameters

BERT is a method of training a general-purpose "language understanding" model on a large text corpus (like Wikipedia), and then using that model for downstream NLP tasks. BERT is the first unsupervised, deeply bidirectional system for pre-training NLP. Pre-trained models can be either context-free or contextual. Models like word2vec & GloVe are Context-free models. These models generate a single "word embedding" representation for each word in the vocabulary. But, based on the context in which a word is used it might have a different meaning & hence a different representation, i.e., more than one representations of same word should be possible, which is handled by Contextual models.

BERT is a contextual model & takes learnings from techniques like Semi-supervised Sequence Learning, Generative Pre-Training, ELMo, and ULMFit. Major enhancement comes with being *Deeply Bidirectional*, which means, while computing the representation of a word, it takes into account both left & right context in a "deep" manner.

**Problem statement**

Givent the document, anaphora, and two antecedents or precedents, resolve, which of the two antecedents/precedents does the anaphora refer to. 

**Solution approach**
Here, we try to use BERT to solve the above mentioned probelm. The problem is formulated as a binary classification problem with two antecedents/precedents being the two classes.

Features used:
- Anaphora
- First antecedent or precedent
- Second antecedent or precedent
- Preceeding n words from anaphora
- Preceeding n words from first antecedent or precedent
- Preceeding n words from second antecedent or precedent
- Head word for anaphora
- Head word for first antecedent or precedent
- Head word for second antecedent or precedent

We run forward propogation through BERT on our data and extract the output of last layer. This output is the embeddings that we use for creating our features. Once all the feature words/expression are extracted, we extract BERT embeddings for these features. These embeddings are then concatenated & used as features to train the Multi-layer perceptron model.

**Steps in detail**

To start with you must be having development & test sets. Development set will be used for training the model & test set will be used for final score.

1. Create input dataset for BERT by setting "create_bert_input=True"
This step will create data in format neeeded by BERT & save it to a text file.
```
from coreference import CoreferenceResolution
CoreferenceResolution.create_bert_input(development_data, 'dev')
```
2. input_<data_type>.txt file will be generated in temp folder of project
3. Use the input file to train BERT embeddings. 
4. Do the above steps for all the datasets (train/test)
5. If you already have BERT embeddings, set create_bert_input=False & run the script.
6. Once you have the embeddings, extract relevant features. Currently we extract above mentioned features from the text. This step has scope for improvements. 
```
development_data['feature_words'] = development_data.apply(CoreferenceResolution.extract_features, axis=1)
test_data['feature_words'] = test_data.apply(CoreferenceResolution.extract_features, axis=1)
```
7. Features that we have created are words/phrases, we need to convert them to numerical format using embeddings. So, extract embeddings for these features using:
```
feature_em_dev   = CoreferenceResolution.extract_bert_embedding_for_word(development_data, 'dev')
feature_em_test  = CoreferenceResolution.extract_bert_embedding_for_word(test_data, 'test')
```
8. Concatenate embeddings to create features
```
dev_emb_all = CoreferenceResolution.merge_all_features(development_emb, feature_em_dev)
test_emb_all = CoreferenceResolution.merge_all_features(test_emb, feature_em_test)
```
9. We can add additional features if want to use to our dataframe at this step.

10. Next we train the classifier using the above created features. 


**References**
- https://nlp.stanford.edu/courses/cs224n/2003/fp/iqsayed/project_report.pdf
- https://github.com/google-research/bert

[BERT_Large_Uncased]: https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
[BERT_Large_Cased]: https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip
