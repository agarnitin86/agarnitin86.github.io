---
layout: post
title:  "Text classification using CNN : Example"
date:   2016-11-23 17:49:34
categories: jekyll update
---

Suppose we have two sentences for our classification task.

Sentence 1 : The camera quality is very good  
Sentence 2 : The battery life is good

Here we have two sentences of length 6 and 5 respectively.  
We also assume that their are two classes for our classification model, Positive and Negative.

### Step 1 : Building the vocabulary
Since the sentences are of different length, we pad our sentences with special `<PAD>` token.
So now we have our sentences modified as :  
Sentence 1 : the camera quality is very good  
Sentence 2 : the battery life is good `<PAD>`.
Now, both the sentences are of same length. We proceed to build the vocabulary index.  
Vocabulary index is a mapping of integer to each unique word in the corpus.
In our case, size of vocabulary index will be 9, since there are 9 unique tokens.
In tensorflow, tensorflow.contrib.learn.preprocessing.VocabularyProcessor is used for building the vocabulary.  

For our case, suppose the vocabulary index is :
`<PAD>` the camera quality is very good battery life  

Next, each sentence is converted into vector of integers.
Sentence 1 : [1, 2, 3, 4, 5, 6]
Sentence 2 : [1, 7, 8, 4, 6, 0]

I will try to explain the network in the same order as it is in code.  

1. Embedding Layer
2. Convolution Layer
3. Pooliing Layer
4. Dropout Layer
5. Output Layer

##Embedding Layer  

###Configurable parameters for embedding layer

* **batch_size** = 2 (Since we have only two sentences here)  
* **sequence length** = 6 (Max length of the senteces)  
* **num_classes**       = 2 (Number of output classes. Positive and negative in our case)  
* **vocabulary size (V)** = 9  

###Trainable parameters for embedding layer

* **Embedding vector size (E)** = 128 (Size of the embedding vector)  
* **Embedding matrix (W) of shape** = [V * E] = [9 * 128]

###Input
**shape of input (input_x)** = [batch_size, sequence_length] = [2, 6]  
**shape of input (input_y)** = [batch_size, num_classes] = [2, 2]  

###Working of embedding layer

As per the code in the [wildml blog][wildml_text_cnn]:  
{% highlight  python %}
self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
{% endhighlight %}

Input to the embedding layer is a tuple whose length is equal to the batch size, i.e, number of sentences in a batch.  
Each element of the tuple is an array of numbers, representing sentence in the form of indexes into vocabulary.  
For example :   
input_x = ( ## Figure)  

###Embedding Lookup
Statement from the [wildml blog][wildml_text_cnn]
{% highlight  python %}
self.embedded_chars = tf.nn.embedding_lookup(w, self.input_x)
{% endhighlight %}
This statement takes as input, input_x from the previous step.  
For each sentence, and for each word, it looks up (indexes) the embedding matrix (which is 9 * 128 in our case) (w) and returns the corresponding embedding vector for each word of each sentence.  
When i printed the variable `embedded_chars`, it is actually a list structured as follows :   

* embedded_chars is a list of size 1,  
* its 0th element is a list of size batch_size, i.e, number of sentences, (2 for this ex)  
* each element of this list is a list of size sequence_length, (6 for this ex)
* and, each element of this list is a list of size embedding vector (128 for this ex.)  

{% highlight  python %}
[
	[
		[  [0,1,.......127] ],
		[  [0,1,.......127] ],
		[  [0,1,.......127] ],
		[  [0,1,.......127] ],
		[  [0,1,.......127] ],
		[  [0,1,.......127] ],
	],
	[
		and so on.
	]
]
{% endhighlight %}

If this looks difficult, it can be understood with the following diagram,

###Figure

Here, the shape of tensor is `[batch_size, sequence_length, embedding_vector_length]`, i.e, `[2 * 9 * 128]`.   
Each element of the word vector is a real value. In the next satement of the blog, i.e,
{% highlight  python %}
self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
{% endhighlight %}
tensor is reshaped to `[batch_size, sequence_length, embedding_vector_length, 1]`,  
i.e, `[2 * 9 * 128 * 1]`. So that, each element is itself a list of size 1, instead of a real number.  
This completes the explanation of embedding layer

###Output of embedding layer : 

* Tensor of shape : `[batch_size, sequence_length, embedding_vector_length, 1]`, i.e, `[2 * 9 * 128 * 1]`.
Next layer in the model is the convolution layer.

##Convolution Layer

###Configurable parameters for convolution layer

* **filter_sizes** : 3, 4, 5
* **num_filters**: 128
* **Stride length** : 
* **Padding** :   
As oppossed to 2D filters in images, here in text classification we use 1D filters. We will be using filters of sizes 3,4,5. First we will go through the process of a single filter. Same will work for the other two sizes. Later we will see, how to merge the output from all the three filter sizes.

###Trainable parameters of convolution layer
* **Weight matrix (W)** : its shape is same as shape of filter.
* **Bias(b)** : Since there are 128 filters, there will be 128 bias values.

**Now, what is the shape of filter ?**
Lets take filter of size, `filter_size = 3`. In this case, shape of the filter is `[filter_height, fiter_width]` or `[filter_size, embedding_vector_len]` or `[3 * 128]`. This means that the filter sees 3 words (or embedding vectors) at  a time, and based on the stride keeps seeing the 3 word vectors.  
We have only one input channel, therefore, num_input_channels = 1  
Number of output channels or Number of filters is 128.
Therefore, shape of the filter tensor becomes : `[filter_size, embedding_vector_len, num_input_channels, num_filters] = [ 3 * 128 * 1 * 128]`
The following figure shows the mapping between input matrix and filter of size 3 : 

###Figure

I will explain how to apply convolution filter on the input in the next step, but, before that let us see how to compute the output shape for this filter of size 3.  

**Now, what is the shape of output filter ?**  
For 1D case, as in text,  
`out_rows = (W1- F + 2 * P) / S + 1`  
where,  
W1 = sequence_length (Number of input words) = 6 (in our case)  
F = height of filter or filter size = 3 (in our case)  
P = padding = 0 (in our case)  
Stride = 1 (in our case)  

Therefore,  
`out_rows = (6 - 3 + 2 * 0) / 1 + 1 = 4`  
Since the filter moves only in 1D, shape of output = 4 * 1, for a single sentence with a fiter_size = 3 and for one output_channel.  
Now, since we have batch of sentences and num_filters = 128, we get 4 * 1 outputs for each combination. Therefore, shape of output tensor for fiter_size (3) is  
`[batch_size * 4 * 1 * num_filters] = [2 * 4 * 1 * 128]`  

Similarly, we can compute the output shapes for filters of size 4 and 5 as follows :   
**For  filter_size (4)** `(6 - 4 + 2 * 0) / 1 + 1 = 3` or `[2 * 3 * 1 * 128]`  
**For  filter_size (5)** `(6 - 5 + 2 * 0) / 1 + 1 = 2` or `[2 * 2 * 1 * 128]`  
Now let us see a simple calculation showing how filter is applied to input and what is the role of weight matrix and bias.  

###Working of convolution layer

####Input
* Output of the expanded embedding lookup step. Its shape is  
`[batch_size, sequence_length, embedding_vector_length, 1]`, i.e, `[2 * 9 * 128 * 1]`
####Filter 

You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve --watch`, which launches a web server and auto-regenerates your site when a file is updated.

To add new posts, simply add a file in the `_posts` directory that follows the convention `YYYY-MM-DD-name-of-post.ext` and includes the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll’s dedicated Help repository][jekyll-help].

[wildml_text_cnn]:     http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-help]: https://github.com/jekyll/jekyll-help
