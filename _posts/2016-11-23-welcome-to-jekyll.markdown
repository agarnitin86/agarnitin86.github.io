---
layout: post
title:  "CNN's with example"
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
Sentence 1 : The camera quality is very good  
Sentence 2 : The battery life is good `<PAD>`.
Now, both the sentences are of same length. We proceed to build the vocabulary index.  
Vocabulary index is a mapping of integer to each unique word in the corpus.
In our case, size of vocabulary index will be 9, since there are 9 unique tokens.
In tensorflow, tensorflow.contrib.learn.preprocessing.VocabularyProcessor is used for building the vocabulary.  
Next, each sentence is converted into vector of integers.
For our case, suppose the vocabulary index is :


### Values of the variables for our example

###Embedding Layer 

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

[jekyll]:      http://jekyllrb.com
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-help]: https://github.com/jekyll/jekyll-help
