# STTM: A Library of Short Text Topic Modeling
This is a Java based open-source library for short text topic modeling algorithms. The library is designed to facilitate the development of short text topic modeling algorithms and make comparisons between the new models and existing ones available. STTM is open-sourced at [Here](https://github.com/qiang2100/STTM).

STTM is maintained by [Jipeng Qiang](https://github.com/qiang2100/) (Yangzhou, China).

<center style="padding: 40px"><img width="70%" src="https://github.com/qiang2100/STTM/blob/master/Architecture.png" /></center>


Table of Contents
=================
  * [Full Documentation](https://arxiv.org/abs/1808.02215)
  * [Algorithms](#algorithms)
  * [Datasets](#datasets)
  * [Evaluation](#evaluation)
  * [Quickstart](#quickstart)
  * [Citation](#citation)
  

## Algorithms

* Short text topic models: Dirichlet Multinomial Mixture ([DMM](http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf)) in conference KDD2014, Biterm Topic Model ([BTM](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.4032&rep=rep1&type=pdf)) in journal TKDE2016, Word Network Topic Model ([WNTM](https://arxiv.org/abs/1412.5404) ) in journal KAIS2018, Pseudo-Document-Based Topic Model ([PTM](http://www.kdd.org/kdd2016/papers/files/rpp1190-zuoA.pdf)) in conference KDD2016, Self-Aggregation-Based Topic Model ([SATM](https://ijcai.org/Proceedings/15/Papers/321.pdf)) in conference IJCAI2015, ([ETM](https://arxiv.org/abs/1609.08496)) in conference PAKDD2017, Generalized P´olya Urn (GPU) based Dirichlet Multinomial Mixturemodel ([GPU-DMM](https://dl.acm.org/citation.cfm?id=2911499)) in conference SIGIR2016, Generalized P´olya Urn (GPU) based Poisson-based Dirichlet Multinomial Mixturemodel ([GPU-PDMM](https://www.researchgate.net/profile/Aixin_Sun/publication/319277811_Enhancing_Topic_Modeling_for_Short_Texts_with_Auxiliary_Word_Embeddings/links/59daddef0f7e9b12b36d90b4/Enhancing-Topic-Modeling-for-Short-Texts-with-Auxiliary-Word-Embeddings.pdf)) in journal TIS2017 and Latent Feature Model with DMM ([LF-DMM](http://aclweb.org/anthology/Q15-1022)) in journal TACL2015. 

* Long text topic models: Latent Dirichlet Allocation ([LDA]()) and Latent Feature Model with LDA ([LF-LDA](http://aclweb.org/anthology/Q15-1022)) in journal TACL2015.

Here, DMM is from jLDADMM package (https://github.com/datquocnguyen/jLDADMM). LF-DMM and LF-LDA are from https://github.com/datquocnguyen/LFTM.
## Datasets

We provided the following six short text datasets for evaluation. The summary statistics and semantic topics of these datasets (SearchSnippets, StackOverflow and Biomedical) are described in the [paper](https://arxiv.org/pdf/1701.00185.pdf). The statistics of the two datasets (Tweet and GoogleNews) are described in the "DMM" paper.
* SearchSnippets: This dataset was selected from the results of web search transaction using predefined phrases of 8 different domains.

* StackOverflow: This is the challenge data published in Kaggle.com. The raw dataset consists 3,370,528 samples through July 31st, 2012 to August 14, 2012. Here, the dataset randomly select 20,000 question titles from 20 different tags.

* Biomedical. Biomedical use the challenge data published in BioASQ's official website.

* Tweet: In the 2011 and 2012 microblog tracks at Text REtrieval Conference (TREC)2 , totally 109 queries were used. Using
a standard polling strategy, the NIST assessors evaluated the tweets submitted for each query by the participants into:
spam, not relevant, relevant, and highly-relevant. We regard the queries as clusters and the highly-relevant tweets of each
query as documents in each cluster. After removing the queries with none highly-relevant tweets, we constructed a
dataset with 89 clusters and totally 2,472 tweets.

* GoogleNews: In the Google News, the news articles are grouped into clusters (stories) automatically. We took a snapshot of the Google News on November 27, 2013, and crawled the titles and snippets of 11,109 news articles belonging to 152 clusters.

* Pascal_Flickr: The Pascal Captions dataset are sets of captions solicited from Mechanical Turkers for photographs from Flickr and from the Pattern Analysis, Statistical Modeling, and Computational Learning (PASCAL) Visual Object Classes
Challenge (Everingham et al., 2010). PAS includes twenty categories of images and 4834 captions. Each category has fifty images with approximately five captions for each image. We use the category as the gold standard cluster.

## Evaluation

* Topic coherence: Computing topic coherence, additional dataset (Wikipedia) as a single
meta-document is needed to score word pairs using term cooccurrence in the paper (Automatic Evaluation of Topic Coherence). Here, we calculate the pointwise mutual
information (PMI) of each word pair, estimated from the entire corpus of over one million English Wikipedia articles. Using a sliding window of 10-
words to identify co-occurrence, we computed the PMI of all a given word pair. The wikipedia can downloaded [Here](https://dumps.wikimedia.org/enwiki/20180720/). Then, we can transfer the dataset from html to text using the [code](https://github.com/qiang2100/STTM/blob/master/process_wiki.py) through executing "python process_wiki.py enwiki-latest-pages-articles.xml.bz2 wiki.en.text". Finally, due to the large size, we only choose a part of them.

* Cluster Evaluation (Purity and NMI): By choosing the maximum of topic probability for each text, we can get the cluster label for each text. Then, we can compare the cluster label and the golden label using metric Purity and NMI.

* Classification Evaluation: With topic modeling, we can represent each document with its topic distribution p(z|d). Hence, the quality of the topics can be assessed by the accuracy of text classification using topic-level representation, as an indirect evaluation. A better classification accuracy means the learned topics
are more discriminative and representative. Here, we employ a linear kernel Support Vector Machine (SVM) classifier in [LIBLINEAR](https://liblinear.bwaldvogel.de/) with default parameter settings. The classification accuracy is computed through fivefold cross-validation on both datasets.

## Quickstart

### Step 1: Infer latent topics from the corpus

Users can find the pre-compiled file `STTM.jar` and source codes in folders `src`, respectively. The users can recompile the source codes by Exclipse or IDEA.

**File format of input corpus:**  Similar to file `corpus.txt`  in the `dataset` folder, STTM assumes that each line in the input corpus represents a document. Here, a document is a sequence of words/tokens separated by white space characters. The users should preprocess the input corpus before training the short text topic models, for example: down-casing, removing non-alphabetic characters and stop-words, removing words shorter than 3 characters and words appearing less than a certain times.  

**Now, we can train the algorithms in STTM tool by executing:**

	$ java [-Xmx1G] -jar jar/STTM.jar –model <LDA or BTM or PTM or SATM or DMM or WATM> -corpus <Input_corpus_file_path> [-ntopics <int>] [-alpha <double>] [-beta <double>] [-niters <int>] [-twords <int>] [-name <String>] [-sstep <int>]

!!! note "Note"
    If users train these models based word embeddings, users need to download the Pre-trained word embeddings. In the package, the code is based on [Global Vectors](https://nlp.stanford.edu/projects/glove/).

    $ java [-Xmx1G] -jar jar/STTM.jar –model <GPUDMM or GPU-PDMM or LFDMM or LFLDA> -corpus <Input_corpus_file_path> -vectors <Input_Word2vec_file_Path> [-ntopics <int>] [-alpha <double>] [-beta <double>] [-niters <int>] [-twords <int>] [-name <String>] [-sstep <int>]

where parameters in [ ] are optional.

`-model`: Specify the topic model LDA or DMM

`-corpus`: Specify the path to the input corpus file.

`-vectors`: Specify the path to the word2vec file.

`-ntopics <int>`: Specify the number of topics. The default value is 20.

`-alpha <double>`: Specify the hyper-parameter `alpha`. Following [6, 8], the default  `alpha` value is 0.1.

`-beta <double>`: Specify the hyper-parameter `beta`. The default `beta` value is 0.01 which is a common setting in  literature. 

`-niters <int>`: Specify the number of Gibbs sampling iterations. The default value is 2000.

`-twords <int>`: Specify the number of the most probable topical words. The default value is 20.

`-name <String>`: Specify a name to the topic modeling experiment. The default value is `model`.

`-sstep <int>`: Specify a step to save the sampling outputs. The default value is 0 (i.e. only saving the output from the last sample).

**Examples:**

	$ java -jar jar/STTM.jar -model BTM -corpus dataset/corpus.txt -name corpusBTM

The output files are saved in the "results" folder containing `corpusBTM.theta`, `corpusBTM.phi`, `corpusBTM.topWords`, `corpusBTM.topicAssignments` and `corpusBTM.paras` referring to the document-to-topic distributions, topic-to-word distributions, top topical words, topic assignments and model parameters, respectively. 

### Step 2: Evaluation the inferring models using Clustering, Coherence or Classification


#### For clustering, we treat each topic as a cluster, and we assign every document the topic with the highest probability given the document. To get the Purity and NMI clustering scores, we perform:

	$ java –jar jar/STTM.jar –model ClusteringEval –label <Golden_label_file_path> -dir <Directory_path> -prob <Document-topic-prob/Suffix>

`–label`: Specify the path to the ground truth label file. Each line in this label file contains the golden label of the corresponding document in the input corpus. See files `corpus.LABEL` and `corpus.txt` in the `dataset` folder.

`-dir`: Specify the path to the directory containing document-to-topic distribution files.

`-prob`: Specify a document-to-topic distribution file OR a group of document-to-topic distribution files in the specified directory.

**Examples:**

	$ java -jar jar/STMM.jar -model ClusteringEval -label dataset/corpus.LABEL -dir results -prob corpusBTM.theta

#### For coherence, we perform:
	$ java –jar jar/STTM.jar –model CoherenceEval –label <Wikipedia_file_path> -dir <Directory_path> -topWords <Document-TopWord/Suffix>

`–label`: Specify the path to the Wikipedia file. How to obtain Wikipedia file, please check the above section "Evaluation".
`–topword`: Specify the path to the top words file, e.g., `corpusBTM.topWords`.

#### For classification, we perform:
	$ java –jar jar/STTM.jar –model ClassificationEval –label <Golden_label_file_path> -dir <Directory_path> -prob <Document-topic-prob/Suffix>


The above commands will produce the clustering scores for the ".theta" file for each model  in the `dataset` folder, separately. The following command

	$ java -jar jar/STTM.jar -model ClusteringEval -label dataset/corpus.LABEL -dir results -prob theta

will produce the clustering scores for all document-to-topic distribution files with their names ending in `theta`.

Similarly, we perform
    $ java -jar jar/STTM.jar -model CoherenceEval -label dataset/Wikipedia -dir results -topWords topWords
    $ java -jar jar/STTM.jar -model ClassificationEval -label dataset/corpus.LABEL -dir results -prob theta


### Step 3: Topic inference on new/unseen corpus

To infer topics on a new/unseen corpus using a pre-trained LDA or DMM or LFDMM or LFLDA topic model, we perform:

`$ java -jar jar/STTM.jar -model <LDA_inf or DMM_inf or LFLDA_inf or LFDMM_inf> -paras <Hyperparameter_file_path> -corpus <Unseen_corpus_file_path> [-niters <int>] [-twords <int>] [-name <String>] [-sstep <int>]`

* `-paras`: Specify the path to the hyper-parameter file produced by the pre-trained LDA/DMM topic model.

<b>Examples:</b>

`$ java -jar jar/STTM.jar -model LDAinf -paras results/corpusLDA.paras -corpus dataset/unseenTest.txt -niters 100 -name LDAinf`




## Citation

[STTM technical report](https://arxiv.org/abs/1808.02215)

```
@article{qiang2018STTP,
  title =  {STTM: A Tool for Short Text Topic Modeling },
  author = {Qiang, Jipeng and 
            Li, Yun and 
            Yuan, Yunhao and 
            Liu, Wei and 
            Wu, Xindong},
  journal = {arXiv preprint arXiv:1808.02215},
  year  =  {2018}
}
```




