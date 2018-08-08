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

## Datasets

We provided the following short text datasets for evaluation. The summary statistics and semantic topics of these datasets (SearchSnippets, StackOverflow and Biomedical) are described in the [paper](https://arxiv.org/pdf/1701.00185.pdf).
* SearchSnippets: This dataset was selected from the results of web search transaction using predefined phrases of 8 different domains.

* StackOverflow: This is the challenge data published in Kaggle.com. The raw dataset consists 3,370,528 samples through July 31st, 2012 to August 14, 2012. Here, the dataset randomly select 20,000 question titles from 20 different tags.

* Biomedical. Biomedical use the challenge data published in BioASQ's official website.

## Evaluation

* Topic coherence: Computing topic coherence, additional dataset (Wikipedia) as a single
meta-document is needed to score word pairs using term cooccurrence in the paper (Automatic Evaluation of Topic Coherence). Here, we calculate the pointwise mutual
information (PMI) of each word pair, estimated from the entire corpus of over one million English Wikipedia articles. Using a sliding window of 10-
words to identify co-occurrence, we computed the PMI of all a given word pair. The wikipedia can downloaded [Here](https://dumps.wikimedia.org/enwiki/20180720/). Then, we can transfer the dataset from html to text using the [code](https://github.com/qiang2100/STTM/blob/master/process_wiki.py) through executing "python process_wiki.py enwiki-latest-pages-articles.xml.bz2 wiki.en.text". Finally, due to the large size, we only choose a part of them.

* Cluster Evaluation (Purity and NMI): By choosing the maximum of topic probability for each text, we can get the cluster label for each text. Then, we can compare the cluster label and the golden label using metric Purity and NMI.

* Classification Evaluation: With topic modeling, we can represent each document with its topic distribution p(z|d). Hence, the quality of the topics can be assessed by the accuracy of text classification using topic-level representation, as an indirect evaluation. A better classification accuracy means the learned topics
are more discriminative and representative. Here, we employ a linear kernel Support Vector Machine (SVM) classifier in [LIBLINEAR](https://liblinear.bwaldvogel.de/) with default parameter settings. The classification accuracy is computed through fivefold cross-validation on both datasets.

## Quickstart

## Citation

[STTM technical report](https://arxiv.org/abs/1808.02215)

@article{qiang2018STTP,
  title={STTM: A Tool for Short Text Topic Modeling },
  author={Qiang, Jipeng and Li, Yun and Yuan, Yunhao and Liu, Wei and Wu, Xindong},
  journal={arXiv preprint arXiv:1808.02215},
  year={2018}
}

