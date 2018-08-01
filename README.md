# STTM: A Library of Short Text Topic Modeling
This is a Java based open-source library for short text topic modeling algorithms. The library is designed to facilitate the development of short text topic modeling algorithms and make comparisons between the new models and existing ones available. STTM is open-sourced at https://github.com/qiang2100/STTM.

STTM is maintained by [Jipeng Qiang](https://github.com/qiang2100/) (Yangzhou, China).

<center style="padding: 40px"><img width="70%" src="https://github.com/qiang2100/STTM/blob/master/Architecture.png" /></center>


Table of Contents
=================
  * [Full Documentation](http://opennmt.net/OpenNMT-py/)
  * [Algorithms](#algorithms)
  * [Datasets](#datasets)
  * [Evaluation](#evaluation)
  * [Quickstart](#quickstart)
  * [Citation](#citation)

## Algorithms

* Short text topic models: [DMM](http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf) in conference KDD2014, [BTM](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.4032&rep=rep1&type=pdf) in journal TKDE2016, [WNTM](https://arxiv.org/abs/1412.5404) in journal KAIS2018, [PTM](http://www.kdd.org/kdd2016/papers/files/rpp1190-zuoA.pdf) in conference KDD2016, [SATM](https://ijcai.org/Proceedings/15/Papers/321.pdf) in conference IJCAI2015,  [ETM](https://arxiv.org/abs/1609.08496) in conference PAKDD2017, [GPU-DMM](https://dl.acm.org/citation.cfm?id=2911499) in conference SIGIR2016, [GPU-PDMM](https://www.researchgate.net/profile/Aixin_Sun/publication/319277811_Enhancing_Topic_Modeling_for_Short_Texts_with_Auxiliary_Word_Embeddings/links/59daddef0f7e9b12b36d90b4/Enhancing-Topic-Modeling-for-Short-Texts-with-Auxiliary-Word-Embeddings.pdf) in journal TIS2017 and [LF-DMM](http://aclweb.org/anthology/Q15-1022) in journal TACL2015.

* Long text topic models: [LDA]() and [LF-LDA](http://aclweb.org/anthology/Q15-1022) in journal TACL2015.

## Datasets

We provided the following short text datasets for evaluation. The summary statistics and semantic topics of these datasets (SearchSnippets, StackOverflow and Biomedical) are described in the paper(https://arxiv.org/pdf/1701.00185.pdf).
* SearchSnippets: This dataset was selected from the results of web search
transaction using predefined phrases of 8 different domains.

*StackOverflow: This is the challenge data published in Kaggle.com. The
raw dataset consists 3,370,528 samples through July 31st, 2012 to August 14,
2012. Here, the dataset randomly select 20,000 question titles from 20
different tags.

*Biomedical. Biomedical use the challenge data published in BioASQ's official website.

## Evaluation

## Quickstart

## Citation

[STTM technical report](https://doi.org/10.18653/v1/P17-4012)

