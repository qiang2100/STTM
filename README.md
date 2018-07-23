# STTM: A Library of Short Text Topic Modeling

Along with the emergence and popularity of social communications on the Inter-
net, topic discovery from short texts becomes fundamental to many applications that
require semantic understanding of textual content. As an emerging research direction,
short text topic modeling provides a new, complementary algorithmic methodology to
enrich topic modeling, especially targets to limited word co-occurrence information in
short texts. This paper introduces the first comprehensive open-source library, called
STTM, for use in JAVA that integrates the state-of-the-art models of short text topic
modeling and abundant functions for model inference and evaluation. The library is
designed to facilitate the development of new models in this research direction and
make comparisons between the new models and existing ones available. STTM is
open-sourced at https://github.com/qiang2100/STTM.



Copyright © 2018 Jipeng Qiang



Due to only very limited word co-occurrence information is available in short texts, how to extract topics from short texts remains a challenging research problem. Three major heuristic strategies have been adopted to deal with how to discover the
latent topics from short texts. One follows window-based strategy that two words or all words in one window are sampled from only one latent topic which is totally unsuited to long texts, but it can be suitable for short texts compared to the complex assumption that each text is modeled over a set of topics. Therefore, many models (DMM,BTM, WNTM and PYPM) for short texts were proposed based on thiswindow-based strategy. The second strategy aggregates short texts into long pseudo-texts before topic inference that can help improve word co-occurrence information. In this framework, STTM based self-aggregation strategy includes two algorithms, such as PTM, SATM and ETM. The last scheme directly leverages recent results by word embeddings that obtain vector representations for words trained on very large corpora to improve the word-topic mapping learned on a smaller corpus. Using word-embedding strategy, STTM includes GPU-DMM,GPU-PDMM and LF-DMM, which are the variations of DMM by incorporating the knowledge of word embeddings.
