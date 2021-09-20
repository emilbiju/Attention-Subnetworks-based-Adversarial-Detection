# Attention-Subnetworks-based-Adversarial-Detection

Self-attention heads are characteristic of Transformer models and have been well studied for interpretability and pruning. In this work, we demonstrate an altogether different utility of attention heads, namely for adversarial detection. Specifically, we propose a method to construct input-specific attention subnetworks (IAS) from which we extract three features to discriminate between authentic and adversarial inputs. 

We report results on 10 NLU tasks from the GLUE benchmark (SST2, MRPC, RTE, SNLI, MultiNLI, QQP, QNLI) and elsewhere (Yelp, AG News, IMDb). For each of these tasks, we first 
create a benchmark of adversarial examples combining 11 attack methodologies. These include word-level attacks: deletion (Feng et al., 2018), antonyms, synonyms, embeddings(Mrkšic et al. ´ , 2016), order swap (Pruthi et al.,2019), PWWS (Ren et al., 2019), TextFooler (Jin et al., 2020) and character-level attacks: substitution, deletion, insertion, order swap (Gao et al.,2018). 

To further research in this field,we realease a benchmark that contains 5,686 adversarial examples across tasks and attack types. To the best of our knowledge, this dataset is the most extensive benchmark available on the considered task. For detailed information regarding this work, please visit our [paper](https://openreview.net/pdf?id=HcPfWDZZVuh). 

<p align="center">
   <img src="../gh-pages/assets/images/gesture_sample.jpg" width=400 height=300>
</p>

## Key Contributions

1. A Gesture Path Decoding model that uses a multi-headed Transformer along with LSTM layers for coordinate sequence encoding and a character-level LSTM model for character sequence decoding.
2. A Contrastive Transliteration correction model that uses position-aware character embeddings to measure word proximities and correct spellings of transliterated words.
3. Two datasets of simulated word traces for supporting work on gesture typing for Indic language keyboards including low resource languages like Telugu and Kannada.
4. The accuracies of the proposed models vary from 70 to 89% for English-to-Indic decoding and 86-95% for Indic-to-Indic decoding across the 7 languages used for the study.

## People

This work has been developed by [Anirudh Sriram](https://github.com/anirudhs123), [Emil Biju](https://github.com/emilbiju),[Prof. Mitesh Khapra](https://www.cse.iitm.ac.in/~miteshk/) and [Prof. Pratyush Kumar](https://www.cse.iitm.ac.in/~pratyush/) from the Indian Institute of Technology, Madras. Ask us your questions at [anirudhsriram30799@gmail.com](mailto:anirudhsriram30799@gmail.com) or [emilbiju7@gmail.com](mailto:emilbiju7@gmail.com).
