# Attention-Subnetworks-based-Adversarial-Detection

Self-attention heads are characteristic of Transformer models and have been well studied for interpretability and pruning. In this work, we demonstrate an altogether different utility of attention heads, namely for adversarial detection. Specifically, we propose a method to construct input-specific attention subnetworks (IAS) from which we extract three features to discriminate between authentic and adversarial inputs. 

We propose three sets of features from IAS. The first feature, Fmask, is simply the attention mask that identifies if an attention head is retained or pruned in IAS. The 068
second feature, Fflip, characterizes the output of a “mutated” IAS obtained by toggling the mask used for attention heads in the middle layers of IAS. The third feature, Flw, characterizes the outputs of IAS as obtained layer-wise with a separately trained classification head for each layer. We train a classifier, called AdvNet, with these features as inputs to predict if an input is adversarial.

We report results on 10 NLU tasks from the GLUE benchmark (SST2, MRPC, RTE, SNLI, MultiNLI, QQP, QNLI) and elsewhere (Yelp, AG News, IMDb). For each of these tasks, we first 
create a benchmark of adversarial examples combining 11 attack methodologies. These include word-level attacks: deletion (Feng et al., 2018), antonyms, synonyms, embeddings(Mrkšic et al. ´ , 2016), order swap (Pruthi et al., 2019), PWWS (Ren et al., 2019), TextFooler (Jin et al., 2020) and character-level attacks: substitution, deletion, insertion, order swap (Gao et al., 2018). 

To further research in this field, we realease a benchmark that contains 5,686 adversarial examples across tasks and attack types. To the best of our knowledge, this dataset is the most extensive benchmark available on the considered task. For detailed information regarding this work, please visit our [paper](https://openreview.net/pdf?id=HcPfWDZZVuh). 

## Key Contributions

1. We demonstrate that our method is more accurate for larger models which are likely to have more spurious correlations and thus vulnerable to adversarial attack, and performs well even with modest training sets of adversarial examples.
2. Across all these tasks and attack types, we compare our adversarial detection technique against state-of the-art methods such as DISP (Zhou et al., 2019), NWS (Mozes et al., 2021), and FGWS (Mozes et al., 2021).
3. Our method establishes the best results in all tasks and attack types, with an average improvement of 10.8% over the best method for each task. Our detector achieves an accuracy of 80–90% across tasks suggesting effective defense against adversarial attacks.
4. We develop three sets feature sets, which together comprise of the input to the adversarial detection model.We compare different combinations of the features demonstrating that they are mutually informative and thus combining them all works best.
5. We also show that CutMix data augmentation (Yun et al., 2019) improves accuracy, demonstrating the first use of this method in adversarial detection in NLP task.


## People

This work has been developed by [Anirudh Sriram](https://github.com/anirudhs123), [Emil Biju](https://github.com/emilbiju),[Prof. Mitesh Khapra](https://www.cse.iitm.ac.in/~miteshk/) and [Prof. Pratyush Kumar](https://www.cse.iitm.ac.in/~pratyush/) from the Indian Institute of Technology, Madras. Ask us your questions at [anirudhsriram30799@gmail.com](mailto:anirudhsriram30799@gmail.com) or [emilbiju7@gmail.com](mailto:emilbiju7@gmail.com).
