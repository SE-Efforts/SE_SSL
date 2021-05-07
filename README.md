# FRUGAL: Unlocking SSL for Software Analytics

## Case Studies:

[1] [Static Warning Analysis](https://link.springer.com/article/10.1007/s10664-021-09948-6): Learning to recognize actionable static code warnings (is intrinsically easy)
- [Data](https://github.com/SE-Efforts/SE_SSL/tree/main/intrinsic_dimension/data): originated from [Wang et al.](https://www.researchgate.net/publication/328084908_Is_there_a_golden_feature_set_for_static_warning_identification_an_experimental_evaluation)'s ''Is there a "golden" feature set for static warning identification?: an experimental evaluation'' 


[2] [Issue Close Time](https://www.researchgate.net/publication/348588972_When_SIMPLE_is_better_than_complex_A_case_study_on_deep_learning_for_predicting_Bugzilla_issue_close_time): When SIMPLE is better than complex: A case study on deep learning for predicting Bugzilla issue close time
- [Data](https://github.com/mkris0714/Bug-Related-Activity-Logs): originated and used by [Lee et al.](https://ieeexplore.ieee.org/document/8955829) and [Mani et al.](https://dl.acm.org/doi/10.1145/3297001.3297023)

## Methodology: 
- [CLA/CLAFI+ML](https://github.com/lifove/CLAMI) JAVA implementation from [Nam et. al](https://dl.acm.org/doi/abs/10.1109/ASE.2015.56)'s CLAMI: defect prediction on unlabeled datasets.
- FRUGAL: FRUGAL finds the best combination of unsupervised learners = {CLA, CLA+ML, CLAFI+ML} and 𝐶 = {5% to 95% increments by 5%}. 

![](/CLA_4.png)



## How To Run: 
- ICT:

```cd issue_close_time; python clami.py get_CLAGRID```

- SWA: 

```cd intrinsic_dimension; python clami.py get_CLAGRID```
