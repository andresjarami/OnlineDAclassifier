# Experiments

We evaluate our online classifier through two experiments to determine their performance and
their robustness to mislabeled data. In these experiments, we use five publicly available
datasets that contain sEMG data of hand gestures and three feature sets. We make publicly available the code (shown
in the supplementary material) of this empirical evaluation to easily compare our approach with future approaches in this field.

First: Import the library developed to visualize the results


```python
import Experiments.analysis_experiments as analysis_experiments

```

## Experiment 1

To evaluate the performance of our approach, we define five DA classifiers (LDA/QDA): initial (baseline), online
classifier using labels and pseudo-labels through our soft-labeling, Nigam's soft-labeling,
and thresholding techniques.

The **initial classifier** is a DA classifier trained over a dataset $\mathcal{I}$ that has one gesture per class.
The **online classifier using labels and pseudo-labels** is a DA classifier initially trained over the set $\mathcal{I}$
and sequentially updated *with labeled gestures* and *with pseudo-labeled gestures* by our soft-labeling technique
, respectively.
The **Nigam-based classifier** is the online classifier updated with pseudo-labeled gestures using Nigam's soft-labeling.
In this technique, a gesture is pseudo-labeled using the conditional posterior probability $p_{(c,t)}$
multiplied by a parameter $\lambda$ that decreases the contribution of this probability to minimize the error of
gestures incorrectly pseudo-labeled. The parameter $\lambda$ is in the interval $[0,1]$.
The **thresholding-based classifier** is also the online classifier updated with pseudo-labeled gestures using the
thresholding technique that is commonly used in self-training learning. In this technique, a gesture is labeled based
on the probability $p_{(c,t)}$. If this probability is greater than a threshold $\tau$, then this pseudo-labeled gesture
is used to update the classifier.

In this experiment, we determine the best parameters $\lambda$ and $\tau$ for each dataset and feature set from the set
 $\{0,0.1,\cdots, 1\}$ using grid search optimization.


```python
analysis_experiments.experiment1(best_parameters_Nigam_thresholding=True)

```

    
    FEATURE 1: Nina5
    Best parameter λ (for Nigam-based classifier) is 0.6. Accuracy Difference (wrt initial classifier)=1.9
    Best parameter λ (for Nigam-based classifier) is 1.0. Accuracy Difference (wrt initial classifier)=12.74
    Best parameter τ (for thresholding-based classifier) is 0.4. Accuracy Difference (wrt initial classifier)=2.72
    Best parameter τ (for thresholding-based classifier) is 0.0. Accuracy Difference (wrt initial classifier)=8.29
    FEATURE 2: Nina5
    Best parameter λ (for Nigam-based classifier) is 1.0. Accuracy Difference (wrt initial classifier)=5.11
    Best parameter λ (for Nigam-based classifier) is 1.0. Accuracy Difference (wrt initial classifier)=-4.06
    Best parameter τ (for thresholding-based classifier) is 0.3. Accuracy Difference (wrt initial classifier)=6.05
    Best parameter τ (for thresholding-based classifier) is 0.3. Accuracy Difference (wrt initial classifier)=-13.12
    FEATURE 3: Nina5
    Best parameter λ (for Nigam-based classifier) is 1.0. Accuracy Difference (wrt initial classifier)=5.01
    Best parameter λ (for Nigam-based classifier) is 1.0. Accuracy Difference (wrt initial classifier)=-3.28
    Best parameter τ (for thresholding-based classifier) is 0.3. Accuracy Difference (wrt initial classifier)=5.83
    Best parameter τ (for thresholding-based classifier) is 0.4. Accuracy Difference (wrt initial classifier)=-9.26
    
    FEATURE 1: Capgmyo_dbb
    Best parameter λ (for Nigam-based classifier) is 1.0. Accuracy Difference (wrt initial classifier)=5.65
    Best parameter λ (for Nigam-based classifier) is 1.0. Accuracy Difference (wrt initial classifier)=5.13
    Best parameter τ (for thresholding-based classifier) is 0.5. Accuracy Difference (wrt initial classifier)=5.56
    Best parameter τ (for thresholding-based classifier) is 0.0. Accuracy Difference (wrt initial classifier)=2.78
    FEATURE 2: Capgmyo_dbb
    Best parameter λ (for Nigam-based classifier) is 1.0. Accuracy Difference (wrt initial classifier)=5.94
    Best parameter λ (for Nigam-based classifier) is 1.0. Accuracy Difference (wrt initial classifier)=-4.12
    Best parameter τ (for thresholding-based classifier) is 0.5. Accuracy Difference (wrt initial classifier)=5.82
    Best parameter τ (for thresholding-based classifier) is 0.5. Accuracy Difference (wrt initial classifier)=-5.46
    FEATURE 3: Capgmyo_dbb
    Best parameter λ (for Nigam-based classifier) is 1.0. Accuracy Difference (wrt initial classifier)=6.38
    Best parameter λ (for Nigam-based classifier) is 0.8. Accuracy Difference (wrt initial classifier)=0.67
    Best parameter τ (for thresholding-based classifier) is 0.0. Accuracy Difference (wrt initial classifier)=6.04
    Best parameter τ (for thresholding-based classifier) is 0.7. Accuracy Difference (wrt initial classifier)=-0.46
    
    FEATURE 1: Cote
    Best parameter λ (for Nigam-based classifier) is 0.7. Accuracy Difference (wrt initial classifier)=1.4
    Best parameter λ (for Nigam-based classifier) is 1.0. Accuracy Difference (wrt initial classifier)=4.77
    Best parameter τ (for thresholding-based classifier) is 0.7. Accuracy Difference (wrt initial classifier)=1.47
    Best parameter τ (for thresholding-based classifier) is 0.5. Accuracy Difference (wrt initial classifier)=4.64
    FEATURE 2: Cote
    Best parameter λ (for Nigam-based classifier) is 0.7. Accuracy Difference (wrt initial classifier)=2.27
    Best parameter λ (for Nigam-based classifier) is 1.0. Accuracy Difference (wrt initial classifier)=8.25
    Best parameter τ (for thresholding-based classifier) is 0.0. Accuracy Difference (wrt initial classifier)=2.21
    Best parameter τ (for thresholding-based classifier) is 0.4. Accuracy Difference (wrt initial classifier)=6.14
    FEATURE 3: Cote
    Best parameter λ (for Nigam-based classifier) is 1.0. Accuracy Difference (wrt initial classifier)=1.92
    Best parameter λ (for Nigam-based classifier) is 1.0. Accuracy Difference (wrt initial classifier)=4.82
    Best parameter τ (for thresholding-based classifier) is 0.7. Accuracy Difference (wrt initial classifier)=1.82
    Best parameter τ (for thresholding-based classifier) is 0.5. Accuracy Difference (wrt initial classifier)=4.85
    
    FEATURE 1: LongTerm3DC
    Best parameter λ (for Nigam-based classifier) is 1.0. Accuracy Difference (wrt initial classifier)=0.93
    Best parameter λ (for Nigam-based classifier) is 1.0. Accuracy Difference (wrt initial classifier)=2.59
    Best parameter τ (for thresholding-based classifier) is 0.8. Accuracy Difference (wrt initial classifier)=0.74
    Best parameter τ (for thresholding-based classifier) is 0.4. Accuracy Difference (wrt initial classifier)=0.62
    FEATURE 2: LongTerm3DC
    Best parameter λ (for Nigam-based classifier) is 1.0. Accuracy Difference (wrt initial classifier)=1.85
    Best parameter λ (for Nigam-based classifier) is 1.0. Accuracy Difference (wrt initial classifier)=-2.02
    Best parameter τ (for thresholding-based classifier) is 0.6. Accuracy Difference (wrt initial classifier)=1.16
    Best parameter τ (for thresholding-based classifier) is 0.3. Accuracy Difference (wrt initial classifier)=-4.62
    FEATURE 3: LongTerm3DC
    Best parameter λ (for Nigam-based classifier) is 1.0. Accuracy Difference (wrt initial classifier)=1.81
    Best parameter λ (for Nigam-based classifier) is 0.1. Accuracy Difference (wrt initial classifier)=-2.29
    Best parameter τ (for thresholding-based classifier) is 0.6. Accuracy Difference (wrt initial classifier)=1.21
    Best parameter τ (for thresholding-based classifier) is 0.3. Accuracy Difference (wrt initial classifier)=-3.78
    
    FEATURE 1: EPN_120
    Best parameter λ (for Nigam-based classifier) is 1.0. Accuracy Difference (wrt initial classifier)=4.96
    Best parameter λ (for Nigam-based classifier) is 1.0. Accuracy Difference (wrt initial classifier)=14.48
    Best parameter τ (for thresholding-based classifier) is 0.5. Accuracy Difference (wrt initial classifier)=5.78
    Best parameter τ (for thresholding-based classifier) is 0.0. Accuracy Difference (wrt initial classifier)=11.24
    FEATURE 2: EPN_120
    Best parameter λ (for Nigam-based classifier) is 1.0. Accuracy Difference (wrt initial classifier)=14.81
    Best parameter λ (for Nigam-based classifier) is 1.0. Accuracy Difference (wrt initial classifier)=11.59
    Best parameter τ (for thresholding-based classifier) is 0.4. Accuracy Difference (wrt initial classifier)=14.73
    Best parameter τ (for thresholding-based classifier) is 0.3. Accuracy Difference (wrt initial classifier)=4.84
    FEATURE 3: EPN_120
    Best parameter λ (for Nigam-based classifier) is 1.0. Accuracy Difference (wrt initial classifier)=13.17
    Best parameter λ (for Nigam-based classifier) is 1.0. Accuracy Difference (wrt initial classifier)=13.74
    Best parameter τ (for thresholding-based classifier) is 0.3. Accuracy Difference (wrt initial classifier)=13.22
    Best parameter τ (for thresholding-based classifier) is 0.0. Accuracy Difference (wrt initial classifier)=7.33
    

For the five DA classifiers, the following figure shows the average classification accuracy of the users in the five
datasets using the three feature sets described above.
To determine if the accuracy differences between the methods tested are statistically significant, we use the 2-tailed
Wilcoxon signed ranks test at $p_{value}<0.5$.
As we excepted, the accuracy of the online classifier using labels is higher than the accuracy of the other classifiers
that use pseudo-labeled gestures. Note that this accuracy is equal to the accuracy of a DA classifier trained with all
data (the initial set $\mathcal{I}$ and all labeled gestures) in full batch fashion, as we established in Theorem 1.
The accuracies of the online classifier using labels and pseudo-labels are higher than the accuracy of the initial
classifier, so the updating proposed by us improves the performance of a DA classifier trained with few samples (one gesture per class).
In contrast, the Nigam-based and thresholding-based classifiers perform worse than the initial classifier when the
DA classifier is QDA as we can see, for example, in NinaPro5 and Long-Term 3DC using the feature sets FS2 and FS3, and
in Capgmyo\_dbb using the feature set FS2.



```python
analysis_experiments.experiment1(graph_acc=True)
```

    
    ANALYSIS WILCOXON (CONFIDENCE LEVEL 95%) shows the accuracy of two classifiers that come from the same distribution (Nina5): 
    
    FEATURE 1: Nina5
    p value 0.970219757029658 , LDA_oursusing pseudo-labels and LDA_ours_threshold_0.4 and comes from the same distribution
    
    FEATURE 2: Nina5
    p value 0.2958775226696384 , LDA_oursusing pseudo-labels and LDA_ours_threshold_0.3 and comes from the same distribution
    p value 0.7651984444971875 , QDA_oursusing pseudo-labels and QDA_weak and comes from the same distribution
    p value = 0.7651984444971875 , QDA_initial_classifier and QDA_ours_soft_labels come from the same distribution
    p value = 0.7089053094753638 , QDA_initial_classifier and QDA_ours_probs_1.0 come from the same distribution
    p value 0.11688763780953301 , QDA_oursusing pseudo-labels and QDA_ours_probs_1.0 and comes from the same distribution
    
    FEATURE 3: Nina5
    p value 0.13535690010210896 , LDA_oursusing pseudo-labels and LDA_ours_threshold_0.3 and comes from the same distribution
    
    ANALYSIS WILCOXON (CONFIDENCE LEVEL 95%) shows the accuracy of two classifiers that come from the same distribution (Capgmyo_dbb): 
    
    FEATURE 1: Capgmyo_dbb
    p value 0.506996170454357 , LDA_oursusing pseudo-labels and LDA_ours_probs_1.0 and comes from the same distribution
    p value 0.9412383091320182 , LDA_oursusing pseudo-labels and LDA_ours_threshold_0.5 and comes from the same distribution
    
    FEATURE 2: Capgmyo_dbb
    p value 0.9061928202383859 , LDA_oursusing pseudo-labels and LDA_ours_probs_1.0 and comes from the same distribution
    p value 0.8268342101798088 , LDA_oursusing pseudo-labels and LDA_ours_threshold_0.5 and comes from the same distribution
    
    FEATURE 3: Capgmyo_dbb
    p value 0.37047409261896025 , LDA_oursusing pseudo-labels and LDA_ours_threshold_0.0 and comes from the same distribution
    p value = 0.31499483027906106 , QDA_initial_classifier and QDA_ours_probs_0.8 come from the same distribution
    p value = 0.675460813735282 , QDA_initial_classifier and QDA_ours_threshold_0.7 come from the same distribution
    
    ANALYSIS WILCOXON (CONFIDENCE LEVEL 95%) shows the accuracy of two classifiers that come from the same distribution (Cote): 
    
    FEATURE 1: Cote
    p value 0.132861236687201 , LDA_oursusing pseudo-labels and LDA_ours_labels and comes from the same distribution
    p value 0.6575663850187541 , QDA_oursusing pseudo-labels and QDA_ours_probs_1.0 and comes from the same distribution
    p value 0.6732787363156975 , QDA_oursusing pseudo-labels and QDA_ours_threshold_0.5 and comes from the same distribution
    p value 0.3744792736563748 , QDA_oursusing pseudo-labels and QDA_ours_labels and comes from the same distribution
    
    FEATURE 2: Cote
    p value 0.3359328511803926 , LDA_oursusing pseudo-labels and LDA_ours_probs_0.7 and comes from the same distribution
    p value 0.6435165948165775 , LDA_oursusing pseudo-labels and LDA_ours_threshold_0.0 and comes from the same distribution
    p value 0.4366330161611942 , QDA_oursusing pseudo-labels and QDA_ours_labels and comes from the same distribution
    
    FEATURE 3: Cote
    p value 0.32814280480777647 , LDA_oursusing pseudo-labels and LDA_ours_probs_1.0 and comes from the same distribution
    p value 0.6829616948348547 , LDA_oursusing pseudo-labels and LDA_ours_threshold_0.7 and comes from the same distribution
    p value 0.11281874216705877 , LDA_oursusing pseudo-labels and LDA_ours_labels and comes from the same distribution
    
    ANALYSIS WILCOXON (CONFIDENCE LEVEL 95%) shows the accuracy of two classifiers that come from the same distribution (LongTerm3DC): 
    
    FEATURE 1: LongTerm3DC
    p value 0.16434084243582547 , LDA_oursusing pseudo-labels and LDA_ours_probs_1.0 and comes from the same distribution
    p value = 0.13176969837471053 , QDA_initial_classifier and QDA_ours_threshold_0.4 come from the same distribution
    
    FEATURE 2: LongTerm3DC
    
    FEATURE 3: LongTerm3DC
    p value 0.2540662583957104 , LDA_oursusing pseudo-labels and LDA_ours_probs_1.0 and comes from the same distribution
    
    ANALYSIS WILCOXON (CONFIDENCE LEVEL 95%) shows the accuracy of two classifiers that come from the same distribution (EPN_120): 
    
    FEATURE 1: EPN_120
    p value 0.05464694846814326 , LDA_oursusing pseudo-labels and LDA_ours_probs_1.0 and comes from the same distribution
    
    FEATURE 2: EPN_120
    
    FEATURE 3: EPN_120
    


![png](output_6_1.png)


We also perform the analysis of time of the batch classifier using labels and our online classifier using labels and pseudo-labels.


```python
analysis_experiments.experiment1(analysis_time=True)

```

    
    FEATURE 1: Nina5
    time[s]: LDA_batch 439.06 ± 27.07
    time[s]: LDA_ours_soft_labels 0.99 ± 0.15
    time[s]: LDA_ours_labels 0.69 ± 0.04
    time[s]: QDA_batch 437.53 ± 27.07
    time[s]: QDA_ours_soft_labels 0.42 ± 0.19
    time[s]: QDA_ours_labels 0.08 ± 0.0
    
    FEATURE 2: Nina5
    time[s]: LDA_batch 524.69 ± 40.06
    time[s]: LDA_ours_soft_labels 1.25 ± 0.19
    time[s]: LDA_ours_labels 0.9 ± 0.04
    time[s]: QDA_batch 523.07 ± 40.04
    time[s]: QDA_ours_soft_labels 0.46 ± 0.17
    time[s]: QDA_ours_labels 0.09 ± 0.01
    
    FEATURE 3: Nina5
    time[s]: LDA_batch 523.24 ± 43.38
    time[s]: LDA_ours_soft_labels 1.23 ± 0.22
    time[s]: LDA_ours_labels 0.89 ± 0.05
    time[s]: QDA_batch 521.6 ± 43.33
    time[s]: QDA_ours_soft_labels 0.44 ± 0.19
    time[s]: QDA_ours_labels 0.1 ± 0.01
    
    FEATURE 1: Capgmyo_dbb
    time[s]: LDA_batch 77.21 ± 3.39
    time[s]: LDA_ours_soft_labels 0.79 ± 0.07
    time[s]: LDA_ours_labels 0.68 ± 0.06
    time[s]: QDA_batch 75.8 ± 3.35
    time[s]: QDA_ours_soft_labels 0.19 ± 0.07
    time[s]: QDA_ours_labels 0.08 ± 0.01
    
    FEATURE 2: Capgmyo_dbb
    time[s]: LDA_batch 83.66 ± 4.33
    time[s]: LDA_ours_soft_labels 0.83 ± 0.09
    time[s]: LDA_ours_labels 0.69 ± 0.07
    time[s]: QDA_batch 82.23 ± 4.29
    time[s]: QDA_ours_soft_labels 0.23 ± 0.09
    time[s]: QDA_ours_labels 0.08 ± 0.01
    
    FEATURE 3: Capgmyo_dbb
    time[s]: LDA_batch 86.54 ± 5.84
    time[s]: LDA_ours_soft_labels 0.86 ± 0.09
    time[s]: LDA_ours_labels 0.71 ± 0.05
    time[s]: QDA_batch 85.11 ± 5.8
    time[s]: QDA_ours_soft_labels 0.24 ± 0.09
    time[s]: QDA_ours_labels 0.09 ± 0.01
    
    FEATURE 1: Cote
    time[s]: LDA_batch 489.87 ± 21.63
    time[s]: LDA_ours_soft_labels 0.78 ± 0.05
    time[s]: LDA_ours_labels 0.62 ± 0.02
    time[s]: QDA_batch 488.46 ± 21.59
    time[s]: QDA_ours_soft_labels 0.17 ± 0.06
    time[s]: QDA_ours_labels 0.07 ± 0.0
    
    FEATURE 2: Cote
    time[s]: LDA_batch 526.97 ± 28.46
    time[s]: LDA_ours_soft_labels 0.8 ± 0.04
    time[s]: LDA_ours_labels 0.65 ± 0.02
    time[s]: QDA_batch 525.52 ± 28.45
    time[s]: QDA_ours_soft_labels 0.19 ± 0.08
    time[s]: QDA_ours_labels 0.08 ± 0.0
    
    FEATURE 3: Cote
    time[s]: LDA_batch 532.9 ± 26.68
    time[s]: LDA_ours_soft_labels 0.85 ± 0.07
    time[s]: LDA_ours_labels 0.67 ± 0.03
    time[s]: QDA_batch 531.38 ± 26.66
    time[s]: QDA_ours_soft_labels 0.19 ± 0.07
    time[s]: QDA_ours_labels 0.08 ± 0.01
    
    FEATURE 1: LongTerm3DC
    time[s]: LDA_batch 181.33 ± 8.96
    time[s]: LDA_ours_soft_labels 0.9 ± 0.09
    time[s]: LDA_ours_labels 0.73 ± 0.05
    time[s]: QDA_batch 179.85 ± 8.94
    time[s]: QDA_ours_soft_labels 0.24 ± 0.1
    time[s]: QDA_ours_labels 0.08 ± 0.01
    
    FEATURE 2: LongTerm3DC
    time[s]: LDA_batch 198.33 ± 8.26
    time[s]: LDA_ours_soft_labels 0.94 ± 0.09
    time[s]: LDA_ours_labels 0.77 ± 0.06
    time[s]: QDA_batch 196.81 ± 8.24
    time[s]: QDA_ours_soft_labels 0.28 ± 0.1
    time[s]: QDA_ours_labels 0.09 ± 0.01
    
    FEATURE 3: LongTerm3DC
    time[s]: LDA_batch 198.5 ± 7.87
    time[s]: LDA_ours_soft_labels 0.95 ± 0.1
    time[s]: LDA_ours_labels 0.79 ± 0.06
    time[s]: QDA_batch 196.96 ± 7.85
    time[s]: QDA_ours_soft_labels 0.26 ± 0.1
    time[s]: QDA_ours_labels 0.09 ± 0.01
    
    FEATURE 1: EPN_120
    time[s]: LDA_batch 611.42 ± 200.34
    time[s]: LDA_ours_soft_labels 1.88 ± 4.41
    time[s]: LDA_ours_labels 1.32 ± 3.62
    time[s]: QDA_batch 608.3 ± 199.37
    time[s]: QDA_ours_soft_labels 0.48 ± 2.24
    time[s]: QDA_ours_labels 0.08 ± 0.01
    
    FEATURE 2: EPN_120
    time[s]: LDA_batch 676.62 ± 219.69
    time[s]: LDA_ours_soft_labels 1.84 ± 4.23
    time[s]: LDA_ours_labels 1.77 ± 4.57
    time[s]: QDA_batch 672.41 ± 219.36
    time[s]: QDA_ours_soft_labels 0.54 ± 2.33
    time[s]: QDA_ours_labels 0.08 ± 0.01
    
    FEATURE 3: EPN_120
    time[s]: LDA_batch 662.59 ± 201.74
    time[s]: LDA_ours_soft_labels 1.88 ± 4.27
    time[s]: LDA_ours_labels 1.09 ± 2.88
    time[s]: QDA_batch 659.47 ± 201.23
    time[s]: QDA_ours_soft_labels 0.5 ± 2.24
    time[s]: QDA_ours_labels 0.17 ± 1.29
    

## Experiment 2

In this experiment, we show the robustness to mislabeled data of our online classifier due to how we calculate the
 covariance matrix (COV). Traditionally, the COV of the DA classifiers is calculated w.r.t. the mean vector of all
 gestures, whereas we calculate w.r.t. the mean vectors of each gesture.

Concretely, we compare the performance of the weighted batch classifier and a version of this weighted classifier
(from now on called traditional classifier), in which the covariance matrix (COV) is calculated w.r.t.
the mean vector of all gestures.
Note that our online classifier using pseudo-labeled gestures has the same performance as the weighted batch classifier,
 as shown Theorem 2.

We show the accuracy difference of these two compared batch classifiers w.r.t. the initial classifier using labeled
and pseudo-labeled gestures as shown in the next figure.
When the accuracy of any of these classifiers is higher than the initial classifier, the accuracy difference
is positive; otherwise, it is negative.
We use the 2-tailed Wilcoxon signed ranks test at $p_{value}<0.5$ to determine if the accuracy differences are
statistically significant.
When these two batch classifiers use labeled gestures, their performance is similar for LDA and QDA. In contrast,
when they use pseudo-labeled gestures, the performance of the weighted batch classifier is significantly higher
than the difference of the traditional classifier.
In fact, the performance of the traditional classifier is worse than the performance of the initial classifier in several cases.


```python
analysis_experiments.experiment2()
```

    
    ANALYSIS WILCOXON (CONFIDENCE LEVEL 95%) shows the accuracy of two classifiers that come from the same distribution (Nina5): 
    
    FEATURE 1: Nina5
    p value:  0.2958775226696384 , LDA: weighted and traditional batch classifiers using pseudo-labels come from the same distribution
    p value=  0.654158944417145 , LDA: weighted and traditional batch classifiers using labels come from the same distribution
    
    FEATURE 2: Nina5
    p value=  0.8227604017844778 , QDA: weighted and traditional batch classifiers using labels come from the same distribution
    
    FEATURE 3: Nina5
    p value=  0.5015913016269502 , QDA: weighted and traditional batch classifiers using labels come from the same distribution
    
    ANALYSIS WILCOXON (CONFIDENCE LEVEL 95%) shows the accuracy of two classifiers that come from the same distribution (Capgmyo_dbb): 
    
    FEATURE 1: Capgmyo_dbb
    p value=  0.5186995762757332 , QDA: weighted and traditional batch classifiers using labels come from the same distribution
    
    FEATURE 2: Capgmyo_dbb
    p value=  0.05127352900893355 , LDA: weighted and traditional batch classifiers using labels come from the same distribution
    p value=  0.5407934646543326 , QDA: weighted and traditional batch classifiers using labels come from the same distribution
    
    FEATURE 3: Capgmyo_dbb
    p value=  0.07257441007347816 , LDA: weighted and traditional batch classifiers using labels come from the same distribution
    p value=  0.1922847281429566 , QDA: weighted and traditional batch classifiers using labels come from the same distribution
    
    ANALYSIS WILCOXON (CONFIDENCE LEVEL 95%) shows the accuracy of two classifiers that come from the same distribution (Cote): 
    
    FEATURE 1: Cote
    p value=  0.35049563131847306 , LDA: weighted and traditional batch classifiers using labels come from the same distribution
    p value=  0.32518649095806207 , QDA: weighted and traditional batch classifiers using labels come from the same distribution
    
    FEATURE 2: Cote
    p value=  0.8754242567176749 , LDA: weighted and traditional batch classifiers using labels come from the same distribution
    p value=  0.5857749837558817 , QDA: weighted and traditional batch classifiers using labels come from the same distribution
    
    FEATURE 3: Cote
    p value=  0.3281623065662794 , LDA: weighted and traditional batch classifiers using labels come from the same distribution
    p value=  0.9426105124576512 , QDA: weighted and traditional batch classifiers using labels come from the same distribution
    
    ANALYSIS WILCOXON (CONFIDENCE LEVEL 95%) shows the accuracy of two classifiers that come from the same distribution (LongTerm3DC): 
    
    FEATURE 1: LongTerm3DC
    p value:  0.9509263906838966 , LDA: weighted and traditional batch classifiers using pseudo-labels come from the same distribution
    
    FEATURE 2: LongTerm3DC
    p value:  0.9302662950606848 , LDA: weighted and traditional batch classifiers using pseudo-labels come from the same distribution
    p value=  0.08524622674182489 , QDA: weighted and traditional batch classifiers using labels come from the same distribution
    
    FEATURE 3: LongTerm3DC
    
    ANALYSIS WILCOXON (CONFIDENCE LEVEL 95%) shows the accuracy of two classifiers that come from the same distribution (EPN_120): 
    
    FEATURE 1: EPN_120
    p value=  0.2858891290534964 , LDA: weighted and traditional batch classifiers using labels come from the same distribution
    
    FEATURE 2: EPN_120
    p value=  0.9932886367973275 , LDA: weighted and traditional batch classifiers using labels come from the same distribution
    
    FEATURE 3: EPN_120
    p value=  0.2599006724426639 , LDA: weighted and traditional batch classifiers using labels come from the same distribution
    


![png](output_10_1.png)


## Friedman rank test

We also perform the Friedman test's average ranks of the three pseudo-labeleing techniques (Nigam's technique, thresholding
and our soft-labelling technique) and of the two batch classifiers (weighted and traditional classifiers).


```python
analysis_experiments.experiment1(friedman=True)

```

    
    
    FRIEDMAN TOTAL (CONFIDENCE LEVEL 95%) soft-labelling techniques
    
    Type DA classifier: LDA
    ours_soft_labelling: 2.0
    Nigam's technique: 2.1
    thresholding: 1.9
    
     The best classifier is:  thresholding
                                                p    sig
    thresholding vs Nigam's technique    0.000008   True
    thresholding vs ours_soft_labelling  0.055365  False
    
    Type DA classifier: QDA
    ours_soft_labelling: 1.7
    Nigam's technique: 2.1
    thresholding: 2.2
    
     The best classifier is:  ours_soft_labelling
                                                p   sig
    ours_soft_labelling vs Nigam's technique  0.0  True
    ours_soft_labelling vs thresholding       0.0  True
    
    
    FRIEDMAN TOTAL (CONFIDENCE LEVEL 95%) batch classifiers
    
    Type DA classifier: LDA
    labels
    weighted classifier: 1.5
    traditional classifier: 1.5
    
     The best classifier is:  traditional classifier
                                                          p    sig
    traditional classifier vs weighted classifier  0.061532  False
    
    Type DA classifier: QDA
    labels
    weighted classifier: 1.5
    traditional classifier: 1.5
    
     The best classifier is:  traditional classifier
                                                          p   sig
    traditional classifier vs weighted classifier  0.034556  True
    
    Type DA classifier: LDA
    soft_labels
    weighted classifier: 1.2
    traditional classifier: 1.8
    
     The best classifier is:  weighted classifier
                                                     p   sig
    weighted classifier vs traditional classifier  0.0  True
    
    Type DA classifier: QDA
    soft_labels
    weighted classifier: 1.1
    traditional classifier: 1.9
    
     The best classifier is:  weighted classifier
                                                     p   sig
    weighted classifier vs traditional classifier  0.0  True
    
