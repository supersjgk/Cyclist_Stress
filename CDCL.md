### Formula to calculate Cross Domain Contrastive Loss
![cdcl](https://github.com/supersjgk/Cyclist_Stress/assets/75927878/5fd94404-ada7-4943-82a9-e885cef11f69)

For each target anchor $x$<sub>$t$</sub><sup>$i$</sup> with a pseudo-label $\hat{y}$<sub>$t$</sub><sup>$i$</sup> in a mini-batch, loss $L$<sub>$CDC$</sub><sup>$t,i$</sup> is calculated using the above formula.
- $Z$<sub>$t$</sub><sup>$i$</sup> represents the flattened representation of the target anchor obtained from encoder + projection network.
- $τ$ is the temperature hyper-parameter
- $P$<sub>$s$</sub>($\hat{y}$<sub>$t$</sub><sup>$i$</sup>) is the set of positive samples, i.e. samples from source domain with same label as the pseudo label of target anchor.
- $Z$<sub>$s$</sub><sup>$p$</sup> reppresents the flattened representation of a positive sample from source domain obtained from encoder + projection network
- $I$<sub>$s$</sub> is the set of all samples whether positive or negative in the mini-batch.
- The encircled formula shown below represents the Cosine-Similarity between the flattened vectors in both numerator and denominator.<br>

    ![cs](https://github.com/supersjgk/Cyclist_Stress/assets/75927878/8e68e3e1-35d2-451d-994a-40578ca0cebb)
