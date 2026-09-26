# Lecture 3 — Verification, process supervision, and reliable selection

*Independent textbook chapter accompanying [CS329A Part 3](https://www.youtube.com/watch?v=p7TdPUcPoik). Timestamped headings follow the recording. The fraction problem is recording-derived; the repair service, probability examples, ensemble calculations, and exercises are teaching constructions.*

## 1. A verifier learns a target, not truth in general [00:00](https://www.youtube.com/watch?v=p7TdPUcPoik&t=0s)

The median-repair service now generates twenty candidate patches. Some return the upper middle element for even-length inputs. Others compute the correct value but mutate the caller's list, violating the contract. A verifier trained only on returned values will label the second group correct. Its predictions may be accurate relative to its labels while failing to measure the property the user needs.

Let $q$ be a task specification and available context, $y$ a complete candidate, and $z\in\{0,1\}$ a correctness label. A learned verifier $v_\phi(q,y)\in(0,1)$ predicts the probability of label one, using parameter vector $\phi$. An **outcome reward model**, or ORM, scores the completed result. A **process reward model**, or PRM, scores intermediate steps. Neither label establishes the evaluator's reliability; it names where supervision is applied.

For labeled examples drawn from distribution $\mathcal D$, binary cross-entropy is

$$L_{\rm ver}(\phi)=-\mathbb E_{(q,y,z)\sim\mathcal D}
\left[z\log v_\phi(q,y)+(1-z)\log(1-v_\phi(q,y))\right].$$

Here $\mathbb E$ means expectation and $\log$ is the natural logarithm. A correct example contributes $-\log v$; an incorrect one contributes $-\log(1-v)$. Confident mistakes are expensive: predicting 0.99 for a wrong patch contributes about 4.605, while predicting 0.5 contributes about 0.693. This encourages predictions consistent with training labels, assuming the optimizer and model can represent them.

[Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168) investigates sampling candidate mathematical solutions and learning to select them. Its use of correctness supervision and an auxiliary language-modeling objective illustrates that verifier design includes data generation, labels, architecture, and training loss. A scalar score attached to a language model does not remove the need to define the supervised event.

### Ranking and calibration are different properties

A verifier is **calibrated** on a distribution if, among examples assigned a score near $u$, approximately fraction $u$ receive label one. A verifier ranks well when correct candidates tend to score above incorrect ones. A strictly increasing transformation of scores preserves ranking but can destroy calibration. For example, squaring scores preserves their order on $[0,1]$ while changing a prediction 0.8 to 0.64.

This distinction matters operationally. Choosing the highest-scoring patch needs comparative accuracy within a candidate batch. Releasing any patch above threshold 0.99 needs a meaningful relation between scores and error probabilities. A verifier trained to rank may be useful for the former and unsuitable for the latter.

Training and evaluation distributions must also match the intended claim. A verifier can learn stylistic clues correlated with correctness in one generator's outputs. When another generator produces polished wrong answers, that shortcut may fail. Evaluate on candidates from the actual search procedure, including difficult near-misses and candidates selected for unusually high scores.

### Why cross-entropy targets the conditional probability

The loss can be understood without invoking a neural network. Fix one observable task–candidate representation and let $p_*$ be the true probability of label one among examples with that representation. Suppose the verifier reports a constant $v\in(0,1)$. Its expected loss is $\ell(v)=-p_*\log v-(1-p_*)\log(1-v)$. Differentiating gives

$$\ell'(v)=-\frac{p_*}{v}+\frac{1-p_*}{1-v}
=\frac{v-p_*}{v(1-v)}.$$

The denominator is positive. The derivative is negative below $p_*$ and positive above it, so the unique interior minimum is $v=p_*$. At boundary probabilities zero or one, the infimum occurs at the corresponding boundary. This is why cross-entropy is called a **proper scoring rule**: in expectation, reporting the true conditional probability minimizes the loss.

The conclusion has specific limits. A finite model may not represent the conditional probability, finite training data may estimate it poorly, and optimization may not reach the minimum. More fundamentally, $p_*$ refers to the supplied labels. If labels ignore mutation, the optimal verifier can confidently accept mutation bugs. Statistical consistency with a label distribution does not repair a missing contract clause.

For a numerical check, suppose the conditional correctness rate is 0.8. Reporting 0.8 gives expected loss approximately 0.5004; reporting 0.5 gives 0.6931; reporting 0.99 gives approximately 0.9291. Overconfidence can therefore be worse than an uninformative score. Calibration should be checked on the selected candidate distribution as well as on random samples, because the conditions used to estimate $p_*$ can change after search.

### Build labels that expose the real contract

For the repair task, label at least returned-value correctness, non-mutation, accepted-input behavior, and resource limits separately. A combined label can require all mandatory conditions, but retaining the components makes errors diagnosable. An outcome of zero then carries more information than a generic rejection if it can be traced to a failed invariant.

A concrete counterexample is a function that calls `values.sort()` before returning the median. It may pass every numeric output test and still alter a list the caller intended to reuse. A verifier trained on output-only tests cannot be expected to learn non-mutation reliably from those labels. The missing property must appear in the specification and evaluation evidence.

## 2. Rare correctness makes small verifier errors expensive [12:00](https://www.youtube.com/watch?v=p7TdPUcPoik&t=720s)

Let $p$ be the fraction of generated candidates that are truly correct. Let $t=P(\text{accept}\mid\text{correct})$ be the true-positive rate and $f=P(\text{accept}\mid\text{wrong})$ the false-positive rate at a specified threshold. All three quantities lie in $[0,1]$. **Acceptance precision** is the conditional probability that an accepted candidate is correct.

A randomly generated candidate enters the accepted-correct group with probability $pt$. It enters the accepted-wrong group with probability $(1-p)f$. If the sum is positive, conditioning on acceptance yields

$$P(\text{correct}\mid\text{accept})=\frac{pt}{pt+(1-p)f}.$$

With $p=0.02$, $t=0.90$, and $f=0.05$, the numerator is 0.018 and the false-positive contribution is 0.049. Precision is $0.018/0.067\approx0.2687$. A service accepting these candidates would return mostly wrong answers. Lowering the false-positive rate to 0.001 raises precision to about 0.9484 at the same true-positive rate.

The comparison does not imply that thresholds can lower $f$ without lowering $t$. A threshold generally changes both. The correct evaluation reports their tradeoff and the candidate prevalence. An aggregate accuracy score can conceal the relevant behavior: rejecting everything on a population with 2% correct candidates attains 98% binary accuracy and delivers no useful patch.

### Deriving the required false-positive rate

Suppose the service requires acceptance precision at least $\tau\in(0,1)$. Rearranging the precision inequality gives

$$pt\ge\tau[pt+(1-p)f],\qquad
f\le\frac{pt(1-\tau)}{\tau(1-p)}.$$

The first step moves all accepted-candidate mass into the denominator; the second isolates the permitted false-positive contribution. At $p=0.02$, $t=0.90$, and $\tau=0.95$, the required $f$ is at most approximately 0.0009667. A verifier that appears strong on balanced examples can be inadequate when correct proposals are rare.

The shared [numerical lab](numerical_lab.py) implements this conditional-probability calculation. Its inputs are probabilities, its output is a probability or `None` if no candidates are accepted, and it performs no external action. Returning an undefined result in the no-acceptance case avoids reporting spurious perfect precision.

```python
from numerical_lab import acceptance_precision

precision = acceptance_precision(0.02, 0.90, 0.05)
assert abs(precision - 0.2686567164) < 1e-9
assert acceptance_precision(0.02, 0.0, 0.0) is None
```

### Search exposes the verifier's tail

Selecting the maximum of many scores is different from applying one fixed threshold to one random candidate. Even if a verifier makes rare extreme mistakes on random wrong answers, a large batch supplies many chances to encounter them. With independent false-alarm probability $f=0.01$ per wrong candidate, one hundred wrong candidates contain at least one false alarm with probability $1-0.99^{100}\approx0.6340$.

This is a multiple-opportunity calculation, not an exact formula for best-of-one-hundred accuracy. That accuracy also depends on the score distribution of correct candidates and how all scores compare. The example nevertheless identifies a failure mechanism: optimizing a noisy proxy changes the population on which the proxy must work.

In the median service, a long explanation can receive a high learned score while missing the mutation constraint. Repeated generation may eventually produce an especially persuasive version of that mistake. A useful stress test deliberately includes such near-correct candidates, rather than evaluating the verifier only on easy failures. The [weak-verifier ensemble work](https://arxiv.org/abs/2506.18203) addresses this generation–verification gap; it should be read as an empirical method, not a guarantee that ensemble scores remain calibrated under arbitrary search.

## 3. Process supervision asks where reasoning first fails [21:00](https://www.youtube.com/watch?v=p7TdPUcPoik&t=1260s)

An outcome label says whether the final answer meets a criterion. A process label evaluates an intermediate transition or assertion. For the median repair, a process might identify the sorted array, distinguish parity, choose indices, compute an average, and preserve the input. A final numeric match can occur despite an invalid intermediate step, so the two supervision targets are not interchangeable.

Write a solution as steps $y_1,\ldots,y_T$. Let $E_t$ be the event that step $t$ is valid under the problem's rules. A true chain-rule factorization of all-step validity is

$$P(E_1\cap\cdots\cap E_T)=\prod_{t=1}^{T}
P(E_t\mid E_1,\ldots,E_{t-1}).$$

The symbol $\cap$ means intersection: every event must occur. No independence assumption is needed for this identity. However, the conditional probabilities must refer to these events. Multiplying arbitrary learned step scores does not automatically instantiate the identity.

If each step were independently valid with probability 0.98, twenty steps would all be valid with probability $0.98^{20}\approx0.6676$. Independence is an additional assumption in that numerical example. In real reasoning, a mistaken definition can corrupt many later steps, producing strongly dependent errors.

[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) compares outcome and process supervision on mathematical reasoning and releases the PRM800K step-feedback dataset. Its empirical advantage for process supervision is evidence in the studied setting. The general mechanism is more localized credit: the training signal can distinguish an early invalid move from later text built upon it.

### A worked algebraic check

Consider the recording-derived equation $x/(3x-7)=2/5$, with real unknown $x$. The original expression is defined only when $3x-7\ne0$, so $x\ne7/3$. Multiplying by the nonzero denominator and by five gives $5x=2(3x-7)$. Expanding gives $5x=6x-14$, and subtracting $5x$ yields $x=14$. Substitution checks $14/(42-7)=14/35=2/5$.

The domain restriction matters before cross-multiplication. On this problem it does not exclude the final answer, but omitting such restrictions in other equations can introduce extraneous solutions. A process verifier can identify whether a transformation preserves the solution set; a final-answer verifier may only see that 14 is correct.

Suppose a solution incorrectly expands $2(3x-7)$ as $6x-7$, then later changes the resulting answer to 14 without justification. Its outcome is right and its derivation is invalid. Conversely, a valid prefix can lead to a final arithmetic mistake. These cases give different training signals depending on whether the objective is final answer accuracy, reliable explanations, or proof validity.

### Aggregation is a modeling choice

A product of step scores penalizes long chains when scores stay below one. A minimum emphasizes the weakest-scored step. An average can hide one catastrophic error among many easy valid steps. These rules encode different assumptions. If one invalid step destroys a proof, averaging is generally not a faithful probability model of proof validity.

For the repair service, use a typed checklist rather than pretending every textual sentence has the same role. “Input unchanged” is a postcondition to test; “sorted values are…” is an observation to verify; “therefore the index is…” is a derivation to inspect. A step scorer trained on one form of reasoning may not transfer to another without additional evidence.

Active learning selects examples for labeling based on expected informativeness rather than sampling uniformly. In process supervision, disagreement or uncertainty can help focus expensive human labels. But uncertainty scores themselves can be miscalibrated. Keep an independently sampled evaluation set so the selected training examples do not redefine the population on which success is claimed.

## 4. Rollouts estimate future success, not local logical validity [37:00](https://www.youtube.com/watch?v=p7TdPUcPoik&t=2220s)

Human annotation of every step is expensive. An alternative is to complete a partial solution several times and use eventual success to label the prefix. A **rollout** is one sampled continuation from a specified partial state. Let $h$ be a solution prefix and $\pi$ the continuation policy. Define its continuation value $V^\pi(h)=P(\text{final success}\mid h,\text{continue with }\pi)$.

This value depends on the continuation policy. A mathematically valid prefix may have low value for a weak solver unable to finish it. A flawed prefix may have positive value if the solver can detect and repair the error. Therefore $V^\pi(h)$ measures recoverable success under a policy, not the truth of the last sentence in isolation.

Suppose $N$ independent continuations yield success indicators $Z_1,\ldots,Z_N$, each zero or one. A **soft rollout label** is the mean $\widehat V(h)=N^{-1}\sum_{j=1}^{N}Z_j$. It has expectation $V^\pi(h)$ and variance $V^\pi(h)(1-V^\pi(h))/N$ under the independent identical sampling assumption. A **hard rollout label** is one when at least one continuation succeeds and zero otherwise. Its probability of being one is

$$P(\text{hard label}=1)=1-(1-V^\pi(h))^N.$$

The soft label estimates a probability; the hard label estimates whether a finite search finds a success. At $V^\pi(h)=0.1$, four rollouts produce a positive hard label with probability 0.3439. Thirty-two rollouts raise that probability to 0.9657 without changing the prefix or its single-rollout value. Label meaning depends on the rollout budget.

[Math-Shepherd](https://arxiv.org/abs/2312.08935) uses automatically constructed process supervision and studies both reranking and reinforcement learning. Its mechanism motivates the distinction between step labels obtained by search and direct human judgments of a step's validity. The paper's performance results do not eliminate this conceptual difference.

### Worked repair example

Prefix A correctly identifies the even-length branch but has not chosen the indices. Prefix B inserts a wrong averaging formula yet leaves the code open to revision. A continuation policy may finish A successfully only 20% of the time because indexing is difficult, while fixing B successfully 50% of the time because the error message is highly diagnostic. A rollout label then ranks B above A.

That ranking can be useful for choosing where a repair agent should continue. It would be misleading if interpreted as evidence that B's current formula is more mathematically valid. The intended use determines the right label. A search controller wants expected downstream utility; an explanation checker may want local validity; a final selector wants terminal correctness.

Changing the continuation policy after collecting labels introduces another mismatch. A prefix considered unrecoverable by an older model may become easy for a stronger model. A verifier trained on old rollout values can undervalue it. Periodic relabeling or policy-aware evaluation can diagnose this issue, but relabeling cost belongs in the training budget.

The existing lab can check hard-label behavior directly. The calculation consumes a stipulated rollout probability and count, returns an analytical label probability, and does not sample or certify a real solution.

```python
from numerical_lab import coverage

assert abs(coverage(0.1, 4) - 0.3439) < 1e-12
assert abs(coverage(0.1, 32) - 0.9656631618) < 1e-9
```

Increasing rollout count reduces one form of label noise, but it does not repair an incorrect final evaluator. If the evaluator accepts the upper-middle median bug, more rollouts can make that wrong target appear more reliably attainable.

## 5. Ensembles help when errors contain independent information [51:00](https://www.youtube.com/watch?v=p7TdPUcPoik&t=3060s)

An **ensemble** combines several verifiers. Let $M$ be the number of verifiers, $v_j(q,y)$ the score from verifier $j$, and $w_j$ its combination weight. A weighted score is $s(q,y)=\sum_{j=1}^{M}w_jv_j(q,y)$. The units and scale of component scores matter. One verifier's logits and another's probabilities cannot be added meaningfully without a specified normalization or fitted combination rule.

[Shrinking the Generation–Verification Gap with Weak Verifiers](https://arxiv.org/abs/2506.18203) studies combining weaker verifiers rather than assuming a single sufficiently strong judge. The linked landing page can contain a revision newer than the recording; its system should not be identified with every detail of a historical slide. The supported methodological point is that complementary verification signals can improve selection in evaluated settings.

### A derivation of the correlation ceiling

To isolate averaging, suppose each verifier score equals a shared target plus zero-mean error. Let every error have variance $\sigma^2$, and let the correlation between any two distinct errors be $\rho$. The variance of an average includes $M$ individual variance terms and $M(M-1)$ covariance terms. Dividing by $M^2$ gives

$$\operatorname{Var}(\text{average error})
=\frac{M\sigma^2+M(M-1)\rho\sigma^2}{M^2}
=\sigma^2\left(\rho+\frac{1-\rho}{M}\right).$$

Here variance is expected squared deviation from the mean; correlation is covariance normalized by standard deviations. At zero correlation, averaging reduces variance by a factor $M$. At perfect correlation, it reduces nothing. For ten verifiers with $\rho=0.8$, the factor is 0.82 rather than 0.1. Merely increasing the number of similar judges offers limited independent evidence.

This derivation concerns unbiased numerical errors with an equicorrelation structure. It does not establish classification accuracy for a nonlinear selector, and averaging does not remove a shared bias. If all verifiers misunderstand the median contract, agreement strengthens the same wrong conclusion.

### Likelihood ratios explain complementary evidence

Let $Z$ be the event that a candidate is correct, with prior probability $p\in(0,1)$. Let $e_1,\ldots,e_M$ be verifier observations. Under conditional independence of the observations given correctness and also given incorrectness, Bayes' rule yields posterior odds

$$\frac{P(Z\mid e_1,\ldots,e_M)}{P(\neg Z\mid e_1,\ldots,e_M)}
=\frac{p}{1-p}\prod_{j=1}^{M}\frac{P(e_j\mid Z)}{P(e_j\mid\neg Z)}.$$

The symbol $\neg Z$ means not correct. Each ratio compares how likely that evidence is under a correct versus incorrect candidate. Independent evidence multiplies odds. Repeatedly asking the same judge about the same explanation generally does not satisfy the conditional-independence assumption.

For the repair service, a parity test, a mutation test, and a check of the formal contract can reveal different defects. Their complementarity is more meaningful than three differently worded requests to one model. Even executable tests can be dependent if they exercise the same branch. Estimate joint errors on a relevant candidate population rather than assuming independence from different names or prompts.

A useful allocation question is whether the next resource unit should generate another patch or verify an existing one more carefully. The answer depends on whether the current bottleneck is availability or selection. Keep both measurements: high coverage with poor delivered accuracy points toward verification; low coverage with strong selection points toward generation or better feedback.

## 6. Limits of a verified result

Verification is always relative to a target and evidence procedure. A formal proof has a specification and trusted checker; a test has an input distribution and coverage boundary; a human label has a rubric and annotator variation; a learned score has a training and evaluation distribution. Naming those boundaries makes a successful check useful rather than vague.

For the median case, acceptance should report which contract clauses were checked. It can state that parity and mutation tests passed without claiming exhaustive correctness over all finite real-valued sequences. A broader claim needs a proof or a justified argument that bridges from finite tests to the full domain.

Process supervision, rollout labeling, and ensembles each improve a different part of the decision procedure. Their effects can combine, but their guarantees do not combine automatically. A model trained on biased rollout labels and judged by correlated critics may still be confidently wrong. The right diagnostic is a counterexample tied to the contract, not another declaration of confidence.

## 7. Exercises and solutions

1. A verifier has true-positive rate 0.8 and false-positive rate 0.02. Compute acceptance precision when correctness prevalence is 0.5 and when it is 0.01.
2. For prevalence 0.01 and true-positive rate 0.8, derive the maximum false-positive rate compatible with 99% acceptance precision.
3. A prefix has continuation success probability 0.2. Compare the expectation of its soft label with the probability of a positive hard label after five independent rollouts. What does each estimate?
4. Sixteen equally weighted verifier errors have equal variance and pairwise correlation 0.25. Compute the average-error variance factor and its limiting value as the ensemble grows.
5. Construct one valid-but-low-value prefix and one invalid-but-recoverable prefix for a repair agent. Design an evaluation that distinguishes logical validity from continuation value.

<details><summary>Fully worked solutions</summary>

<p><strong>1.</strong> At prevalence 0.5, accepted-correct mass is $0.5(0.8)=0.4$ and accepted-wrong mass is $0.5(0.02)=0.01$, giving precision $0.4/0.41\approx0.9756$. At prevalence 0.01, the two masses are 0.008 and 0.0198, giving $0.008/0.0278\approx0.2878$. The verifier's conditional error rates did not change. The candidate population changed, making false positives dominate the accepted pool.</p>

<p><strong>2.</strong> Substitute $p=0.01$, $t=0.8$, and $\tau=0.99$ into $f\le pt(1-\tau)/[\tau(1-p)]$. This gives $f\le0.00008/0.9801\approx0.00008162$, or about 0.00816%. A false-positive rate of 0.02 misses this requirement by roughly a factor of 245. Raising the threshold may reduce false positives but may also reduce true positives, so both rates must be remeasured together.</p>

<p><strong>3.</strong> The mean of five independent Bernoulli outcomes remains unbiased for the single-rollout value, so its expectation is 0.2. Its variance is $0.2(0.8)/5=0.032$. The hard label is positive with probability $1-0.8^5=0.67232$. It estimates whether five attempts find any success, not a 67.2% single-continuation success rate. Neither quantity alone certifies the last step's logical validity.</p>

<p><strong>4.</strong> The variance factor is $0.25+0.75/16=0.296875$. Sixteen independent errors would instead yield 0.0625. As ensemble size grows, the factor tends to 0.25 because the shared correlated component remains. The result assumes equal variances, equal pairwise correlations, and zero-mean errors. A common bias survives even if the variance becomes small.</p>

<p><strong>5.</strong> A valid prefix may correctly reduce an even-length median to averaging two indexed elements, while a weak policy frequently chooses the wrong indices afterward. An invalid prefix may contain an off-by-one index that a clear exception reliably exposes and the policy repairs. Have independent reviewers label the prefix's current validity, then run a fixed continuation policy under a fixed budget and an independent final test suite. Cross-tabulate validity with measured continuation success. Repeating the experiment with a stronger policy tests whether the value labels are policy-dependent without changing the logical-validity labels.</p>

</details>

## 8. Primary references and source boundaries

- Cobbe et al., [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168): learned outcome verification.
- Lightman et al., [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050): process supervision and PRM800K.
- Wang et al., [Math-Shepherd](https://arxiv.org/abs/2312.08935): automatically constructed process labels, reranking, and reinforcement learning.
- [Shrinking the Generation–Verification Gap with Weak Verifiers](https://arxiv.org/abs/2506.18203): weak-verifier combination and selection.

The recording supplies these four research topics, their progression, and the fraction example. Bayes calculations, correlation analysis, likelihood-ratio derivation, and repair-service counterexamples are independent mathematical development. The distinction between local validity and policy-dependent continuation value is essential when interpreting rollout-derived labels; numerical examples here do not reproduce a paper's benchmark table.
