# Lecture 6 — Training reasoning policies: STaR, GRPO, and DAPO

*Independent textbook chapter accompanying [CS329A Part 6](https://www.youtube.com/watch?v=yVnmHSAy3ck). The recording supplies the progression from successful-trace bootstrapping to group-relative reinforcement learning and training-system details. The finite-policy derivations, repair examples, and calculations are original teaching extensions.*

## 1. What should persist after a successful repair? [00:00](https://www.youtube.com/watch?v=yVnmHSAy3ck&t=0s)

The median-repair service eventually finds a correct patch after twelve attempts. On the next task it repeats the same mistakes. Search improved one run without changing the policy. Training aims to make useful behavior more probable on future tasks: identifying parity, checking mutation, responding to failures, and stopping after justified verification.

Let $q$ be a task drawn from distribution $\mathcal D$, $y$ a complete response or interaction trajectory, and $\pi_\theta(y\mid q)$ its probability under parameters $\theta$. Let $R(q,y)$ be a scalar reward from a specified evaluator. A training objective can be written as $J(\theta)=\mathbb E_{q\sim\mathcal D,y\sim\pi_\theta}[R(q,y)]$, where expectation averages over tasks and generated trajectories.

The objective requires a choice of reward. Returning the correct median, preserving input data, and providing an honest verification report are distinct properties. If the reward measures only one visible numeric example, optimization can improve that score without producing a generally correct repair. The evaluator defines the behavior being reinforced.

**Supervised fine-tuning**, or SFT, increases the likelihood of selected demonstrations. **Reinforcement learning**, or RL, uses rewards to alter action or response probabilities. **Distillation** trains one model to imitate information supplied by another model or procedure. These mechanisms can be combined, but they consume different information. A filtered successful trace supplies a positive example; an RL update can compare differently rewarded outputs from the same task.

### A training claim needs an independent comparison

Keep a final task set outside generation, filtering, reward fitting, and hyperparameter selection. A dataset whose scores drive parameter updates contributes training supervision even when its answers stay out of the prompt. Evaluate the resulting policy at matched inference budgets. Otherwise an apparent training improvement may come from additional samples or a stronger selector.

For the median service, divide task families as well as individual examples. Near-duplicate even-length tests in both training and evaluation can make generalization look stronger than it is. Include held-out mechanisms, such as preserving a caller-owned sequence, to test whether the learned behavior transfers beyond memorized answer patterns.

The central question is not whether a model can generate a successful trace somewhere. It is how training changes the probability of useful trajectories, and which parts of the task distribution benefit or regress. The following objectives make that question explicit.

## 2. STaR converts selected reasoning into training data [16:00](https://www.youtube.com/watch?v=yVnmHSAy3ck&t=960s)

**Self-Taught Reasoner**, or STaR, bootstraps reasoning using generated rationales and answer checks. A **rationale** is an intermediate explanation accompanying a final answer. [The STaR paper](https://arxiv.org/abs/2203.14465) generates rationales, retains those producing correct answers, and uses answer-conditioned rationalization for some initially failed problems before further training.

Let $a^*(q)$ be a known target answer, $z$ a generated rationale, and $a$ the generated final answer. A filter retains $(q,z,a)$ when $a=a^*(q)$. SFT then increases $\pi_\theta(z,a\mid q)$ on the retained dataset. The filtering event changes the training distribution: frequently solved problems and common successful strategies may be overrepresented.

For failures, supplying the correct answer can help generate a rationale that reaches it. The answer hint is used while creating training data; the desired deployed model must reason from the original question without that hint. A rationale conditioned on the answer may contain useful structure, but it may also rationalize backward from a conclusion without providing a valid derivation.

### Answer correctness does not certify the rationale

Suppose the median service produces the right answer 5 for `[1, 9]` while explaining that the upper middle element is 5. The outcome check passes and the explanation is false. Training on it can reinforce a misleading reasoning pattern. A richer filter can check intermediate claims, but that introduces another evaluator with its own error and cost.

A successful trace also need not be the best demonstration. It may contain irrelevant detours, repeated failed edits, or accidental recovery. Whether to retain such material depends on the desired policy. Removing all failures can erase examples of recovery; retaining every token can teach inefficient habits. Treat data curation as a behavioral intervention rather than clerical preprocessing.

### Deriving the distribution induced by filtering

For a fixed task, let $\pi(y)$ be the generation distribution and let $A(y)\in\{0,1\}$ indicate acceptance by the filter. Let $Z=\sum_y\pi(y)A(y)>0$ be the acceptance probability. The retained distribution is

$$\pi_{\rm retained}(y)=\frac{\pi(y)A(y)}{Z}.$$

This is ordinary conditioning. If only correct outputs are accepted, it concentrates on correct outputs already sampled from the generator. It does not directly create demonstrations of a strategy that never appears. In a finite dataset, rare useful strategies may disappear even when their true probability is positive.

Across tasks, acceptance rates also change task frequencies. Suppose half of source tasks are easy with acceptance probability 0.8 and half are hard with acceptance probability 0.1. Retained easy mass is 0.4 and retained hard mass 0.05, so easy tasks constitute $0.4/0.45\approx88.9\%$ of the retained set. A nominally balanced source distribution becomes heavily skewed.

This calculation suggests a diagnostic: report acceptance rates and retained counts by task family. Reweighting or collecting more hard-task attempts can reduce the skew, but each changes cost and may introduce noisier data. The STaR mechanism motivates bootstrapping; it does not remove the need to inspect what the filter actually teaches.

## 3. Deriving the policy-gradient signal [41:00](https://www.youtube.com/watch?v=yVnmHSAy3ck&t=2460s)

Consider one task and a finite response set for clarity. Let $\pi_\theta(y)>0$ be differentiable in parameter vector $\theta$, and assume reward $R(y)$ does not itself depend on $\theta$. Define $\nabla_\theta$ as the vector of partial derivatives. The expected reward is $J(\theta)=\sum_y\pi_\theta(y)R(y)$. Differentiation gives

$$\nabla_\theta J=\sum_yR(y)\nabla_\theta\pi_\theta(y)
=\sum_y\pi_\theta(y)R(y)\nabla_\theta\log\pi_\theta(y)
=\mathbb E[R(y)\nabla_\theta\log\pi_\theta(y)].$$

The hinge is the identity $\nabla\pi=\pi\nabla\log\pi$. It turns a derivative of an expectation into an expectation that can be estimated from sampled responses. With an unbounded response set, exchanging differentiation and summation needs regularity conditions; the finite derivation avoids hiding that issue.

### Why a baseline can reduce noise without changing the mean

Let $b(q)$ be a baseline independent of the sampled response given the task. Then

$$\mathbb E[b(q)\nabla_\theta\log\pi_\theta(y\mid q)]
=b(q)\nabla_\theta\sum_y\pi_\theta(y\mid q)=0.$$

The final equality holds because probabilities sum to one. Thus replacing reward by $R-b$ leaves the expected gradient unchanged. The **advantage** is a reward relative to a baseline; positive advantage favors the sampled response, while negative advantage discourages it. A baseline that depends on the sampled response requires more care and cannot automatically use this argument.

A **critic** is a learned estimator of expected return used to construct advantages. Training a separate critic can add memory and optimization cost. [DeepSeekMath](https://arxiv.org/abs/2402.03300) introduces Group Relative Policy Optimization, or GRPO, using rewards within a sampled group to form relative advantages rather than relying on the same separate value-model structure as conventional actor–critic training.

### A two-response model with an exact gradient

Let the policy choose between a correct repair with reward one and an incorrect repair with reward zero. Let real parameter $u$ be the logit difference, and define $p=\sigma(u)=1/(1+e^{-u})$ as the probability of the correct repair. Expected reward is $J(u)=p$. Differentiating the logistic function gives

$$\frac{dJ}{du}=p(1-p).$$

At $u=0$, $p=0.5$ and the gradient is 0.25. The score-function expression gives the same result: a correct sample has log-probability derivative $1-p$, an incorrect sample has derivative $-p$, and its zero reward contributes nothing. Expected contribution is $p(1-p)$.

The following code checks the analytic derivative against a symmetric finite difference. Its input is a finite logit, output is a probability, and it has no external effects. The step size is a numerical approximation parameter, not a learning rate.

```python
import math


def success_probability(logit):
    return 1.0 / (1.0 + math.exp(-logit))


step = 1e-5
finite_difference = (success_probability(step)
                     - success_probability(-step)) / (2 * step)
assert abs(finite_difference - 0.25) < 1e-9
```

This toy model gives the gradient a concrete meaning: it increases the log-odds of the higher-reward response. A neural model shares parameters across many tasks and tokens, so improving one response can alter others. The simple derivation does not imply that every update monotonically improves every task.

## 4. Group-relative advantages and clipped updates [44:00](https://www.youtube.com/watch?v=yVnmHSAy3ck&t=2640s)

For one task, sample $G\ge2$ responses with rewards $R_1,\ldots,R_G$. Let $\bar R=G^{-1}\sum_iR_i$ be their mean and $s_R=\sqrt{G^{-1}\sum_i(R_i-\bar R)^2}$ their population standard deviation. A common group-relative advantage is $A_i=(R_i-\bar R)/s_R$ when $s_R>0$. If all rewards agree, the normalized expression is undefined; implementations use an explicit zero-signal convention or numerical stabilization.

For rewards $(1,0,1,0)$, the mean and standard deviation are both 0.5, giving advantages $(1,-1,1,-1)$. For $(1,1,1,1)$, all centered rewards are zero. A stabilizing denominator prevents division by zero but does not invent a comparative learning signal. The shared lab adopts zero advantages in that case.

```python
from numerical_lab import group_advantages

assert group_advantages([1, 0, 1, 0]) == [1, -1, 1, -1]
assert group_advantages([1, 1, 1, 1]) == [0, 0, 0, 0]
```

The helper accepts a nonempty sequence of finite rewards and returns a list of relative values. It does not train a model. A group-dependent baseline is not identical to the response-independent baseline proved above: each response contributes to its own group statistics. Group normalization is an algorithmic design with its own statistical effects, not a free consequence of the baseline identity.

### Why clip a likelihood ratio?

Suppose responses were sampled from an older policy $\pi_{\rm old}$. For a sampled action or token, define the **importance ratio** $r(\theta)=\pi_\theta/\pi_{\rm old}$ on that same event and context. It measures how the new policy changes its probability. Let $A$ be a fixed advantage estimate and $\epsilon\in(0,1)$ a clipping width. The clipped surrogate used in [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) has the form

$$L_{\rm clip}(\theta)=\mathbb E\left[
\min\left(r(\theta)A,\operatorname{clip}(r(\theta),1-\epsilon,1+\epsilon)A\right)\right],$$

where the clip operator truncates a value to the stated interval. This limits the incentive for certain large probability changes. It does not constrain every realized ratio to remain inside the interval and is not a formal guarantee that the whole policy stays close.

For positive $A=1$ and $\epsilon=0.2$, a ratio of 1.5 contributes the minimum of 1.5 and 1.2, namely 1.2. Further increasing that ratio does not improve this sample's surrogate contribution. For negative $A=-1$ and ratio 0.5, the contributions are -0.5 and -0.8, so the minimum is -0.8. The incentive to reduce the probability of this negative-advantage event is clipped once the ratio falls below 0.8.

The sign matters. A ratio moving in an undesirable direction is not protected by the same flat region. This asymmetry is why one should compute the two terms rather than remember clipping as a generic hard bound on parameter changes.

### Token and sequence conventions must be explicit

A language-model response contains many token probabilities. Algorithms may form token-level ratios and average their surrogate losses, while using a response-level reward-derived advantage at every token. A sequence-level ratio is a product of token ratios and behaves differently. Do not substitute one into the other without re-deriving the objective and its variance.

The practical training recipe also includes reference-policy regularization, sampling parameters, batching, reward evaluation, and numerical stabilization. GRPO is not fully specified by one advantage formula. The [DeepSeekMath paper](https://arxiv.org/abs/2402.03300) is the primary source for its particular formulation and experimental context.

## 5. DAPO shows that implementation details alter the objective [53:00](https://www.youtube.com/watch?v=yVnmHSAy3ck&t=3180s)

[DAPO](https://arxiv.org/abs/2503.14476) studies a large-scale RL training recipe including asymmetric clipping, dynamic sampling, token-level loss aggregation, and handling of overlong outputs. These choices affect which examples supply gradient, how much weight each token receives, and what behavior earns penalties. They are part of the learning problem rather than interchangeable engineering settings.

Let lower and upper clipping widths be $\epsilon_{\rm low}$ and $\epsilon_{\rm high}$. The permitted surrogate interval becomes $[1-\epsilon_{\rm low},1+\epsilon_{\rm high}]$. Increasing the upper width leaves a larger incentive range for raising the probability of positive-advantage tokens. This can influence exploration, but its effect depends on the full objective and data; it is not a theorem that larger upper clipping always improves diversity.

### How often does a group contain a useful contrast?

For binary independent rewards with single-response success probability $p$, a group of size $G$ is uninformative under pure centered relative rewards when all responses fail or all succeed. These disjoint events have probabilities $(1-p)^G$ and $p^G$. Thus the probability of a mixed group is

$$P(\text{mixed group})=1-p^G-(1-p)^G.$$

At $p=0.01$ and $G=8$, this is approximately 0.0773. Most groups contain no reward contrast. At $p=0.5$, the probability is $1-2(0.5)^8=0.9922$. Dynamic sampling can favor groups with useful contrast, but discarded generation still costs compute and changes the effective task distribution.

For the median service, once parity repair is almost always solved, parity-only groups contribute little centered binary signal. Harder mutation tasks may provide contrast, but tasks the model always fails also provide none. This creates a curriculum pressure toward intermediate difficulty. It does not ensure that those tasks best match the desired deployment distribution.

### Sequence averages and token averages weight data differently

Let response $i$ contain $L_i$ tokens and let $\ell_{it}$ be the per-token surrogate contribution at token $t$. A sequence-normalized objective averages $G^{-1}\sum_i L_i^{-1}\sum_t\ell_{it}$. A token-normalized objective averages $\left(\sum_iL_i\right)^{-1}\sum_i\sum_t\ell_{it}$. The first gives every response equal total weight; the second gives every token equal weight.

For lengths 10 and 100, a token in the short response receives ten times the weight of a token in the long response under sequence normalization. Under token normalization, all 110 tokens receive equal weight. This changes gradient allocation. Token normalization by itself is not a penalty for long output: a long response contributes more tokens and therefore more total weight. An explicit length or truncation policy is a separate mechanism.

### Truncation changes the meaning of failure

A response cut off at a maximum length may be unfinished rather than logically wrong. Assigning the same penalty to every truncated response can conflate a promising long derivation with an unproductive loop. Conversely, ignoring truncation can reward behavior that never produces a usable answer. Define the desired tradeoff between success, latency, and output length, then test how the penalty changes behavior.

The repair analogue is a timeout. A test runner that times out supplies a resource-limit observation, not a proof of an incorrect numeric result. Training may reasonably discourage timeouts because the service needs bounded completion, but that reward should be interpreted as operational utility rather than pure mathematical correctness.

## 6. Does RL create capability or concentrate it? [61:00](https://www.youtube.com/watch?v=yVnmHSAy3ck&t=3660s)

A policy update can increase the probability of already available successful strategies, reduce diversity, change generalization, or expose behaviors not seen in a finite pre-training sample. These effects require different evidence. A higher single-sample accuracy does not settle whether the support of useful behaviors expanded.

Let $p_q$ be single-sample success for task $q$ before training and $p'_q$ afterward. Independent-sampling coverage at budget $k$ is $1-(1-p_q)^k$ or $1-(1-p'_q)^k$. If every $p'_q\ge p_q$, coverage cannot decrease on any task under these assumptions. But average pass@1 can rise even while some hard tasks lose probability mass, causing high-budget coverage to worsen.

Consider two equally weighted task types. Before training their probabilities are 0.5 and 0.1; afterward they are 0.9 and 0.001. Average single-sample success rises from 0.3 to 0.4505. At twenty samples, average coverage before training is approximately $(1-0.5^{20}+1-0.9^{20})/2=0.9392$. Afterward it is approximately $(1-0.1^{20}+1-0.999^{20})/2=0.5099$. A headline pass@1 gain conceals a large loss on the harder type.

This construction is mathematical, not a claim about a named model. It shows why full task-level curves and diversity diagnostics matter. Finite sampling cannot establish that a strategy had exactly zero pre-update probability. Statements about genuinely new capability need carefully specified operational tests, not metaphysical claims inferred from a few missing samples.

### Finite samples cannot identify a support boundary

Suppose a fixed policy produces no successful repair on a task in $n$ independent attempts. For any hypothesized probability $p$, the chance of that observation is $(1-p)^n$. A small positive probability can easily produce no observed successes. At $p=0.001$ and $n=100$, that probability is about 0.9048. Declaring the task impossible from this experiment would confuse a sampling limit with a capability boundary.

Conversely, one success after training does establish that the updated procedure can produce a successful response under the tested conditions. It does not establish that the earlier procedure assigned exactly zero probability to that response. A stronger empirical claim compares success probabilities, generalization, and resource requirements with uncertainty, rather than claiming to observe the full support of an enormous distribution.

For the repair service, keep a task-by-task matrix of pre-update and post-update success at several sample budgets. Add behavior-family labels identifying parity, mutation, and input-contract failures. A gain concentrated in parity repairs and a loss in mutation handling becomes visible even when average pass@1 improves. This provides a more useful research conclusion than one binary label of capability creation or concentration.

### A falsifiable training ablation

An **ablation** removes or changes one component to isolate its effect. Compare successful-trace SFT, group-relative RL, and unchanged weights at matched training data access and reported compute. Then evaluate each with both single sampling and the same fixed search harness. Freeze the final verifier across all conditions.

If RL improves single-sample accuracy but its advantage disappears under large equal-budget search, one supported interpretation is that it made useful trajectories easier to elicit. If it also improves held-out hard-task coverage, the evidence supports a broader operational gain. Neither result by itself reveals all internal representations. Inspecting concrete newly solved tasks, negative transfer, and verifier failures makes the claim more precise.

## 7. Exercises and solutions

The finite-policy derivation establishes an identity under explicit assumptions. Practical neural training adds shared parameters, finite batches, clipping, group-dependent normalization, imperfect rewards, and changing data distributions. Measure the complete recipe, not just the name of the optimizer.

1. Two equally common task families have acceptance rates 0.9 and 0.1. What fraction of filtered demonstrations comes from each? How could this bias a self-training loop?
2. Derive the exact policy gradient for the two-response logistic model and check it at success probability 0.8.
3. Compute group-relative advantages for rewards $(1,1,0,0)$ and explain why all-zero groups supply no centered reward contrast.
4. For binary success probability 0.1 and group size four, compute the probability of a mixed group. If mixed groups alone are retained, what expected number of groups is needed to collect one?
5. Explain why token-level averaging is not inherently a length penalty, and design a measurement that separates length efficiency from correctness.

<details><summary>Fully worked solutions</summary>

<p><strong>1.</strong> The retained masses are $0.5(0.9)=0.45$ and $0.5(0.1)=0.05$. Normalizing by total acceptance 0.5 gives 90% easy-family and 10% hard-family demonstrations. Further training can reinforce the easier behavior and reduce attention to the hard family. Report acceptance by family, then consider balanced collection or reweighting, checking whether the added hard examples have reliable labels.</p>

<p><strong>2.</strong> With rewards one and zero, expected reward is $J(u)=\sigma(u)$. Differentiation gives $dJ/du=\sigma(u)[1-\sigma(u)]$. At success probability 0.8, this is 0.16. The same result follows from the score-function estimator: successful samples occur with probability 0.8 and have derivative $1-0.8=0.2$, giving $0.8(0.2)=0.16$. Failed samples have zero reward in this unbaselined calculation.</p>

<p><strong>3.</strong> The mean is 0.5 and population variance is 0.25, so standard deviation is 0.5. The advantages are $(1,1,-1,-1)$. With all-zero rewards, every centered reward is zero and the standard deviation is zero. A zero-advantage convention avoids undefined arithmetic. Adding a small denominator constant alone cannot create a nonzero numerator or a comparative signal.</p>

<p><strong>4.</strong> Mixed-group probability is $1-0.1^4-0.9^4=0.3438$. Independent group draws therefore require $1/0.3438\approx2.91$ groups on average for one mixed group. At four responses per group, expected generation is about 11.63 responses. Reporting only the four retained responses would omit most of the sampling expense. Correlated responses would change the group probability.</p>

<p><strong>5.</strong> Token averaging weights each token equally; longer responses contribute more tokens. It does not directly assign a negative reward to length. Measure final correctness, output tokens, latency, and truncation separately at matched task difficulty. Compare explicit length penalties or stopping policies only after defining an acceptable correctness tradeoff. A shorter incorrect answer is not evidence of more efficient reasoning.</p>

</details>

## 8. Primary references and source boundaries

- Zelikman et al., [STaR](https://arxiv.org/abs/2203.14465): successful-rationale bootstrapping and rationalization.
- Shao et al., [DeepSeekMath](https://arxiv.org/abs/2402.03300): mathematical model training and GRPO.
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347): the clipped surrogate objective.
- [DAPO](https://arxiv.org/abs/2503.14476): a training recipe with dynamic sampling, asymmetric clipping, and token-level aggregation.

The recording provides the method sequence and capability-versus-concentration question. Filtering bias, score-function proofs, the logistic finite-difference check, group-yield calculations, and the two-task coverage counterexample are original development. The distinction between token weighting and a length penalty corrects an easy conceptual conflation; no benchmark gain is assumed universal.
