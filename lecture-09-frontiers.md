# Lecture 9 — Frontiers: diversity, self-verification, curricula, and efficiency

*Independent textbook chapter accompanying [CS329A Part 9](https://www.youtube.com/watch?v=AyO6wyu4DEg). The recording supplies the research sequence and open questions. The repair curriculum, covariance derivation, proof examples, energy calculations, and proposed experiment are original teaching constructions.*

## 1. Self-improvement needs new information, not just more copies [00:00](https://www.youtube.com/watch?v=AyO6wyu4DEg&t=0s)

The median-repair service trains on its own accepted patches. After several rounds, its outputs become more uniform. They pass familiar parity tests but repeatedly miss mutation and unusual input contracts. The loop has become better at reproducing its evaluator's preferred strategy without necessarily expanding the range of problems it can solve.

Let $\pi_k(y\mid q)$ be the distribution of trajectories $y$ for task $q$ at training round $k$. A filter accepts selected trajectories, and an update produces $\pi_{k+1}$. A **self-improvement loop** changes a persistent component using experience generated partly by the system itself. Its success requires a useful signal about behavior, not merely an increase in agreement among generated responses.

**Diversity** can mean different wordings, different algorithms, different information sources, or complementary errors. Only some forms help a particular task. Renaming variables in ten upper-middle implementations does not diversify their failure on even-length inputs. A second strategy that checks mutation may supply genuinely different evidence.

[Multiagent Finetuning](https://arxiv.org/abs/2501.05707) studies models that begin from a common base but specialize through independently constructed training data from multiagent interactions. Its evaluated results support a strategy for maintaining diverse reasoning across training rounds. The existence of several agents alone does not establish independent knowledge or independent mistakes.

### Deriving an effective sample count

Let $X_1,\ldots,X_n$ be exchangeable estimates with common variance $\sigma^2$ and common pairwise correlation $\rho$. Exchangeability means their joint distribution is invariant to reordering. Let $\bar X=n^{-1}\sum_iX_i$ be their mean. Summing individual variances and pairwise covariances gives

$$\operatorname{Var}(\bar X)=\frac{n\sigma^2+n(n-1)\rho\sigma^2}{n^2}
=\frac{\sigma^2}{n}[1+(n-1)\rho].$$

Define **effective sample size** $n_{\rm eff}$ by equating this variance to $\sigma^2/n_{\rm eff}$ for independent estimates. Then

$$n_{\rm eff}=\frac{n}{1+(n-1)\rho}.$$

Ten estimates with $\rho=0.5$ have $n_{\rm eff}=10/5.5\approx1.82$. Ten agent identities can therefore provide less than two independent estimates in this specific variance sense. The calculation does not directly predict majority-vote accuracy, and a shared bias remains even when variance decreases.

For the repair loop, measure disagreement on meaningful behaviors and joint errors on held-out task families. Distinct prompts may still induce the same misconception. Different tools, training experiences, or explicit hypotheses can create more useful diversity, but the benefit must be demonstrated in outcomes.

### Preserve exploration while filtering

A filter that rewards only the dominant strategy can remove rare alternatives before they improve. Keeping every alternative is not the answer either: many are simply wrong. Track how accepted strategy families change over rounds, whether rare successful approaches survive, and whether gains transfer to tasks outside the training generator's habits.

The most informative comparison holds total generation and verification cost fixed. A multiagent system that uses ten times the computation may outperform a single run without establishing that specialization is the reason. Compare against an equal-budget baseline and inspect where joint errors actually decrease.

## 2. Verification itself becomes a learning target [15:00](https://www.youtube.com/watch?v=AyO6wyu4DEg&t=900s)

A proof generator can produce a correct final answer through invalid reasoning. A verifier can assign the right overall score while citing an error that does not exist. Improving verification therefore requires judging the assessment itself, not only whether its binary verdict happened to match a label.

Let $q$ be a proposition, $p$ a proposed proof, and $V(q,p)$ a verifier output containing a score and identified issues. A **meta-verifier** $M(q,p,V(q,p))$ evaluates whether the claimed issues are justified and whether the score follows from them. This adds another evidence-processing stage. It does not create an infallible foundation merely by placing a judge above another judge.

[DeepSeekMath-V2](https://arxiv.org/abs/2511.22570) studies learned proof verification, meta-verification of proof analyses, and training a generator using verification feedback. Its natural-language verification should be distinguished from a formal proof certificate. The research target is more accurate and faithful assessment of mathematical reasoning.

### A counterexample must satisfy the proposition's assumptions

Consider the statement $\sqrt{x^2}=x$ for every real number $x$. The square-root symbol denotes the nonnegative square root. Taking $x=-2$ yields $\sqrt4=2\ne-2$, so the statement is false. The correct identity over all real numbers is $\sqrt{x^2}=|x|$, where $|x|$ is absolute value.

Now change the proposition to the domain $x\ge0$. The same objection is invalid because -2 is outside the allowed domain. A meta-verifier should check that the alleged counterexample meets the assumptions before crediting the critique. Familiarity with a common algebraic pitfall is insufficient without reading the actual statement.

For median repair, a critic may object that empty input raises an exception. That is a valid defect only if the contract requires another behavior. If empty input is explicitly excluded, the critique changes the task rather than finding an error. Verification quality includes respecting the specification.

### Formal and informal checking warrant different claims

**Formal proof checking** verifies a derivation in a specified logical system using a trusted checker. **Informal verification** evaluates natural-language reasoning, often with implicit conventions and omitted steps. An informal proof can be correct and insightful without formalization. A formally checked statement can be irrelevant if its assumptions or encoded specification miss the intended claim.

A learned meta-verifier may reduce false objections but can share the generator's or verifier's blind spots. Evaluate it on both genuine errors and valid unusual arguments. A judge trained mostly on standard proofs may reject novel but sound reasoning, creating a feedback loop that suppresses useful diversity.

### Coupled optimization changes the test distribution

As the generator improves, it produces harder-to-detect errors. A verifier trained on old easy mistakes can become inadequate. Conversely, a stronger verifier can change which trajectories the generator learns. The pair co-evolves, so static evaluator accuracy does not describe the whole training process.

Maintain an independent audit set that includes new failure modes and valid counterexamples to the verifier's heuristics. Do not train every component on every audit result and then claim the same audit remains independent. The verification boundary is an experimental design problem, not just an architectural diagram.

## 3. A model can propose its own curriculum [23:00](https://www.youtube.com/watch?v=AyO6wyu4DEg&t=1380s)

A **curriculum** determines which tasks a learner sees and in what proportions. A proposer–solver system lets one component generate tasks and another attempt them. A trusted environment can validate whether a task is well formed and whether an answer satisfies an executable criterion.

For a program $f$, input $x$, and output $y=f(x)$, three reasoning directions are useful. **Deduction** predicts $y$ from $f$ and $x$. **Abduction** finds an $x$ consistent with known $f$ and $y$. **Induction** proposes an $f$ consistent with input–output examples. These are different inverse problems and can have different numbers of valid solutions.

Take $f(x)=x^2+1$ over real inputs. Deduction at $x=3$ gives ten. Abduction from output ten admits both three and minus three. Induction from pairs $(0,1)$ and $(1,2)$ does not distinguish $x^2+1$ from $x+1$. A task generator must account for this non-uniqueness rather than reject valid alternatives because they differ from one hidden construction.

[Absolute Zero](https://arxiv.org/abs/2505.03335) studies a proposer–solver loop in an execution environment, generating tasks and validating them through code. Its “zero data” framing concerns the post-training task-data regime. A pretrained model, programming language, execution environment, and reward design still supply substantial prior structure.

### Intermediate difficulty supplies contrast

Suppose a proposed task has independent single-attempt success probability $p$, and a group contains $G\ge2$ attempts. The group contains both successes and failures with probability $1-p^G-(1-p)^G$. This can provide a relative learning signal. Groups of all failures or all successes supply no centered binary reward contrast.

Let $g(p)=1-p^G-(1-p)^G$. Differentiation gives $g'(p)=G[(1-p)^{G-1}-p^{G-1}]$. It is positive below one half, zero at one half, and negative above one half. Thus mixed-group probability is maximized at $p=0.5$. This is a theorem about reward contrast under the stated model, not a theorem that 50%-solvable tasks maximize long-term learning.

A proposer can generate ambiguous or noisy tasks whose outcomes vary for unhelpful reasons. Such tasks have high reward variance without teaching a useful skill. A curriculum should combine validity, relevance, novelty, and evidence of learning progress rather than optimizing variance alone.

### A repair curriculum with controlled generalization

Begin with parity bugs, then introduce mutation defects, duplicate values, negative values, and explicit empty-input policies. Keep task families distinguishable and preserve held-out combinations. A solver that learns each isolated rule may still fail when several constraints interact.

Measure performance before and after training on a fixed probe set. Let $p_k(q)$ be estimated success on task family $q$ after round $k$. A local progress estimate is $p_{k+1}(q)-p_k(q)$. It is noisy, depends on the probes, and can favor tasks that were temporarily underestimated. Use repeated evaluation or uncertainty estimates before letting a proposer chase small fluctuations.

The numerical lab checks the mixed-group calculation directly. Inputs are a probability and group size, output is the analytical chance of a reward contrast, and no model is trained.

```python
from numerical_lab import mixed_group_probability

assert abs(mixed_group_probability(0.5, 4) - 0.875) < 1e-12
assert mixed_group_probability(0.0, 4) == 0.0
assert mixed_group_probability(1.0, 4) == 0.0
```

The zero boundary cases expose why an always-solved curriculum and an always-failed curriculum can both stall this particular signal. They do not imply that other learning objectives cannot learn from those examples.

## 4. Slow verification changes the optimal experiment [34:00](https://www.youtube.com/watch?v=AyO6wyu4DEg&t=2040s)

Code execution can provide feedback quickly. A chip simulation, materials experiment, or biological assay may take hours or days. Such tasks are not necessarily unverifiable; their verification is expensive, delayed, or noisy. The search policy must account for that resource structure.

Let $V_0$ be a cheap approximate screen and $V_1$ an expensive higher-fidelity evaluator. Suppose $N$ candidates each cost $c_0$ units to screen, and fraction $r$ proceed to evaluation costing $c_1$ each. Total expected cost is

$$C=Nc_0+rNc_1.$$

For one thousand candidates, screening cost one, expensive evaluation cost one hundred, and pass fraction 0.05, total cost is $1000+50(100)=6000$ units. Evaluating every candidate directly would cost 100,000. This is an accounting saving, not yet evidence of equal scientific value.

### False negatives can hide the best discovery

Let $p$ be the fraction of truly valuable candidates, and let $t$ be screen sensitivity, the probability a valuable candidate passes. Expected valuable candidates reaching the expensive evaluator are $Npt$. A screen with low $t$ can discard the most useful unusual ideas while producing an impressive cost reduction.

For $N=1000$, $p=0.02$, and $t=0.5$, only ten of the expected twenty valuable candidates reach the second stage. If the screen's errors concentrate on novel mechanisms, the loss can be worse than a random half of successes. A surrogate trained on familiar data may be least reliable where discovery is most valuable.

Audit a random subset of rejected candidates with the high-fidelity evaluator. This estimates false negatives under the audit sampling scheme. Also reserve some budget for uncertain or novel candidates. Pure exploitation of predicted score can repeatedly confirm the surrogate's preferences while learning little about its blind spots.

### Delay and throughput are not the same cost

If an expensive evaluator runs many candidates in parallel, its wall-clock delay differs from total compute or laboratory expense. A two-stage cascade may reduce total evaluations but add a serial screening stage. Batch size, queueing, and instrument availability affect the preferred design.

For the median service, these issues are small because tests are cheap. The transferable structure is to distinguish proposal cost, screening cost, definitive evaluation cost, delay, and false rejection. A learned reward model is useful as a surrogate when its uncertainty and failure modes are measured against a stronger source of evidence.

A reward prediction cannot become definitive merely because definitive evaluation is slow. When reporting a result, label whether it is predicted, simulated, or experimentally observed. That distinction is central to scientific agent systems.

## 5. Power, energy, and correct completions answer different questions [40:00](https://www.youtube.com/watch?v=AyO6wyu4DEg&t=2400s)

[Intelligence per Watt](https://arxiv.org/abs/2511.07885) studies task accuracy relative to average power draw as an efficiency measure for model–hardware configurations. Such a metric motivates examining local inference and heterogeneous hardware. It must be interpreted alongside runtime and measurement boundaries.

**Power** is energy per unit time. One watt equals one joule per second. Let $P(t)$ be power in watts at time $t$ seconds during a run of duration $T$. Energy in joules is $E=\int_0^TP(t)\,dt$. Under constant average power $\bar P$, energy is $E=\bar PT$.

Let accuracy $a\in[0,1]$ be the fraction of tasks correctly completed under a stated evaluation. Accuracy per watt is $a/\bar P$. Energy per attempted task is $\bar PT$. Under a homogeneous workload with expected success probability $a>0$ and expected energy $E$, aggregate energy divided by expected correct completions is $E/a$. These quantities have different units and can rank systems differently.

### A worked ranking reversal

System A has accuracy 0.8, average power 20 W, and runtime ten seconds. System B has accuracy 0.9, power 100 W, and runtime one second. Accuracy per watt favors A: 0.04 versus 0.009. Energy per attempt favors B: 200 J versus 100 J. Energy per expected correct completion is 250 J for A and about 111.1 J for B.

There is no contradiction. The power metric values operating under a low power draw; the energy metric also accounts for how long that draw persists. A device with a hard power envelope and a service minimizing total energy face different objectives.

The following deterministic check makes the units explicit. It uses stipulated average power and duration, returns joules per expected correct completion, and does not measure hardware.

```python
power_a, seconds_a, accuracy_a = 20, 10, 0.8
power_b, seconds_b, accuracy_b = 100, 1, 0.9
energy_a = power_a * seconds_a / accuracy_a
energy_b = power_b * seconds_b / accuracy_b
assert energy_a == 250
assert abs(energy_b - 111.1111111111) < 1e-9
assert accuracy_a / power_a > accuracy_b / power_b
assert energy_a > energy_b
```

### Measure the whole configured service

Record hardware, model, numerical precision, quantization, batch size, context length, output length, idle-power treatment, and measurement boundary. Chip power, accelerator-board power, and whole-system power are different quantities. A local desktop measurement and a cloud chip measurement are not directly comparable without a common boundary.

Count verification, retries, and routing. If a cheap local model fails frequently and sends most tasks to a remote fallback, the combined service may use more energy than direct remote inference. A per-call saving does not establish a per-completed-task saving.

### Deriving a routing decision

For task $q$, let model option $m$ have cost $C_m(q)$ and success probability $p_m(q)$. Let $\ell$ be the loss of failure in the same utility units as cost. A simple decision minimizes $C_m(q)+\ell[1-p_m(q)]$. For local option L and remote option R, choose R when

$$C_R-C_L<\ell(p_R-p_L).$$

The extra cost must be smaller than the value of the success-probability improvement. With costs one and five, probabilities 0.7 and 0.9, and failure loss thirty, the improvement is worth six units, exceeding the extra cost four. The stronger option is preferred under this model.

The probabilities must be calibrated on the relevant tasks. A router that is confidently wrong about its local model's competence can defeat an excellent fallback policy. Evaluate the router and fallback together, including cases where uncertainty should trigger escalation.

## 6. Continual learning must preserve and revise prior competence [52:00](https://www.youtube.com/watch?v=AyO6wyu4DEg&t=3120s)

**Continual learning** updates a system across successive experiences or task families. The updated component might be external memory, a prompt, an adapter, or the full parameter vector. These mechanisms differ in persistence, reversibility, retrieval dependence, and generalization. Naming the changed component is necessary to interpret an improvement claim.

Let $A_{i,j}$ be accuracy on task family $j$ after training phase $i$. After $T$ phases, a simple forgetting measure for previously introduced families is

$$F_T=\frac1{T-1}\sum_{j=1}^{T-1}
\left(\max_{j\le i<T}A_{i,j}-A_{T,j}\right),\qquad T>1.$$

The maximum uses only phases after family $j$ was introduced and before the final phase. The difference compares its best previous performance with current performance. This signed convention can be negative when the final system improves beyond every previous score. A nonnegative forgetting metric would clamp each difference at zero; state which convention is used.

For two old families with previous best accuracies 0.9 and 0.8 and final accuracies 0.85 and 0.9, signed average forgetting is $(0.05-0.10)/2=-0.025$. The aggregate improvement hides a five-point regression on the first family. Report family-level changes alongside the average.

### Retention is not always the right target

A system should retain valid old skills but revise obsolete facts and policies. Perfectly preserving a stale API convention can be harmful after the environment changes. Evaluate both retention of stable competence and adaptation to changed requirements. A memory system needs provenance and freshness, not merely a growing archive.

For median repair, preserve arithmetic and non-mutation skills while allowing a new project to specify a different empty-input policy. The model should distinguish transferable reasoning from project-specific constraints. Blindly reusing an old successful patch can violate the new contract.

External memory can improve performance only if relevant information is retrieved and used correctly. A weight update can generalize without retrieval but can also interfere with older behaviors. Compare these mechanisms under fixed resource budgets and fresh tasks, including misleading memories and incorrect experiences. More stored text is not itself evidence of learning.

### A falsifiable solo research project

Build a small, deterministic repair benchmark with known candidate distributions and trusted finite evaluators. Compare three equal-budget systems: more candidate generation with one checker, fewer candidates with complementary checkers, and adaptive allocation between the two. Introduce a deliberately imperfect learned-score surrogate so the generation–verification gap is measurable.

The hypothesis is specific: extra verification helps most when correct candidates are frequently available but selection is weak. It should fail when generation rarely produces a correct candidate or the added checkers share the same errors. Record coverage, delivered success, compute, latency, and joint failures. A useful result identifies the boundary rather than announcing one universal winner.

A second phase can learn from accepted traces and repeat the frozen evaluation, checking both new-task performance and retention. Training is optional for the first experiment; a synthetic generator and the standard-library lab suffice to test the allocation logic without paid model calls.

## 7. Exercises and solutions

Self-generated tasks, learned judges, and specialized agents can create useful feedback loops. They can also reinforce a shared misconception. Independent evaluation, explicit specifications, and evidence from the environment remain necessary. Slow verification changes how evidence is allocated; it does not convert prediction into observation.

1. Compute effective sample size for twenty exchangeable estimates with pairwise correlation 0.1. State what this number does not measure.
2. Evaluate the objection $x=-2$ against both versions of the square-root proposition. Explain the meta-verifier's role.
3. Show why mixed-group probability is maximized at success probability one half for group size at least two. Give a reason this need not maximize learning progress.
4. A screen costs two units, a definitive evaluator costs fifty, and 20% of five hundred candidates pass. Compute total cost and expected valuable survivors if prevalence is 0.04 and sensitivity is 0.75.
5. Design an experiment distinguishing useful continual learning from simply appending more text to memory. Include a retention and a stale-information test.

<details><summary>Fully worked solutions</summary>

<p><strong>1.</strong> The denominator is $1+19(0.1)=2.9$, so $n_{\rm eff}=20/2.9\approx6.90$. It is the independent sample count giving the same mean-estimation variance under the equicorrelated model. It does not directly measure vote accuracy, distinct strategies, or absence of shared bias. Twenty similarly mistaken judges can still agree on a false claim.</p>

<p><strong>2.</strong> For all real $x$, minus two is allowed and gives $\sqrt{x^2}=2\ne-2$, so it refutes the statement. Under the restriction $x\ge0$, minus two is not an admissible counterexample. The meta-verifier checks whether the objection actually addresses the proposition and whether the verdict follows. A learned meta-verifier can itself make mistakes; this is not formal proof checking.</p>

<p><strong>3.</strong> Differentiating $1-p^G-(1-p)^G$ gives $G[(1-p)^{G-1}-p^{G-1}]$. For $G\ge2$, the derivative is positive below one half and negative above it, establishing the maximum. A task with random or ambiguous labels can also produce a 50% success rate and frequent mixed groups while teaching nothing useful. Validity and transfer matter in addition to reward contrast.</p>

<p><strong>4.</strong> Screening costs $500(2)=1000$. One hundred candidates reach the definitive evaluator, costing $100(50)=5000$, for total 6000 versus 25,000 without screening. Expected valuable survivors are $500(0.04)(0.75)=15$. There were twenty valuable candidates in expectation, so five are lost at screening. The cost saving must be weighed against that loss and whether false negatives concentrate on especially valuable novelty.</p>

<p><strong>5.</strong> Hold the model, retrieval budget, and memory capacity fixed. Add relevant experiences for one new task family and compare against shuffled, irrelevant, and no-update memories on unseen tasks. Test older families for regression. Then change one project-specific rule and check that the system follows the new authoritative rule rather than stale memory. Improvement should depend on relevant transferable information while preserving stable skills and revising obsolete constraints.</p>

</details>

## 8. Primary references and source boundaries

- Subramaniam et al., [Multiagent Finetuning](https://arxiv.org/abs/2501.05707): specialization and diverse self-improvement data.
- Shao et al., [DeepSeekMath-V2](https://arxiv.org/abs/2511.22570): learned proof verification and meta-verification.
- [Absolute Zero](https://arxiv.org/abs/2505.03335): task proposal and solving in an execution environment.
- [Intelligence per Watt](https://arxiv.org/abs/2511.07885): joint model–hardware capability and power measurement.

The recording supplies diversity, self-verification, self-generated curricula, expensive feedback, efficiency, and continual learning in that sequence. The formal models and numerical examples are independent extensions. Guest sessions mentioned in the recording are outside the supplied playlist and are not reconstructed as if their transcripts had been available.
