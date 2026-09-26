# Lecture 1 — From language modeling to self-improving agents

*Independent textbook chapter accompanying [CS329A Part 1](https://www.youtube.com/watch?v=6YnLB0XbTnI). The recording determines the topic order. The recurring median-repair service and its numerical parameters are teaching constructions. The tennis-ball problem is recording-derived; research claims have primary-source links.*

## 1. Scaling changes the distribution of possible solutions [02:00](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=120s)

A software service receives a bug report: its median function returns the upper middle element of an even-length list. The intended result is the average of the two middle values after sorting. A language model might explain the error, propose a patch, or generate a convincing but defective implementation. The engineering question is whether the complete service can reliably return a verified repair within a resource budget.

A **token** is a discrete unit in the model's text vocabulary. A **language model** assigns probabilities to token sequences. Let $q$ be the task description and supplied context, $y=(y_1,\ldots,y_T)$ an output of $T$ tokens, and $\theta$ the vector of learned numerical parameters. Let $y_{<t}$ denote the tokens before position $t$. By the probability chain rule, the conditional sequence probability is

$$\pi_\theta(y\mid q)=\prod_{t=1}^{T}\pi_\theta(y_t\mid q,y_{<t}).$$

The product multiplies conditional next-token probabilities. It does not assume the tokens are independent. In fact each factor can depend on all preceding tokens. A patch that is globally coherent can emerge from this autoregressive process, but the objective used to train the factors need not directly measure whether the patch satisfies its specification.

Let $\mathcal D$ be a distribution over training examples, and let $\mathbb E$ denote expectation, or averaging with respect to that distribution. **Pretraining** commonly minimizes negative log-likelihood,

$$L_{\rm pre}(\theta)=-\mathbb E_{(q,y)\sim\mathcal D}\left[\sum_{t=1}^{T}\log\pi_\theta(y_t\mid q,y_{<t})\right].$$

The natural logarithm turns the product into a sum. A token assigned probability 0.5 contributes approximately 0.693 units of negative log-loss; one assigned probability 0.01 contributes 4.605. Increasing the probability of observed tokens lowers the loss. This explains the training signal, but it does not imply that the observed text is correct, safe, or appropriate for the user's task.

### What a scaling law actually describes

Let $N$ be parameter count and $D$ the number of training tokens. An illustrative fitted loss model is $L(N,D)=L_\infty+aN^{-\alpha}+bD^{-\beta}$, with positive constants $a,b,\alpha,\beta$ and limiting loss $L_\infty$. A **scaling law** is an empirical relationship between quantities such as loss, model size, data, and compute over a measured regime. It is not an unconditional law governing all future architectures or every downstream skill. [Kaplan et al.](https://arxiv.org/abs/2001.08361) studied such relationships; [Hoffmann et al.](https://arxiv.org/abs/2203.15556) examined compute-optimal allocation between model size and training data.

The allocation problem can be derived under a deliberately simplified training-cost approximation. Suppose compute $C$ is proportional to $ND$, and absorb the proportionality constant into the units so $D=C/N$. Substituting gives $L(N,C/N)=L_\infty+aN^{-\alpha}+bC^{-\beta}N^\beta$. Treating $N$ as continuous, differentiation and setting the derivative to zero yield

$$-\alpha aN^{-\alpha-1}+\beta bC^{-\beta}N^{\beta-1}=0,
\qquad N^{\alpha+\beta}=\frac{\alpha a}{\beta b}C^\beta.$$

Thus the preferred model size scales as $C^{\beta/(\alpha+\beta)}$ within this model. The hinge is the fixed-compute substitution: enlarging the model leaves fewer training tokens affordable. If the exponents are equal, model size and token count both scale as the square root of compute. This is an analytical teaching result, not a re-estimation of any paper's fitted constants.

For the repair service, lower pretraining loss may improve code familiarity and instruction comprehension. It does not prove correctness on the even-length case. A benchmark's thresholded score can also make gradual improvements look sudden. A model can cross a test threshold without acquiring an entirely discontinuous internal capability. Keep the measured quantity and evaluation rule beside every claim about “emergence.”

## 2. Prompting supplies information without changing weights [06:00](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=360s)

**Zero-shot prompting** supplies instructions without worked demonstrations in the prompt. **Few-shot prompting** includes a small number of task examples. In both cases the model can condition on new information while keeping $\theta$ fixed. This is different from gradient-based training, which updates the parameters. If the median specification appears in the prompt, its influence comes through the conditional distribution $\pi_\theta(y\mid q)$ rather than a persistent weight change.

A **reasoning trace** is an intermediate sequence used while constructing an answer. In chain-of-thought prompting, demonstrations or instructions encourage such a trace. The [original chain-of-thought study](https://arxiv.org/abs/2201.11903) reports improvements on evaluated reasoning tasks for particular model scales and prompting settings. Those findings do not make every fluent explanation faithful to the computation that determined an answer.

The recording's tennis-ball example asks for a total after starting with five balls and buying two packs containing three balls each. The arithmetic is $5+2(3)=11$. The intermediate multiplication identifies the contribution of the packs; addition combines it with the initial stock. This tiny example distinguishes an operation sequence from a bare final number. It does not establish that printing more intermediate text will help every problem.

### Marginalizing over reasoning paths

Let $z$ denote an intermediate reasoning sequence and $a$ the final answer. A joint generation model factors as $\pi_\theta(z,a\mid q)=\pi_\theta(z\mid q)\pi_\theta(a\mid q,z)$. The probability of an answer, after allowing all possible traces, is

$$\pi_\theta(a\mid q)=\sum_z\pi_\theta(z\mid q)\pi_\theta(a\mid q,z).$$

The sum integrates over discrete traces. Sampling one trace uses one route through this distribution; sampling several and selecting an answer can explore several routes. The equation is a probability identity. It does not certify that a trace is logically valid or that its text captures all causes of the output.

For the median task, a useful trace distinguishes odd and even lengths, specifies zero-based indices, and checks `[1, 9]`. A misleading trace might say “the median is the middle element” and never confront the even-length case. Both can look orderly. External checks are needed because linguistic structure and semantic validity are distinct properties.

### Examples can specify the wrong task

Suppose every demonstration has odd length. A learner can infer either the intended median rule or the upper-middle rule and agree with all examples. The observations do not identify which hypothesis is intended. An even-length demonstration breaks this equivalence. The practical lesson is not simply to provide more examples; provide examples that discriminate among plausible interpretations.

A prompt should state whether the input may be empty, whether sorting may mutate the caller's list, and how nonfinite values are handled. These choices are part of the contract. Inferring them from ordinary examples is unreliable. A generated explanation can conceal ambiguity by choosing one interpretation confidently, leaving the resulting tests internally consistent but irrelevant to the intended specification.

A check of the arithmetic needs no model and no dependency. The input values are counts; multiplication gives the number purchased, and addition gives the final count. This code has no external effects and verifies the recording-derived example directly.

```python
initial_balls = 5
packs = 2
balls_per_pack = 3
final_balls = initial_balls + packs * balls_per_pack
assert final_balls == 11
```

The check is deliberately small. Its role is to distinguish a computed result from a merely repeated claim. Later chapters use the same principle for probabilistic and optimization calculations whose errors are harder to notice mentally.

## 3. Post-training changes which behaviors are probable [12:00](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=720s)

A pretrained model can complete many kinds of text. A useful assistant must respond to instructions, follow a task's constraints, and produce outputs that evaluators prefer. **Supervised fine-tuning**, abbreviated SFT, trains on selected prompt–response pairs. Its negative log-likelihood has the same form as the pretraining objective, but its data distribution concentrates on desired behavior.

Let $\mathcal D_{\rm SFT}$ be that demonstration distribution. Minimizing $-\mathbb E_{(q,y)\sim\mathcal D_{\rm SFT}}\log\pi_\theta(y\mid q)$ raises the probability of demonstrated responses. For the median service, demonstrations might contain correct patches with tests and explanations. If they consistently omit mutation checks, SFT may reinforce that omission. Demonstration quality is part of the objective in practice because the loss rewards imitation of whatever examples are provided.

**Reinforcement learning from human feedback**, or RLHF, uses human judgments to construct a reward signal for training. A **reward model** $r_\phi(q,y)\in\mathbb R$ assigns a real score using learned parameters $\phi$. A standard preference model predicts which of two responses a human will prefer. Let $y^+$ be the preferred response and $y^-$ the rejected response. Define the logistic function $\sigma(u)=1/(1+e^{-u})$. The pairwise model is

$$P(y^+\text{ preferred to }y^-\mid q)=\sigma\big(r_\phi(q,y^+)-r_\phi(q,y^-)\big).$$

A reward difference of zero predicts probability 0.5; a difference of $\log3$ predicts 0.75. The score difference, rather than either absolute score, determines the prediction. Pairwise preference data alone therefore do not identify an absolute zero of reward. The [InstructGPT paper](https://arxiv.org/abs/2203.02155) provides a primary account of demonstrations, preference modeling, and reinforcement learning in instruction following.

### Reward optimization with a reference policy

A reward model is imperfect. Optimizing its score without constraint can find responses that exploit its errors. Let $\pi_{\rm ref}(y\mid q)$ be a fixed reference policy with positive probability wherever the new policy places mass. For discrete responses, the **Kullback–Leibler divergence** is

$$D_{\rm KL}(\pi\Vert\pi_{\rm ref})=\sum_y\pi(y\mid q)\log\frac{\pi(y\mid q)}{\pi_{\rm ref}(y\mid q)}.$$

This nonnegative quantity measures a directional distributional difference and is zero when the distributions agree. Let $\beta>0$ be a penalty weight in reward units. An idealized per-prompt objective chooses a policy to maximize expected reward minus $\beta D_{\rm KL}$.

The optimum can be derived by imposing the normalization constraint $\sum_y\pi(y\mid q)=1$. Differentiating the objective with respect to each probability and introducing a Lagrange multiplier for normalization gives $r_\phi(q,y)-\beta[\log(\pi(y\mid q)/\pi_{\rm ref}(y\mid q))+1]+\lambda=0$. Rearranging and normalizing yields

$$\pi^*(y\mid q)=\frac{\pi_{\rm ref}(y\mid q)e^{r_\phi(q,y)/\beta}}{Z(q)},
\qquad Z(q)=\sum_y\pi_{\rm ref}(y\mid q)e^{r_\phi(q,y)/\beta}.$$

Here $Z(q)$ is the normalizing constant. This is a distribution-level solution, not a claim that a finite neural optimizer reaches it. It shows how reward tilts a reference distribution. Larger $\beta$ resists change; smaller $\beta$ gives the learned reward more influence.

For two initially equally likely patches, rewards one and zero with $\beta=1$ produce probabilities $e/(e+1)\approx0.7311$ and $1/(e+1)\approx0.2689$. If the reward model prefers the defective patch because its explanation is more polished, the same mathematics increases the wrong behavior. Better optimization cannot repair a misdefined target.

Keep correctness, helpfulness, and constraint adherence as separately measurable properties before combining them. Human preference is valuable evidence about human judgments. It is not automatically a formal correctness label for code, and a passing test is not automatically evidence that an explanation is useful or that the requested scope was respected.

## 4. More inference can reveal capability without learning [20:00](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=1200s)

**Inference** uses a trained model to produce outputs. **Test-time compute** is the computation spent during that use: additional samples, longer reasoning, tool calls, verification, or search. Holding $\theta$ fixed isolates improvement from changes to the inference procedure. Updating $\theta$ changes the generator itself. These are different interventions and should be named separately.

Let $p$ be the probability that one independent sample is correct for a fixed task. With $k$ independent attempts, all attempts fail with probability $(1-p)^k$, so the probability of at least one success is $1-(1-p)^k$. At $p=0.02$, one hundred samples give approximately 86.74% coverage. Chapter 2 distinguishes this oracle quantity from the accuracy of an available selector.

The infinite-monkey analogy has a precise probabilistic form. If each independent trial has fixed positive success probability, the probability of never succeeding is the limit of $(1-p)^k$, which is zero. Under those assumptions, eventual success has probability one. This statement is provable; it does not require claiming that an actual model has support on every desired solution or that infinite compute is available.

The [Large Language Monkeys study](https://arxiv.org/abs/2407.21787) supplies empirical motivation for repeated sampling. Its results also make verification central: the service must recognize a successful candidate. A hundred median patches do not constitute a hundred verified repairs. If all are judged with only odd-length examples, the original bug may survive every selection stage.

### Elicitation and learning have different persistence

**Elicitation** changes the information or procedure used to obtain behavior from a fixed model. **Learning** changes a persistent component, such as parameters or an explicitly maintained memory that influences future tasks. A successful repair produced after twenty attempts is evidence that this run found a solution. It is not evidence that the next fresh run has become more likely to succeed.

To claim improvement across tasks, compare a pre-update system and a post-update system on untouched tasks under the same inference budget. Otherwise training effects and extra search are confounded. A system that uses twice as many samples after an update may score better even if its single-sample distribution is unchanged.

Let $C_{\rm train}$ be an additional training cost, $N$ the number of later tasks served, and $c_{\rm infer}$ the average inference cost per task. Amortized cost is $C_{\rm train}/N+c_{\rm infer}$. Training can make repeated service cheaper if it lowers needed inference sufficiently, but that conclusion depends on reuse. A one-off task may favor extra search; a recurring workload may justify learning. Cost must use compatible units before terms are added.

### Self-generated data still needs an evaluator

A **self-improvement loop** proposes outputs, evaluates them, retains or revises selected data, updates a persistent component, and tests the result. The evaluator can be execution, human judgment, a proof checker, or a learned model. Each changes what improvement means. A self-generated explanation with an unverified final answer cannot certify its own truth merely by being reused as training data.

For the median case, keep the original task specification and a held-out evaluation set outside the generation loop. Retain successful patches with evidence showing which tests passed. If the generator creates both patches and tests, include independent tests that can expose shared misunderstandings. Chapter 6 develops training objectives; Chapter 9 examines what happens when the system also generates its own curriculum.

The runnable numerical check uses the shared lab's independent-sampling model. It accepts a probability and integer count and returns an analytical probability. It makes no model calls and does not measure a real generator.

```python
from numerical_lab import coverage

assert abs(coverage(0.02, 100) - 0.8673804441) < 1e-9
assert coverage(0.0, 100) == 0.0
assert coverage(0.8, 0) == 0.0
```

The boundary cases matter: additional compute cannot create success when the stipulated generator has zero success probability, and no attempts create no candidate. A numerical implementation should preserve these properties before it is used to interpret experiments.

## 5. An agent closes the action–observation loop [41:00](https://www.youtube.com/watch?v=6YnLB0XbTnI&t=2460s)

An **agent** repeatedly chooses actions in pursuit of a task while receiving observations from an environment. A **tool** exposes an external operation such as reading a file, running a test, or querying a database. A **harness** is the software controlling the model–tool loop, including context assembly, validation, execution, and stopping. A model's proposed tool call is not yet an executed action.

Let $x_t$ be the environment state before step $t$, $h_t$ the recorded history, $b_t$ the remaining budget, and $m_t$ a persistent memory available to the harness. A context builder forms $q_t=C(h_t,m_t,b_t)$, where $C$ is a software function. The policy proposes an action $a_t$ from $\pi_\theta(a\mid q_t)$. A gate checks whether the action is permitted, and the environment returns observation $o_{t+1}$ while transitioning to $x_{t+1}$. The history becomes $h_{t+1}=h_t\Vert(a_t,o_{t+1})$, where $\Vert$ denotes sequence concatenation.

The true state may be only partially observed. Reading one source file does not reveal every dependency; a test timeout does not reveal whether the code would eventually pass. The agent must reason from observations with known limitations. Generated text saying “all tests passed” is not a substitute for an actual test-runner result.

The median repair can proceed through a concrete trace: inspect the function, identify its accepted inputs, run `[1, 9]`, observe 9 instead of 5, edit the even-length branch, rerun parity tests, and report the remaining limitations. Each observation changes what a justified next action would be. A fixed one-shot answer lacks these environment-dependent branches.

### Workflows and adaptive control

A **workflow** fixes much of the operation sequence in software. **Routing** chooses among specialized operations based on the task. **Parallelization** runs independent operations concurrently. An **orchestrator–worker** arrangement decomposes work and combines worker outputs. These patterns can be combined; they do not form a universal ladder of increasing intelligence.

For a known median bug, a short workflow may suffice: inspect, patch, test. A repository-wide failure with uncertain origin may require adaptive retrieval and repeated diagnosis. More autonomy adds decisions that can fail. Use adaptive control where observations can materially change the next step, and evaluate that benefit against the extra cost and failure modes.

Let $s_i$ be the probability that stage $i$ succeeds conditional on all preceding stages succeeding. For $T$ necessary stages, the probability of complete success is $\prod_{i=1}^{T}s_i$ by the chain rule. This formula does not require stage independence because the probabilities are conditional. If all conditional stage probabilities equal 0.98 across twenty stages, end-to-end success is $0.98^{20}\approx0.6676$. High local reliability can still produce weak overall reliability.

Recovery changes the model. If a stage has initial success probability $s$, detects its failure with probability $d$ conditional on failure, and repairs a detected failure with probability $r$, its one-recovery success probability is $s+(1-s)dr$. With $s=0.9$, $d=0.8$, and $r=0.9$, this is 0.972. The additional success comes from a specific path through detection and repair; assuming recovery always works would exaggerate it.

### Capability is a property of the whole configured system

The output distribution depends on model parameters, prompts, tools, memory, stopping rules, and validators. When one component changes, the evaluated object changes. A benchmark score for a bare model does not determine the performance of a repair harness, and a successful harness does not prove that its model alone can reproduce the outcome.

A useful experiment logs candidate generation, executed actions, returned observations, budgets, and final evaluation. If a run fails, these records help distinguish inability to propose a patch from missing evidence, invalid tool use, poor selection, and premature stopping. That distinction guides the next improvement: more training is not necessarily the answer to a missing test or an incorrect specification.

## 6. Limits and study discipline

The formal models separate distribution learning, inference search, and environment interaction. Their boundaries are intentional. Token likelihood does not certify truth; reward preference does not certify code correctness; candidate coverage does not certify delivered accuracy; tool access does not certify effective use.

The nine recordings cover a selection of the [Autumn 2025 CS329A syllabus](https://cs329a.stanford.edu/). Additional syllabus sessions are not reconstructed here. Historical examples of model scale or product capability are treated as context rather than current specification sheets. The analysis instead focuses on mechanisms that can be tested in a small service.

For independent study, preserve one repair task through the sequence. Track its candidate distribution, verifier, feedback loop, search budget, training data, and evaluation protocol. The same failure can move between these components: a missing even-length case can be a prompt omission, a test omission, a reward-label error, or an evaluation blind spot. Diagnosing its location is more informative than attributing every failure to “reasoning.”

## 7. Exercises and solutions

1. Under the fitted loss model with equal exponents, derive how the preferred parameter and token counts change when compute quadruples. State two reasons the prediction can fail empirically.
2. A demonstration set contains only odd-length lists. Construct two rules consistent with it that disagree on `[1, 9]`. What new evidence separates them?
3. Two patches have equal reference probability and rewards one and zero. Compute the idealized KL-regularized probability of the higher-reward patch for $\beta=1$ and $\beta=0.5$.
4. Derive the one-recovery stage-success expression and compute end-to-end success for ten stages with $s=0.9$, $d=0.8$, and $r=0.9$, assuming the same conditional stage probability at each stage.
5. Design an experiment that distinguishes improved elicitation from improved parameters in the median-repair service.

<details><summary>Fully worked solutions</summary>

<p><strong>1.</strong> With $\alpha=\beta$, the allocation equation gives $N\propto C^{1/2}$. Since $D=C/N$, token count also scales as $C^{1/2}$. Quadrupling compute therefore doubles each count within the simplified model. The conclusion can fail when the fitted exponents change outside the observed regime, when data quality changes as more tokens are added, or when compute is not proportional to $ND$. It predicts a loss-optimal allocation under assumptions, not guaranteed downstream task accuracy.</p>

<p><strong>2.</strong> Rule A sorts and returns the usual median, averaging the two middle values for even length. Rule B sorts and always returns index $\lfloor n/2\rfloor$ under zero-based indexing, where the floor operator rounds down. They agree on all odd-length sequences. On `[1, 9]`, A returns 5 and B returns 9. An explicit even-length specification or this labeled counterexample separates them. More odd-length examples do not resolve the ambiguity.</p>

<p><strong>3.</strong> The reference probabilities cancel in the ratio, so the higher-reward probability is $e^{1/\beta}/(1+e^{1/\beta})$. For $\beta=1$, it is about 0.7311; for $\beta=0.5$, it is about 0.8808. Reducing the penalty concentrates more mass on the higher reward. If the reward label is wrong, the same calculation quantifies how stronger optimization amplifies the error.</p>

<p><strong>4.</strong> Success occurs immediately with probability $s$, or after initial failure, detection, and successful repair with probability $(1-s)dr$. These disjoint paths add to $0.9+0.1(0.8)(0.9)=0.972$. Ten necessary stages with that conditional success probability yield $0.972^{10}\approx0.7528$. The calculation does not include repeated repairs or failures caused by the repair itself. Those require additional states and conditional probabilities.</p>

<p><strong>5.</strong> Evaluate four conditions on untouched tasks: original weights with original inference, original weights with improved inference, updated weights with original inference, and updated weights with improved inference. Match resource budgets within comparisons and keep the final evaluator fixed and hidden from adaptation. The first contrast estimates an inference-procedure effect; the second isolates a parameter effect under the original procedure. Their interaction reveals whether the update helps specifically with the improved harness. A single before–after score with both changes cannot identify either cause.</p>

</details>

## 8. Primary references and source boundaries

- Kaplan et al., [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361).
- Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556).
- Wei et al., [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903).
- Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155).
- Brown et al., [Large Language Monkeys](https://arxiv.org/abs/2407.21787).

The recording supplies scaling, prompting, post-training, inference scaling, and agent workflows in that order. The probability model, KL-regularized derivation, repair trace, recovery calculation, and controlled experiment are original teaching extensions. The infinite-trial statement is a mathematical result under explicit independence and positive-probability assumptions; it is not a claim of universal model capability.
