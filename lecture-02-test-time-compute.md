# Lecture 2 — Test-time compute, coverage, and adaptive search

*Independent textbook chapter accompanying [CS329A Part 2](https://www.youtube.com/watch?v=-Ggc37xLj_Y). Recording links identify the topic sequence. The median-repair case, calculations, code, and exercises are original teaching constructions; empirical research is attributed where used.*

## 1. A correct candidate is not yet a correct service [00:00](https://www.youtube.com/watch?v=-Ggc37xLj_Y&t=0s)

A coding service must repair a median function. The specification accepts a nonempty finite sequence of finite real numbers, sorts it, returns the middle value for odd length, and returns the arithmetic mean of the two middle values for even length. Returning the upper middle element works on many examples and fails on every even-length input whose middle values differ. Generating a correct implementation somewhere in a large batch does not help the user if the service delivers this defective one.

Let $q$ denote the repair task, including its specification and available evidence. A **candidate** $Y$ is a proposed complete patch. A fixed generator has conditional distribution $\pi(Y\mid q)$ over patches. The expression $c(q,Y)\in\{0,1\}$ denotes true correctness under the specification, with one meaning correct. In general this function is unavailable to the service. For this mathematical experiment, assume an evaluator can determine it after the run. Define the single-attempt success probability $p_q=P(c(q,Y)=1)$, where $P$ denotes probability over generation randomness.

**Repeated sampling** draws $k$ candidates independently from the same distribution, where $k$ is a nonnegative integer. Independence means that learning one sample's outcome does not change the probability distribution of another sample. The **oracle coverage** $C_q(k)$ is the probability that at least one generated candidate is correct. “Oracle” identifies the information required to recognize success; it is not an available component of the deployed service.

There is no correct candidate exactly when every draw fails. Each failure has probability $1-p_q$, and independence allows multiplication. Taking the complementary event gives

$$C_q(k)=1-P(\text{all }k\text{ candidates fail})=1-(1-p_q)^k.$$

At $k=0$, coverage is zero. At $p_q=0$, no finite or infinite sequence of independent attempts from this unchanged distribution can produce success. For $p_q>0$, coverage approaches one as $k$ grows without bound, but this limit says nothing about affordable budgets. With $p_q=0.001$, reaching 90% coverage requires 2,302 attempts. A statement about nonzero support can therefore coexist with practical inability.

### The missing selector

A **selector** $S$ reads the task and candidates and returns an index $I=S(q,Y_1,\ldots,Y_k)$ when $k\ge1$. Its delivered accuracy is $A_q(k)=P(c(q,Y_I)=1)$. With no candidates, no answer is returned and set $A_q(0)=0$. When coverage is positive, let $s_q(k)$ be the conditional probability that selection succeeds given that at least one correct candidate exists. Set $s_q(k)=0$ by convention when coverage is zero; no empirical conditional probability exists on that zero-probability event. The probability chain rule gives

$$A_q(k)=C_q(k)s_q(k),\qquad 0\le A_q(k)\le C_q(k).$$

This factorization does not assume independence between candidate quality and selection. It conditions on availability. The **generation–verification gap** is $G_q(k)=C_q(k)-A_q(k)$; it measures success lost between finding and returning a valid patch.

Suppose the median-repair generator succeeds with probability 0.1 per independent draw. Twenty attempts give $C_q(20)=1-0.9^{20}=0.8784$. If the selector succeeds on 70% of batches containing a valid candidate, delivered accuracy is $0.8784(0.7)=0.6149$. Improving selection to perfection would recover 26.35 percentage points on this hypothetical task. The conditional selection rate need not stay constant when the batch becomes larger. More candidates can expose the selector to more persuasive mistakes.

The [Large Language Monkeys experiments](https://arxiv.org/abs/2407.21787) motivate this distinction by studying repeated generation and the difficulty of turning its coverage into useful selection. Their empirical curves concern particular models and benchmarks. They do not establish that arbitrary errors in a live repository are detectable by the same evaluators. A **fuser**, which writes a new answer from several candidates, changes the output space: its answer need not equal any input candidate, so the selector upper bound does not apply to its output.

### Estimating coverage without pretending to deploy an oracle

Suppose an evaluation generates $n$ independent candidates and marks $c$ of them correct. Here $c$ is a count, distinct from the earlier correctness function $c(q,Y)$. For an integer $k$ satisfying $0\le k\le n$, let $\binom nk$ denote the number of subsets of size $k$ from $n$ items. Of all such subsets, $\binom{n-c}{k}$ contain only failures. Therefore the average success indicator across these subsets is

$$\widehat{\operatorname{pass@}k}=1-\frac{\binom{n-c}{k}}{\binom nk}.$$

The numerator is zero if fewer than $k$ failures exist. For two successes among ten candidates, estimated pass@3 is $1-56/120=0.5333$. This estimator, used in [the Codex evaluation paper](https://arxiv.org/abs/2107.03374), is unbiased for independent identical sampling of candidates. The subset calculation conditional on the observed batch does not itself require independence; the interpretation as an estimator of future independent sampling does.

A service cannot obtain the same performance merely by computing the estimator. It still needs a selector. Keep the offline correctness labels unavailable to selection, or the evaluation silently measures a more informed system than the one being deployed.

## 2. Why easy examples disappear and hard examples remain [08:00](https://www.youtube.com/watch?v=-Ggc37xLj_Y&t=480s)

The repair queue contains more than median bugs. Some patches require changing one index; others require understanding an undocumented numerical convention. Let $p$ denote a task's single-attempt success probability, and suppose tasks are drawn from a distribution with density $f(p)$ on the interval $[0,1]$. A density assigns probability mass by integration; its integral over the full interval equals one. Average failure after $k$ attempts per task is

$$F(k)=\int_0^1(1-p)^k f(p)\,dp.$$

The integration averages the fixed-task failure probability over task difficulty. It must occur after exponentiation. Substituting the average success probability into the fixed-task formula loses the difficult tail of the population.

For a transparent example, assume $p$ is uniformly distributed, meaning $f(p)=1$ throughout $[0,1]$. Integration gives

$$F(k)=\left[-\frac{(1-p)^{k+1}}{k+1}\right]_{p=0}^{p=1}=\frac1{k+1}.$$

Every task with fixed positive $p$ becomes exponentially unlikely to remain unsolved, yet population failure falls only as the reciprocal of the budget. After ten attempts, the actual failure rate is $1/11=0.09091$. Replacing the task distribution by its mean probability, 0.5, instead predicts $0.5^{10}=0.000977$. The error is nearly a factor of 93. The remaining failures are disproportionately hard tasks, so the original average task is no longer representative of them.

### Deriving the difficult-tail mechanism

An **asymptotic equivalence** $u(k)\sim v(k)$ means their ratio tends to one as $k$ tends to infinity. Suppose the difficulty density behaves as $f(p)\sim ap^{\alpha-1}$ near zero, where $a>0$ and $\alpha>0$ are constants. The **gamma function** is defined for positive $\alpha$ by $\Gamma(\alpha)=\int_0^\infty e^{-u}u^{\alpha-1}\,du$. This integral will appear when the hard-task region is rescaled.

For small $p$, the natural logarithm satisfies $\log(1-p)=-p+O(p^2)$; the remainder notation means its magnitude is bounded by a constant times $p^2$ near zero. Thus $(1-p)^k$ is close to $e^{-kp}$ in the region $p$ of order $1/k$. Tasks with probabilities bounded away from zero make exponentially small contributions. In the remaining region, substitute $u=kp$, so $p=u/k$ and $dp=du/k$. Under regularity conditions permitting this limiting approximation,

$$F(k)\sim a\int_0^\infty e^{-kp}p^{\alpha-1}\,dp
=a k^{-\alpha}\int_0^\infty e^{-u}u^{\alpha-1}\,du
=a\Gamma(\alpha)k^{-\alpha}.$$

The consequential step is the factor $k^{-\alpha}$ introduced by the change of scale. Increasing the budget keeps moving attention to harder tasks near $p=1/k$; the amount of probability mass there controls the population curve. This is the distributional explanation developed in [How Do Large Language Monkeys Get Their Power (Laws)?](https://arxiv.org/abs/2502.17578).

That paper discusses the negative logarithm of average success as well as failure. These are not identical at finite budget. If failure $F$ is small, $-\log(1-F)=F+O(F^2)$, so they have the same leading asymptotic behavior. Confusing the two quantities outside this regime can produce an apparently excellent but incorrectly interpreted scaling fit.

### What a curve cannot certify

Suppose a fraction $w\in[0,1]$ of tasks has exactly zero success probability. Those tasks contribute $w$ to average failure forever. The remaining population may still follow a power law over the budgets that were measured. Extrapolating that finite range to zero failure would erase a genuine floor.

Even distinguishing zero from very small probability is difficult. If a task produces no success in $n$ independent attempts, the observation has probability $(1-p)^n$ for any hypothesized $p$. Inverting the one-sided 5% tail gives the upper bound $p\le1-0.05^{1/n}$ at the corresponding confidence level. For $n=100$, the upper bound is about 0.0295. Zero observed successes do not establish zero support.

For the median service, group tasks by mechanisms that matter: parity mistakes, mutation of caller data, overflow conventions, and unsupported input types. Such stratification may reveal why the aggregate curve changes. Do not invent a mechanistic explanation solely because a log–log plot looks straight. Temperature changes, repeated prompts, and feedback also alter the sampling distribution, invalidating the fixed-$p$ interpretation unless modeled separately.

## 3. Verification is a statistical decision with a specification [12:00](https://www.youtube.com/watch?v=-Ggc37xLj_Y&t=720s)

A **verifier** maps a candidate and task to a score or decision. An executable test runner observes behavior on selected inputs. A formal proof checker checks a derivation against a formal specification. A learned verifier estimates quality from examples. Each answers a different question. Passing a finite test suite does not prove correctness on an infinite input domain; proving a formal specification does not establish that it captures the user's intended behavior.

For the median repair, an odd-length test such as `[1, 2, 100]` returns 2 under both the correct implementation and the upper-middle bug. Repeating that input adds no evidence. The even-length input `[1, 9]`, whose required answer is 5, separates them immediately. Verification quality depends on discrimination between plausible failure mechanisms, not just the number of tests executed.

If a defective implementation fails independently on a fraction $r\in(0,1)$ of inputs drawn from a stated test distribution, then $m$ random tests miss the defect with probability $(1-r)^m$. To make this at most a target $\delta\in(0,1)$, take logarithms and divide by the negative number $\log(1-r)$, reversing the inequality:

$$m\ge\frac{\log\delta}{\log(1-r)}.$$

At $r=0.01$ and $\delta=0.01$, at least 459 tests are required. This is a conditional detection calculation, not a universal software guarantee. If all tests use odd-length lists, the relevant failure fraction under that test distribution is zero and no sample count detects the even-length bug.

### Why voting and scoring can disagree with truth

**Majority voting** selects the most common normalized answer. It estimates a mode of the generator's answer distribution. Consider answer probabilities 0.45 for the upper-middle patch, 0.35 for a lower-middle patch, and 0.20 for the correct patch. As the sample count grows, observed proportions approach these probabilities and the vote favors the wrong upper-middle patch. Oracle coverage nevertheless approaches one.

The failure is not random noise that more voting removes. The generator's most probable answer is wrong. [Compute-allocation experiments by Snell and colleagues](https://arxiv.org/abs/2408.03314) study approaches beyond simple repetition, including revision and verifier-guided search. Their task-dependent results are a reason to measure a selection mechanism, not a guarantee that a learned verifier supplies truth.

A score threshold has its own base-rate problem. Let $p$ be the prevalence of correct candidates, $t$ the probability of accepting a correct candidate, and $f$ the probability of accepting a wrong candidate. These last two quantities are the true-positive and false-positive rates. If any candidates are accepted, the probability that an accepted candidate is correct is

$$P(\text{correct}\mid\text{accepted})=\frac{pt}{pt+(1-p)f}.$$

The numerator is the joint probability of correctness and acceptance. The denominator adds all routes to acceptance. At $p=0.02$, $t=0.9$, and $f=0.05$, accepted-candidate precision is only 26.87%. Most accepted candidates are wrong despite a 90% true-positive rate. Chapter 3 develops this problem further; it is already enough to show why impressive aggregate verifier accuracy can be misleading.

### Selection changes the distribution being verified

A verifier measured on ordinary candidates will face a different distribution when an optimizer searches for its highest scores. Suppose each wrong candidate independently has probability $f=0.01$ of receiving a misleadingly high score. Among 100 wrong candidates, the probability of at least one such event is $1-0.99^{100}=0.6340$. This calculation counts opportunities for false acceptance; it does not by itself determine whether that candidate outranks a correct one.

For the repair service, keep a held-out evaluation suite outside the adaptive search loop. A visible test may guide repair and still be useful, but repeated optimization against it makes its passing rate training feedback rather than independent evidence. Record which tests influenced selection. When a candidate fails a new boundary case, the failure should update the development suite for future versions while a separate untouched set remains available for evaluation.

## 4. Width, depth, and the next useful unit of compute [26:00](https://www.youtube.com/watch?v=-Ggc37xLj_Y&t=1560s)

**Parallel width** is the number of independently initiated candidate branches. **Sequential depth** is the number of dependent revision stages within a branch. Revision conditions a new proposal on previous proposals and feedback, so its success probability cannot be obtained by substituting its step count into the independent-sampling formula.

Let $n$ be the branch count, $d$ the stages per branch, and $c_g,c_v>0$ the generation and verification cost per stage in common resource units. Under constant costs, a budget $B$ permits $nd(c_g+c_v)\le B$. This is an accounting approximation. Shared prefixes, variable output length, batching, and unequal tool costs require a richer cost model.

Let $p_d$ be the probability that a branch produces an acceptable result after its $d$ dependent stages, including its internal selection rule. If branches are independent, at least one branch succeeds with probability $1-(1-p_d)^n$. A final imperfect selector can still reduce delivered success. The response curve $d\mapsto p_d$ must be measured or estimated; extra tokens do not determine it.

Suppose twelve generation-and-check stages are affordable. Twelve one-step branches with $p_1=0.1$ have coverage $1-0.9^{12}=0.7176$. Four three-step branches with $p_3=0.4$ have coverage $1-0.6^4=0.8704$. If instead $p_3=0.15$, the latter falls to $1-0.85^4=0.4780$. Feedback quality changes the preferred allocation even when the total step budget is identical.

### Work and latency are separate resources

With enough workers, twelve independent branches can finish in roughly one stage's latency; three dependent stages require roughly three stage latencies. These idealized statements assume no queueing or shared bottleneck. Total work may be equal while response time differs substantially. A service answering an interactive query can prefer lower latency even when a deeper search offers higher eventual success.

The experiments in [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) compare compute strategies conditional on estimated problem difficulty. Difficulty is not directly observed at deployment. If estimating it requires additional model calls, include their cost. If an offline experiment uses known correctness to estimate difficulty, label that information advantage when interpreting its results.

### Deriving a budget allocation rule

Consider several independent tasks indexed by $i$, each with known single-attempt probability $p_i$. Let $k_i$ be its allocated attempt count. The objective is the sum of oracle coverage across tasks, all tasks have equal importance, and every attempt costs one unit. One additional sample changes task $i$'s coverage by

$$\Delta_i(k_i)=[1-(1-p_i)^{k_i+1}]-[1-(1-p_i)^{k_i}]
=p_i(1-p_i)^{k_i}.$$

Each task's marginal gains form a nonincreasing sequence. Choosing the largest currently available marginal gain is optimal for this separable integer-budget problem: a later gain on a task cannot be larger than its earlier gains, so selecting the globally largest gains automatically respects the prerequisite order. An exchange argument replaces any smaller chosen gain with a larger feasible unchosen one until no improvement remains.

For $p_1=0.2$ after three attempts, the next gain is $0.2(0.8)^3=0.1024$. For a new task with $p_2=0.05$, it is 0.05. The first task gets the next attempt under the stated objective. A fairness requirement, unequal task values, uncertain probabilities, or an unreliable selector changes the optimization problem.

The following implementation extends the existing [numerical lab](numerical_lab.py). It accepts known probabilities and an integer budget, returns counts in input order, and has no external effects. The imported helper checks the probability domain. A zero budget returns zero allocations; an empty task collection cannot receive a positive budget.

```python
from numerical_lab import allocate_samples, coverage

allocation = allocate_samples([0.2, 0.05], 4)
assert allocation == [4, 0]
assert abs(coverage(0.2, 4) - 0.5904) < 1e-12
```

The answer may seem unfair: all four samples go to the easier task. That is precisely what maximizing expected number of covered tasks can require. This small check exposes a policy choice hidden inside an apparently neutral “optimal compute” rule. It does not recommend that rule for allocating service across people.

## 5. Searching the inference architecture [45:00](https://www.youtube.com/watch?v=-Ggc37xLj_Y&t=2700s)

An inference system can contain generators, critics, rankers, fusers, and executable checks. A **directed acyclic graph** represents operations as nodes and data dependencies as directed edges without a directed cycle. A repair graph might generate four patches, test them, ask a critic to explain failures, revise two survivors, and rank the final pair. Changing this graph changes what each component can observe and how much computation it consumes.

Let $a$ denote an architecture from a set $\mathcal A$ of permitted graphs. Let $Q(a)$ be its expected delivered quality on a specified task distribution, and let $C(a)$ be expected cost in stated units. For a budget $B$, the architecture-selection problem is

$$a^*\in\operatorname*{arg\,max}_{a\in\mathcal A,\ C(a)\le B}Q(a).$$

The operator returns one or more maximizing architectures. In experiments, $Q$ is estimated from finite validation tasks. Searching more graphs can improve the best observed score while also increasing selection bias. The graph must be evaluated on fresh tasks after development choices are fixed.

[Archon](https://arxiv.org/abs/2409.15254) makes inference composition an explicit search space. Its components supply a concrete vocabulary for varying the graph. The durable methodological idea is to optimize a system of inference operations under a budget; a historical benchmark improvement is not a universal coefficient that can be inserted into the median service's forecast.

### Dominance and lifecycle cost

An architecture is **Pareto-dominated** when another has at least equal quality and no greater cost, with strict improvement in at least one coordinate. Suppose architectures A, B, and C have illustrative cost–accuracy pairs $(2,0.70)$, $(4,0.78)$, and $(5,0.76)$. B dominates C. Neither A nor B dominates the other. Adding latency as a third coordinate may restore C to consideration if it runs much faster.

Search itself also costs resources. Let $C_{\rm search}$ be the development cost, $N>0$ the number of future requests served, and $C_{\rm run}$ the average cost per request. Amortized lifecycle cost per request is $C_{\rm run}+C_{\rm search}/N$. A costly architecture search can be sensible for a widely reused service and wasteful for one disposable task. Keep development cost separate before deciding whether amortization is appropriate.

The bracket-checking example exposes a deeper boundary than optimization. For strings whose alphabet contains only opening and closing parentheses, odd length makes balance impossible. For strings that may contain letters ignored by the parser, the string `a()` is balanced and has odd total length. A test generator cannot settle this disagreement until the input language is specified. Optimizing agreement among generator, critic, and judge can strengthen a shared mistake about that language.

For the median service, freeze the accepted input domain, mutation policy, and error behavior before measuring architectures. A correct mathematical median over real numbers is not automatically a correct floating-point implementation under overflow, missing values, or empty inputs. Those are specification choices rather than details an architecture search should silently invent.

## 6. Limits and experimental design

The useful object is a joint record of generation, selection, cost, and delay. Oracle coverage diagnoses whether a successful candidate was available. Delivered accuracy diagnoses the complete decision procedure. Verifier error identifies why the two differ. None can replace the others.

A comparison of width and depth should use the same task split, model version, final correctness criterion, and total resource accounting. Log public feedback separately from held-out outcomes. Repeat stochastic runs and report uncertainty at the task level; generating many candidates for one task does not create many independent tasks. Record when a method times out or exhausts its budget rather than dropping those runs.

The derivations use fixed distributions, independent branches, and simplified costs. They explain mechanisms and yield testable predictions under those assumptions. They do not show that today's model has nonzero probability on every desired answer, that its verifier remains calibrated after optimization, or that a curve extrapolates beyond the measured budget. For the running case, the strongest immediate improvement may be one discriminating even-length test rather than another thousand generated patches.

## 7. Exercises and solutions

1. Derive the smallest number of independent attempts needed for 95% coverage when single-attempt success is 0.02. Explain what changes when half the attempts are exact copies of earlier samples.
2. Compute pass@3 from ten generated patches containing two correct patches. Explain why this number cannot be used as the service's accuracy without additional evidence.
3. Derive population failure for uniformly distributed task probabilities. Add a 10% mass of impossible tasks and compute the failure after nine attempts on each task.
4. Compare two equal-cost repair designs: twelve independent one-step branches with success 0.1, and four independent three-step branches with success 0.4. Give a selection-quality condition that reverses their oracle ranking.
5. Audit an architecture search whose developer reports only the best score among 200 graphs on one benchmark. Design a minimum credible follow-up evaluation.

<details><summary>Fully worked solutions</summary>

<p><strong>1.</strong> Require $1-0.98^k\ge0.95$, so $0.98^k\le0.05$. Taking natural logarithms gives $k\log(0.98)\le\log(0.05)$. Division by the negative logarithm reverses the inequality, yielding $k\ge148.284\ldots$. The smallest integer is 149. Exact duplicates provide no new success event: if each independent patch is copied once, 150 submitted patches contain only 75 independent opportunities, whose coverage is $1-0.98^{75}\approx0.7802$. The raw number of outputs is not the number of independent draws.</p>

<p><strong>2.</strong> There are $\binom{10}{3}=120$ three-patch subsets and $\binom83=56$ all-failure subsets. Therefore estimated coverage is $64/120=8/15$. This is a probability of availability under the sampling interpretation. To infer delivered accuracy, separately estimate the probability of choosing a correct patch when one exists. For example, conditional selection accuracy 0.75 would give $(8/15)(0.75)=0.4$, subject to the same evaluation distribution.</p>

<p><strong>3.</strong> The antiderivative of $(1-p)^k$ is $-(1-p)^{k+1}/(k+1)$, so uniform-mixture failure is $1/(k+1)$. With impossible-task mass 0.1 and the remaining 0.9 distributed uniformly, failure is $0.1+0.9/(k+1)$. At $k=9$, it equals 0.19. The limiting failure is 0.1, not zero. This construction separates an observed improvement trend from an assumption about eventual solvability.</p>

<p><strong>4.</strong> The two coverages are approximately 0.7176 and 0.8704. Write their conditional selection accuracies as $s_1$ and $s_3$. The deeper design loses when $0.8704s_3<0.7176s_1$, or $s_3/s_1<0.8245$. If $s_1=0.9$ and $s_3=0.7$, delivered success is about 0.6458 versus 0.6093. The deeper design produces more successful batches but delivers fewer successful answers. Equal stage counts also do not imply equal latency.</p>

<p><strong>5.</strong> Freeze the chosen graph and its prompts, thresholds, model versions, and stopping rules. Evaluate it once on an untouched task set alongside the original baseline at matched cost, with repeated runs where stochasticity matters. Report delivered correctness, coverage if an evaluator can establish it, latency, resource use, and failure categories. Include all development evaluations in the search-cost total. A held-out improvement supports the selected graph on that population; it does not prove it is globally optimal among all possible graphs.</p>

</details>

## 8. Primary references and source boundaries

- Brown et al., [Large Language Monkeys](https://arxiv.org/abs/2407.21787): repeated-sampling experiments and the gap between coverage and selection.
- Schaeffer et al., [How Do Large Language Monkeys Get Their Power (Laws)?](https://arxiv.org/abs/2502.17578): the distributional explanation of aggregate scaling.
- Snell et al., [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314): difficulty-dependent revision and search strategies.
- Saad-Falcon et al., [Archon](https://arxiv.org/abs/2409.15254): inference architecture composition and search.
- Chen et al., [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374): code evaluation and pass@$k$ estimation.

The recording supplies repeated sampling, task heterogeneity, verification, sequential revision, and architecture search in that order. The median example, resource-allocation proof, numerical comparisons, and experimental-design exercises are teaching extensions. Power-law mechanisms and named system designs are researched extensions supported by the linked papers. All numerical parameters in the repair case are illustrative, not measured model performance.
