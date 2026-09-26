# Lecture 8 — Evaluating agents: reliability, professional work, and evidence

*Independent textbook chapter accompanying [CS329A Part 8](https://www.youtube.com/watch?v=8JAqLnTaZu4). The recording supplies METR task horizons, GDPval, and DeepScholar-Bench. The repair-service experiment, numerical horizon model, reliability calculations, and exercises are original teaching constructions.*

## 1. Evaluate the configured system and the decision it supports [00:00](https://www.youtube.com/watch?v=8JAqLnTaZu4&t=0s)

The median-repair service passes 90% of a development suite. That number does not yet answer whether it can be trusted with a repository. Were tasks sampled from deployment? Did the agent see the tests? Were failed runs excluded? How many attempts and human interventions were allowed? A score without these conditions describes an incomplete experiment.

A **system configuration** includes the model version, prompts, tools, memory, inference budget, stopping rule, and evaluator. Let $x_i$ be task $i$, $s$ a fixed configuration, and $Y_i(s)$ its observed outcome. The outcome can be binary correctness or a vector of properties. Randomness in generation, tools, and evaluation means repeated runs can differ even on the same task.

Let $\mathcal D$ be the target task distribution. The estimand—the quantity the experiment aims to estimate—might be $A(s)=\mathbb E_{x\sim\mathcal D}[P(\text{success}\mid x,s)]$. This averages success probability over a specified population. A benchmark samples or constructs tasks intended to represent some part of that population. It does not automatically represent all work a user might delegate.

For $n$ independent tasks with binary outcomes $Y_i\in\{0,1\}$, the sample mean is $\widehat A=n^{-1}\sum_iY_i$. If outcomes are identically distributed Bernoulli variables with success probability $A$, its variance is $A(1-A)/n$. An estimated standard error is $\sqrt{\widehat A(1-\widehat A)/n}$. Correlated tasks, repeated variants, or selected examples can invalidate that simple uncertainty calculation.

At 90 successes in 100 independent tasks, the estimated standard error is 0.03. A rough normal interval spans about 0.841 to 0.959. This approximation is not ideal near zero or one, and it does not account for benchmark selection bias. It is still enough to show that a small difference in headline percentages may not establish a meaningful system improvement.

### Paired comparisons preserve task difficulty

When comparing configurations A and B, run both on the same tasks and define $D_i=Y_i(B)-Y_i(A)$. The average difference estimates the paired performance change. Its uncertainty depends on the distribution of $D_i$, especially tasks where the systems disagree. Treating the two means as independent throws away pairing information.

For the repair service, log correctness, runtime, tokens, tool calls, and required human repair separately. A system may improve correctness while doubling delay or increasing review burden. Combining them into one utility score requires an application-specific decision rule. Report the raw dimensions first.

### Work through the disagreement table

Let $b$ be the number of tasks solved by B but not A and $c$ the number solved by A but not B. On the other tasks, paired difference $D_i$ is zero. Thus mean change is $(b-c)/n$, and mean squared difference is $(b+c)/n$. An unbiased sample variance of the differences is

$$s_D^2=\frac{(b+c)-n[(b-c)/n]^2}{n-1},\qquad n>1.$$

The estimated standard error of the mean difference is $s_D/\sqrt n$. This follows by summing squared deviations around the mean and using the independent-task variance rule. It is a large-sample diagnostic; small discordant counts call for an exact paired analysis or a carefully justified resampling method.

For one hundred tasks, suppose B uniquely solves twenty and A uniquely solves ten. The mean gain is 0.10. The variance estimate is $(30-1)/99\approx0.2929$, giving standard error about 0.0541. A rough 95% normal interval is approximately -0.006 to 0.206. The observed ten-point gain is promising but not decisive under this approximation. Generating more samples on the same tasks can reduce run noise, but it does not replace collecting more independent tasks from the intended population.

### Missing outcomes are outcomes

A timeout, budget exhaustion, malformed artifact, or unrecoverable tool error should appear in the result table. Dropping incomplete runs can inflate success. If a task is genuinely unevaluable, report why and distinguish that from an agent failure. The denominator is part of the claim.

## 2. Human-duration horizons measure task difficulty, not runtime [05:00](https://www.youtube.com/watch?v=8JAqLnTaZu4&t=300s)

A **task horizon** summarizes the human-baseline duration of tasks an agent can complete at a specified reliability. Human duration is used as a proxy for task difficulty or scope. It is not the number of minutes the agent runs, and it is not a direct estimate of labor replaced.

[The original METR long-task study](https://arxiv.org/html/2503.14499v1) fits success against human task duration for its evaluated task collection. This link is pinned to version 1 so later revisions are not silently treated as the methodology discussed in the recording. A horizon depends on the task suite, human baselines, model configuration, fit, and chosen reliability threshold.

Let $t>0$ be human-baseline task duration in minutes. Let $h_{50}>0$ be the duration at which modeled success is 50%, and let $\beta>0$ be a slope parameter. Define the logistic function $\sigma(z)=1/(1+e^{-z})$. An illustrative model consistent with a logistic fit in log duration is

$$p(t)=\sigma\left(\beta\log\frac{h_{50}}{t}\right).$$

The ratio inside the logarithm is dimensionless because both durations use the same units. At $t=h_{50}$, the argument is zero and success is 0.5. Shorter tasks give a positive argument and higher success; longer tasks give lower success. Larger $\beta$ makes the transition steeper.

### Deriving a reliability-specific horizon

Let $r\in(0,1)$ be the required success probability and $h_r$ the corresponding duration. The **logit** of a probability is $\operatorname{logit}(r)=\log[r/(1-r)]$. Inverting the logistic model gives

$$\operatorname{logit}(r)=\beta\log\frac{h_{50}}{h_r},
\qquad h_r=h_{50}\exp\left[-\frac{\operatorname{logit}(r)}{\beta}\right].$$

The negative sign is consequential: requiring reliability above 50% shortens the horizon. With illustrative values $h_{50}=60$ minutes and $\beta=1$, the 80% horizon is $60e^{-\log4}=15$ minutes. The 95% horizon is $60/19\approx3.16$ minutes. Reporting only the one-hour 50% horizon hides how much shorter the task must be under a stricter reliability requirement.

These numbers are a teaching example, not a fitted result for any named model. The [numerical lab](numerical_lab.py) implements the inversion. It requires positive finite duration and slope and reliability strictly between zero and one, returns minutes, and makes no network or model calls.

```python
from numerical_lab import horizon

assert abs(horizon(60, 1, 0.8) - 15) < 1e-12
assert abs(horizon(60, 1, 0.95) - 60 / 19) < 1e-12
```

### Horizon uncertainty grows away from the observed center

Write $\eta=\log h_{50}$ and $z=\operatorname{logit}(r)$. Then $\log h_r=\eta-z/\beta$. A small perturbation in fitted parameters changes log horizon approximately by $d\log h_r=d\eta+(z/\beta^2)d\beta$. Here $d$ denotes a small differential, not task duration. The derivative exposes sensitivity to both the center and the slope.

At reliability one half, $z=0$, so slope uncertainty does not affect the horizon definition. At high reliability, $z$ grows and uncertainty in $\beta$ matters more. With $\beta=1$ and $r=0.95$, $z\approx2.944$. A slope perturbation of 0.1 alone changes log horizon by roughly 0.294, corresponding to about a 34% multiplicative change under this first-order approximation.

The parameters are generally statistically correlated, so their uncertainties should not be added as if independent without checking the fit. The practical consequence is simple: a precise-looking horizon at an extreme reliability level can be fragile when the data poorly determine the slope. Report uncertainty from the fitted model and task sampling, and show observations near the reliability region used for the decision.

### Baseline aggregation and extrapolation

Human durations often vary greatly across people and tasks. A **geometric mean** of positive times $t_1,\ldots,t_n$ is $\exp(n^{-1}\sum_i\log t_i)$. It averages in log space and treats multiplicative differences symmetrically. For ten and ninety minutes, the geometric mean is thirty minutes, while the arithmetic mean is fifty. The choice changes the task's location on the duration axis.

A fitted curve also compresses heterogeneous tasks into one predictor. A ten-minute task requiring hidden context may be harder for an agent than a longer self-contained coding exercise. Residual analysis—examining errors unexplained by duration—can reveal such structure. A single horizon should be accompanied by task composition and fit uncertainty.

Extrapolating beyond observed durations or projecting future horizons from a historical trend adds another layer of assumptions. A good fit on the current suite does not establish that longer real-world tasks have the same failure mechanisms. The horizon is an informative measurement, not a universal clock for automation.

## 3. Reliability requires recovery and state management [20:00](https://www.youtube.com/watch?v=8JAqLnTaZu4&t=1200s)

A long task can fail through an early misunderstanding, lost evidence, repeated ineffective actions, or premature stopping. Longer context can help preserve information, but it does not ensure correct use or recovery. The evaluation should identify which mechanism failed rather than treating every failure as insufficient context length.

Let $s_i$ be the probability that stage $i$ succeeds conditional on all preceding necessary stages succeeding. For $T$ stages, the chain rule gives total success $\prod_{i=1}^{T}s_i$. Independence is unnecessary because these are conditional probabilities. If every stage has conditional success 0.98, twenty stages succeed with probability $0.98^{20}\approx0.6676$.

### Recovery changes the per-stage probability

Let $s$ be initial stage success, $d$ failure-detection probability conditional on failure, and $r$ repair success conditional on detection. With at most one repair and no regression of already successful stages, effective stage success is $s+(1-s)dr$.

For $s=0.98$, $d=0.9$, and $r=0.8$, this becomes $0.9944$. Twenty stages then succeed with probability $0.9944^{20}\approx0.8938$. The improvement depends on an operational detector and repair procedure. A model merely claiming that it checks its work does not supply either measured probability.

For the median service, the detector is a test exposing an incorrect branch; repair changes the implicated code; regression tests check that the correction did not break earlier behavior. Logging these transitions permits direct estimation of detection, repair, and regression rates.

### Repeated failure is not independent persistence

An agent can repeatedly choose the same action after the same failure because its state representation does not encode the failed hypothesis or because its policy still ranks that action highest. Ten retries of an unchanged deterministic operation do not give ten independent chances of success.

A recovery state should distinguish the last attempted action, its observed failure, the current hypothesis, and the next action expected to provide different evidence. Detecting an identical-state loop can terminate an unproductive run and report an unresolved issue. That may reduce nominal completion counts while improving honest reliability reporting.

Memory can fail by omission or corruption. An omitted mutation constraint can make a numerically correct patch unacceptable; a corrupted test result can make an incorrect patch appear verified. Evaluate continuation after compaction or memory retrieval using downstream tasks that require the retained facts. Summary fluency is not the outcome of interest.

### Reliability is conditional on the allowed assistance

Human hints, approval interventions, and manual recovery can materially change success. An assisted-agent score should identify those interventions. A system that succeeds after an expert supplies the missing algorithm is different from one that discovers it independently.

For practical delegation, assistance may be entirely appropriate. The measurement should reflect the intended workflow: autonomous completion, completion with review, or accelerated human work. These are different estimands. A benchmark cannot choose the right one without a deployment question.

## 4. Professional output quality requires a different evaluator [27:00](https://www.youtube.com/watch?v=8JAqLnTaZu4&t=1620s)

A correct program has a relatively sharp execution criterion. A professional report, spreadsheet, or presentation can vary in completeness, usability, style, and factual quality. **Pairwise evaluation** compares two artifacts under a rubric, often asking an expert which better satisfies a task.

[GDPval](https://arxiv.org/abs/2510.04374) evaluates outputs on specified economically relevant professional tasks. Its comparisons provide evidence about artifacts produced from those task packages. They do not automatically establish that an agent can perform an entire occupation, obtain unstated organizational context, negotiate requirements, or handle every exception in a live workflow.

Let $W$, $T$, and $L$ denote counts of wins, ties, and losses for an agent in $n=W+T+L$ comparisons. Strict win rate is $W/n$. Win-or-tie rate is $(W+T)/n$. A half-tie score is $(W+0.5T)/n$. These answer different questions and should not be interchanged.

If the counts are 30 wins, 40 ties, and 30 losses, strict win rate is 30%, win-or-tie is 70%, and half-tie score is 50%. The same experiment can therefore produce very different headline percentages. Name the metric and denominator beside the result.

### A rubric should expose the reason for preference

An expert may prefer a polished artifact despite a hidden factual error, or reject a correct one because it is difficult to use. Preserve component judgments when feasible: factual correctness, completeness, requested format, usability, and unnecessary content. A single preference label can train or compare systems, but it hides which property changed.

For the median repair, the professional deliverable includes the patch, tests, and explanation of residual limits. A technically correct patch with no integration instructions may be less useful than a slightly slower but complete repair package. A clear report falsely claiming exhaustive tests should still fail factual evaluation.

Blinding evaluators to system identity reduces one source of bias. It does not eliminate preferences for length, formatting, or familiar style. Randomize presentation order and inspect disagreement among evaluators. Disagreement can reveal ambiguous task requirements rather than mere annotator error.

### From artifact quality to economic utility

Let $p$ be the probability a delivered artifact is accepted without substantial repair, $V$ the value of such an artifact, $C$ generation cost, $R$ expected review cost, and $L$ expected loss conditional on an unacceptable artifact. All quantities except $p$ use compatible monetary or utility units. A simple expected net value is

$$U=pV-C-R-(1-p)L.$$

This is a decision model, not a benchmark metric. If $p=0.8$, $V=100$, $C=5$, $R=20$, and $L=50$, net value is $80-5-20-10=45$. Reducing review cost can matter as much as raising a leaderboard score. Conversely, rare high-consequence errors can make an apparently strong average unacceptable.

The calculation also explains why occupational replacement cannot be inferred directly from artifact win rates. Real work includes selecting tasks, obtaining context, coordinating with people, reviewing outputs, and bearing consequences. Those activities must be measured or explicitly excluded from the claim.

## 5. Research synthesis needs support and completeness [51:00](https://www.youtube.com/watch?v=8JAqLnTaZu4&t=3060s)

A research answer can be fluent, well cited, and still omit the central finding. It can also cite real papers that do not support its claims. **Citation validity** asks whether a reference exists and is accessible. **Citation support** asks whether the cited passage supports the attached claim. **Coverage** asks whether the answer includes the important facts needed for the question. These properties require separate evaluation.

[DeepScholar-Bench](https://arxiv.org/abs/2508.20033) targets generative research synthesis and evaluates multiple aspects of evidence use and output quality. Its motivation fits a research agent that must retrieve, integrate, and support information, rather than simply retrieve a document containing a known short answer.

Let $C=\{c_1,\ldots,c_m\}$ be substantive claims in a report. Let $s_j\in\{0,1\}$ indicate whether claim $c_j$ is adequately supported by its cited evidence. A simple support rate is $m^{-1}\sum_js_j$. Let $F=\{f_1,\ldots,f_n\}$ be a reference set of required facts and let $r_i\in\{0,1\}$ indicate whether fact $f_i$ is correctly covered. Fact recall is $n^{-1}\sum_ir_i$.

These are teaching definitions rather than a claim to reproduce every benchmark's exact scoring implementation. A report can score perfectly on support by making only one narrow, well-supported statement while missing most required facts. It can have high recall while adding unsupported claims. Both dimensions matter.

### Weighting evidence by consequence

Not every claim has equal importance. Let $w_j>0$ be a consequence weight for claim $j$. Weighted support is $\sum_jw_js_j/\sum_jw_j$. The weights express an evaluation policy and should be specified before inspecting system outputs. Otherwise the evaluator can retrospectively favor one system by changing what counts as important.

For the median-repair report, the core claims are that even-length behavior is fixed, inputs remain unchanged, and stated tests were actually run. A minor stylistic sentence should not compensate for an unsupported correctness claim. An evidence ledger ties each major assertion to a test result or source passage and makes the scoring unit explicit.

### A worked evidence audit

Suppose a report has four substantive claims with consequence weights $(3,3,2,1)$. Three are supported, but the second weight-three claim is unsupported. Weighted support is $(3+2+1)/9=2/3$. Unweighted support is $3/4$. The difference reveals that the unsupported claim is central rather than peripheral.

Now suppose the required fact set has five items and the report covers only three. Fact recall is 0.6. Neither its 75% unweighted support nor its polished prose repairs the missing coverage. An evaluator should report the pair rather than hide the tradeoff in an unexplained composite score.

The following code checks the arithmetic. Inputs are manually audited support labels and predeclared weights; output is a scalar summary. It does not automate semantic entailment or decide which facts are required.

```python
weights = [3, 3, 2, 1]
supported = [1, 0, 1, 1]
weighted_support = sum(w * s for w, s in zip(weights, supported))
weighted_support /= sum(weights)
assert abs(weighted_support - 2 / 3) < 1e-12
assert sum(supported) / len(supported) == 0.75
```

### Merging research can lose attribution

When several workers summarize sources, the final synthesis may detach a claim from the passage that supported it. A citation attached to the nearest paragraph is not necessarily the source of every statement in that paragraph. Preserve claim–source relationships through merging, then audit the final text rather than assuming the intermediate notes guarantee it.

A primary-source reference list is necessary for verification but insufficient for support. The evaluator must inspect the actual claim, passage, scope, and qualifiers. Numerical improvements should retain their baseline, denominator, and experimental conditions. A correct number with a changed denominator is a different claim.

## 6. Limits and a minimum credible evaluation package

A useful evaluation package identifies the frozen configuration, task selection, assistance policy, resource budget, success rubric, all outcomes, uncertainty, and known exclusions. It records source and artifact versions so another evaluator can reconstruct what was assessed.

For the repair service, use untouched task families and paired configurations. Report successful repairs, regressions, timeouts, review effort, and costs. For its research reports, separately audit support and coverage. A task horizon can summarize difficulty, while professional preference and citation metrics answer different questions about the output.

These measures complement rather than replace one another. No single score establishes general intelligence, universal reliability, or economic substitution. The value of a benchmark lies in the decision it informs and the limits it makes visible.

## 7. Exercises and solutions

1. Under the illustrative logistic model with $h_{50}=120$ minutes and $\beta=2$, derive the 80% horizon.
2. Two systems each solve 80 of 100 tasks. Explain why their usefulness can differ and why a paired comparison needs the disagreement table.
3. Compute twenty-stage success with initial per-stage success 0.98, detection 0.9, and one-repair success 0.8. State the omitted failure mechanisms.
4. An evaluation has 45 wins, 20 ties, and 35 losses. Compute strict win, win-or-tie, and half-tie scores.
5. Construct a research report with perfect citation support and poor fact recall. Explain why adding more citations alone does not solve the problem.

<details><summary>Fully worked solutions</summary>

<p><strong>1.</strong> The logit of 0.8 is $\log4$. Substitution gives $h_{80}=120\exp[-\log4/2]=120/2=60$ minutes. The duration is the human-baseline length of tasks at the modeled reliability, not agent runtime. The larger slope makes the decline around the 50% horizon steeper, so the 80% horizon is closer to it than when $\beta=1$.</p>

<p><strong>2.</strong> The systems may solve the same 80 tasks or complementary subsets. Equal averages conceal differences by task type, cost, delay, and failure consequence. In a paired table, tasks solved by B but not A contribute +1, and tasks solved by A but not B contribute -1. Their counts determine the distribution of paired differences and reveal whether one system helps on the tasks that matter to deployment.</p>

<p><strong>3.</strong> One-recovery stage success is $0.98+0.02(0.9)(0.8)=0.9944$. Twenty stages with that same conditional success probability yield approximately 0.8938. The model omits false alarms, repairs that damage previously correct work, multiple recovery attempts, and shared failures such as a wrong initial specification. These omissions should be tested before applying the number to a real agent.</p>

<p><strong>4.</strong> The total is 100. Strict win rate is 45%, win-or-tie is 65%, and half-tie score is $(45+10)/100=55\%$. None is the unique correct summary; each encodes a different treatment of ties. A comparison must use the same definition for both systems and explain how ties were assigned.</p>

<p><strong>5.</strong> Suppose a question requires five major findings and the report states only one, with an exact supporting passage. Citation support is 100%, but required-fact recall is 20%. Adding more citations to the same narrow claim does not cover the other four findings. The remedy is targeted retrieval and synthesis for missing facts, followed by support checks for the new claims.</p>

</details>

## 8. Primary references and source boundaries

- METR, [Measuring AI Ability to Complete Long Tasks, version 1](https://arxiv.org/html/2503.14499v1): human-duration task horizons and their methodology.
- [GDPval](https://arxiv.org/abs/2510.04374): evaluation of specified professional work artifacts.
- [DeepScholar-Bench](https://arxiv.org/abs/2508.20033): evidence-aware evaluation of research synthesis.

The recording supplies these evaluation perspectives and their limitations. The logistic example, paired-comparison analysis, recovery model, economic utility equation, and weighted evidence audit are independent teaching constructions. They do not reproduce current benchmark rankings or forecast when occupations will be automated.
