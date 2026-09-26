# Lecture 5 — Planning, tree search, and parallel execution

*Independent textbook chapter accompanying [CS329A Part 5](https://www.youtube.com/watch?v=Ml_fp9XkB8Y). The recording supplies language-agent tree search, SPRINT, and SWiRL. The repair tree, scheduling examples, derivations, and code are original teaching constructions.*

## 1. A plan proposes future interaction [00:00](https://www.youtube.com/watch?v=Ml_fp9XkB8Y&t=0s)

The median-repair agent can patch the even-length branch immediately or inspect the caller contract to determine whether mutation is allowed. The first action may produce a quick passing example; the second may prevent an incomplete repair. Planning asks which sequence is likely to reach the objective under limited evidence and a limited budget.

Let $h$ denote the history of actions and observations, and $a$ an allowed next action. A **search node** represents a history or a state derived from it. A **search edge** represents an action and its resulting observation. A **trajectory** is a path through subsequent interactions. Let terminal reward $R\in[0,1]$ measure success under a specified evaluator. A **policy** $\pi$ chooses actions from histories. Its action value is

$$Q^\pi(h,a)=\mathbb E[R\mid h,a,\text{then follow }\pi],$$

where $\mathbb E$ averages over policy and environment randomness. This is a counterfactual quantity: what would happen if this action were chosen and the continuation policy followed? It is generally unknown. Search estimates it using simulations, executions, learned scores, or combinations of them.

A language model can propose several plausible actions without knowing their downstream values. One branch might inspect edge cases, another run tests, and another change code. [Language Agent Tree Search, or LATS](https://arxiv.org/abs/2310.04406), combines language-model proposals, tree search, feedback, and reflection. Generation expands a decision tree while evaluation guides where to spend subsequent search effort.

### The state must preserve consequential distinctions

Two histories can produce identical visible code while differing in unresolved obligations. One may have verified non-mutation; the other may not. Merging them solely because their source files match loses information that changes whether stopping is justified. A sufficient state retains distinctions relevant to future decisions.

Conversely, retaining every sentence of every previous explanation can make equivalent situations look different. Search then wastes budget revisiting them. A practical state may include the artifact version, known failing tests, passed obligations, pending effects, and remaining budget. It should summarize evidence without inventing certainty.

Consider a small repair tree. The root records the known upper-middle bug. Branch A adds an even-length average. Branch B first checks mutation and discovers sorting in place. A child of A may pass the numeric test but later fail mutation; a child of B may make both corrections together. The value of inspection depends on how often it prevents rework and how costly it is.

### Search scores are not acceptance certificates

A learned value score predicts whether a branch is worth exploring. A promising partial patch can receive a high value while still being wrong. The controller uses estimates to allocate attention; a final verifier checks the completed result against the contract.

This distinction prevents termination merely because a model predicts that its plan will succeed. Prediction can justify the next unit of computation. It cannot establish that an action executed or that its result satisfied the specification. Keep proposed, executed, and verified states distinct throughout the tree.

## 2. Monte Carlo tree search balances estimates and exploration [08:00](https://www.youtube.com/watch?v=Ml_fp9XkB8Y&t=480s)

**Monte Carlo tree search**, abbreviated MCTS, repeatedly selects a node, expands actions, evaluates a continuation, and propagates its score through visited nodes. Monte Carlo refers to sampled estimation. A **backup** updates stored value estimates; it is not a gradient update to the language model.

Let $N(h)$ be visits to parent $h$, $N(h,a)$ visits to action edge $a$, and $\widehat Q(h,a)$ its mean observed return. The **upper confidence bound applied to trees**, or UCT, uses the selection index

$$U(h,a)=\widehat Q(h,a)+c\sqrt{\frac{\log N(h)}{N(h,a)}},$$

for visited edges, where $c>0$ controls exploration and $\log$ denotes natural logarithm. Unvisited edges receive explicit priority rather than division by zero. The mean favors observed success; the bonus favors less-explored alternatives.

### A worked decision and backup

Suppose the parent has twenty visits. A has mean return 0.8 after ten visits; B has mean 0.6 after two visits. With $c=0.5$, A scores $0.8+0.5\sqrt{\log(20)/10}\approx1.0737$, while B scores $0.6+0.5\sqrt{\log(20)/2}\approx1.2119$. Search chooses B despite its lower mean because the exploration bonus is larger.

Scores can exceed one even though returns lie in $[0,1]$. They are selection indices, not probabilities. Interpreting 1.21 as 121% confidence would confuse an exploration rule with a calibrated estimate.

If an edge has mean $Q$ after $n$ observations and receives return $r$, then

$$Q_{\rm new}=\frac{nQ+r}{n+1}=Q+\frac{r-Q}{n+1}.$$

The new observation pulls the mean toward itself with weight $1/(n+1)$. For $Q=0.6$, $n=2$, and $r=1$, the new mean is $0.7333$. Only a search statistic changed; no model parameter changed.

The shared [numerical lab](numerical_lab.py) implements this index with input validation. Its inputs are a finite mean, integer visit counts, and a nonnegative exploration coefficient. It returns infinity for an unvisited edge. It performs no environment action.

```python
from numerical_lab import uct

score_a = uct(0.8, parent_visits=20, visits=10, exploration=0.5)
score_b = uct(0.6, parent_visits=20, visits=2, exploration=0.5)
assert score_b > score_a
assert uct(0.0, 20, 0, 0.5) == float("inf")
```

### A tree iteration should leave an auditable trace

Suppose branch B is chosen by the computed UCT scores. It inspects the contract and discovers non-mutation is mandatory. Expansion proposes two patches: sort in place and restore afterward, or sort a copy. A finite test evaluator gives the first terminal return zero after an exception prevents restoration, and the second return one after all development checks pass. These are observations about two candidate artifacts, not evaluations of two verbal intentions.

The controller records the successful return on every edge traversed for that rollout. If B previously had total return 1.2 from two visits, its new total becomes 2.2 and visit count three, yielding mean 0.7333. Its sibling A remains at its previous mean and count. Updating a sibling as though it had received the same evidence would contaminate the counterfactual comparison.

The root count also increases. Its change affects the exploration bonuses of all children on the next iteration, even though their own returns did not change. This explains why an underexplored action can become attractive later: the controller increasingly demands evidence before leaving it neglected. The mechanism allocates experiments, not just final answers.

If the successful branch is retained as the best verified artifact, the system can return it when the remaining budget expires. It should not replace it with the branch having the largest exploration index: that index intentionally rewards uncertainty. Search selection and final recommendation need different rules. A final rule might choose the highest independently verified score, with explicit tie-breaking and an unresolved status when no candidate satisfies the contract.

### When should search stop?

Let $V$ be the value of an additional successful repair, $\Delta$ the predicted increase in delivered-success probability from one more search iteration, and $c$ its cost in the same utility units. Continuing has positive estimated net value when $V\Delta>c$. For $V=100$, $\Delta=0.01$, and $c=2$, the extra iteration costs more than its estimated benefit. This is a decision rule under stipulated estimates, not a calibrated property of UCT.

The marginal gain is difficult to predict and can be overestimated by the same judge driving search. A hard budget remains useful even when a learned stopping policy is present. Report both the stopping reason and the evidence attached to the returned candidate; doing so distinguishes an accepted repair from the least-bad proposal available at exhaustion.

### Why the formula does not import every classical guarantee

Confidence arguments assume conditions about bounded returns and sampling. A language-agent tree can violate them: prompts change after reflection, judges behave differently across contexts, and branches share correlated errors. UCT can remain a useful heuristic without satisfying the assumptions of a simpler theorem.

Reflection records why a branch failed and influences future proposals. Remembering that sorting in place violates non-mutation can prevent the same defect. It also changes the proposal distribution, so later samples are not identical to earlier samples. Record this intervention when interpreting search estimates.

Count unsuccessful branches, verifier calls, and reflection in the resource budget. A comparison against one cheap direct sample answers a different question from a matched-budget comparison. The [LATS experiments](https://arxiv.org/abs/2310.04406) motivate the design; a local deployment decision requires evaluation on the relevant task population.

## 3. Branching requires isolation or a credible simulator [19:00](https://www.youtube.com/watch?v=Ml_fp9XkB8Y&t=1140s)

A search tree imagines alternatives. Real environments do not always permit them. Reading a file is usually repeatable; editing an isolated working copy is reversible; sending a message or placing an order creates an external effect that local rollback cannot erase.

Let $T(x,a)$ denote the transition from environment state $x$ after action $a$. Exploring alternative actions from the same state requires a simulator for $T$, an isolated copy of $x$, or a reliable reset operation. If branch A changes shared state before branch B starts, B evaluates a different counterfactual unless that change is included in its state.

For median repair, isolated code copies and deterministic tests make branching straightforward. Each branch begins at the same source version. Production database migration is different: replaying a branch can duplicate writes or change data another branch reads. Representing an operation as a tree edge does not make it reversible.

### Error accumulates along simulated plans

A **simulator** predicts a transition without performing the real operation. Couple simulated and real executions so that each step has conditional probability at most $\epsilon$ of a relevant mismatch while earlier steps match. On a plan of $T$ steps, the union bound gives probability of any mismatch at most $\min(1,T\epsilon)$.

The union bound adds event probabilities without requiring independence. With $\epsilon=0.02$ and $T=20$, the bound is 0.4. It may be loose, but a seemingly small one-step error can still undermine a long plan. A planner also deliberately seeks unusual states, where a simulator estimated on ordinary trajectories may be less accurate.

If rewards lie in $[0,1]$ and agree whenever no relevant mismatch occurs, expected reward difference is at most the mismatch probability. On matching runs the difference is zero; on mismatching runs its absolute value is at most one. This derivation connects transition error to value error under the stated coupling. Independent reward-model mistakes require an additional term.

### Unknown effects must remain unknown

Information gathering, reversible local mutation, and external commitment need different execution policies. Search can explore the first two in isolated environments. An external commitment needs authorization and a reliable record. A timeout after submission may mean the outcome is unknown, not that execution failed.

An **idempotency key** identifies one logical operation so a service can deduplicate retries. It does not certify that the action was appropriate. It prevents an uncertain retry from becoming another execution. A planner that forgets the unresolved operation can treat a dangerous duplicate as a harmless fresh branch.

The repair task is deliberately low impact and easy to reset. What transfers is the discipline of labeling each branch as proposed, simulated, or executed, and tracking which state it used. A high predicted value cannot erase those operational distinctions.

## 4. Parallel reasoning reduces delay only where dependencies permit [23:00](https://www.youtube.com/watch?v=Ml_fp9XkB8Y&t=1380s)

The agent can inspect parity and mutation concurrently if both checks read the same immutable candidate. It cannot verify the final patch before that patch exists. Parallelism follows actual information and state dependencies, not the presence of a numbered plan.

A **directed acyclic graph**, or DAG, represents tasks as nodes and dependencies as edges without a directed cycle. Let node $v$ take $c_v>0$ seconds. **Work** is $W=\sum_vc_v$, total processor time in this model. **Span** $S$ is the largest sum of durations along any dependency path, also called the critical-path length. With $P$ identical workers and elapsed time $T_P$,

$$T_P\ge\max(W/P,S).$$

The work bound follows because $P$ workers can perform at most $PT_P$ work. The span bound follows because one dependency path cannot overlap with itself. These are lower bounds, not guaranteed completion times. Communication, queueing, and discrete task assignment can make the actual schedule slower.

### A numerical schedule

An initial inspection takes two seconds, three independent tests each take three seconds, and final synthesis takes two seconds after all tests. Work is thirteen seconds and span is seven seconds. Three workers attain seven seconds under the ideal assumptions, giving speedup $13/7\approx1.86$, not three.

If one test takes ten seconds, work becomes twenty seconds and span fourteen. The slow **straggler** dominates completion because synthesis waits for all required results. Starting synthesis speculatively helps only if late evidence can still be incorporated correctly.

[SPRINT](https://arxiv.org/abs/2506.05745) studies interleaved planning and parallelized execution in reasoning models. Structured trajectories expose independent segments that can be scheduled concurrently. This reorganizes reasoning; it does not remove autoregressive dependencies inside each segment or make arbitrary subproblems independent.

### Work and sequential generation can move in opposite directions

Let $n_j$ be generated tokens in branch $j$. Total generated tokens are $\sum_jn_j$. Under constant token rate and independent parallel workers, branch latency is controlled by $\max_jn_j$ plus shared planning and integration. Total tokens can rise while sequential generation and delay fall.

That tradeoff can be worthwhile for an interactive service. It can also waste resources when branches duplicate analysis or use incompatible assumptions. Report both total work and latency. A speed improvement is not necessarily a compute saving.

All test workers should receive the same candidate version and contract. Their observations should identify that version. A merge step must reject stale evidence rather than assembling a report from tests of different artifacts. Parallelism makes provenance more important because completion order differs from logical order.

### Computing the graph quantities

The lab accepts a duration map and dependency lists, rejects missing nodes and cycles, and returns work and span. It does not schedule real workers or predict model-serving latency. This graph implements the numerical example:

```python
from numerical_lab import work_span

durations = {"inspect": 2, "odd": 3, "even": 3, "mutation": 3,
             "report": 2}
dependencies = {"inspect": [], "odd": ["inspect"],
                "even": ["inspect"], "mutation": ["inspect"],
                "report": ["odd", "even", "mutation"]}
assert work_span(durations, dependencies) == (13, 7)
```

A cycle is an invalid input for this DAG model, not a long critical path. Iterative workflows can contain loops, but they require bounded unrolling or a different analytical model. The implementation raises an error rather than pretending a cyclic plan has a finite span.

## 5. Learning actions from stored trajectories [50:00](https://www.youtube.com/watch?v=Ml_fp9XkB8Y&t=3000s)

Search spends inference resources on a current task. Training changes behavior on future tasks. **Offline training** uses previously collected interactions. **On-policy interaction** collects new trajectories from the current policy. The distinction determines which states and consequences the learner observes.

Let a stored trajectory be $(h_1,a_1,\ldots,h_K,a_K)$, with each history containing previous actions and tool observations. Decomposing it produces $K$ history–action cases. Let $\mu(h)$ be the stored-history distribution and $r_\phi(h,a)$ a step-quality score. An idealized stepwise objective is

$$J_{\rm step}(\theta)=\mathbb E_{h\sim\mu}
\mathbb E_{a\sim\pi_\theta(\cdot\mid h)}[r_\phi(h,a)].$$

The policy can propose a new action in a stored history and receive a judgment. This objective does not automatically include the actual transition caused by the new action or its terminal reward. It optimizes local judged quality on the stored context distribution.

[SWiRL](https://arxiv.org/abs/2504.04736) collects synthetic multi-step tool-use trajectories, decomposes them into stepwise examples, filters data, and trains using process judgments. This avoids waiting for live tool execution during every optimization update. The judgment concerns a proposed step given prior context; it is not an observation of a new tool result that was never executed.

### Better actions create new histories

Suppose stored median trajectories never test mutation. A new policy learns to request that test. Its observation creates a history absent from training. Local training can reward the request while leaving the response to its result poorly learned. Improving an action changes the states the system will visit.

Let $\nu_\theta$ be the history distribution induced by deployment. A large difference between $\mu$ and $\nu_\theta$ means strong average step scores on stored contexts need not imply strong deployed behavior. Evaluate complete trajectories on held-out tasks to determine whether local improvements survive this shift.

A terminal-success filter retains only trajectories ending correctly. It may discard useful diagnostic prefixes from failed runs. A process filter retains trajectories judged to contain reasonable steps, preserving different information. Terminal success can hide lucky bad reasoning; process judgment can endorse plausible but unproductive steps. Neither filtering rule is universally best.

### Credit assignment across a plan

**Credit assignment** determines which choices deserve reinforcement after an outcome. A final failure can originate in a bad query, a misread observation, or an incorrect synthesis. Terminal reward measures the completed result but may not localize the cause. Stepwise supervision localizes feedback but can miss delayed consequences.

In the repair service, a failed final mutation test identifies an earlier in-place sort. A process critic can separately recognize that inspecting the contract was useful, even in a failed trajectory. Retaining the full trace permits both judgments. The learning target must specify whether it values local validity, information gathering, eventual success, or an explicit combination.

## 6. Limits and a planning experiment

Tree search changes proposal allocation, parallel execution changes scheduling, and offline learning changes future action probabilities. A comparison should hold task splits and final evaluators fixed, account for every branch and judgment, and report delivered success, work, and latency separately.

Search requires a credible simulator or safe reset. Parallelism requires genuine independence and versioned evidence. Offline learning requires testing the new histories reached by the updated policy. None of these conditions follows from generating a longer plan.

For the median task, compare direct repair, sequential feedback, and bounded tree search at the same total generation-and-test budget. Add parallel execution after identifying independent checks. The trace then reveals whether an improvement came from another patch, a more informative test, or simply more computation.

## 7. Exercises and solutions

1. Compute UCT scores with parent count 50, means 0.7 and 0.5, visit counts 20 and 2, and exploration coefficient 0.4.
2. An edge has mean return 0.6 after four visits. Derive its updated mean after return one. Why is this not model training?
3. A graph has a two-second prefix, four independent five-second jobs, and a three-second join. Compute work, span, and ideal elapsed time with two workers.
4. A simulator has conditional relevant-mismatch probability at most 0.01 per step. Bound mismatch probability on a thirty-step plan and explain when this bounds reward error.
5. A policy trained on stored histories starts issuing a new query type. Why does high step reward not establish high deployed success? Propose an evaluation.

<details><summary>Fully worked solutions</summary>

<p><strong>1.</strong> The first score is $0.7+0.4\sqrt{\log(50)/20}\approx0.8769$. The second is $0.5+0.4\sqrt{\log(50)/2}\approx1.0594$. The second edge is selected because its exploration bonus outweighs its lower mean. This is an allocation decision, not a calibrated probability that the second edge is better.</p>

<p><strong>2.</strong> Four visits at mean 0.6 have total return 2.4. Adding one gives 3.4 over five visits, so the new mean is 0.68. Equivalently, $0.6+(1-0.6)/5=0.68$. This changes a statistic in the search controller. The language-model parameters remain unchanged unless a separate learning update occurs.</p>

<p><strong>3.</strong> Work is $2+4(5)+3=25$ seconds and span $2+5+3=10$ seconds. The generic two-worker lower bound is $\max(12.5,10)=12.5$ seconds. The actual graph requires two waves of five-second jobs between prefix and join, giving $2+10+3=15$ seconds. A lower bound need not be achievable with discrete tasks and these dependencies.</p>

<p><strong>4.</strong> The union bound gives at most $30(0.01)=0.3$. Independence is unnecessary under the stated conditional bounds. To bound expected reward difference by the same quantity, couple real and simulated runs so rewards agree when no mismatch occurs and both rewards lie in $[0,1]$. Then only mismatching runs contribute, each by at most one. Independent reward-model errors require separate analysis.</p>

<p><strong>5.</strong> The step judge evaluates a query in an old history. The actual query can produce a new observation and a history outside the training distribution. Execute complete trajectories from fresh tasks and assess final correctness with an independent evaluator under a fixed budget. Compare visited states and failures with the stored data. This tests downstream consequences rather than only local plausibility.</p>

</details>

## 8. Primary references and source boundaries

- Zhou et al., [Language Agent Tree Search](https://arxiv.org/abs/2310.04406): tree search, feedback, and reflection.
- [SPRINT](https://arxiv.org/abs/2506.05745): interleaved planning and parallelized reasoning.
- Goldie et al., [Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use](https://arxiv.org/abs/2504.04736): SWiRL and stepwise optimization.

The mechanism sequence and distinction between live interaction and stored stepwise learning are recording-grounded. UCT arithmetic, work–span derivations, simulation-error bounds, effect tracking, and repair examples are original mathematical and engineering extensions, not reproduced benchmark results.
