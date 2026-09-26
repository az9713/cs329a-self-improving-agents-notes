# Lecture 7 — Research agents: search over programs and evidence

*Independent textbook chapter accompanying [CS329A Part 7](https://www.youtube.com/watch?v=Uni9dqyuuDM). The recording supplies AlphaCode, AlphaCode 2, and Search-o1. The median behavior signatures, evidence ledger, probability decompositions, and exercises are original teaching constructions.*

## 1. A large candidate pool is only the beginning [00:00](https://www.youtube.com/watch?v=Uni9dqyuuDM&t=0s)

A coding agent generates ten thousand median implementations but may submit only ten. A research agent retrieves a hundred documents but can cite only the few that actually support its conclusions. Both problems require reducing a large candidate pool while preserving the information needed for a correct final result. Volume is useful only if the selection procedure can distinguish relevant diversity from repetition and error.

Let $q$ be a task, $\mathcal Y=\{y_1,\ldots,y_N\}$ a generated pool of $N$ candidates, and $c(q,y_i)\in\{0,1\}$ the true correctness of candidate $i$. A **filter** removes candidates violating observable conditions. A **ranker** orders survivors by an estimated quality score. A **selector** returns a subset $S\subseteq\mathcal Y$ with size at most $m$, the submission budget. These operations can use different evidence.

For programs, compilation and example tests provide cheap rejection signals. Passing them is necessary in many tasks but not sufficient for full correctness. For research documents, an inaccessible page or irrelevant topic can be rejected early, while claim support requires reading the actual passage. Filtering should preserve possible good candidates while removing obvious failures at acceptable cost.

[AlphaCode](https://arxiv.org/abs/2203.07814) combines trained code generation, large-scale sampling, execution-based filtering, and behavioral clustering to select a small submission set. Its competitive-programming evaluation concerns a defined problem and submission regime. It is evidence about that system under those conditions, not a measurement of autonomous maintenance in a live repository.

### The one-of-many training objective

A programming task can have many valid implementations, while the submission objective requires finding only one. Maximum-likelihood training encourages representing the observed reference distribution; successful selection under a small submission budget is a different objective. AlphaCode also used a variant of **GOLD**, an offline reinforcement-learning method, to reweight training toward tokens the model already considers more likely. Its softened weighting and training transition are specified in [Appendix C.3 of the AlphaCode paper](https://arxiv.org/abs/2203.07814).

For the repair service, this creates a tension: concentrating on a reliable implementation family can improve immediate yield, but suppressing alternatives can reduce useful search diversity. Evaluate both single-candidate quality and selected-set success. A training loss, a sampling policy, and a behavioral selector contribute different parts of that outcome.

### Distinguish generation and submission budgets

Let $N$ be generated programs and $m$ submitted programs. Oracle pass@$N$ asks whether the full pool contains a correct program. Actual success asks whether the selection procedure places a correct program among the $m$ submissions. Increasing $N$ can improve the first quantity while leaving the second unchanged if the selector cannot identify useful candidates.

The notation “10 at one million” in such a setting refers to selecting ten submissions from a much larger generated pool. It does not mean that a user received one million independent attempts against the final evaluator. Generation, filtering, and submission costs all belong in the resource account.

For the median task, thousands of textually different implementations may share the same upper-middle error. Submitting ten of them provides little diversity. Conversely, one correct but unusually written implementation can be discarded if a ranker equates familiar style with correctness. The selector needs a representation of behavior relevant to the specification.

### Candidate identity and evidence identity

Keep every test result attached to the exact candidate version that produced it. A filter can otherwise accept a modified program using stale results from an earlier version. For documents, the analogous mistake is citing a page whose current content differs from the passage originally read. Stable identifiers, dates, and saved evidence references support reproducible selection.

This is an engineering requirement before any sophisticated ranking method. An excellent score applied to the wrong artifact is still wrong. The formal candidate set must correspond to the objects actually evaluated and delivered.

## 2. Behavioral diversity is more useful than textual variety [06:00](https://www.youtube.com/watch?v=Uni9dqyuuDM&t=360s)

Choose test inputs $x_1,\ldots,x_K$, where $K$ is a positive integer. A program $y$ has a **behavior signature** $\phi(y)=(y(x_1),\ldots,y(x_K))$, with failures represented by explicit error outcomes. Programs with the same signature agree on these tests. This defines an equivalence relation relative to the test set, not semantic equivalence on all inputs.

For median candidates, take `[1, 9]` and `[1, 2, 9]`. The correct implementation has signature $(5,2)$; an upper-middle implementation has $(9,2)$; a lower-middle implementation has $(1,2)$. Variable renaming changes none of these signatures. A mutation defect remains invisible unless the signature includes post-call input state as well as returned values.

Behavioral clustering groups candidates by observed signatures or another similarity rule. Selecting across groups can avoid spending every submission on the same mistake. But the grouping is only as informative as the inputs. With odd-length tests alone, the three implementations above collapse into one group.

### Deriving the value and limit of diverse selection

Suppose selected candidates have correctness probabilities $p_1,\ldots,p_m$ and their correctness events are independent. The probability at least one succeeds is $1-\prod_{i=1}^{m}(1-p_i)$. If they are exact duplicates with identical behavior on every input, their correctness events are identical and the independent formula is wrong: repeated submission gives only the original probability.

Independence is rarely known. Behavioral disagreement can indicate different failure modes, but disagreement on selected tests does not prove independent correctness. It can also arise from two unrelated wrong algorithms. Diversity is a resource for search, not a substitute for validity.

A useful teaching contrast has two strategy families. Family A is correct on 80% of tasks in a target population and family B on 60%. If their success events are independent, submitting one from each succeeds on $1-(0.2)(0.4)=0.92$ of tasks. Two identical A programs still succeed on 0.8. If B succeeds only where A already succeeds, the mixed pair also remains at 0.8. Marginal accuracies alone cannot determine the benefit; joint errors matter.

### Signature grouping in the cumulative lab

The following code groups already-observed signatures. It accepts a mapping from candidate names to immutable signature tuples and returns groups of names. It does not execute generated code or infer correctness. The example checks that two textually different candidates with identical observed behavior remain in one group.

```python
from numerical_lab import group_signatures

signatures = {"correct_a": (5, 2), "correct_b": (5, 2),
              "upper": (9, 2), "lower": (1, 2)}
groups = group_signatures(signatures)
assert sorted(map(len, groups)) == [1, 1, 2]
```

The result says there are three observed behavior classes. It does not identify which is correct without expected outputs. Adding a new discriminating test can split a group; removing tests can merge groups. This makes test design part of the diversity mechanism.

### Expected distinct behaviors explain saturation

Suppose the generator produces one of $J$ behavior classes, with class probabilities $p_1,\ldots,p_J$ summing to one. After $N$ independent samples, class $j$ has appeared unless all samples avoided it, so its appearance probability is $1-(1-p_j)^N$. Let $I_j$ be one if the class appeared and zero otherwise. The number of represented classes is $D_N=\sum_jI_j$, and linearity of expectation gives

$$\mathbb E[D_N]=\sum_{j=1}^{J}[1-(1-p_j)^N].$$

The indicators need not be independent for this sum of expectations. Independence is used only in computing the probability of repeatedly avoiding one class. This distinction matters because class appearance events compete within a finite batch.

If two classes have probabilities 0.9 and 0.1, ten samples yield expected distinct count $[1-0.1^{10}]+[1-0.9^{10}]\approx1.6513$. Increasing to one hundred samples makes both classes almost certain to appear, but cannot create a third behavior absent from the generator. Additional samples mostly duplicate existing classes.

For median repair, a rare correct class can be worth searching for, while a rare wrong class adds diversity without value. The expected distinct count is therefore a diagnostic of exploration, not an objective that should replace correctness. Combine it with class-specific validity and selection success. If distinct behavior saturates early but delivered accuracy remains low, merely increasing textual sample count may be a poor use of the next budget unit.

### Filtering can create blind spots

A filter that rejects all programs disagreeing with an incorrectly generated test removes every truly correct program for that case. More samples cannot recover from a deterministic false-rejection rule applied to all of them. Check generated tests against the specification or an independent reference before treating them as authoritative.

This risk has a research analogue. Filtering papers solely by a query term can exclude work using different terminology. A high-precision filter may create poor recall. Keep track of what the filter is designed to exclude and audit a sample of rejections for systematic omissions.

## 3. Stronger generation and learned scoring solve different bottlenecks [25:00](https://www.youtube.com/watch?v=Uni9dqyuuDM&t=1500s)

A stronger generator can increase the probability that a useful program enters the pool. A stronger ranker can improve the probability that a useful program survives selection. These are different interventions. For a selector restricted to the generated pool, let $C$ be oracle coverage and let $s$ be conditional selection success given a correct candidate exists. Delivered success is $Cs$ by the probability chain rule.

Suppose coverage is 0.9 and conditional selection is 0.5, giving delivered success 0.45. Improving coverage to 0.95 at unchanged selection gives 0.475. Improving selection to 0.7 at unchanged coverage gives 0.63. These hypothetical numbers show why the larger model is not always the highest-value component to improve.

The [AlphaCode 2 technical report](https://storage.googleapis.com/deepmind-media/AlphaCode2/AlphaCode2_Tech_Report.pdf) describes Gemini-based generation combined with filtering, clustering, and a scoring model that selects candidates within a small set of clusters. Its reported competitive-programming results belong to that complete pipeline. Attributing them to model weights alone would omit the search and selection procedure.

### Training likelihood and selection utility

A generator trained to assign high likelihood to reference code is learning a token distribution. A ranker trained on successful and unsuccessful programs is learning a selection signal. The two objectives need not favor the same candidate. An uncommon but correct algorithm can have low likelihood; a common template can be confidently wrong on a boundary case.

Let $\ell(y)=-\log\pi_\theta(y\mid q)$ be negative log-likelihood and let $v(q,y)$ be a correctness score. Low $\ell$ means the generator considers the sequence probable. High $v$ means the verifier estimates quality under its labels. Neither mathematical definition implies monotonic agreement between them.

In the median service, the shortest familiar implementation may omit even-length handling. A longer implementation may correctly cover the contract. Length-normalized likelihood can alter the ranking again. Specify whether a reported score is total likelihood, per-token likelihood, or a separately trained quality estimate before interpreting it.

### Adaptive compute requires a measured response curve

A service can spend additional compute on sampling, revision, testing, or ranking. Let $A_j(b)$ be delivered accuracy of strategy $j$ at budget $b$ in common resource units. The appropriate choice at a fixed budget maximizes $A_j(b)$ under latency and other constraints. These curves must be measured; they cannot be inferred from the number of components in a diagram.

The repair task offers a simple comparison. If the generator rarely proposes the correct average, more generation may help. If correct patches are common but mutation tests are missing, better verification may help. If failures are highly diagnostic and repairable, sequential revision may help. An aggregate final score without intermediate evidence cannot identify which mechanism produced the gain.

A credible experiment records the generated pool, filtered survivors, cluster membership, selected submissions, and final outcomes. This permits counterfactual analysis of the selector on a fixed pool. It also reveals whether an apparent gain came from generating more candidates rather than selecting more effectively.

## 4. Knowledge gaps require external evidence [46:00](https://www.youtube.com/watch?v=Uni9dqyuuDM&t=2760s)

The coding agent now needs to explain whether its median convention matches an external library specification. No amount of internal sampling can establish the current contents of an unread document. A research agent must identify the missing fact, retrieve a source, interpret the relevant passage, and integrate it into the answer.

Let $q$ be the research question, $h_t$ the current evidence history, and $a_t$ a search or reading action. A retrieval tool returns documents $D_t$. An evidence processor extracts relevant passages and their provenance into $E_t$. The next state is $h_{t+1}=h_t\Vert(a_t,D_t,E_t)$, where $\Vert$ means concatenation. This state should distinguish source text, the agent's interpretation, and unresolved questions.

[Search-o1](https://arxiv.org/abs/2501.05366) integrates agentic search into a reasoning process and uses a document-processing mechanism to turn retrieved material into useful information for continued reasoning. The motivating issue is a knowledge gap encountered during solving, rather than a requirement to retrieve the same fixed number of documents for every task.

### Retrieval success is not answer success

Let $R$ be the event that a relevant source is retrieved, $X$ that its relevant content is extracted correctly, and $S$ that the final synthesis uses it correctly. The probability of completing all three stages is

$$P(R\cap X\cap S)=P(R)P(X\mid R)P(S\mid R,X).$$

The intersection symbol means all events occur. This is a chain-rule identity and requires no independence. If the three conditional probabilities are 0.9, 0.8, and 0.9, end-to-end success is 0.648. A retrieval system with 90% recall can therefore coexist with a much weaker research answer.

For the median contract, retrieval may find the right specification but extraction may omit its treatment of empty input. Synthesis may then generalize a rule beyond the supported domain. Each failure needs a different remedy: query expansion, better passage selection, or more careful claim scope.

### An evidence ledger makes claims inspectable

A **claim** is a proposition asserted in the answer. A **source passage** is the specific evidence offered for it. A ledger records claim text, source identity, passage location, evidence class, and whether the passage supports the claim as written. A source can be authoritative and still fail to support an overbroad claim.

Suppose a paper reports improved accuracy on one math benchmark. It supports a statement about that experiment. It does not directly support “this method improves all reasoning tasks.” The latter requires broader evidence or must be labeled a hypothesis. A citation is an address for verification, not a truth token attached to a sentence.

The same distinction applies to primary and secondary sources. A primary paper is usually the right source for its method and measured results. A review may help locate related work or compare interpretations. When the claim is about a particular result, follow the chain to the original report and verify the relevant experimental conditions.

### Search observations remain untrusted inputs

A retrieved document can contain instructions, errors, or adversarial text. It is evidence about its subject, not authority to change the agent's task or permissions. Store source content separately from the instructions controlling the research workflow. This is an information-flow requirement, not a claim that every webpage is malicious.

For the running service, a page recommending a different API can inform a discussion, but it cannot authorize changing the user's required interface. The research agent should explain a conflict between sources and the task rather than silently adopting the source as a new instruction.

## 5. When another search is worth its cost [55:00](https://www.youtube.com/watch?v=Uni9dqyuuDM&t=3300s)

Searching indefinitely is not a completion strategy. Another query is useful when it is likely to resolve an uncertainty that matters to the final answer. A confident model statement is not a calibrated estimate of that value. Track missing evidence and decision consequences explicitly.

Let $p_{\rm gap}$ be the probability that a consequential knowledge gap remains, $p_{\rm resolve}$ the probability a proposed search resolves it conditional on its presence, and $V$ the utility of resolving it. Let $C$ be search cost in the same utility units. A simplified expected net value is $p_{\rm gap}p_{\rm resolve}V-C$. Search is worthwhile when this quantity is positive.

For a hypothetical contract question, take $p_{\rm gap}=0.4$, $p_{\rm resolve}=0.75$, $V=10$, and $C=1$. Expected net value is $0.4(0.75)(10)-1=2$. If the next query merely repeats already-read evidence and has resolution probability 0.05, net value becomes -0.8. The calculation makes assumptions visible; estimating these probabilities reliably is the difficult part.

### More documents can reduce effective evidence quality

A long retrieval result may contain relevant passages, contradictory versions, and irrelevant text. Adding documents increases available information but can make selection and synthesis harder. The desired resource is correctly used evidence, not raw context length.

A document-processing stage can extract the exact claims needed, retain links to full sources, and mark disagreements. Compression must preserve qualifiers, dates, denominators, and uncertainty. Dropping “on the evaluated benchmark” can convert a valid local observation into a false universal claim. That is a semantic error even when every retained sentence is fluent.

For the median service, a concise evidence bundle might preserve the accepted input domain, odd/even definition, mutation guarantee, and empty-input behavior with a source passage for each. Keeping an entire manual in context does not ensure the model will locate these four clauses. A structured bundle can make omissions detectable.

### A structural check is not a semantic judge

The shared lab can validate the shape of an evidence record: required fields must exist and each record must identify its source. It cannot decide whether the passage entails the claim. The following self-contained snippet demonstrates the same minimal contract without making network requests.

```python
record = {"claim": "The example uses an even-length input.",
          "source": "worked-example",
          "passage": "The input is [1, 9].",
          "status": "supported"}
required = {"claim", "source", "passage", "status"}
assert required <= record.keys()
assert all(record[key] for key in required)
```

The check succeeds because provenance fields are populated. A record with a false claim could also pass, so a semantic audit remains necessary. Keeping these two validation layers separate prevents a software schema from being mistaken for an evidence guarantee.

### Synthesis must preserve disagreement

If two sources use different task definitions, their results may be incomparable rather than contradictory. If they use the same definition and disagree, report the disagreement and relevant evidence. Averaging claims into a smooth paragraph can hide the very uncertainty the user needs to understand.

A useful stopping condition is that every consequential claim has adequate support or an explicit uncertainty label, major contrary evidence has been considered, and further search has low expected decision value under the remaining budget. This is a judgment grounded in a ledger, not a fixed document quota.

## 6. Limits of transferring the program-search analogy

Programs can often be executed against exact tests. Research synthesis usually requires evaluating claims, source quality, scope, and completeness. Behavioral clustering has a clear operational definition for program outputs; clustering documents by textual similarity does not establish independent corroboration.

The shared structure is generation or retrieval followed by filtering, selection, and verification. The evidence standards differ by domain. A program accepted by a judge and a research claim supported by a passage are different kinds of result, and both retain boundaries around the evaluator.

For a small independent project, the most informative improvement is often a traceable candidate-and-evidence pipeline. Record where useful material was lost and where unsupported claims entered. That turns a failed final answer into a diagnosis of retrieval, extraction, selection, or synthesis rather than a vague complaint about model intelligence.

## 7. Exercises and solutions

1. Three median programs agree on every odd-length test. Construct a test that separates correct averaging, upper-middle, and lower-middle behavior.
2. Two strategy families succeed with probabilities 0.8 and 0.6. Compute their joint coverage under independence and under the assumption that every B success is also an A success.
3. A selector has coverage 0.85 and conditional selection success 0.6. Compare improving coverage to 0.95 with improving selection to 0.75.
4. Retrieval, extraction conditional on retrieval, and synthesis conditional on both have probabilities 0.95, 0.7, and 0.8. Compute end-to-end evidence use and explain why independence is unnecessary.
5. A cited source reports a benchmark-specific improvement, while the answer claims universal superiority. Repair the claim and specify what additional evidence a broader claim would require.

<details><summary>Fully worked solutions</summary>

<p><strong>1.</strong> Use `[1, 9]`, whose middle elements differ. Correct averaging returns 5, upper-middle returns 9, and lower-middle returns 1. An even-length list with equal middle elements would not separate the rules. To expose mutation as well, preserve a copy of an unsorted input such as `[9, 1]` and compare it after execution. Returned-value signatures alone miss that property.</p>

<p><strong>2.</strong> Under independence, failure of both has probability $0.2(0.4)=0.08$, so coverage is 0.92. If B succeeds only on tasks already solved by A, the union of success events equals A's event and coverage remains 0.8. The same marginal accuracies support different selection benefits because joint error structure differs.</p>

<p><strong>3.</strong> Initial delivered success is $0.85(0.6)=0.51$. Improving coverage alone gives $0.95(0.6)=0.57$, a six-percentage-point gain. Improving selection alone gives $0.85(0.75)=0.6375$, a 12.75-point gain. This compares stipulated changes, not their cost. A deployment choice must measure how expensive and achievable each change is.</p>

<p><strong>4.</strong> The product is $0.95(0.7)(0.8)=0.532$. Each probability is conditioned on the preceding required events, so multiplication follows the chain rule. If the values were instead three unconditional marginal probabilities, multiplication would require additional assumptions. The result identifies extraction as one possible bottleneck despite strong retrieval.</p>

<p><strong>5.</strong> State that the method improved the reported metric on the specified benchmark, under the paper's model, budget, and evaluation setup. Preserve the baseline and uncertainty when available. A broader claim needs independent tasks, domains, model families, and matched-resource comparisons, including counterexamples and failure regimes. A link to a primary source cannot expand the scope of what that source measured.</p>

</details>

## 8. Primary references and source boundaries

- Li et al., [Competition-Level Code Generation with AlphaCode](https://arxiv.org/abs/2203.07814): generation, execution filtering, and behavioral clustering.
- Google DeepMind, [AlphaCode 2 Technical Report](https://storage.googleapis.com/deepmind-media/AlphaCode2/AlphaCode2_Tech_Report.pdf): generation and learned selection in a competitive-programming pipeline.
- [Search-o1](https://arxiv.org/abs/2501.05366): search-enhanced reasoning and processing retrieved documents.

The recording supplies the transition from program search to external knowledge search. Behavior signatures, joint-error examples, evidence-ledger design, chain-rule analysis, and search-value calculations are original extensions. The structural code check verifies record shape only; claim support still requires reading and judgment.
