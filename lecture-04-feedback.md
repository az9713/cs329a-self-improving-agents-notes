# Lecture 4 — Tools, execution feedback, and principle-guided learning

*Independent textbook chapter accompanying [CS329A Part 4](https://www.youtube.com/watch?v=Lxh9RF5S-K0). The recording supplies ReAct, execution-feedback learning, and Constitutional AI in that order. The median-repair trace, information calculation, recovery model, and code are original teaching constructions.*

## 1. A thought is not an observation [00:00](https://www.youtube.com/watch?v=Lxh9RF5S-K0&t=0s)

The repair agent proposes a median function and writes, “the even-length test now passes.” If no test runner executed, that sentence is another model output. It contains no new measurement. If a runner actually returns 5 for `[1, 9]`, the agent has obtained evidence about an implementation. The distinction determines whether an iterative loop can correct itself or merely elaborate its first mistake.

Let $x_t$ be the environment state before step $t$, including the current code and files. Let $a_t$ be an action, such as reading a function, applying a patch, or running a test. Let $o_{t+1}$ be the observation returned after the action. The agent does not generally observe $x_t$ directly; it sees a history $h_t=(o_0,a_0,o_1,\ldots,a_{t-1},o_t)$. A policy $\pi_\theta(a\mid h_t)$ assigns probabilities to the next action using parameters $\theta$.

A **reasoning trace** is generated text used to organize hypotheses and proposed actions. A **tool observation** is data returned by an executed operation. ReAct interleaves these forms of information. [The ReAct paper](https://arxiv.org/abs/2210.03629) studies reasoning and task-specific actions in question answering, fact verification, and interactive environments. The mechanism is a feedback loop: actions obtain information that can change subsequent reasoning.

An observation does not become infallible because it comes from a tool. A cached page may be stale; a parser may drop a field; a timeout may hide a result. The correct distinction is provenance and semantics. A test runner's structured result warrants claims about that run under its inputs and environment, while a generated prediction warrants only a hypothesis until checked.

### A trace that changes the next decision

Suppose the current implementation returns the element at index `len(values) // 2` after sorting. The agent first reads the function and identifies two hypotheses: it either implements the intended median or the upper-middle rule. It runs `[1, 9]`. Observing 9 rules out the intended result under this input and suggests a missing even-length branch. After patching, observing 5 supports the repair but leaves mutation behavior untested.

A second action records the input list, calls the function on `[9, 1]`, and compares the list afterward. If it becomes `[1, 9]`, the agent has found a separate violation. The next patch should preserve the caller's data. A loop that only repeats the first numeric test would miss this failure indefinitely.

The trace separates hypothesis, intervention, and measurement. It also records which question each action answers. This is more useful than requiring a long verbal explanation before every trivial call. Reasoning has operational value when it selects a discriminating action or updates a belief from evidence.

### Information can be useful without changing the world

An action may collect information rather than directly improve the artifact. Let $H$ be a random hypothesis with possible values “correct median” and “upper-middle bug,” each initially assigned probability one half. **Entropy** measures uncertainty as $\mathcal H(H)=-\sum_hP(h)\log_2P(h)$, in bits because the logarithm has base two. The initial entropy is one bit.

Under a noiseless deterministic test, `[1, 9]` returns 5 under the correct rule and 9 under the bug, so either outcome identifies the hypothesis. Expected posterior entropy is zero and information gain is one bit. An odd-length input such as `[1, 2, 9]` returns 2 under both rules, leaving one bit of uncertainty and giving zero information gain.

This calculation is a two-hypothesis teaching model. Real programs admit many defects and tests can be noisy. It nevertheless explains why an action's value depends on which uncertainty it resolves. More tool calls are not necessarily more information.

## 2. Valid actions and bounded feedback loops [08:00](https://www.youtube.com/watch?v=Lxh9RF5S-K0&t=480s)

A tool interface defines an **action schema**: the operation name, required fields, field types, and permitted values. A valid schema instance can still be inappropriate for the task. “Delete file” may be syntactically valid and unauthorized. A harness therefore needs both input validation and a policy deciding which actions may execute in the current state.

Let $\mathcal A(h_t)$ be the set of actions permitted after history $h_t$. A proposal outside that set is rejected before execution. Restricting generation to enumerated actions can reduce malformed calls, but it cannot guarantee that the selected valid action is useful. In the median task, both “run odd-length test” and “run even-length test” can be valid while only the latter distinguishes the current hypotheses.

A **budget** $B$ limits an explicitly named resource, such as tool calls, wall-clock seconds, or tokens. These units should not be mixed implicitly. Let $c(a_t)>0$ be action cost in the chosen units and $b_t$ the remaining budget. Execution updates $b_{t+1}=b_t-c(a_t)$. A loop must decide what happens when no useful action remains or the next action is unaffordable.

### A finite implementation of the model

The following lab interface evaluates a bounded sequence of already-defined patch candidates. A candidate is a callable accepting a list and returning a number. The checker returns structured success or failure evidence for each candidate, including value and mutation checks. The inputs are trusted local teaching functions; this is not a sandbox for arbitrary generated Python. It executes no network or filesystem operation and makes no model call.

```python
from numerical_lab import check_median


def upper_middle(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def repaired(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


assert not check_median(upper_middle)["passed"]
assert check_median(repaired)["passed"]
```

`check_median` is an extension of the existing [numerical lab](numerical_lab.py). It tests specified finite inputs and records failures. Its success result means those cases passed; it is not a proof for all real-number sequences. Exceptions become failure observations, allowing a controller to distinguish a crash from a numeric mismatch. Actual untrusted program execution needs isolation and resource controls beyond this teaching function.

The model–tool boundary should preserve observation identity. If a test result refers to candidate A, the controller must not attach it to a later candidate B. Include candidate versions or hashes in real traces. Otherwise a valid test result can support the wrong artifact, creating an error that additional reasoning cannot repair from the corrupted record.

### Stopping is part of the policy

A successful local check may justify stopping only if it matches the stated completion criterion. Passing the even-length test does not discharge non-mutation or empty-input requirements. Conversely, continuing after all required checks pass consumes budget and may introduce new errors. The stopping rule should depend on verified obligations and unresolved uncertainty.

A simple bounded controller can stop on acceptance, on budget exhaustion, or on repeated lack of progress. These are distinct terminal statuses. Reporting “completed” after exhaustion hides the most important fact: the system stopped searching without establishing success. Keep the last verified artifact available even if a later revision fails.

## 3. Learning to use execution feedback [27:00](https://www.youtube.com/watch?v=Lxh9RF5S-K0&t=1620s)

Giving a model an error message does not ensure it can repair the cause. It may repeat the same patch, change unrelated code, or optimize only the visible test. **Execution feedback** is the result of running a candidate: output differences, exceptions, timeouts, or test failures. Learning from such feedback aims to improve the conditional policy after those observations.

Let a trajectory $\tau=(h_0,a_0,o_1,\ldots,a_{T-1},o_T)$ be one interaction of at most $T$ actions. Let $R(\tau)$ be its scalar reward under a defined evaluator. A reinforcement-learning objective is $J(\theta)=\mathbb E_{\tau\sim\pi_\theta}[R(\tau)]$. The expectation includes both policy randomness and any environmental randomness. Training changes $\theta$ so that better rewarded trajectories become more likely, subject to the chosen optimization algorithm and constraints.

[RLEF](https://arxiv.org/abs/2410.02089) trains code models to use multi-turn execution feedback. Its central contribution is not merely attaching a test runner; training exposes the model to the feedback-conditioned repair process. The distinction matters when comparing a trained repair policy with a base model that receives errors but has not learned how to exploit them.

### Public feedback and reward labels have different roles

A visible test result can enter the next prompt and guide revision. A private reward test can evaluate the trajectory for learning without being shown as detailed feedback. But if its results repeatedly drive parameter updates, that test is part of training supervision. It is not an untouched final evaluation set.

For the median service, maintain three roles: development cases visible to the repair loop, training reward cases used to score trajectories, and held-out evaluation tasks reserved for the finished system. They need not be three literal files, but their information flow must be distinct. Calling a reward test “private” only says the agent did not directly see its contents; it does not make its feedback statistically independent of training.

A short programming puzzle and a repository repair also differ in environment complexity. A puzzle may supply a complete specification and a compact function interface. Repository work can require finding the relevant files, preserving public interfaces, and understanding incomplete tests. Results on one setting provide evidence about a mechanism, not a universal estimate of software-engineering performance.

### Deriving the benefit and cost of recovery

Let $s$ be the probability of initial success on a stage. Conditional on failure, let $d$ be the probability of detecting the failure. Conditional on detection, let $r$ be the probability that one repair succeeds. Immediate success and successful recovery are disjoint events, so stage success after at most one repair is

$$s_{\rm recover}=s+(1-s)dr.$$

If initial success is 0.9, detection is 0.9, and repair success is 0.8, then $s_{\rm recover}=0.972$. A failure that is undetected cannot benefit from this recovery policy. Raising repair quality has little value when the detector rarely triggers; raising detection without a useful repair can increase cost without increasing success.

Let $c_0$ be initial stage cost and $c_r$ additional repair cost, in the same resource units. If every detected initial failure receives one repair, expected cost is $c_0+(1-s)dc_r$. With $c_0=1$ and $c_r=2$, the example costs $1+0.1(0.9)(2)=1.18$ units. The success increase and cost increase should be reported separately before deciding whether the intervention is worthwhile.

This model assumes detection does not falsely trigger on successes and repair cannot corrupt an already successful candidate. Real systems can violate both assumptions. A critic that frequently “fixes” correct code needs an explicit regression term in the model and an experiment measuring it.

## 4. Critique and principles supply another kind of feedback [46:00](https://www.youtube.com/watch?v=Lxh9RF5S-K0&t=2760s)

Executable tests are useful when desired behavior can be specified mechanically. Many response qualities require judgments: whether an answer is needlessly evasive, whether its claims are adequately supported, or whether its tone follows a stated policy. A **principle** describes a desired property; a **critique** identifies where a candidate violates it; a **revision** attempts to address the identified problem.

[Constitutional AI](https://arxiv.org/abs/2212.08073) uses an explicit collection of principles in a supervised critique-and-revision phase and an AI-preference reinforcement-learning phase. In the first, revised outputs supply fine-tuning data. In the second, model-generated preferences train a preference model that supplies reward. The method changes how supervision is produced; it does not eliminate human choices about principles or evaluation.

A repair assistant can have principles such as preserving the requested interface, reporting evidence accurately, and disclosing unresolved assumptions. These differ from a numeric median test. A function can compute the right value while the accompanying report falsely claims exhaustive verification. The artifact and the report need different checks.

### Formalizing a rubric without hiding tradeoffs

Let $u_j(y)$ be a score for property $j$ of response $y$, for $j=1,\ldots,m$. A rubric initially produces a vector $(u_1(y),\ldots,u_m(y))$. Combining it into $R(y)=\sum_jw_ju_j(y)$ requires weights $w_j$ in compatible units. Those weights express a decision policy; they are not implied by the existence of the individual scores.

Some requirements should be constraints rather than tradable rewards. If the task requires preserving a function signature, no amount of eloquent explanation should compensate for changing it. Let $\mathcal F$ be the set of responses satisfying mandatory conditions. A constrained choice maximizes quality over $y\in\mathcal F$. A weighted penalty approximation can be convenient, but a finite penalty may still allow violations when another score is large enough.

Suppose patch A passes all numeric tests but changes the interface, while patch B passes and preserves it. A rubric that gives A extra points for explanation length can prefer A unless interface preservation is enforced or weighted adequately. The mathematical structure reveals a policy failure that a generic “helpfulness” score may obscure.

### Critic dependence and unsupported certainty

Self-critique can expose errors when the critic has a different perspective, additional evidence, or a useful rubric. It can also repeat the generator's misconception. Two calls to the same model are not automatically independent sources of evidence, especially when both read the same flawed explanation.

A concrete critique should name a violated condition and an observable consequence. “This code may be wrong” gives little guidance. “Sorting in place changes the caller's list; compare the input before and after the call” creates a testable repair target. The latter turns a principle into an operational check.

Changing the principles after training raises a separate problem. Updating a prompt can change current conditioning, but it does not erase behavior encoded in weights. Evaluate the updated system against both new requirements and important retained capabilities. Do not infer complete removal of old behavior from the new document's wording alone.

## 5. Feedback quality is a property of a loop [58:00](https://www.youtube.com/watch?v=Lxh9RF5S-K0&t=3480s)

Three feedback sources now coexist: environment measurements, learned outcome scores, and principle-guided critiques. Their evidence classes differ. An executed test measures behavior on specified inputs; an outcome score estimates a labeled property; a critique argues for a revision under a rubric. A robust controller records which kind of evidence justified each decision.

For the median task, a useful cycle is to generate a patch, run parity and mutation checks, inspect structured failures, revise only the implicated behavior, and rerun the full development suite. A separate final evaluator assesses generalization. If a learned critic says the code is correct while execution returns the wrong result, the conflict is resolved by examining the specification and actual behavior, not by averaging incompatible claims.

### Value of information and value of correction

Let $U(a,H)$ be the utility of action $a$ when the true hypothesis is $H$. Before a diagnostic test, the best expected utility is $\max_a\mathbb E[U(a,H)]$. Let $O$ denote the possible test observation. After seeing it, the controller can choose a different action, giving expected value $\mathbb E_O[\max_a\mathbb E[U(a,H)\mid O]]$. The difference is the **expected value of information** before subtracting test cost:

$$\operatorname{VOI}=\mathbb E_O\left[\max_a\mathbb E[U(a,H)\mid O]\right]
-\max_a\mathbb E[U(a,H)].$$

The inner maximization adapts the decision to the observation. If the same action is best regardless of the result, information may have no decision value even when it reduces uncertainty. A costly test is worthwhile under this criterion only when its expected decision improvement exceeds its cost in compatible utility units.

In the two-hypothesis median example, a perfect parity test can prevent an unnecessary patch or identify the needed one. If every permitted action would replace the entire function identically regardless of the result, the test's diagnostic information may not change the action. It can still provide evaluation evidence, but that is a different purpose to include in the utility.

This distinction helps avoid ritual tool use. A tool call should gather needed evidence, change the artifact, or verify an obligation. Repeated calls that cannot change any justified decision consume resources without advancing the task.

### A numerical decision separates information from reassurance

Suppose there are two equally likely states: the median implementation is correct or it has the upper-middle bug. The controller can release it or repair it. Releasing a correct implementation has utility ten; releasing the bug has utility minus ten. Repairing either state produces a correct artifact but costs three utility units, so its utility is seven. Before testing, release has expected utility zero and repair has utility seven; repair is preferred.

A perfect even-length diagnostic permits release when the implementation is correct and repair when it is defective. Expected utility after observing the result, before test cost, is $0.5(10)+0.5(7)=8.5$. The value of information is therefore $8.5-7=1.5$. A test costing one unit is worthwhile; one costing two is not under these assumptions.

An odd-length test produces the same observation in both states. It leaves the best decision unchanged and has zero value of information in this model, even though it may make the controller feel reassured. This is a reason to evaluate tests by decisions they can change rather than by the number of green checkmarks they add.

The utility values are original teaching assumptions. In practice, repairing a correct function can introduce regressions, the diagnostic may be noisy, and releasing a bug can have asymmetric consequences. Those changes belong in the action–outcome table before calculating value. The model's value is that it forces these assumptions into the open.

### A causal comparison of feedback methods

To test whether a new feedback loop helps, hold the generator, task distribution, and total budget fixed while varying the feedback mechanism. Include a no-feedback baseline with an equivalent opportunity to spend compute, rather than comparing a multi-turn agent with a single cheap sample. Otherwise extra attempts can masquerade as a feedback benefit.

Measure correction and regression separately. Let $p_{\rm fix}$ be the probability a wrong candidate becomes correct after revision, and $p_{\rm break}$ the probability a correct candidate becomes wrong. If initial correctness is $s$, one revision yields correctness $s(1-p_{\rm break})+(1-s)p_{\rm fix}$. With $s=0.8$, $p_{\rm fix}=0.4$, and $p_{\rm break}=0.15$, revised correctness is $0.8(0.85)+0.2(0.4)=0.76$. An apparently helpful critic makes the system worse because it breaks too many already-correct candidates.

## 6. Limits and implementation discipline

The toy code checks a finite set of trusted functions. It is intentionally not an execution service for arbitrary generated programs. A real tool runner needs isolation, timeouts, resource limits, and explicit handling of external side effects. These are part of making the observation trustworthy, not optional decorations around the model.

The recovery and information models assume clearly defined outcomes. In open-ended work, human evaluation may be noisy and requirements may conflict. Preserve that uncertainty instead of turning every judgment into a binary fact. A detailed trace makes disagreements inspectable, but trace length alone does not establish correctness.

The important improvement target is the entire loop: what the agent observes, how it interprets that observation, which revision it makes, and whether the revision survives independent evaluation. A fluent critique is useful only when this chain produces better outcomes under the stated cost and scope constraints.

## 7. Exercises and solutions

1. Compute the information gain of an odd-length and an even-length test under the two equally likely deterministic median hypotheses.
2. Derive one-recovery success and expected cost for $s=0.85$, $d=0.8$, $r=0.75$, $c_0=1$, and $c_r=3$.
3. A critic repairs 30% of wrong answers and breaks 5% of correct answers. For what initial correctness rates does revision improve expected accuracy?
4. Explain why a private reward test used throughout training is not an untouched final evaluation. Design a corrected split.
5. A principle says “preserve the interface,” but a weighted reward prefers a signature-changing patch. Give two remedies and state their tradeoffs.

<details><summary>Fully worked solutions</summary>

<p><strong>1.</strong> The prior entropy is $-2(0.5\log_2 0.5)=1$ bit. An odd-length test has the same output under both hypotheses, so the posterior remains one half–one half and information gain is zero. An even-length input with unequal middle values produces different outputs, so the posterior concentrates on one hypothesis and its entropy is zero. Information gain is one bit. Noisy execution or additional hypotheses would change this calculation.</p>

<p><strong>2.</strong> Immediate success contributes 0.85. The disjoint recovery path contributes $0.15(0.8)(0.75)=0.09$, giving 0.94 total success. Repair is attempted with probability $0.15(0.8)=0.12$, so expected cost is $1+0.12(3)=1.36$. The intervention buys nine percentage points of success for 0.36 additional expected cost units. Whether that is worthwhile depends on the value of success and any latency constraint.</p>

<p><strong>3.</strong> Revised accuracy is $s(0.95)+(1-s)(0.30)=0.30+0.65s$. It exceeds $s$ when $0.30>0.35s$, or $s<6/7\approx0.8571$. At very high initial accuracy, there are few errors to repair and many correct answers that can be damaged. This calculation motivates selective revision rather than treating critique as universally beneficial.</p>

<p><strong>4.</strong> Repeated reward outcomes influence parameters, so the trained policy adapts to that supervision even if it never reads the test source. Reserve final tasks and their evaluator outcomes until the policy, prompts, stopping rule, and thresholds are frozen. Use visible development tests for interaction, separate reward cases for learning, and untouched tasks for final comparison. If the final set influences another update, it becomes development data and a new final set is needed for an independent claim.</p>

<p><strong>5.</strong> Enforce signature preservation as a hard feasibility constraint, rejecting violating patches before quality ranking. This is reliable when the condition is mechanically checkable but may reject an otherwise useful redesign that the user could have authorized separately. Alternatively increase its penalty and calibrate the reward tradeoff, which preserves flexibility but provides no absolute guarantee at finite penalty. The first remedy matches a mandatory interface requirement more directly.</p>

</details>

## 8. Primary references and source boundaries

- Yao et al., [ReAct](https://arxiv.org/abs/2210.03629): interleaved reasoning, actions, and observations.
- [RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning](https://arxiv.org/abs/2410.02089): training feedback-conditioned code repair.
- Bai et al., [Constitutional AI](https://arxiv.org/abs/2212.08073): critique/revision supervision and AI preference feedback.

The recording supplies the mechanisms and comparisons among these approaches. The information-theoretic derivation, recovery economics, regression calculation, and median implementation are teaching extensions. They establish consequences of explicit toy assumptions; they are not measurements of the named systems.
