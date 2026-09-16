What We Tested, in Plain English
======================================================================

Start here. No knowledge of machine learning is needed for this page.
The numbers come from completed runs reported by the operator. Test 22 is
now complete; test 23 is implemented and awaiting submission/results.

The Problem We Are Trying to Solve
----------------------------------------------------------------------

We measure a galaxy's light. We want to learn its properties: for example,
how much stellar mass it contains and how much dust hides its light.

Different galaxies can produce very similar measurements. So the program
must return **many possible answers and how much we should trust each one**.
Returning one answer that reproduces the light is not enough.

There are two main parts to the program:

* A neural network quickly suggests possible galaxies.
* A physics calculation predicts the light from each suggestion. We compare
  that light with the measurements to assess the suggestion.

The difficulty was this: the program could suggest thousands of galaxies,
but after this comparison almost all the weight could fall on just one or a
few of them. We then had too little useful information to trust the reported
uncertainties.

Why There Were So Many Tests
----------------------------------------------------------------------

We did not know which part was responsible. We therefore asked four questions,
in order:

1. Is the neural network able to suggest the right kinds of galaxies?
2. Are the calculations used to improve those suggestions working correctly?
3. Does spending more time on each galaxy solve the problem?
4. When the program changes its answers, does that change actually help?

Below, **what it means** says what each result allows us to conclude. It does
not claim that the whole problem is solved. Run numbers are only included so
that we can find the original records.

.. contents:: Find a test
   :local:
   :depth: 1

01. Can the Network Learn From More Simulated Examples?
----------------------------------------------------------------------

**We wanted to know:** could better training make the network suggest more
useful answers?

**We did:** create simulated galaxies and their light, then train the network
to infer the galaxies from that light. We kept the rules used to generate
the galaxies unchanged.

**We saw:** some results improved. Out of 1,024 suggestions, the effective
number contributing after weighting rose from about 6 to about 13. This is
a measure of how spread out the weights are, not a count of correct answers.
Thirteen was still very small compared with 1,024.

**What it means:** this training helped, but did not solve the main problem.
We next checked whether the network's construction was limiting what it
could learn.

Record: September 5-6 frozen-parent sleep experiment, ``warm_start`` reference.

02. Could Every Part of the Network Change?
----------------------------------------------------------------------

**We wanted to know:** were some parts of the suggested answer left unchanged
by the network blocks that were supposed to adjust them?

**We did:** count how often those blocks transformed each of the 15 numbers
used to describe a galaxy.

**We saw:** seven numbers were never transformed by those blocks. We changed
the network's construction so that every number was transformed.

**What it means:** we found and fixed a real limitation in the network.
This does not mean all its answers became correct. It means the network
no longer has that particular restriction.

Records: jobs 1829245, 1829247, 1890513 and 1890514; code change ``ecf3fb2``.

03. Does Asking for a Better Match to the Light Help?
----------------------------------------------------------------------

**We wanted to know:** could we improve the answers by explicitly asking the
network to reproduce the measured light during training?

**We did:** compare several combinations of learning from simulated examples
and learning to match the light.

**We saw:** some predictions matched the light much better. But after weighting
the suggestions, too few still contributed. None of the four tested versions
passed the checks for reliable weights.

**What it means:** matching the light and getting trustworthy uncertainties
are not the same thing. Before trying longer training, we checked the
calculations that tell the program how to improve.

Records: jobs 1893047-1893049.

04. Is the Program Told to Change Its Answers in the Right Direction?
----------------------------------------------------------------------

**We wanted to know:** can we trust the calculation that tells training which
way to change a galaxy's properties?

**We did:** compare two ways of finding that direction. One uses the program's
automatic calculation. The other makes a small change to the input and
directly measures what happens to the result.

**We saw:** the two methods did not agree clearly enough for us to approve
the next training step. We stopped before that training began.

**What it means:** we did not yet know whether the calculation or the check
was at fault. We needed to test smaller parts separately.

Records: jobs 1913341 and 1913854. This kind of check is called a gradient audit.

05. Is the Problem in Predicting the Light or in Comparing It?
----------------------------------------------------------------------

**We wanted to know:** which of these two operations caused the disagreement?

**We did:** test the formula that compares predicted and measured light on its
own. Then we tested it together with the calculation that predicts the light.

**We saw:** the comparison formula passed. The difficulty remained when we
included the prediction of light and changed the galaxy's redshift. Redshift
describes how the expansion of the Universe shifts the light's wavelengths.

**What it means:** we had narrowed the search. We needed to inspect how the
predicted light changes with redshift, rather than rewrite the comparison
formula.

Record: job 1914143.

06. Which Part of the Redshift Calculation Causes Trouble?
----------------------------------------------------------------------

**We wanted to know:** where does the disagreement arise when redshift changes?

**We did:** separate the different effects of redshift. In one test we kept
the emitted light unchanged and tested only how it would be measured through
the telescope's filters.

**We saw:** a disagreement remained in this simpler test. Using more numerical
precision in the tested setup was not enough to remove it.

**What it means:** the way we added up light through the filters needed a
separate check. The problem was not simply "use more precise numbers everywhere".

Record: job 1915987.

07. Can We Check the Filter Calculation Another Way?
----------------------------------------------------------------------

**We wanted to know:** can we calculate the light passing through each filter
in a way that agrees with an independent reference calculation?

**We did:** compare two new ways of adding up the light with that reference.
We also checked how their answers changed when the inputs changed slightly.

**We saw:** both methods agreed with the reference. All 54 tested checks of
these changes passed, across three example points.

**What it means:** this supported the replacement calculation at those points.
We still had to check the rest of the physical model.

Record: job 1918919. Technical name: photometry quadrature reference.

08. Does the Complete Calculation Now Pass?
----------------------------------------------------------------------

**We wanted to know:** after fixing the filter calculation, does the whole
calculation behave as expected?

**We did:** reconnect the parts and repeat the checks on the complete model.

**We saw:** some checks passed, but nine checks involving stellar metallicity
failed. Metallicity describes the abundance of elements heavier than helium
in the stars. Other checks still had no clear answer.

**What it means:** fixing one part exposed another problem. We could not yet
treat the entire calculation as checked.

Record: job 1920226.

09. Does More Precision Fix the Metallicity Calculation?
----------------------------------------------------------------------

**We wanted to know:** were rounding effects causing the metallicity problem?

**We did:** use more numerical precision in the part that describes the
distribution of stellar metallicities. We compared it with a simple reference.

**We saw:** all the metallicity checks passed. Four of the six complete test
points passed, while two still had no clear answer.

**What it means:** the targeted change resolved the metallicity checks.
It did not resolve everything, so we examined the remaining failures separately.

Record: job 1921589.

10. Was One of Our Checks Making the Wrong Assumption?
----------------------------------------------------------------------

**We wanted to know:** could the checking code itself be rejecting useful
measurements?

**We did:** inspect how it decided whether a small numerical change was large
enough to measure reliably.

**We saw:** it assumed a lower precision than the result actually used. We
corrected that assumption. Four remaining checks then passed, but one
redshift-related problem remained.

**What it means:** we fixed a problem in the test, not in the physics. Checking
the model is only useful if the checks are implemented correctly too.

Record: job 1922142.

11. Can We Resolve the Last Redshift Example?
----------------------------------------------------------------------

**We wanted to know:** why did that particular redshift check still have no
clear answer?

**We did:** repeat the redshift calculation with more precision while keeping
the other physical properties fixed.

**We saw:** the check passed. The change in predicted light was tiny compared
with the measurement uncertainty: at most about 0.000022 times that uncertainty.

**What it means:** this was encouraging for this specific calculation. But
training changes several linked quantities, not just redshift alone. We still
needed to test the combined calculation used by training.

Record: job 1922455.

12. Does the Revised Physics Calculation Make Training Work?
----------------------------------------------------------------------

**We wanted to know:** once the numerical changes are combined, do we also
get more reliable galaxy distributions?

**We did:** combine the changes, check the complete calculation, and run
several training versions using it.

**We saw:** the six tested numerical examples passed. The training versions
finished, but their weights were still too concentrated. Version C behaved
better than the other tested versions in some checks, so it became a fixed
starting point for the next experiments.

**What it means:** we made progress on the correctness of the calculation.
That did not automatically make the network's uncertainty estimates reliable.

Record: job 1923347; ``frozen_parent_precision_night_v1``.

13. Can We Improve the Answer for One Galaxy at a Time?
----------------------------------------------------------------------

**We wanted to know:** perhaps the shared network is only a rough starting
point. Would extra work on each individual galaxy improve its answers?

**We did:** start from the network's suggestions and adjust them separately
for each galaxy. We compared different step sizes and numbers of samples
used to calculate a step. We repeated this from two starting states.

**We saw:** the predicted light often improved. The weights did not improve
reliably. All 48 final checks for the observed galaxies raised a warning
about a few samples dominating the answer.

**What it means:** extra work per galaxy helped the fit, but the tested short
runs did not solve the uncertainty problem.

.. figure:: _static/feniks_debug/trajectories.png
   :width: 100%

   How to read this plot: lower residuals mean a closer match to the light.
   Higher ESS means the weights are spread over more samples. In these
   examples, the first improves without a steady improvement in the second.

.. figure:: _static/feniks_debug/final_support.png
   :width: 100%

   Different settings and starting states still leave few samples carrying
   much of the weight. This is why we did not call the better fits a solution.

Records: jobs 1938818 and 1948458. Technical name: local variational inference.

14. Are the Suggested Answers Too Similar to Each Other?
----------------------------------------------------------------------

**We wanted to know:** perhaps the network was looking in too small a range
of possible galaxy properties.

**We did:** spread the suggestions over a wider range. We also mixed in
suggestions from the original network. We did not train anything in this test.

**We saw:** wider ranges sometimes helped, but often added galaxies that
matched the light poorly. The weights remained unreliable in most cases.

**What it means:** simply making every uncertainty wider is not enough.
The important answers may lie in particular combinations of properties,
not in every direction around the original suggestions.

Record: job 1952467; ``frozen_parent_support_probe_v1``.

15. Would a Much Longer Run Per Galaxy Solve It?
----------------------------------------------------------------------

**We wanted to know:** had the previous runs simply stopped too early?

**We did:** increase the number of adjustment steps from 64 to 512. We saved
results along the way and kept both starting states.

**We saw:** several galaxies had much better light predictions. But the
reliability of the weights still varied strongly. Some difficult galaxies
remained difficult.

**What it means:** more time helped some fits, but did not consistently solve
the main problem. We next checked whether the apparent successes depended
on which random samples happened to be drawn.

Record: job 1957394; ``frozen_parent_long_local_vi_v1``.

16. Do the Good Results Survive a New Set of Samples?
----------------------------------------------------------------------

**We wanted to know:** could an apparently good result be due to a lucky
set of random suggestions?

**We did:** keep the trained models unchanged. Draw 4,096 new samples instead
of 1,024 and repeat the evaluation.

**We saw:** some models that previously passed a weight check no longer
passed. At the larger sample count, 31 of the 32 locally adjusted models
raised that warning. Their light predictions were more stable than their weights.

**What it means:** the earlier occasional successes were not strong enough
evidence. Drawing more samples exposed the weakness; it did not repair the model.

.. figure:: _static/feniks_debug/replay_examples.png
   :width: 100%

   Same models, new random samples. The match to the light can stay similar
   while the effective number of weighted samples changes substantially.

Record: job 1959175; ``frozen_parent_long_replay_v1``.

17. Have We Checked the Entire Calculation Used to Adjust the Network?
----------------------------------------------------------------------

**We wanted to know:** the physics had been checked, but was the complete
calculation used for each network adjustment also behaving correctly?

**We did:** check how the network assigns probabilities, how those probabilities
combine with the physics, and how the full result changes with small inputs.

**We saw:** 31 of 32 starting states passed. One still gave no clear answer.
The program therefore did not begin the planned comparison of training methods.

**What it means:** a remaining numerical uncertainty stopped us from confidently
interpreting that comparison. We investigated how precisely the network did
its internal calculations.

Record: job 1960443. Technical name: complete VI objective audit.

18. Does More Precision Inside the Network Resolve That Check?
----------------------------------------------------------------------

**We wanted to know:** could rounding inside the network explain the last
unclear result?

**We did:** keep the same starting states and random inputs, but perform the
relevant internal network calculation with more precision. We also repeated
the old calculation for comparison.

**We saw:** the new version passed all 32 checks. The old version reproduced
its one unclear result. Both had already returned high-precision outputs;
the important change was inside the calculation.

**What it means:** we could now run the planned comparison with better numerical
evidence. This checked the starting states, not every state the network might
reach later.

.. figure:: _static/feniks_debug/transport_resolution.png
   :width: 100%

   A closer look at the earlier unclear check. The horizontal axis changes
   the size of the small test perturbation. The point is to look for agreement
   over several sizes, not to choose one size that happens to look good.

Record: job 1961888. The new calculation is called ``transport64`` in the logs.

19. Which of Two Ways of Improving the Answers Works Better?
----------------------------------------------------------------------

**We wanted to know:** should we improve the current suggestions directly,
or learn from a set of weighted suggestions?

**We did:** compare two methods on the same 16 galaxies and two starting states:

* The first method changes the current suggestions using the physics-based
  training score. It is called **reverse** in the plots.
* The second method draws suggestions, weights them using the observations,
  and trains the network to give more probability to the highly weighted ones.
  It is called **wake** in the plots.

Both were allowed the same number of new galaxy-light calculations for
their adjustment steps. This does not mean their total computing costs were equal.

**We saw:** wake rejected most changes because too few samples carried useful
weight. Only 9 of 32 runs changed their parameters. Those nine ended with a
lower effective weighted sample count than their starting models. Neither
method consistently resolved the weight warnings.

**What it means:** we now had a specific change to investigate. When wake
did accept an adjustment, why could the result get worse?

Record: job 1962310; ``frozen_parent_objective_transport64_pilot_v1``.

20. Why Did the Overnight Run Stop After Two Minutes?
----------------------------------------------------------------------

**We wanted to know:** were the short-run results good enough to justify a
much longer run?

**We did:** make the overnight job check the completed short run before
starting more training. We required enough actual updates and an improvement
in the effective weighted sample count.

**We saw:** those requirements were not met. The job ended without starting
the long training.

**What it means:** this was an intentional stop, not a crash and not a completed
night of learning. It prevented us from spending hours repeating a method
that had not shown convincing improvement.

Record: job 1962505; result ``NIGHT_EXTENSION_NOT_STARTED``.

21. Was the Accepted Change Simply Too Large?
----------------------------------------------------------------------

**We wanted to know:** did wake choose the wrong direction, or take too large
a step in a useful direction?

**We did:** replay the previous runs with exactly the same random choices.
We inspected the first accepted change in each of the nine runs that changed.
We tried the full change, one tenth of it, one hundredth of it and no change.

**We saw:** all 32 replays reproduced the original final parameters exactly.
For the nine inspected changes, a very small movement in the chosen direction
would improve the training score. Yet seven of the nine full changes made
that same score worse.

**What it means:** in these cases, the step went too far. The direction was
locally useful, but that did not make the complete step safe.

.. figure:: _static/feniks_debug/wake_derivatives.png
   :width: 100%

   The two calculations of the local direction agree when the test change
   becomes small. This helps rule out a wrong local direction in these examples.

.. figure:: _static/feniks_debug/wake_loss.png
   :width: 100%

   Bars to the right of zero mean the training score became worse after
   the accepted step. This happened in seven of the nine inspected changes.

There was a second lesson. Two full changes improved the training score but
still worsened the match to the light on fresh evaluation samples. Smaller
changes sometimes helped, but no tested size improved everything for every galaxy.

.. figure:: _static/feniks_debug/wake_scales.png
   :width: 100%

   Compare each smaller change with doing nothing. On the left, below one
   means a better match to the light. On the right, above one means more
   effective weighted samples. A change can help one measure and hurt the other.

**What it means for the next test:** we need to prevent steps that worsen
their own training score. Then we must separately check whether the answers
improve on samples that were not used to choose the step.

Record: job 1965476; :download:`recorded first-step values
<../wake_forensic_evidence.csv>`. The other 23 runs had no accepted change.

22. Does Checking the Step Before Keeping It Help?
----------------------------------------------------------------------

**We want to know:** can we prevent the harmful large changes without stopping
all useful learning?

**We changed the code:** before keeping an adjustment, the program checks
whether it reduces the training score on the same samples. If it does not,
the program tries a smaller adjustment. If none of the allowed sizes works,
it keeps the old parameters and the old training state.

**We are testing:** the same galaxies, starting states and sample budgets
as before. We keep the other training method unchanged for comparison.

**We saw:** all accepted changes reduced their own training loss. But 23 of
32 trajectories did not change. Of the nine that changed, three improved the
effective weighted sample count and six worsened it. None improved both that
count and the match to the light in this evaluation. The run took about 90 minutes.

**What it means:** the step-safety correction worked on this run. It did not
solve the reliability of the answers. We now need to check whether an update
helps only the examples used to calculate it, rather than other examples too.

Record: ``frozen_parent_wake_descent_v1``. Code name: ``wake_armijo_v1``.
Commands: :download:`run instructions <../feniks_wake_descent_runbook.md>`.

23. Does the Change Help on Different Examples Too?
----------------------------------------------------------------------

**We want to know:** does a change that helps its training examples also help
on another set of examples that did not choose the change?

**We will do:** repeat the guarded run and add two fresh sets of 256 suggestions
after each attempted change. For each set, compare the score before and after
the change using the same suggestions and weights. These scores cannot accept,
reject or resize a change. They only measure what happened.

**We will also check:** whether a few suggestions dominate each new set. If
they do, that set cannot give a reliable answer to our question. We keep these
cases in the report rather than discarding them.

**What it can tell us:** better training scores but worse scores on informative
new sets would show that the changes do not reliably help beyond their training
examples. If almost no new set has useful weights, we still cannot answer that
question: we first need better examples to learn and evaluate from.

Status: implemented, not submitted here. One H100; no shared-network training.
:download:`Launch and readback instructions <../feniks_wake_holdout_runbook.md>`.

What We Know Now
----------------------------------------------------------------------

**Problems we found and corrected:**

* The network blocks did not transform all 15 numbers used to describe a galaxy.
  They now do.
* Several parts of the numerical calculation needed changes or more precision.
  The revised versions passed the checks described above.
* One numerical check used an incorrect precision assumption. We corrected it.

**Problem we reproduced and wrote a correction for:**

* Some accepted wake changes were too large and worsened their own training
  score. The current code checks and reduces the change before keeping it.
  The completed test confirms descent on the training examples, but not a
  consistent improvement in the independent results.

**Problem we have not yet shown to be solved:**

* Too few proposed galaxies can still carry almost all the weight. Until this
  improves reliably, we cannot claim that the reported uncertainties are correct.

**The main lesson:** better light predictions, correct numerical calculations,
safe training steps and trustworthy uncertainties are different achievements.
We need all of them. Completing one does not automatically complete the others.

More detail: :doc:`feniks_debug_meeting` contains the starting-point plots and
run paths. :doc:`feniks_debug_metrics` contains the equations. The
:download:`technical log <../feniks_decoder_debug_log.md>` retains the earlier
implementation details and numerical results.
