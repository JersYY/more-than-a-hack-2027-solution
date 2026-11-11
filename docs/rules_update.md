# Additional rules supplied by the team, 2026-11-11

Source: the two organiser-rule screenshots supplied in this conversation.
These supplement the repository's earlier RULES.md. The stricter rule applies
where older project decisions conflict with them.

- Do not check predictions against the provided video.
- No manual labels, manual corrections or other manually created data on the
  provided game.
- Do not upload the provided data to AI services. Local scripts may process it;
  only aggregate diagnostics and PASS/FAIL are reported to the assistant.
- Training, fitting and parameter tuning must use external datasets only, never
  provided-game ground truth or the provided video.
- Automatic checks on the provided data and algorithmically generated inputs
  are allowed. Such checks do not select model parameters in the new workflow.
- Check usage rights of every external dataset and acknowledge sources.

## Parameter provenance correction

Historical D-026 selected stop/restart costs from aggregate plausibility checks
on the provided match. Even without labels or video comparison, that is not the
external-only selection procedure required for this workflow. Original final,
timing_v1 and final_v2 inherit those costs and are retained as historical
artifacts, not the recommended configuration under the clarified policy.

The new validation starts from `configs/hybrid_v2.toml` (E6 external benchmark
calibration): stop cost 6.9, restart cost 5.9. Timing adjustment selection and
LOMO are repeated on external matches. Competition-file generation and format,
hash and determinism checks happen only after choices are fixed.

Synthetic cases are code-generated behaviours derived from the general task
definition, not hand annotations or corrections of the provided match. The
tracking-loss profile uses aggregate input quality to model benchmark conditions;
it is not a manually corrected dataset or a fitted possession model on the target.
The new LOMO procedure evaluates the timing adjustment with other parameters
fixed; it does not redo every historical E6 calibration fold.

## External data usage check

- IDSSE/DFL: CC BY 4.0 confirmed by the authors' dataset paper,
  [Bassek et al., Scientific Data 12, 195 (2025)](https://www.nature.com/articles/s41597-025-04505-y).
  Existing project credits identify the source; local benchmark transformations
  are ball/player loss and smoothing. No dataset redistribution is performed.
- Metrica Sample Game 3: the publisher's
  [README](https://github.com/metrica-sports/sample-data#legal-stuff) offers sample
  data to the analytics community and requests source acknowledgement for public
  use. The root listing has no standalone LICENSE file; do not describe this as
  MIT or CC BY. Used as the publisher's sample-data benchmark with Metrica
  attribution, without redistributing source files. Broader rights are not inferred.

No video has been opened, no output has been compared against video, and no
competition ground-truth scoring was used during this implementation round.
Historical checks and decisions remain documented rather than being rewritten.

Verification complete: 311 tests pass, including parameter-provenance checks
and byte-identical reruns of final_external. Its CSV passes official/strict
format validation. This is evidence of implementation checks, not a guarantee
of competition eligibility or target-match accuracy.
