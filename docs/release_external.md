# External-only release, 11 November 2026

Status: **ready for team review; not published or submitted**.

Build from the repository root, with the data bundle set via MTAH_DATA_DIR:

```powershell
.\.venv\Scripts\python.exe scripts/release.py
```

The default configuration is `configs/final_external.toml`. The technical
document source is `docs/submission_report.md`. The package is written to
`outputs/submission_external/`, separate from the old release:

- `Aziz goes to barcelona.csv`: 199,319 algorithm-generated labels.
- `Aziz goes to barcelona.md`: current standalone technical report.
- `config.toml`: exact configuration used for this generation.
- `Aziz goes to barcelona.csv.manifest.json`: input/output hashes and provenance.
- `release_report.json`: checks, hashes, configuration digest and environment.
- `submission.zip`: the five files above, selected by an explicit allowlist.

CSV SHA-256:
`f6f78ee82869ba63044548e7884d544455396e872e349834747560200998852b`.
The script checks the exact packaged CSV with the official validator, strict
structure checks, LF endings, and an independent second generation. It verifies
that the technical report declares the same CSV hash and configuration digest.
A mismatching document fails the build and no current submission.zip is issued.
An existing ZIP is retained under a timestamped backup name before a rebuild.

The archive is for local review. The submission form receives the CSV and MD
separately, together with the public source repository URL. No match tracking,
video, ground truth, benchmark datasets or credentials are added to the archive.
The manifest contains hashes and aggregate diagnostics, not tracking rows.

Validation: 54 release/rules checks passed, including a real package build,
ZIP-content verification, and rejection of a mismatching technical document.
The previous model verification passed 311 tests; no model code changed in this
packaging round. Repository data/history restrictions are checked by test_rules.

## Team review still needed

- Confirm the registered team name and review the report and source notices.
- The source changes are local/uncommitted. Review and commit the revision before
  publishing so the remote repository can actually reproduce this CSV.
- Review repository visibility and solution-code licensing; data rights remain
  separate from code rights. External sources and usage notices are in
  `docs/rules_update.md` and the standalone report.
- A teammate should reproduce the CSV and compare its hash on another machine.
- Publishing and submitting the form are separate actions; neither was performed.

The generated release report records the actual commit and dirty flag. A dirty
checkout is explicitly a review artifact, not a claim that the current remote
already contains the packaged model.
