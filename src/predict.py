"""The whole pipeline -- tracking in, one label per frame out -- and its command line.

    python -m src.predict --config configs/hybrid_v1.toml --output outputs/candidates/hybrid_v1.csv

predict() is pure: arrays in, labels out, no I/O, so the synthetic acceptance tests and the real
match run exactly the same code. run() adds the I/O. It loads the match through games.py (set
MTAH_DATA_DIR), writes the CSV with src.submission.write (LF, refused if invalid), and writes
<output>.manifest.json beside it: config digest, git commit, input and output SHA-256,
environment, runtime and aggregate diagnostics -- never frame-level values (D-002).
"""

import argparse
import sys


def predict(tracking, context, config, tags=None):
    """(n,) '<U1' labels H/A/D/X for every frame. Ticket T6; tests/test_predict.py.

    tracking -- src.data_loader.Tracking;  context -- src.preprocess.FrameContext;
    config   -- src.config.Config;          tags -- Smart Tagging events, used only if
                config.smart_tagging.enabled (P1, ticket T9).

    decoder "majority": src.features.nearest_team -> src.temporal.majority. With
        output.ungraded = "keep" this is the official baseline file, frame for frame.
    decoder "viterbi": src.features.build_features -> team_evidence, dead_evidence (+ tag evidence)
        -> src.temporal.decode -> letters.
    output.ungraded = "X": X on every frame outside context.blocks; "keep": decoder output there.
    Deterministic: the same inputs give the same labels, bit for bit.
    """
    raise NotImplementedError("Phase 3, ticket T6 -- docs/tickets.md")


def run(config_path, output_path, data_dir=None):
    """Predict the tactical match and write the CSV and its manifest. Returns the CSV path.

    The manifest is <output_path>.manifest.json: run name, config path and digest, git commit and
    dirty flag, SHA-256 of the ATD files read and of the CSV written, Python and package versions,
    runtime, and aggregate diagnostics (label counts, possessions per half, mean duration).
    """
    raise NotImplementedError("Phase 3, ticket T6 -- docs/tickets.md")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="a configs/*.toml file")
    parser.add_argument("--output", required=True, help="CSV to write; the manifest goes beside it")
    parser.add_argument("--data-dir", help="the data bundle, if not in MTAH_DATA_DIR or ./data")
    args = parser.parse_args(argv)
    run(args.config, args.output, args.data_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
