"""Download the other-match benchmark data (docs/experiments.md §5) into outputs/omb/.

    python scripts/omb_fetch.py                 # all seven IDSSE/DFL matches, ~3 GB
    python scripts/omb_fetch.py J03WMX          # one match

IDSSE / DFL open data: Bassek, M., Rein, R., Weber, H. et al. "An integrated dataset of
spatiotemporal and event data in elite soccer." Sci Data 12, 195 (2025), CC BY 4.0. Seven German
Bundesliga / 2. Bundesliga matches, 2022/23, 25 fps positions with the ball, per-frame BallStatus
and BallPossession -- labels for *other* matches, which RULES.md allows in any amount.

Files come from ndownloader.figshare.com: kloppy's springernature.figshare.com URLs answer
HTTP 202 to scripts (D-017). The file ids are the ones kloppy 3.18 lists. outputs/ is gitignored,
and nothing here is redistributed.
"""

import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Params
DOWNLOAD_URL = "https://ndownloader.figshare.com/files/{}"
DFL_DIR = os.path.join(ROOT, "outputs", "omb", "dfl")
# match id -> figshare file ids (kloppy/_providers/sportec.py, get_IDSSE_url)
DFL_FILES = {
    "J03WMX": {"meta": 51643475, "tracking": 51643514},   # 1. FC Köln - FC Bayern München
    "J03WN1": {"meta": 51643472, "tracking": 51643517},   # VfL Bochum 1848 - Bayer 04 Leverkusen
    "J03WPY": {"meta": 51643487, "tracking": 51643526},   # Fortuna Düsseldorf - 1. FC Nürnberg
    "J03WOH": {"meta": 51643478, "tracking": 51643520},   # Fortuna Düsseldorf - SSV Jahn Regensburg
    "J03WQQ": {"meta": 51643484, "tracking": 51643529},   # Fortuna Düsseldorf - FC St. Pauli
    "J03WOY": {"meta": 51643481, "tracking": 51643523},   # Fortuna Düsseldorf - F.C. Hansa Rostock
    "J03WR9": {"meta": 51643490, "tracking": 51643532},   # Fortuna Düsseldorf - 1. FC Kaiserslautern
}
ATTEMPTS = 3
BLOCK = 1 << 20


def fetch(file_id, path):
    """Download one file to path via a .part file, resuming a .part left by an interrupted run;
    skip it if path already exists."""
    if os.path.isfile(path):
        return "present"
    part = path + ".part"
    for attempt in range(1, ATTEMPTS + 1):
        try:
            done = os.path.getsize(part) if os.path.isfile(part) else 0
            request = urllib.request.Request(DOWNLOAD_URL.format(file_id))
            if done:
                request.add_header("Range", "bytes={}-".format(done))
            with urllib.request.urlopen(request, timeout=120) as response:
                resumed = done and response.status == 206
                expected = int(response.headers.get("Content-Length") or 0) + (done if resumed else 0)
                with open(part, "ab" if resumed else "wb") as handle:
                    written = done if resumed else 0
                    for block in iter(lambda: response.read(BLOCK), b""):
                        handle.write(block)
                        written += len(block)
            if expected and written != expected:
                raise IOError("got {} of {} bytes".format(written, expected))
            os.replace(part, path)
            return "{:.1f} MB".format(written / 1e6)
        except Exception as error:  # network trouble: retry, then report and move on
            if attempt == ATTEMPTS:
                return "FAILED ({}: {})".format(type(error).__name__, error)
            time.sleep(5 * attempt)


def main(argv):
    matches = argv or list(DFL_FILES)
    for match in matches:
        folder = os.path.join(DFL_DIR, match)
        os.makedirs(folder, exist_ok=True)
        for kind, file_id in DFL_FILES[match].items():
            started = time.time()
            status = fetch(file_id, os.path.join(folder, kind + ".xml"))
            print("{} {:<8} {}  ({:.0f} s)".format(match, kind, status, time.time() - started),
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
