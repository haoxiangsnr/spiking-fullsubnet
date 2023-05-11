import argparse
from pathlib import Path

import librosa


def format_output(lines, logfile, index):
    logfile = Path(logfile).expanduser().absolute()
    CRED = "\x1b[6;30;42m"
    CEND = "\x1b[0m"

    print(f"{CRED}{index + 1} - {logfile}{CEND}")
    print(*lines)


def main(args):
    logfiles = librosa.util.find_files(args.check_dir, ext=["log"])
    for index, logfile in enumerate(logfiles):
        with open(logfile) as f:
            lines = f.readlines()
            lines.reverse()
            for i, line in enumerate(lines):
                if "best" in line:
                    outputs = lines[i : i + args.line_offset]
                    outputs.reverse()
                    format_output(outputs, logfile, index)
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        epilog="Example: python tools/seek_best_score_from_logfile.py -D ./recipes"
    )
    parser.add_argument("-D", "--check_dir", type=str, required="./recipes")
    parser.add_argument("-L", "--line_offset", type=int, default=12)
    args = parser.parse_args()
    main(args)
