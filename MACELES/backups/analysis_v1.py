import json
import argparse
from pathlib import Path
import re, os

def dedup_hills_by_time(input_file: Path, output_file: Path, keep="last"):
    headers = []
    data = []

    with open(input_file, "r") as fin:
        for line in fin:
            s = line.strip()
            if not s:
                continue

            if s.startswith("#") or s.startswith("@"):
                headers.append(line)
                continue

            cols = line.split()
            t = float(cols[0])
            data.append((t, line))

    # sort by first column (time)
    data.sort(key=lambda x: x[0])

    dedup = {}

    if keep == "last":
        for t, line in data:
            dedup[t] = line
    elif keep == "first":
        for t, line in data:
            if t not in dedup:
                dedup[t] = line
    else:
        raise ValueError("keep must be 'last' or 'first'")

    times = sorted(dedup.keys())

    with open(output_file, "w") as fout:
        # keep only one copy of header lines
        seen_header = set()
        for h in headers:
            if h not in seen_header:
                fout.write(h)
                seen_header.add(h)

        for t in times:
            fout.write(dedup[t])

    print(f"Input entries : {len(data)}")
    print(f"Unique entries: {len(times)}")
    print(f"Removed       : {len(data) - len(times)}")
    print(f"Wrote         : {output_file}")


def convert_hills_line_to_global(line: str, initial_step: int, dt_fs: float) -> str:
    s = line.strip()

    # keep blank or header lines unchanged
    if not s or s.startswith("#") or s.startswith("@"):
        return line

    cols = line.split()
    local_time_fs = float(cols[0])

    local_step = int(round(local_time_fs / dt_fs))
    global_step = initial_step + local_step
    global_time_fs = global_step * dt_fs

    cols[0] = f"{global_time_fs:.8f}"
    return " ".join(cols) + "\n"


def load_meta(meta_path: Path):
    with open(meta_path, "r") as f:
        meta = json.load(f)

    required = ["segment_id", "initial_step", "last_step"]
    for key in required:
        if key not in meta:
            raise ValueError(f"{meta_path}: missing required key '{key}'")

    return meta


def find_hills_for_segment(meta_path: Path, segment_id: int, hills_prefix: str = "HILLS"):
    """
    Default convention:
      meta_0003.json  <->  HILLS_0003
    You can change this if your naming differs.
    """
    candidate1 = meta_path.with_name(f"{hills_prefix}_{segment_id:04d}")
    candidate2 = meta_path.with_name(f"{hills_prefix}_{segment_id:02d}")
    candidate3 = meta_path.with_name(f"{hills_prefix}_{segment_id}")

    for c in (candidate1, candidate2, candidate3):
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Could not find HILLS file for segment_id={segment_id} near {meta_path}"
    )


def parse_hills_step(line: str, dt_fs: float):
    cols = line.split()
    if len(cols) == 0:
        raise ValueError("Empty HILLS data line")

    t_fs = float(cols[0])
    step = int(round(t_fs / dt_fs))
    return step, t_fs

def merge_hills(meta_files, output_path, outdir):
    metas = [load_meta(Path(p)) for p in meta_files]
    metas.sort(key=lambda m: m["segment_id"])

    header_written = False
    prev_last_step = None

    with open(output_path, "w") as fout:
        for meta_path_str, meta in zip(meta_files, metas):
            meta_path = Path(meta_path_str)

            seg = int(meta["segment_id"])
            initial_step = int(meta["initial_step"])
            last_step = int(meta["last_step"])
            dt_fs = float(meta["dt_fs"])

            hills_path = outdir / f"HILLS_seg{seg:04d}"  # strongly recommended

            if prev_last_step is not None and initial_step != prev_last_step:
                print(f"[WARN] discontinuity: seg {seg} initial={initial_step}, prev={prev_last_step}")

            with open(hills_path, "r") as fin:
                for line in fin:
                    s = line.strip()
                    if not s:
                        continue

                    if s.startswith("#") or s.startswith("@"):
                        if not header_written:
                            fout.write(line)
                        continue

                    cols = s.split()
                    t_fs = float(cols[0])
                    step = int(round(t_fs / dt_fs))

                    if 0 < step <= last_step-initial_step:
                        newline=convert_hills_line_to_global(line, initial_step, dt_fs)
                        fout.write(newline)

            header_written = True
            prev_last_step = last_step

            print(f"[seg {seg}] used window ({initial_step}, {last_step}] dt={dt_fs}")

    print(f"Written merged HILLS to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        required=True,
        help="Locations of output files and meta files"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Merged HILLS output file"
    )

    args = parser.parse_args()

    merge_hills(
        meta_files=sorted(Path(args.outdir).glob("meta_seg*")),
        output_path=Path(args.output),
        outdir=Path(args.outdir),
    )
    tmp = Path(args.output).with_suffix(".tmp")
    dedup_hills_by_time(Path(args.output), tmp)
    os.replace(tmp, Path(args.output))



if __name__ == "__main__":
    main()



