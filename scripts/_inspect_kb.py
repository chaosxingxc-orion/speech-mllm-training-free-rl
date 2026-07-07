"""Throwaway: inspect passage/question/answer text fields in candidate QA sets, to pick a
boundary-clean external-knowledge KB for the T6 retrieve->gate->inject->generate loop."""
import pyarrow.parquet as pq, glob, os

D = os.environ["SPEECHRL_DATA_DIR"] + "/datasets"
for name in ["heysquad", "spoken-squad", "uro-bench/SQuAD-zh", "uro-bench"]:
    fs = sorted(glob.glob(f"{D}/{name}/**/*.parquet", recursive=True))
    print("==", name, "n_files", len(fs))
    for f in fs[:2]:
        try:
            s = pq.read_schema(f)
        except Exception as e:
            print("  (schema err)", type(e).__name__); continue
        print("  ", os.path.relpath(f, D), "->", s.names)
    if fs:
        try:
            t = pq.read_table(fs[0]).slice(0, 1).to_pylist()[0]
            for k, v in t.items():
                vs = f"<{type(v).__name__}>" if isinstance(v, (dict, bytes, list)) else str(v)[:90]
                print(f"     {k}: {vs}")
        except Exception as e:
            print("  (row err)", type(e).__name__)
