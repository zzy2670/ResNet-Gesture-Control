# Twelve-model CPU numerical audit preparation

Status: PREPARED / WAITING for the unique real `label_variants_cpu_benchmark.json`.
This note is an internal source preparation receipt, not a benchmark or a passing result audit.

Writer-owned new source: `audit_label_variants_cpu.py`.
Source SHA256: `988a4ce50d2921d46b29381d4c31fd09764028165a1d6ac1ddbf657f0a4e5033`.
Producer and frozen protocol remain unchanged. Their identities are taken from the real source-preflight report and checked against live bytes during the eventual audit.

After root completes the one authorized real benchmark, run:

```powershell
& E:/python/python.exe C:/Users/Administrator/Desktop/6216_VISION/MA6216_CA3_final_project/Vision_CA/runs/MERGED_20261102/audit_label_variants_cpu.py --run
```

Default and `--help` use only the standard library. Even `--run` performs no ML imports, model loading, tensor construction, inference, calibration, accuracy evaluation, or timing. Missing real input prints WAITING, exits 2 and leaves the canonical review path absent. An existing canonical review is never overwritten.

The eventual output is `label_variants_cpu_independent_review.json`. Its successful schema contains `status=PASS`, `issues=[]`, `data_sha256`, `audit_source_sha256`, `bound_file_sha256`, `checks`, 24 independently recomputed summaries, all 72 original CPU-background records with independently recomputed estimates, execution timestamps, environment and explicit limitations. Failure preserves `status=FAIL`, exception and traceback in the exclusively created output; it never remeasures or chooses a faster run. Success bytes are materialized first, then every live bound file is rehashed immediately before exclusive write, flush and fsync.

The audit checks complete/unique 12 identities, fixed 100 photo and 100 GT crop identities and order, frozen tensor SHA equality with the old reference, live source/crop/index/annotation/weight/calibration/parameter/scientific-review bindings, six merged scientific reports and old unmerged checkpoint identity, selected TPE parameter equality, architecture parameter counts, six outputs, CPU/float32/8 threads/batch1, 20 warmups, three fixed forward/reverse/rotate6 orders, and 72 blocks with 7,200 positive finite raw ms values. All means, p50/p95 (linear interpolation), ddof=1 sample SD and three repeat means are independently recomputed. Exact journal events must show scientific readiness and actual dependency import before all 12 loads, then warmup and every block in fixed scope/order. Every journal block must exactly match its corresponding result block. Workload records retain each whole-photo input order and accepted/detected counts. Slow blocks are never excluded.

Preparation checks actually completed without model imports or real audit execution:

- AST parse; default and `--help` exit 0; no torch/torchvision/numpy/psutil/PIL modules imported.
- Pure numerical example `1..300`: n=300, mean=p50=150.5, p95=285.05, sample SD=86.74675786448736.
- Nonfinite, zero, negative and boolean timing examples rejected.
- In-memory CPU-counter example recomputed 80% busy / 40% estimated other processes; a supplied 39% estimate rejected.
- Real benchmark and canonical independent review were absent at preparation completion; no fake real report was produced.

Limits: This is a numerical and live-identity audit. Journal chronology and pinned producer source establish recorded gate order, not independent OS attestation. Tensor SHA equality is checked against the frozen historical reference; this standard-library audit does not reconstruct tensors. Background counters cannot prove Windows system silence or exclude thermal/scheduler effects. These are repeated timings of fixed A-val inputs, not new fair accuracy results. Frozen thresholds preserve actual workload differences and do not support causal speed claims.

Diagnostic correction: an initial ad hoc read of A18 `fixed_head_summary.json` omitted explicit UTF-8 and produced a Windows-default-decoding error. Reading actual bytes as UTF-8-sig succeeds; 52,390 bytes and SHA256 `f4e2fad19117993728a9547cd5a4558484af80cdc480cd4d40918d657df664be` match root's verification. No scientific artifact defect was established and no original file was changed. The audit's JSON reader always uses UTF-8-sig.

Root, as the non-author, owns the independent source review and any eventual decision to run against the completed real benchmark. This preparation does not satisfy the CAM/portable/document result gate.
