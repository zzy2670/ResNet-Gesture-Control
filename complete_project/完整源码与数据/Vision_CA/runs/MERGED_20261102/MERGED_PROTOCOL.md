# Merged gesture experiment

This is a separate six-model experiment under `MERGED_20261102/{A,B}/{resnet18,resnet34,resnet50}`. All original A/B programs, splits, classifiers, detector weights, TPE records and evaluation results remain unchanged.

Class 1 (forward) combines `peace`, `peace_inverted`, `two_up`, and `two_up_inverted`. Classes 0, 2, 3, 4 and 5 retain their original meaning. Photo commands are recalculated using the same stop-first and conflicting-action rejection rules.

Each protocol retains its exact original train/validation/test manifests. A uses `data/A_split_verified_20261102/P1_random_image`; B uses `data/B_split_verified_20261102`. New crop indices preserve original image/hand order and paths. Frozen feature tensors are reused only with verified identical official ImageNet V1 initialization, identical tensor values and updated labels. Test feature caches are never used for training.

For each architecture and protocol, only `best_params` from its own original TPE summary are reused. No new TPE search or trial weights are used. A new dropout/linear head is trained for exactly 18 epochs at batch size 512 on the new train labels. Optimizer, learning rate, weight decay, dropout, smoothing, class weighting and cosine schedule follow that fixed configuration and the original algorithm. Validation macro-F1 selects the best new head. `fixed_head_summary.json` and `fixed_head_best.pt` preserve all 18 actual epochs and selected head independently of later fine tuning.

The selected new head is loaded into that architecture's official ImageNet V1 backbone. The original `base.fine_tune` then trains layer4 and fc only, at most 18 epochs, batch size 128, AdamW layer4 LR 0.0001 / head LR 0.0005, original weight decay/smoothing/class weights, cosine schedule, CUDA AMP, gradient clipping 5, and patience 4. Validation must improve on the frozen head to replace it. If fine tuning never improves, the final checkpoint remains the new frozen head; `finetune_summary.json` reports the actual epochs and selected stage without claiming 18 fine-tuning epochs.

The detector is copied from the same original protocol, never cross-protocol and never retrained. Original validation calibration evaluates all 12 detector/classification threshold pairs (3 x 4), then freezes both weight hashes and the chosen thresholds. Only after freezing does full final test evaluation run, preserving GT-crop, complete-photo and detector results, all examples and actual denominators, including zero/stop false motion. No test result selects a model or threshold.

Six CPU smoke stages run before any long CUDA stage: actual official model construction, strict classifier rebuild/load, real train crop forward/backward, strict same-protocol detector load, and real detector forward. The controller then runs each model's head, fine tune, detector copy, calibration and test serially on one GPU. CPU speed comparisons are deliberately scheduled separately after all training by the root coordinator.

Offline initialization defaults to the existing `ResNet项目完整交接_20261102/完整源码与数据/Vision_CA/pretrained` directory. `--pretrained-root` can explicitly choose another local cache. Official filename hash prefixes are verified and socket connections are blocked; there is no fallback download. The default interpreter is the bundled Codex Python; child `PYTHONPATH` is `E:/python/Lib/site-packages`.

`merged_input_identity.json` pins source files, original manifests and annotations, merged data identities/photo truths/crop indices/features, official initialization, original best-parameter summary and same-protocol detector. Source snapshots are immutable. The controller records actual PIDs and process creation times, commands, original logs, exit codes, and SHA256 output success markers. Completed stages are skipped only after rechecking input/source/output/log identities. A failed attempt is preserved and blocks unattended retry; health inspection is read-only. No failure is converted into success or restarted automatically.

Run CPU preflight only:

```text
<bundled-python> run_merged_training.py --preflight-only
```

After independent review, run the full serial training/evaluation controller:

```text
<bundled-python> run_merged_training.py
```

Merged and unmerged targets have different class semantics and different difficulty. Comparison must name the label version, protocol, architecture, samples, actual training/selection procedure, thresholds and model stage. A higher merged score alone does not establish better generalization. Original six-model results must remain present in the final comparison and handover package.
