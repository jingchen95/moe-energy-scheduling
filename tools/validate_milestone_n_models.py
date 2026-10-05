#!/usr/bin/env python3
"""Offline held-out, mode, composition, power and candidate-decision validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.afd_cost_model import fit_binned_curve, fit_linear_power, load_same_run_cost_profile
from src.afd_cost_transaction import execution_fingerprint, read_complete
from src.afd_cost_validation import ResidualRidge, coverage, error_statistics, ranking_statistics, select_feasible, oracle_decision_gate
from src.afd_static_placement import load_placement, sha256_file
from tools.analyze_milestone_n_cost_model import build_power_dataset


MODELS = ("n_only", "mode_binned", "ridge_residual")
FEATURES = ("active_local_experts", "max_tokens_per_local_expert", "routing_concentration",
            "layer", "token_rows", "combined_token_rows", "replay", "mean_active_load",
            "expert_load_cv", "top1_fraction", "top2_fraction", "top4_fraction", "entropy")
SOURCE_PROFILE = Path("results/raw/milestone-m-afd-profile-20260930T191548Z/runs/c-math-20260930T191743.393254Z-9ad557de/expert_counts.npz")


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dump(path, value):
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def save(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else None


def derive_features(profile, mapping, rank):
    layers = profile["layers"].astype(int)
    local = profile["expert_histograms"][mapping[layers] == rank].reshape(-1, 64).astype(float)
    n = local.sum(axis=1)
    active = (local > 0).sum(axis=1)
    fractions = local / np.maximum(n[:, None], 1)
    ordered = np.sort(fractions, axis=1)
    return pd.DataFrame({
        "runtime_step_id": profile["runtime_step_ids"], "stage": profile["stages"], "layer": layers,
        "n": n, "active_local_experts": active, "max_tokens_per_local_expert": local.max(axis=1),
        "routing_concentration": np.square(fractions).sum(axis=1),
        "token_rows": profile["token_rows_by_source_rank"][rank],
        "combined_token_rows": profile["token_rows_by_source_rank"].sum(axis=0),
        "replay": (profile["run_modes_by_rank"][rank] == "replay").astype(int),
        "mean_active_load": n / np.maximum(active, 1),
        "expert_load_cv": local.std(axis=1) / np.maximum(local.mean(axis=1), 1e-12),
        "top1_fraction": ordered[:, -1], "top2_fraction": ordered[:, -2:].sum(axis=1),
        "top4_fraction": ordered[:, -4:].sum(axis=1),
        "entropy": -(fractions * np.log(np.maximum(fractions, 1e-12))).sum(axis=1),
        "local_ms": profile["local_elapsed_ms_by_rank"][rank],
        "full_ms": profile["full_elapsed_ms_by_rank"][rank],
    })


def ridge_x(frame):
    numerical = np.column_stack([
        np.log1p(frame.n), np.log1p(frame.active_local_experts),
        np.log1p(frame.max_tokens_per_local_expert), frame.routing_concentration,
        np.log1p(frame.combined_token_rows), frame.replay,
        frame.expert_load_cv, frame.top4_fraction, frame.entropy,
    ])
    return np.column_stack([numerical, np.eye(48)[frame.layer.to_numpy(dtype=int)]])


def load_formal(root, allocation, output):
    plan = json.loads((allocation / "manifest.json").read_text())
    integrity = root / "results/processed/milestone_n_272216_integrity"
    summary = json.loads((integrity / "summary.json").read_text())
    assert summary["status"] == "pass" and summary["formal_configurations_revalidated"] == 11
    assert summary["source_commit"] == plan["git_commit"]
    prior = pd.read_csv(integrity / "formal-configurations.csv").set_index("config_id")
    runs = pd.read_csv(allocation / "runs.tsv", sep="\t").set_index("config_id")
    assert len(runs) == 11 and set(runs.status) == {"complete"}
    paths = [allocation / "manifest.json", allocation / "runs.tsv", integrity / "summary.json",
             integrity / "formal-configurations.csv", allocation / "reuse-manifest.json"]
    frames, profiles, mappings = [], {}, {}
    study = allocation.parents[1]
    for config in plan["configurations"]:
        name = config["config_id"]
        directory = study / "timing-configurations" / f'{config["order"]:02d}-{name}'
        run = read_complete(directory, execution_fingerprint(plan, config, plan["patch_sha256"]))
        assert str(run) == runs.loc[name, "run_dir"]
        profile_path = run / "afd_same_run_cost_profile.npz"
        assert sha256_file(profile_path) == prior.loc[name, "profile_sha256"]
        profile = load_same_run_cost_profile(profile_path)
        assert len(profile["layers"]) == prior.loc[name, "records_per_rank"]
        mapping = np.asarray(load_placement(str(root / config["placement"]["path"]))[0])
        assert sha256_file(root / config["placement"]["path"]) == config["placement"]["sha256"]
        profiles[name], mappings[name] = profile, mapping
        paths += [directory / "COMPLETE.json", profile_path, run / "afd_same_run_cost_finalize.json",
                  run / "metrics.json", run / "trace.jsonl", root / config["placement"]["path"]]
        for rank in (0, 1):
            frame = derive_features(profile, mapping, rank)
            frame["config_id"], frame["lifecycle"] = name, config["lifecycle"]
            frame["rank"], frame["gpu"] = rank, rank + 2
            frame["frequency_mhz"] = config["frequency_mhz"]
            frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    assert len(frame) == 621120 and frame.local_ms.gt(0).all()
    reuse = json.loads((allocation / "reuse-manifest.json").read_text())
    for source in reuse["primary_energy_power_data"]:
        path = root / source["path"]
        assert sha256_file(path) == source["sha256"], path
        paths.append(path)
    paths.append(root / SOURCE_PROFILE)
    baseline = root / "results/processed/milestone_n_272216_model_validation"
    paths.extend(baseline.glob("milestone_n_*.csv"))
    paths += [baseline / "milestone_n_model_validation.json", baseline / "milestone_n_joined_invocations.csv.gz",
              root / "tools/validate_milestone_n_models.py", root / "src/afd_cost_validation.py"]
    provenance = [{"path": str(path.relative_to(root)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
                  for path in dict.fromkeys(paths)]
    dump(output / "source_provenance.json", {
        "execution_commit": plan["git_commit"], "job": 272216, "profiler_schema": 3,
        "excluded": ["Phase A", "schema-2 job 272171", "earlier failed jobs", "instrumented serving/energy as performance ground truth"],
        "datasets": provenance,
    })
    return frame, profiles, mappings, plan


def latency_analysis(frame, output):
    models, validations, mode_rows, residuals, coefficients, mode_validation = {}, [], [], [], [], []
    for (rank, frequency), index in frame.groupby(["rank", "frequency_mhz"]).groups.items():
        subset = frame.loc[index]
        training = subset[subset.lifecycle == "primary"]
        base = fit_binned_curve(training.n.to_numpy(), training.local_ms.to_numpy())
        mode = {m: fit_binned_curve(part.n.to_numpy(), part.local_ms.to_numpy())
                for m, part in training.groupby("replay")}
        ridge = ResidualRidge.fit(ridge_x(training), training.local_ms.to_numpy() - base.predict(training.n), alpha=10.0)
        models[(rank, frequency)] = {"base": base, "mode": mode, "ridge": ridge}
        base_pred = base.predict(subset.n.to_numpy())
        mode_pred = np.empty(len(subset))
        for m in (0, 1):
            mask = subset.replay.to_numpy() == m
            if mask.any():
                mode_pred[mask] = mode[m].predict(subset.n.to_numpy()[mask])
        richer_raw = base_pred + ridge.predict(ridge_x(subset))
        for name, values in zip(MODELS, [base_pred, mode_pred, np.maximum(richer_raw, 1e-6)]):
            frame.loc[index, f"pred_{name}"] = values
        dump(output / f"model-rank{rank}-f{frequency}.json", {
            "training_configurations": training.config_id.unique().tolist(),
            "baseline": {"centers": base.centers.tolist(), "values": base.values.tolist(), "counts": base.counts.tolist()},
            "mode_binned": {str(m): {"centers": c.centers.tolist(), "values": c.values.tolist(), "counts": c.counts.tolist()} for m, c in mode.items()},
            "ridge_residual": ridge.to_dict(), "fixed_alpha": 10,
            "features": ["log_n", "log_active", "log_max", "concentration", "log_combined_rows", "replay", "expert_cv", "top4", "entropy"] + [f"layer_{i}" for i in range(48)],
            "richer_clipped_fraction": float(np.mean(richer_raw <= 0)),
            "no_holdout_tuning": True,
        })
        for config_id, group in subset[subset.lifecycle != "primary"].groupby("config_id"):
            for name in MODELS:
                prediction = frame.loc[group.index, f"pred_{name}"].to_numpy()
                validations.append({"config_id": config_id, "evaluation": group.lifecycle.iloc[0],
                                    "rank": rank, "gpu": rank + 2, "frequency_mhz": frequency, "model": name,
                                    **error_statistics(group.local_ms, prediction),
                                    **coverage(training.n, group.n, base.centers)})
                for m, mode_group in group.groupby("replay"):
                    mode_validation.append({"config_id": config_id, "evaluation": group.lifecycle.iloc[0],
                        "rank": rank, "frequency_mhz": frequency, "model": name,
                        "mode": "replay" if m else "eager",
                        **error_statistics(mode_group.local_ms, frame.loc[mode_group.index, f"pred_{name}"]),
                        **coverage(training[training.replay == m].n, mode_group.n, mode[m].centers)})
        for config_id, group in subset.groupby("config_id"):
            for m, part in group.groupby("replay"):
                residual = frame.loc[part.index, "pred_n_only"] - part.local_ms
                mode_rows.append({"config_id": config_id, "rank": rank, "frequency_mhz": frequency,
                                  "mode": "replay" if m else "eager", "samples": len(part),
                                  "n_median": part.n.median(), "local_median_ms": part.local_ms.median(),
                                  "local_p95_ms": part.local_ms.quantile(0.95),
                                  "n_conditioned_residual_mean_ms": residual.mean(),
                                  "n_conditioned_residual_median_ms": residual.median()})
            residual = frame.loc[group.index, "pred_n_only"] - group.local_ms
            for feature in FEATURES:
                x = group[feature]
                boundaries = np.unique(np.quantile(x, [0, .25, .5, .75, 1]))
                bin_means = []
                for low, high in zip(boundaries[:-1], boundaries[1:]):
                    mask = (x >= low) & (x <= high)
                    bin_means.append(float(residual[mask].mean()))
                residuals.append({"config_id": config_id, "rank": rank, "frequency_mhz": frequency,
                                  "feature": feature, "samples": len(group), "pearson_residual": correlation(x, residual),
                                  "quartile_residual_mean_range_ms": max(bin_means) - min(bin_means) if bin_means else 0})
        for i, coefficient in enumerate(ridge.coefficients):
            coefficients.append({"rank": rank, "frequency_mhz": frequency, "feature_index": i,
                                 "standardized_residual_coefficient_ms": coefficient})
    save(output / "latency_validation.csv", [r for r in validations if r["evaluation"] == "heldout_repeat"])
    save(output / "latency_distribution_validation.csv", [r for r in validations if r["evaluation"] == "heldout_skew"])
    save(output / "execution_mode_comparison.csv", mode_rows)
    save(output / "execution_mode_heldout_validation.csv", mode_validation)
    save(output / "residual_feature_analysis.csv", residuals)
    save(output / "ridge_coefficients.csv", coefficients)
    table = pd.DataFrame(validations)
    comparison = table.groupby(["model", "evaluation"]).agg(mape_mean=("mape_percent", "mean"),
                   mape_max=("mape_percent", "max"), mae_mean_ms=("mae", "mean"), groups=("config_id", "count")).reset_index()
    comparison.to_csv(output / "latency_model_comparison.csv", index=False)
    # This is a diagnostic retention rule, not training/tuning on held-out rows.
    rich = table[table.model == "ridge_residual"].set_index(["config_id", "rank"])
    simple = table[table.model == "n_only"].set_index(["config_id", "rank"])
    retained = bool((rich.mape_percent <= simple.mape_percent + 2).all()
                    and rich.mape_percent.mean() <= .8 * simple.mape_percent.mean())
    mode = table[table.model == "mode_binned"].set_index(["config_id", "rank"])
    mode_retained = bool((mode.mape_percent <= simple.mape_percent + 2).all()
                        and mode.mape_percent.mean() < simple.mape_percent.mean())
    selected = "ridge_residual" if retained else "mode_binned" if mode_retained else "n_only"
    dump(output / "latency_model_retention.json", {"selected_diagnostic_model": selected,
         "richer_retained": retained, "mode_retained": mode_retained,
         "rule": "Ridge: >=20% lower mean held-out MAPE and no group worsened >2 percentage points; mode: lower mean and no group worsened >2pp. Fixed before extended analysis; no hyperparameter search.",
         "not_nested_validation": "retention uses held-out diagnostics; not an independent final test"})
    return models, selected


def critical_analysis(frame, output):
    keys = ["config_id", "runtime_step_id", "stage", "layer"]
    zero = frame[frame["rank"] == 0].set_index(keys)
    one = frame[frame["rank"] == 1].set_index(keys).reindex(zero.index)
    assert one.local_ms.notna().all() and np.array_equal(zero.replay, one.replay)
    paired = zero[["frequency_mhz", "lifecycle", "combined_token_rows", "replay"]].copy()
    paired["local_critical_ms"] = np.maximum(zero.local_ms, one.local_ms)
    paired["full_critical_ms"] = np.maximum(zero.full_ms, one.full_ms)
    paired["residual_ms"] = paired.full_critical_ms - paired.local_critical_ms
    for model in MODELS:
        paired[f"pred_{model}"] = np.maximum(zero[f"pred_{model}"], one[f"pred_{model}"])
    paired = paired.reset_index()
    summaries, validation = [], []
    medians = paired[paired.lifecycle == "primary"].groupby(["frequency_mhz", "replay"]).residual_ms.median()
    paired["pred_comm_constant_ms"] = [medians.loc[(f, m)] for f, m in zip(paired.frequency_mhz, paired.replay)]
    for name, group in paired.groupby("config_id"):
        residual = group.residual_ms
        summaries.append({"config_id": name, "lifecycle": group.lifecycle.iloc[0],
                          "frequency_mhz": group.frequency_mhz.iloc[0], "paired_invocations": len(group),
                          "local_sum_ms": group.local_critical_ms.sum(), "full_sum_ms": group.full_critical_ms.sum(),
                          "residual_sum_fraction_of_full": residual.sum() / group.full_critical_ms.sum(),
                          "residual_mean_ms": residual.mean(), "residual_median_ms": residual.median(),
                          "residual_p95_ms": residual.quantile(.95), "residual_std_ms": residual.std(),
                          "residual_corr_total_rows": correlation(group.combined_token_rows, residual),
                          "residual_corr_layer": correlation(group.layer, residual),
                          "residual_corr_replay": correlation(group.replay, residual)})
        for model in MODELS:
            for correction in (False, True):
                prediction = group[f"pred_{model}"] + (group.pred_comm_constant_ms if correction else 0)
                validation.append({"config_id": name, "lifecycle": group.lifecycle.iloc[0],
                                   "model": model, "communication_correction": correction,
                                   "summed_prediction_bias_percent": (prediction.sum() / group.full_critical_ms.sum() - 1) * 100,
                                   **error_statistics(group.full_critical_ms, prediction)})
    save(output / "critical_path_summary.csv", summaries)
    save(output / "critical_path_validation.csv", validation)
    heldout = paired[paired.lifecycle != "primary"]
    direct_rows = []
    for name, group in heldout.groupby("config_id"):
        direct_rows.append({"candidate": name, "measured_local_critical_sum_ms": group.local_critical_ms.sum(),
                            "measured_full_critical_sum_ms": group.full_critical_ms.sum(),
                            "known_same_run_invocation_features": True,
                            **{f"score_{model}_ms": group[f"pred_{model}"].sum() for model in MODELS}})
    direct = pd.DataFrame(direct_rows)
    direct.to_csv(output / "profile_candidate_scores.csv", index=False)
    ranking, pairs = [], []
    for model in MODELS:
        for boundary in ("local", "full"):
            statistics, pair_rows = ranking_statistics(direct.candidate.tolist(), direct[f"measured_{boundary}_critical_sum_ms"], direct[f"score_{model}_ms"])
            ranking.append({"model": model, "boundary": boundary, **statistics})
            pairs += [{"model": model, "boundary": boundary, **row} for row in pair_rows]
    save(output / "profile_candidate_ranking_validation.csv", ranking)
    save(output / "profile_candidate_pairwise_comparisons.csv", pairs)
    paired.to_csv(output / "paired_critical_paths.csv.gz", index=False, compression="gzip")
    return paired


def power_analysis(root, frame, mappings, output):
    rows, _ = build_power_dataset(root)
    power = pd.DataFrame(rows)
    # Keep the baseline sampler-power fit unchanged; extended validation uses E_NVML / duration.
    power["nvml_average_power_w"] = power.window_energy_j / power.duration_s
    power = power[power.frequency_mhz.isin([1635, 2070])].copy()
    validation, predictions, fitted_models = [], [], []
    for (rank, frequency), subset in power.groupby(["rank", "frequency_mhz"]):
        intercept, slope = fit_linear_power(subset.load_rate, subset.nvml_average_power_w)
        fitted_models.append({"rank": rank, "frequency_mhz": frequency, "intercept_w": intercept,
                              "slope_w_per_assignment_s": slope, "samples": len(subset),
                              "load_rate_min": subset.load_rate.min(), "load_rate_max": subset.load_rate.max(),
                              "historical_profile_labels_not_same_run": True})
        for config_id, test in subset.groupby("config_id"):
            train = subset[subset.config_id != config_id]
            intercept, slope = fit_linear_power(train.load_rate, train.nvml_average_power_w)
            p = intercept + slope * test.load_rate.to_numpy()
            for i, (_, row) in enumerate(test.iterrows()):
                predictions.append({**row.to_dict(), "predicted_power_w": p[i],
                                    "conditional_predicted_energy_j": p[i] * row.duration_s,
                                    "extrapolation": row.load_rate < train.load_rate.min() or row.load_rate > train.load_rate.max()})
            validation.append({"rank": rank, "frequency_mhz": frequency, "heldout_config": config_id,
                               **error_statistics(test.nvml_average_power_w, p),
                               **coverage(train.load_rate, test.load_rate),
                               **{f"conditional_energy_{k}": v for k, v in error_statistics(test.window_energy_j, p * test.duration_s).items()}})
    save(output / "power_validation.csv", validation)
    save(output / "power_loco_predictions.csv", predictions)
    save(output / "power_models.csv", fitted_models)
    power.to_csv(output / "power_dataset.csv", index=False)
    with np.load(root / SOURCE_PROFILE, allow_pickle=False) as old:
        counts = old["expert_counts"].astype(float)
    diagnostic = []
    for name, group in frame.groupby("config_id"):
        mapping = mappings[name]
        old_load = np.array([counts[mapping == rank].sum() for rank in (0, 1)])
        # Each rank holds a partition of combined same-run routing; do not count source-rank vectors as GPU load.
        actual = group.groupby("rank").n.sum().reindex([0, 1]).to_numpy()
        for rank in (0, 1):
            diagnostic.append({"config_id": name, "rank": rank, "historical_label_assignments": old_load[rank],
                               "same_run_assignments": actual[rank], "assignments_ratio": actual[rank] / old_load[rank],
                               "historical_fraction": old_load[rank] / old_load.sum(),
                               "same_run_fraction": actual[rank] / actual.sum(),
                               "fraction_difference_pp": (actual[rank] / actual.sum() - old_load[rank] / old_load.sum()) * 100,
                               "instrumented_not_historical_run_truth": True})
    save(output / "historical_load_label_sanity.csv", diagnostic)
    return power, pd.DataFrame(predictions)


def predict_features(features, rank, frequency, models, model):
    fitted = models[(rank, frequency)]
    if model == "n_only":
        return fitted["base"].predict(features.n)
    if model == "ridge_residual":
        return np.maximum(fitted["base"].predict(features.n) + fitted["ridge"].predict(ridge_x(features)), 1e-6)
    result = np.empty(len(features))
    for mode in (0, 1):
        mask = features.replay.to_numpy() == mode
        result[mask] = fitted["mode"][mode].predict(features.n.to_numpy()[mask])
    return result


def candidate_analysis(root, profiles, models, selected, power, power_predictions, output):
    symmetric = pd.read_csv(root / "results/processed/milestone_m_symmetric_runs.csv")
    skew = pd.read_csv(root / "results/processed/milestone_m_target_split_runs.csv")
    sources = pd.concat([symmetric.assign(source="symmetric"), skew.assign(source="target_split")], ignore_index=True)
    assert sources.trace_sha256.nunique() == 1 and sources.vllm_version.astype(str).eq("0.26.0").all()
    assert sources.completed_requests.eq(300).all() and sources.failed_requests.eq(0).all()
    sources["f0"] = sources.target_clocks_mhz.str.split("/").str[2].astype(int)
    sources["f1"] = sources.target_clocks_mhz.str.split("/").str[3].astype(int)
    sources = sources[sources.f0.isin([1635, 2070]) & sources.f1.isin([1635, 2070])].copy()
    placement_summary = pd.read_csv(root / "results/processed/milestone_m_target_split_placements.csv").set_index("placement")
    rows = []
    reference = profiles["balanced-f2070-primary"]
    reference_modes = reference["run_modes_by_rank"]
    assert np.array_equal(reference_modes[0], reference_modes[1])
    artifact_sources = []
    for (source, config_id), observations in sources.groupby(["source", "config_id"]):
        representative = observations.iloc[0]
        assert observations.trace_sha256.nunique() == 1
        for target, actual in zip(observations.target_clocks_mhz, observations.actual_clocks_mhz):
            expected = np.array([float(v) for v in target.split("/")])
            measured = np.array([float(v) for v in actual.split("/")])
            assert np.all(np.abs(expected - measured) <= 5) and np.array_equal(expected[:2], [2520, 2520])
        placement = representative.placement
        placement_path = root / ("data/placements/milestone_m/P_balanced.json" if placement == "P_balanced"
                                 else placement_summary.loc[placement, "placement_file"])
        assert sha256_file(placement_path) == representative.placement_sha256
        mapping = np.asarray(load_placement(str(placement_path))[0])
        name = f"{placement}@{int(representative.f0)}-{int(representative.f1)}"
        values = {"candidate": name, "source": source, "config_id": config_id, "placement": placement,
                  "f0": int(representative.f0), "f1": int(representative.f1), "repetitions": len(observations),
                  "placement_sha256": representative.placement_sha256, "trace_sha256": representative.trace_sha256,
                  "measured_tpot_mean_ms": observations.tpot_p99_ms.mean(), "measured_tpot_std_ms": observations.tpot_p99_ms.std(ddof=1),
                  "measured_ttft_mean_ms": observations.ttft_p99_ms.mean(), "measured_ttft_std_ms": observations.ttft_p99_ms.std(ddof=1),
                  "measured_energy_mean_j": observations.total_gpu_energy_j.mean(), "measured_energy_std_j": observations.total_gpu_energy_j.std(ddof=1),
                  "measured_duration_mean_s": observations.duration_s.mean()}
        features = [derive_features(reference, mapping, rank) for rank in (0, 1)]
        for model in MODELS:
            costs = [predict_features(features[r], r, values[f"f{r}"], models, model) for r in (0, 1)]
            values[f"score_{model}_ms"] = float(np.maximum(*costs).sum())
        endpoint = []
        for rank in (0, 1):
            train = models[(rank, values[f"f{rank}"])]["base"].centers
            endpoint.append(float(np.mean((features[rank].n < train.min()) | (features[rank].n > train.max()))))
        values["endpoint_clamp_fraction_max"] = max(endpoint)
        loo = power_predictions[(power_predictions.source == f"{source}_uninstrumented") & (power_predictions.config_id == config_id)]
        # Historical source uses target_split_uninstrumented, not a timing-run configuration.
        assert len(loo) == len(observations) * 2
        predicted_ffn_by_run = loo.groupby("run_id").conditional_predicted_energy_j.sum()
        attention_measured = observations.set_index("run_id").attention_pool_energy_j
        values["conditional_energy_prediction_j"] = float((predicted_ffn_by_run + attention_measured).mean())
        values["conditional_energy_uses_measured_duration_and_attention"] = True
        # A full independent-server lifecycle is the uncertainty unit, not invocation rows.
        for run_id in observations.run_id:
            raw = Path(observations.loc[observations.run_id == run_id, "run_dir"].iloc[0])
            metrics = json.loads((raw / "metrics.json").read_text())
            assert metrics["completed_requests"] == 300 and metrics["failed_requests"] == 0
            assert sha256_file(raw / "trace.jsonl") == representative.trace_sha256
            metadata_path = raw / "server_metadata.json"
            for path in [raw / "metrics.json", raw / "trace.jsonl", raw / "requests.jsonl", raw / "resolved_config.yaml", metadata_path]:
                artifact_sources.append({"path": str(path.relative_to(root)), "sha256": sha256_file(path)})
        artifact_sources.append({"path": str(placement_path.relative_to(root)), "sha256": sha256_file(placement_path)})
        rows.append(values)
    candidates = pd.DataFrame(rows)
    anchor = candidates[(candidates.placement == "P_balanced") & (candidates.f0 == 2070) & (candidates.f1 == 2070)].iloc[0]
    # Explicitly diagnostic: a single-anchor score-to-p99/duration bridge is not a validated queueing model.
    candidates["proxy_tpot_ms"] = candidates[f"score_{selected}_ms"] / anchor[f"score_{selected}_ms"] * anchor.measured_tpot_mean_ms
    candidates["proxy_duration_s"] = candidates[f"score_{selected}_ms"] / anchor[f"score_{selected}_ms"] * anchor.measured_duration_mean_s
    candidates["bridge_calibration_candidate"] = anchor.candidate
    candidates["bridge_is_validated_p99_model"] = False
    proxy_energy = []
    for _, row in candidates.iterrows():
        observations = sources[(sources.source == row.source) & (sources.config_id == row.config_id)]
        representative = observations.iloc[0]
        matching_power = power[(power.source == f"{row.source}_uninstrumented") & (power.config_id == row.config_id)]
        assignment_counts = matching_power.groupby("rank").routed_assignments.first()
        ffn_watts = 0
        for rank in (0, 1):
            train = power[(power["rank"] == rank) & (power.frequency_mhz == row[f"f{rank}"]) & (power.config_id != row.config_id)]
            intercept, slope = fit_linear_power(train.load_rate, train.nvml_average_power_w)
            ffn_watts += intercept + slope * assignment_counts.loc[rank] / row.proxy_duration_s
        anchor_observations = sources[(sources.source == anchor.source) & (sources.config_id == anchor.config_id)]
        attention_watts = float((anchor_observations.attention_pool_energy_j / anchor_observations.duration_s).mean())
        proxy_energy.append((ffn_watts + attention_watts) * row.proxy_duration_s)
    candidates["proxy_energy_j"] = proxy_energy
    candidates.to_csv(output / "candidate_predictions.csv", index=False)
    dump(output / "uninstrumented_provenance.json", artifact_sources)
    ranking, pairs = [], []
    metrics = [(f"latency_{model}", "measured_tpot_mean_ms", f"score_{model}_ms", "measured_tpot_std_ms") for model in MODELS]
    metrics += [("energy_conditional", "measured_energy_mean_j", "conditional_energy_prediction_j", "measured_energy_std_j"),
                ("energy_proxy", "measured_energy_mean_j", "proxy_energy_j", "measured_energy_std_j")]
    for metric, actual, predicted, deviation in metrics:
        stats, inverted = ranking_statistics(candidates.candidate.tolist(), candidates[actual], candidates[predicted], candidates[deviation])
        ranking.append({"metric": metric, **stats})
        pairs += [{"metric": metric, **value} for value in inverted]
    save(output / "candidate_ranking_validation.csv", ranking)
    save(output / "candidate_pairwise_inversions.csv", pairs)
    required = []
    for placement in ["P_balanced", "P_split_40_slow_rank0", "P_split_40_slow_rank1"]:
        for f in (1635, 2070):
            match = candidates[(candidates.placement == placement) & (candidates.f0 == f) & (candidates.f1 == f)]
            required.append({"placement": placement, "f0": f, "f1": f,
                             "uninstrumented_exact_ground_truth": not match.empty,
                             "missing_reason": "" if not match.empty else "Only heterogeneous 1635/2070 or 2070/1635 M runs exist; symmetric instrumented N runs cannot substitute."})
    save(output / "requested_candidate_ground_truth_coverage.csv", required)
    decisions = []
    for frequency in (2070, 1635):
        symmetric_point = candidates[(candidates.placement == "P_balanced") & (candidates.f0 == frequency) & (candidates.f1 == frequency)].iloc[0]
        for factor in (1.0, 1.025, 1.05):
            budget = float(symmetric_point.measured_tpot_mean_ms * factor)
            measured = select_feasible(candidates.candidate.tolist(), candidates.measured_tpot_mean_ms, candidates.measured_energy_mean_j, budget)
            proxy = select_feasible(candidates.candidate.tolist(), candidates.proxy_tpot_ms, candidates.proxy_energy_j, budget)
            winner = candidates[candidates.candidate == proxy].iloc[0] if proxy else None
            best = candidates[candidates.candidate == measured].iloc[0] if measured else None
            within_noise = bool(winner is not None and best is not None and
                abs(winner.measured_energy_mean_j - best.measured_energy_mean_j) <= np.hypot(winner.measured_energy_std_j, best.measured_energy_std_j))
            decisions.append({"scenario": f"symmetric_{frequency}_TPOT_x{factor}", "tpot_budget_ms": budget,
                              "measured_feasible": json.dumps(candidates[candidates.measured_tpot_mean_ms <= budget].candidate.tolist()),
                              "measured_best": measured, "predicted_proxy_best": proxy,
                              "selection_exact": proxy == measured, "candidate_energy_difference_within_noise": within_noise,
                              "selected_measured_tpot_feasible": bool(winner is not None and winner.measured_tpot_mean_ms <= budget),
                              "selected_measured_tpot_ms": winner.measured_tpot_mean_ms if winner is not None else None,
                              "selected_measured_ttft_ms": winner.measured_ttft_mean_ms if winner is not None else None,
                              "measured_energy_gap_j": winner.measured_energy_mean_j - best.measured_energy_mean_j if winner is not None and best is not None else None,
                              "ttft_not_predicted_or_certified": True,
                              "deployment_prediction_supported": False,
                              "bridge_anchor_scenario_is_in_sample": frequency == 2070})
    save(output / "slo_selection_validation.csv", decisions)
    return candidates, ranking, decisions, required


def plots(frame, candidates, output, figures):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for rank in (0, 1):
        for mode in (0, 1):
            axis = axes[rank, mode]
            for frequency in (1005, 1635, 2070, 2220, 2520):
                subset = frame[(frame["rank"] == rank) & (frame.frequency_mhz == frequency) & (frame.lifecycle == "primary") & (frame.replay == mode)].iloc[::100]
                axis.scatter(subset.n, subset.local_ms, s=4, alpha=.4, label=str(frequency))
            axis.set(title=f'GPU{rank + 2} {"replay" if mode else "eager"}', xlabel="Local assignments n", ylabel="Local boundary ms")
            axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figures / "latency_by_mode.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    holdout = frame[frame.lifecycle != "primary"]
    sampled = holdout.iloc[::100]
    axes[0].scatter(sampled.local_ms, sampled.pred_n_only, s=4, alpha=.35, label="n only")
    axes[0].scatter(sampled.local_ms, sampled.pred_ridge_residual, s=4, alpha=.35, label="Ridge")
    low = min(holdout.local_ms.min(), holdout.pred_n_only.min(), holdout.pred_ridge_residual.min())
    high = max(holdout.local_ms.max(), holdout.pred_n_only.max(), holdout.pred_ridge_residual.max())
    axes[0].plot([low, high], [low, high], color="black", linewidth=1)
    axes[0].set(xlabel="Actual local boundary ms (log scale)", ylabel="Predicted ms (log scale)",
                xscale="log", yscale="log", xlim=(low * .9, high * 1.1), ylim=(low * .9, high * 1.1)); axes[0].legend()
    comparison = pd.read_csv(output / "latency_model_comparison.csv")
    for i, evaluation in enumerate(("heldout_repeat", "heldout_skew")):
        subset = comparison[comparison.evaluation == evaluation].set_index("model").reindex(MODELS)
        axes[1].bar(np.arange(3) + i * .35, subset.mape_mean, width=.35, label=evaluation)
    axes[1].set(xticks=np.arange(3) + .175, xticklabels=["n", "mode+n", "Ridge"], ylabel="Mean group MAPE %"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figures / "heldout_model_comparison.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    residual_diagnostics = pd.read_csv(output / "residual_feature_analysis.csv")
    residual_diagnostics["absolute_correlation"] = residual_diagnostics.pearson_residual.abs()
    strongest = residual_diagnostics.groupby("feature").absolute_correlation.mean().sort_values(ascending=False).head(3).index
    for column, feature in enumerate(strongest):
        axes[0, column].scatter(sampled[feature], sampled.local_ms, s=4, alpha=.3)
        axes[0, column].set(xlabel=feature, ylabel="Actual local boundary ms", yscale="log")
        axes[1, column].scatter(sampled[feature], sampled.pred_n_only - sampled.local_ms, s=4, alpha=.3)
        axes[1, column].set(xlabel=feature, ylabel="Prediction - actual ms")
        axes[1, column].set_yscale("symlog", linthresh=.1)
        for row in (0, 1):
            axes[row, column].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(figures / "residual_features.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].errorbar(candidates.measured_tpot_mean_ms, candidates.score_n_only_ms / 1000,
                     xerr=candidates.measured_tpot_std_ms, fmt="o", markersize=4)
    axes[0].set(xlabel="Uninstrumented TPOT p99 mean +/- sample SD (ms)", ylabel="Common routing n-only critical score (s)")
    axes[1].errorbar(candidates.measured_energy_mean_j, candidates.proxy_energy_j,
                     xerr=candidates.measured_energy_std_j, fmt="o", markersize=4)
    low, high = candidates.measured_energy_mean_j.min(), candidates.measured_energy_mean_j.max()
    axes[1].plot([low, high], [low, high], color="black", linewidth=1)
    axes[1].set(xlabel="Uninstrumented total energy mean +/- sample SD (J)", ylabel="Diagnostic proxy total energy (J)")
    fig.tight_layout(); fig.savefig(figures / "candidate_ranking.png", dpi=150); plt.close(fig)
    power = pd.read_csv(output / "power_dataset.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for rank, axis in enumerate(axes):
        for frequency in (1635, 2070):
            subset = power[(power["rank"] == rank) & (power.frequency_mhz == frequency)]
            axis.scatter(subset.load_rate, subset.nvml_average_power_w, label=str(frequency))
        axis.set(title=f"GPU{rank + 2}", xlabel="Historical-profile assignments / measured second", ylabel="NVML energy / window (W)"); axis.legend()
    fig.tight_layout(); fig.savefig(figures / "power_vs_load.png", dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    args = parser.parse_args()
    root, output = Path.cwd().resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frame, profiles, mappings, plan = load_formal(root, args.study_dir.resolve(), output)
    models, selected = latency_analysis(frame, output)
    frame.to_csv(output / "joined_invocations_with_features.csv.gz", index=False, compression="gzip")
    paired = critical_analysis(frame, output)
    power, predictions = power_analysis(root, frame, mappings, output)
    candidates, ranking, decisions, required = candidate_analysis(root, profiles, models, selected, power, predictions, output)
    plots(frame, candidates, output, args.figures_dir)
    # Candidate ordering and the p99/energy bridge are necessary gates, not just invocation MAPE.
    gate = oracle_decision_gate(ranking, decisions, selected_model=selected, bridge_validated=False)
    dump(output / "decision_gate.json", {
        **gate,
        "selected_diagnostic_latency_model": selected,
        "latency_validation": "See separate lifecycle/config/rank holdouts; no random row splits.",
        "candidate_rankings": ranking,
        "diagnostic_scenarios": len(decisions), "diagnostic_exact_selections": sum(r["selection_exact"] for r in decisions),
        "diagnostic_measured_feasible_selections": sum(r["selected_measured_tpot_feasible"] for r in decisions),
        "missing_requested_ground_truth": [r for r in required if not r["uninstrumented_exact_ground_truth"]],
        "uncertainty": "Three windows per M config, one server per config; pooled sample-SD overlap is descriptive, not equivalence or significance.",
        "instrumented_serving_metrics_used_as_ground_truth": False, "gpu_jobs_submitted": 0,
        "runtime_modified": False, "execution_commit": plan["git_commit"],
    })
    print(json.dumps(clean({"rows": len(frame), "selected_model": selected, "candidates": len(candidates), "rankings": ranking, "decision": gate}), indent=2))


if __name__ == "__main__":
    main()
