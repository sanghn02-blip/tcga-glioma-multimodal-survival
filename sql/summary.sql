SELECT 'patients' AS item, COUNT(*) AS value FROM patients;
SELECT project_id, COUNT(*) AS n FROM patients GROUP BY project_id ORDER BY project_id;
SELECT data_type, COUNT(*) AS n, ROUND(SUM(file_size) / 1000000000, 2) AS size_gb
FROM gdc_files
GROUP BY data_type
ORDER BY data_type;
SELECT pancan_subtype, COUNT(*) AS n
FROM cbioportal_covariates
WHERE pancan_subtype IS NOT NULL
GROUP BY pancan_subtype
ORDER BY n DESC;
SELECT run_id, metric_name, metric_value, metric_text
FROM model_metrics
ORDER BY run_id, metric_name;
SELECT comparison_name, COUNT(*) AS n
FROM deg_genes
GROUP BY comparison_name
ORDER BY comparison_name;
SELECT comparison_name, candidate_rank, drug_name, ROUND(repurposing_score, 3) AS repurposing_score, primary_target, approved, antineoplastic
FROM drug_repurposing_candidates
ORDER BY comparison_name, candidate_rank
LIMIT 15;
SELECT comparison_name, COUNT(*) AS n
FROM clue_lincs_reversal_candidates
GROUP BY comparison_name
ORDER BY comparison_name;
SELECT comparison_name, candidate_rank, drug_name, ROUND(final_repurposing_score, 3) AS final_repurposing_score,
       ROUND(local_reversal_prior, 2) AS local_reversal_prior, primary_target, clue_status
FROM clue_lincs_reversal_candidates
ORDER BY comparison_name, candidate_rank
LIMIT 15;
SELECT comparison_name, COUNT(*) AS n
FROM drug_repurposing_shortlist
GROUP BY comparison_name
ORDER BY comparison_name;
SELECT comparison_name, shortlist_rank, candidate_tier, drug_name, primary_target,
       ROUND(final_repurposing_score, 3) AS final_repurposing_score,
       ROUND(clue_tau_score, 2) AS clue_tau_score, validation_status_ko
FROM drug_repurposing_shortlist
ORDER BY comparison_name, shortlist_rank
LIMIT 12;
SELECT comparison_name, COUNT(*) AS n
FROM drug_repurposing_validation_matrix
GROUP BY comparison_name
ORDER BY comparison_name;
SELECT comparison_name, validation_rank, validation_tier, drug_name, primary_target,
       ROUND(validation_score, 3) AS validation_score,
       ROUND(clue_tau_score, 2) AS clue_tau_score, bbb_rule_label, gbm_trial_count
FROM drug_repurposing_validation_matrix
ORDER BY comparison_name, validation_rank
LIMIT 12;
