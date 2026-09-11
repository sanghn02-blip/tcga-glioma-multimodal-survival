CREATE TABLE IF NOT EXISTS patients (
  case_submitter_id VARCHAR(32) PRIMARY KEY,
  project_id VARCHAR(32) NOT NULL,
  os_event TINYINT NULL,
  os_days DOUBLE NULL,
  vital_status VARCHAR(32) NULL,
  gender VARCHAR(32) NULL,
  age_at_diagnosis_days DOUBLE NULL,
  tumor_grade VARCHAR(64) NULL,
  primary_diagnosis VARCHAR(255) NULL,
  wsi_file_count INT NOT NULL DEFAULT 0,
  rna_file_count INT NOT NULL DEFAULT 0,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_patients_project (project_id),
  INDEX idx_patients_survival (os_event, os_days)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS gdc_files (
  file_id VARCHAR(64) PRIMARY KEY,
  file_name VARCHAR(255) NOT NULL,
  case_submitter_id VARCHAR(32) NOT NULL,
  project_id VARCHAR(32) NOT NULL,
  sample_submitter_ids TEXT NULL,
  data_type VARCHAR(128) NOT NULL,
  experimental_strategy VARCHAR(128) NULL,
  workflow_type VARCHAR(128) NULL,
  file_size BIGINT NULL,
  local_path TEXT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  CONSTRAINT fk_gdc_files_patient FOREIGN KEY (case_submitter_id)
    REFERENCES patients(case_submitter_id)
    ON UPDATE CASCADE
    ON DELETE CASCADE,
  INDEX idx_gdc_files_case (case_submitter_id),
  INDEX idx_gdc_files_data_type (data_type),
  INDEX idx_gdc_files_project (project_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS cbioportal_covariates (
  case_submitter_id VARCHAR(32) PRIMARY KEY,
  cbioportal_study_id VARCHAR(128) NULL,
  pancan_subtype VARCHAR(128) NULL,
  histologic_grade VARCHAR(64) NULL,
  cancer_type_detailed VARCHAR(255) NULL,
  tumor_type VARCHAR(255) NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  CONSTRAINT fk_cbio_patient FOREIGN KEY (case_submitter_id)
    REFERENCES patients(case_submitter_id)
    ON UPDATE CASCADE
    ON DELETE CASCADE,
  INDEX idx_cbio_subtype (pancan_subtype),
  INDEX idx_cbio_grade (histologic_grade)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS artifacts (
  artifact_id BIGINT AUTO_INCREMENT PRIMARY KEY,
  artifact_type VARCHAR(64) NOT NULL,
  name VARCHAR(255) NOT NULL,
  path TEXT NOT NULL,
  description TEXT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uq_artifacts_type_name (artifact_type, name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS model_runs (
  run_id VARCHAR(128) PRIMARY KEY,
  model_name VARCHAR(128) NOT NULL,
  modality VARCHAR(64) NOT NULL,
  cohort_name VARCHAR(128) NOT NULL,
  notes TEXT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_model_runs_model (model_name),
  INDEX idx_model_runs_modality (modality),
  INDEX idx_model_runs_cohort (cohort_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS model_metrics (
  run_id VARCHAR(128) NOT NULL,
  metric_name VARCHAR(128) NOT NULL,
  metric_value DOUBLE NULL,
  metric_text TEXT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (run_id, metric_name),
  CONSTRAINT fk_metrics_run FOREIGN KEY (run_id)
    REFERENCES model_runs(run_id)
    ON UPDATE CASCADE
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS model_predictions (
  run_id VARCHAR(128) NOT NULL,
  case_submitter_id VARCHAR(32) NOT NULL,
  fold INT NULL,
  risk_score DOUBLE NULL,
  os_days DOUBLE NULL,
  os_event TINYINT NULL,
  project_id VARCHAR(32) NULL,
  risk_group VARCHAR(32) NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (run_id, case_submitter_id),
  CONSTRAINT fk_predictions_run FOREIGN KEY (run_id)
    REFERENCES model_runs(run_id)
    ON UPDATE CASCADE
    ON DELETE CASCADE,
  CONSTRAINT fk_predictions_patient FOREIGN KEY (case_submitter_id)
    REFERENCES patients(case_submitter_id)
    ON UPDATE CASCADE
    ON DELETE CASCADE,
  INDEX idx_predictions_risk_group (risk_group),
  INDEX idx_predictions_project (project_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS wsi_slides (
  file_id VARCHAR(64) PRIMARY KEY,
  case_submitter_id VARCHAR(32) NOT NULL,
  slide_path TEXT NULL,
  tissue_mask_path TEXT NULL,
  patch_manifest_path TEXT NULL,
  embedding_path TEXT NULL,
  encoder_name VARCHAR(128) NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  CONSTRAINT fk_wsi_file FOREIGN KEY (file_id)
    REFERENCES gdc_files(file_id)
    ON UPDATE CASCADE
    ON DELETE CASCADE,
  CONSTRAINT fk_wsi_patient FOREIGN KEY (case_submitter_id)
    REFERENCES patients(case_submitter_id)
    ON UPDATE CASCADE
    ON DELETE CASCADE,
  INDEX idx_wsi_case (case_submitter_id),
  INDEX idx_wsi_encoder (encoder_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS deg_genes (
  comparison_name VARCHAR(128) NOT NULL,
  gene VARCHAR(64) NOT NULL,
  direction VARCHAR(32) NULL,
  log2_fold_change DOUBLE NULL,
  fdr_bh DOUBLE NULL,
  gbm_mean_tpm DOUBLE NULL,
  control_mean_tpm DOUBLE NULL,
  survival_risk_direction VARCHAR(64) NULL,
  survival_coef DOUBLE NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (comparison_name, gene),
  INDEX idx_deg_direction (direction),
  INDEX idx_deg_fdr (fdr_bh)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS drug_repurposing_candidates (
  comparison_name VARCHAR(128) NOT NULL,
  candidate_rank INT NOT NULL,
  drug_name VARCHAR(255) NOT NULL,
  repurposing_score DOUBLE NULL,
  primary_target VARCHAR(64) NULL,
  targets TEXT NULL,
  target_count INT NULL,
  primary_target_direction VARCHAR(32) NULL,
  primary_target_log2_fc DOUBLE NULL,
  primary_target_fdr DOUBLE NULL,
  interaction_types TEXT NULL,
  sources TEXT NULL,
  approved TINYINT NULL,
  antineoplastic TINYINT NULL,
  immunotherapy TINYINT NULL,
  ranking_note TEXT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (comparison_name, candidate_rank),
  INDEX idx_drug_name (drug_name),
  INDEX idx_drug_target (primary_target)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS clue_lincs_reversal_candidates (
  comparison_name VARCHAR(128) NOT NULL,
  candidate_rank INT NOT NULL,
  drug_name VARCHAR(255) NOT NULL,
  final_repurposing_score DOUBLE NULL,
  repurposing_score DOUBLE NULL,
  primary_target VARCHAR(64) NULL,
  primary_target_direction VARCHAR(32) NULL,
  local_reversal_prior DOUBLE NULL,
  local_reversal_reason VARCHAR(128) NULL,
  clue_tau_score DOUBLE NULL,
  clue_connectivity_direction VARCHAR(64) NULL,
  clue_status VARCHAR(128) NULL,
  approved TINYINT NULL,
  antineoplastic TINYINT NULL,
  ranking_note TEXT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (comparison_name, candidate_rank),
  INDEX idx_clue_drug_name (drug_name),
  INDEX idx_clue_target (primary_target),
  INDEX idx_clue_status (clue_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS drug_repurposing_shortlist (
  comparison_name VARCHAR(128) NOT NULL,
  shortlist_rank INT NOT NULL,
  candidate_tier VARCHAR(64) NULL,
  drug_name VARCHAR(255) NOT NULL,
  primary_target VARCHAR(64) NULL,
  final_repurposing_score DOUBLE NULL,
  repurposing_score DOUBLE NULL,
  primary_target_direction VARCHAR(32) NULL,
  primary_target_log2_fc DOUBLE NULL,
  primary_target_fdr DOUBLE NULL,
  local_reversal_prior DOUBLE NULL,
  mechanism_summary_ko VARCHAR(255) NULL,
  approved TINYINT NULL,
  antineoplastic TINYINT NULL,
  clue_status VARCHAR(128) NULL,
  clue_tau_score DOUBLE NULL,
  clue_connectivity_direction VARCHAR(64) NULL,
  pert_id VARCHAR(64) NULL,
  clue_pert_iname VARCHAR(255) NULL,
  validation_status_ko VARCHAR(255) NULL,
  rationale_ko TEXT NULL,
  next_validation_ko TEXT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (comparison_name, shortlist_rank),
  INDEX idx_shortlist_drug_name (drug_name),
  INDEX idx_shortlist_target (primary_target),
  INDEX idx_shortlist_tier (candidate_tier)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS drug_repurposing_validation_matrix (
  comparison_name VARCHAR(128) NOT NULL,
  validation_rank INT NOT NULL,
  validation_tier VARCHAR(64) NULL,
  drug_name VARCHAR(255) NOT NULL,
  primary_target VARCHAR(64) NULL,
  validation_score DOUBLE NULL,
  final_repurposing_score DOUBLE NULL,
  bbb_rule_score DOUBLE NULL,
  bbb_rule_label VARCHAR(128) NULL,
  molecular_weight DOUBLE NULL,
  xlogp DOUBLE NULL,
  tpsa DOUBLE NULL,
  pert_id VARCHAR(64) NULL,
  clue_pert_iname VARCHAR(255) NULL,
  clue_tau_score DOUBLE NULL,
  gbm_trial_count INT NULL,
  gbm_trial_nct_ids TEXT NULL,
  external_lookup_status VARCHAR(128) NULL,
  validation_comment_ko TEXT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (comparison_name, validation_rank),
  INDEX idx_validation_drug_name (drug_name),
  INDEX idx_validation_target (primary_target),
  INDEX idx_validation_tier (validation_tier)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
