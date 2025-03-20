SELECT
  job_id,
  run_id,
  COUNT(*) AS number_of_tables,
  FIRST(project),
  COLLECT_LIST(STRUCT(table_name, row_count, run_time_secs, extract_type, src_sys_cd)) AS metadata
FROM
  (
    SELECT
      regexp_extract(parent_job_url, '#job/([0-9]+)/run/', 1) AS job_id,
      regexp_extract(parent_job_url, '/run/([0-9]+)', 1) AS run_id,
      *
    FROM
      delta.`s3://tfsdl-edp-common-dims-prod/processed/run_stats_table`
  )
GROUP BY
  job_id,
  run_id
