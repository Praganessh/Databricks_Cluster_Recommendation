import pandas as pd
import os
# from dotenv import load_dotenv
import itertools

# load_dotenv()

# Make sure you have your custom langchain class or import from wherever you define it
from langchain_openai import ChatOpenAI

# Load API key from environment variables
# api_key = os.getenv("OPENAI_API_KEY")
api_key = ''
# Create your LLM model instance
model = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)


# Function to calculate average duration
def calculate_job_durations(df, job_id=None):
    # Convert start_time and end_time to datetime
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    
    # Group by cluster_id and calculate duration
    duration_df = (
        df.groupby('cluster_id')
        .apply(lambda group: (group['end_time'].max() - group['start_time'].min()).total_seconds() / 60)
        .reset_index(name='duration_mins')
    )
 
    new_df = pd.merge(df,duration_df,on="cluster_id")
        
    # Filter by specific job_id if provided
    if job_id:
        filtered_df = new_df[new_df['cluster_job_id'] == job_id]
    else:
        filtered_df = new_df
 
    # Group by cluster_id and calculate total duration for each cluster
    unique_cluster_ids = filtered_df.drop_duplicates(subset='cluster_id')
    average_duration = unique_cluster_ids['duration_mins'].mean()
 
 
    return {
        'average_duration': average_duration
    }

# -------------------------------------------------------------
# Function to validate CSV columns
# -------------------------------------------------------------
def validate_csv_columns(df, required_columns=None):
    """
    Validate that the DataFrame has the required columns
    for cluster utilization analysis.
    """
    required_columns = required_columns or {
        "cluster_id",
        "cluster_name",
        "create_time",
        "driver_node_type",
        "worker_node_type",
        "worker_count",
        "min_autoscale_workers",
        "max_autoscale_workers",
        "cluster_job_id",
        "cluster_run_id",
        "instance_id",
        "start_time",
        "end_time",
        "driver",
        "cpu_user_percent",
        "mem_used_percent",
        "node_type",
        "job_name",
    }

    missing_columns = list(required_columns - set(df.columns))
    if missing_columns:
        return False, missing_columns
    return True, []


# -------------------------------------------------------------
# Function to generate *three* instance recommendations
# -------------------------------------------------------------
def generate_instance_recommendation(
    job_id,
    worker_current_instance,
    worker_cpu_stats,
    worker_memory_stats,
    current_worker_count,
    current_min_workers,
    current_max_workers,
    driver_current_instance,
    driver_cpu_stats,
    driver_memory_stats,
    duration
):
    """
    Generates *3* recommendations for the best instance type for workers
    and the driver, based on utilization metrics. Also suggests updated
    min/max worker counts. (No reason field included in final output)
    """

    # -- prompt requesting 3 different recommendations
    prompt = f"""

    You are a cloud optimization expert specializing in balancing performance and cost-effectiveness 
    for cloud infrastructure. Your task is to analyze the given usage data (CPU, Memory) for both 
    worker nodes and the driver node, along with current worker settings, and produce 3 distinct recommendations.

    Each recommendation must include:
    - A recommended worker instance type (No t-type instances).
    - A recommended driver instance type (No t-type instances).
    - Recommended minimum workers.
    - Recommended maximum workers.
    - Emphasize cost-efficiency but handle peak loads effectively.
    - Prefer Graviton (ARM-based) instances if possible.
    - Ensure that the new instance type does not deviate significantly from the average job duration.

    Notes:
    - If auto-scaling is not enabled in the current instance configuration, the new recommendation 
        should not introduce auto-scaling.
    - The number of workers can still vary based on the configuration and workload.
    - If high CPU utilization is observed, give preference to **C-series (compute-optimized)** instances, 
        while still ensuring cost-efficiency.
    - If high memory utilization is observed, give preference to **R-series (memory-optimized)** instances, 
        while still ensuring cost-efficiency.


     
    The job details are:
      - Job ID: {job_id}
      - Avg Job Duration:{duration}


    -- Worker Node --
      - Current Worker Instance: {worker_current_instance}
      - CPU Stats (Workers):
          * Avg: {worker_cpu_stats['mean']:.1f}%
          * Max: {worker_cpu_stats['max']:.1f}%
          * Min: {worker_cpu_stats['min']:.1f}%
          * Std: {worker_cpu_stats['std']:.1f}%
      - Memory Stats (Workers):
          * Avg: {worker_memory_stats['mean']:.1f}%
          * Max: {worker_memory_stats['max']:.1f}%
          * Min: {worker_memory_stats['min']:.1f}%
          * Std: {worker_memory_stats['std']:.1f}%
      - Current Worker Count: {current_worker_count}
      - Current Min Workers: {current_min_workers}
      - Current Max Workers: {current_max_workers}

    -- Driver Node --
      - Current Driver Instance: {driver_current_instance}
      - CPU Stats (Driver):
          * Avg: {driver_cpu_stats['mean']:.1f}%
          * Max: {driver_cpu_stats['max']:.1f}%
          * Min: {driver_cpu_stats['min']:.1f}%
          * Std: {driver_cpu_stats['std']:.1f}%
      - Memory Stats (Driver):
          * Avg: {driver_memory_stats['mean']:.1f}%
          * Max: {driver_memory_stats['max']:.1f}%
          * Min: {driver_memory_stats['min']:.1f}%
          * Std: {driver_memory_stats['std']:.1f}%
    
    For each of the three distinct recommendations, use the following format exactly:

    -- Recommendation 1 --
    - Worker Instance: <value>
    - Driver Instance: <value>
    - Min Workers: <value>
    - Max Workers: <value>

    -- Recommendation 2 --
    - Worker Instance: <value>
    - Driver Instance: <value>
    - Min Workers: <value>
    - Max Workers: <value>

    -- Recommendation 3 --
    - Worker Instance: <value>
    - Driver Instance: <value>
    - Min Workers: <value>
    - Max Workers: <value>
    """

    try:
        response = model.invoke(prompt)
        parsed_recs = {
            "Worker Instance": [],
            "Driver Instance": [],
            "Min Workers": [],
            "Max Workers": [],
        }

        lines = [line.strip() for line in response.content.split("\n") if line.strip()]

        current_block = None
        for line in lines:
            if line.startswith("-- Recommendation"):
                current_block = line  # e.g. "-- Recommendation 1 --"
                continue

            if current_block:
                if line.startswith("- Worker Instance:"):
                    val = line.split(":", 1)[1].strip()
                    parsed_recs["Worker Instance"].append(val)
                elif line.startswith("- Driver Instance:"):
                    val = line.split(":", 1)[1].strip()
                    parsed_recs["Driver Instance"].append(val)
                elif line.startswith("- Min Workers:"):
                    val = line.split(":", 1)[1].strip()
                    parsed_recs["Min Workers"].append(val)
                elif line.startswith("- Max Workers:"):
                    val = line.split(":", 1)[1].strip()
                    parsed_recs["Max Workers"].append(val)

        # Ensure we have 3 sets
        for key in parsed_recs:
            while len(parsed_recs[key]) < 3:
                parsed_recs[key].append("N/A")

        return parsed_recs

    except Exception as e:
        return {
            "Worker Instance": ["N/A", "N/A", "N/A"],
            "Driver Instance": ["N/A", "N/A", "N/A"],
            "Min Workers": ["N/A", "N/A", "N/A"],
            "Max Workers": ["N/A", "N/A", "N/A"],
        }


# -------------------------------------------------------------
# Bulk Recommendation Function
# -------------------------------------------------------------
def generate_bulk_recommendations(df):
    """
    Loops through all unique cluster_job_id values in df,
    calculates stats, obtains 3 recommendations, and returns
    a DataFrame ready for export.
    """

    # We'll accumulate rows here
    all_rows = []

    for job_id in df["cluster_job_id"].unique():
        job_data = df[df["cluster_job_id"] == job_id]
        if job_data.empty:
            continue  # Skip if no data for this job_id

        # Gather job-level info
        job_name = job_data["job_name"].iloc[0] if not job_data.empty else "Unknown"

        # Worker data
        worker_data = job_data[job_data["driver"] == False]
        # Driver data
        driver_data = job_data[job_data["driver"] == True]

        # If there's no data, skip
        if worker_data.empty and driver_data.empty:
            continue

        # ---------------------------
        # Worker Summaries
        # ---------------------------
        if not worker_data.empty:
            avg_cpu_w = worker_data["cpu_user_percent"].mean()
            peak_cpu_w = worker_data["cpu_user_percent"].max()
            lowest_cpu_w = worker_data["cpu_user_percent"].min()
            std_cpu_w = worker_data["cpu_user_percent"].std()

            avg_memory_w = worker_data["mem_used_percent"].mean()
            peak_memory_w = worker_data["mem_used_percent"].max()
            lowest_memory_w = worker_data["mem_used_percent"].min()
            std_memory_w = worker_data["mem_used_percent"].std()

            first_worker_row = worker_data.iloc[0]
            worker_current_instance = first_worker_row.get("worker_node_type", "N/A")
            current_worker_count = first_worker_row.get("worker_count", 0)
            current_min_workers = first_worker_row.get("min_autoscale_workers", 0)
            current_max_workers = first_worker_row.get("max_autoscale_workers", 0)
        else:
            # default if no worker data
            avg_cpu_w = peak_cpu_w = lowest_cpu_w = std_cpu_w = 0
            avg_memory_w = peak_memory_w = lowest_memory_w = std_memory_w = 0
            worker_current_instance = "N/A"
            current_worker_count = 0
            current_min_workers = 0
            current_max_workers = 0

        worker_cpu_stats = {
            "mean": avg_cpu_w,
            "max": peak_cpu_w,
            "min": lowest_cpu_w,
            "std": std_cpu_w,
        }
        worker_memory_stats = {
            "mean": avg_memory_w,
            "max": peak_memory_w,
            "min": lowest_memory_w,
            "std": std_memory_w,
        }

        # ---------------------------
        # Driver Summaries
        # ---------------------------
        if not driver_data.empty:
            avg_cpu_d = driver_data["cpu_user_percent"].mean()
            peak_cpu_d = driver_data["cpu_user_percent"].max()
            lowest_cpu_d = driver_data["cpu_user_percent"].min()
            std_cpu_d = driver_data["cpu_user_percent"].std()

            avg_memory_d = driver_data["mem_used_percent"].mean()
            peak_memory_d = driver_data["mem_used_percent"].max()
            lowest_memory_d = driver_data["mem_used_percent"].min()
            std_memory_d = driver_data["mem_used_percent"].std()

            first_driver_row = driver_data.iloc[0]
            driver_current_instance = first_driver_row.get("driver_node_type", "N/A")
        else:
            avg_cpu_d = peak_cpu_d = lowest_cpu_d = std_cpu_d = 0
            avg_memory_d = peak_memory_d = lowest_memory_d = std_memory_d = 0
            driver_current_instance = "N/A"

        driver_cpu_stats = {
            "mean": avg_cpu_d,
            "max": peak_cpu_d,
            "min": lowest_cpu_d,
            "std": std_cpu_d,
        }
        driver_memory_stats = {
            "mean": avg_memory_d,
            "max": peak_memory_d,
            "min": lowest_memory_d,
            "std": std_memory_d,
        }

        #Generate average duration of the job
        duration = calculate_job_durations(df,job_id)
        duration = duration.get('average_duration')

        # ---------------------------
        # Generate 3 Recommendations
        # ---------------------------
        multi_recs = generate_instance_recommendation(
            job_id=job_id,
            worker_current_instance=worker_current_instance,
            worker_cpu_stats=worker_cpu_stats,
            worker_memory_stats=worker_memory_stats,
            current_worker_count=current_worker_count,
            current_min_workers=current_min_workers,
            current_max_workers=current_max_workers,
            driver_current_instance=driver_current_instance,
            driver_cpu_stats=driver_cpu_stats,
            driver_memory_stats=driver_memory_stats,
            duration=duration
        )

        # For each recommendation # (1,2,3), store a row in all_rows
        for i in range(3):
            rec_number = f"Rec #{i+1}"
            row = {
                "Cluster Job ID": job_id,
                "Job Name": job_name,
                "Recommendation": rec_number,
                "Old Worker Instance": worker_current_instance,
                "Old Driver Instance": driver_current_instance,
                "Old Min Workers": current_min_workers,
                "Old Max Workers": current_max_workers,
                "New Worker Instance": multi_recs["Worker Instance"][i],
                "New Driver Instance": multi_recs["Driver Instance"][i],
                "New Min Workers": multi_recs["Min Workers"][i],
                "New Max Workers": multi_recs["Max Workers"][i],
            }
            all_rows.append(row)

    # Create a final DataFrame
    final_df = pd.DataFrame(all_rows)
    return final_df


# -------------------------------------------------------------
# Analyze Single Job
# -------------------------------------------------------------
def analyze_job(df, cluster_job_id):
    """
    Analyze a single job and return its stats and recommendations
    """
    # Ensure correct data types
    df["driver"] = df["driver"].astype(bool)
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    
    job_data = df[df["cluster_job_id"] == cluster_job_id]
    job_name = job_data["job_name"].iloc[0] if not job_data.empty else "Unknown Job Name"
    
    results = {
        "job_id": cluster_job_id,
        "job_name": job_name,
        "stats": {},
        "recommendations": None,
        "instance_metrics": None
    }
    
    if not job_data.empty:
        # Worker data
        worker_data = job_data[job_data["driver"] == False]
        # Driver data
        driver_data = job_data[job_data["driver"] == True]
        
        # ---- Summaries for WORKERS ----
        total_clusters = worker_data["cluster_id"].nunique()
        if not worker_data.empty:
            avg_cpu_w = worker_data["cpu_user_percent"].mean()
            peak_cpu_w = worker_data["cpu_user_percent"].max()
            lowest_cpu_w = worker_data["cpu_user_percent"].min()
            std_cpu_w = worker_data["cpu_user_percent"].std()

            avg_memory_w = worker_data["mem_used_percent"].mean()
            peak_memory_w = worker_data["mem_used_percent"].max()
            lowest_memory_w = worker_data["mem_used_percent"].min()
            std_memory_w = worker_data["mem_used_percent"].std()
        else:
            avg_cpu_w = peak_cpu_w = lowest_cpu_w = std_cpu_w = 0
            avg_memory_w = peak_memory_w = lowest_memory_w = std_memory_w = 0

        worker_cpu_stats = {
            "mean": avg_cpu_w,
            "max": peak_cpu_w,
            "min": lowest_cpu_w,
            "std": std_cpu_w,
        }
        worker_memory_stats = {
            "mean": avg_memory_w,
            "max": peak_memory_w,
            "min": lowest_memory_w,
            "std": std_memory_w,
        }

        # Grab a representative worker row for worker settings (if any)
        if not worker_data.empty:
            first_worker_row = worker_data.iloc[0]
            worker_current_instance = first_worker_row.get("worker_node_type", "N/A")
            current_worker_count = first_worker_row.get("worker_count", 0)
            current_min_workers = first_worker_row.get("min_autoscale_workers", 0)
            current_max_workers = first_worker_row.get("max_autoscale_workers", 0)
        else:
            worker_current_instance = "No Worker Data"
            current_worker_count = 0
            current_min_workers = 0
            current_max_workers = 0

        # ---- Summaries for DRIVER ----
        if not driver_data.empty:
            avg_cpu_d = driver_data["cpu_user_percent"].mean()
            peak_cpu_d = driver_data["cpu_user_percent"].max()
            lowest_cpu_d = driver_data["cpu_user_percent"].min()
            std_cpu_d = driver_data["cpu_user_percent"].std()

            avg_memory_d = driver_data["mem_used_percent"].mean()
            peak_memory_d = driver_data["mem_used_percent"].max()
            lowest_memory_d = driver_data["mem_used_percent"].min()
            std_memory_d = driver_data["mem_used_percent"].std()

            first_driver_row = driver_data.iloc[0]
            driver_current_instance = first_driver_row.get("driver_node_type", "N/A")
        else:
            avg_cpu_d = peak_cpu_d = lowest_cpu_d = std_cpu_d = 0
            avg_memory_d = peak_memory_d = lowest_memory_d = std_memory_d = 0
            driver_current_instance = "No Driver Data"

        driver_cpu_stats = {
            "mean": avg_cpu_d,
            "max": peak_cpu_d,
            "min": lowest_cpu_d,
            "std": std_cpu_d,
        }
        driver_memory_stats = {
            "mean": avg_memory_d,
            "max": peak_memory_d,
            "min": lowest_memory_d,
            "std": std_memory_d,
        }

        # Generate average duration of the job
        duration = calculate_job_durations(df, cluster_job_id)
        duration = duration.get('average_duration')

        # Generate *three* recommendations via LLM
        multi_recs = generate_instance_recommendation(
            job_id=cluster_job_id,
            worker_current_instance=worker_current_instance,
            worker_cpu_stats=worker_cpu_stats,
            worker_memory_stats=worker_memory_stats,
            current_worker_count=current_worker_count,
            current_min_workers=current_min_workers,
            current_max_workers=current_max_workers,
            driver_current_instance=driver_current_instance,
            driver_cpu_stats=driver_cpu_stats,
            driver_memory_stats=driver_memory_stats,
            duration=duration
        )

        # Build a DataFrame for the 3 recommendations
        data_for_table = {
            "Recommendation": ["Rec #1", "Rec #2", "Rec #3"],
            "Old Worker Instance": [worker_current_instance] * 3,
            "Old Driver Instance": [driver_current_instance] * 3,
            "Old Min Workers": [current_min_workers] * 3,
            "Old Max Workers": [current_max_workers] * 3,
            "New Worker Instance": multi_recs["Worker Instance"][:3],
            "New Driver Instance": multi_recs["Driver Instance"][:3],
            "New Min Workers": multi_recs["Min Workers"][:3],
            "New Max Workers": multi_recs["Max Workers"][:3],
        }
        recs_df = pd.DataFrame(data_for_table)
        
        # ---- Instance-Level Metrics ----
        instance_metrics = (
            job_data.groupby("instance_id")
            .apply(
                lambda x: pd.Series(
                    {
                        "Is Driver": x["driver"].iloc[0],
                        "Avg CPU Usage (%)": x["cpu_user_percent"].mean(),
                        "Peak CPU Usage (%)": x["cpu_user_percent"].max(),
                        "Lowest CPU Usage (%)": x["cpu_user_percent"].min(),
                        "Avg Memory Usage (%)": x["mem_used_percent"].mean(),
                        "Peak Memory Usage (%)": x["mem_used_percent"].max(),
                        "Lowest Memory Usage (%)": x["mem_used_percent"].min(),
                        "Peak CPU Time": (
                            x.loc[x["cpu_user_percent"].idxmax(), "start_time"]
                            if x["cpu_user_percent"].idxmax() in x.index
                            else None
                        ),
                        "Peak Memory Time": (
                            x.loc[x["mem_used_percent"].idxmax(), "start_time"]
                            if x["mem_used_percent"].idxmax() in x.index
                            else None
                        ),
                    }
                )
            )
            .reset_index()
        )
        
        # Store all the results
        results["stats"] = {
            "worker": {
                "total_clusters": total_clusters,
                "cpu": worker_cpu_stats,
                "memory": worker_memory_stats,
                "current_instance": worker_current_instance,
                "current_worker_count": current_worker_count,
                "current_min_workers": current_min_workers,
                "current_max_workers": current_max_workers
            },
            "driver": {
                "cpu": driver_cpu_stats,
                "memory": driver_memory_stats,
                "current_instance": driver_current_instance
            },
            "duration": duration
        }
        results["recommendations"] = recs_df
        results["instance_metrics"] = instance_metrics
    
    return results


# -------------------------------------------------------------
# Main function to process CSV
# -------------------------------------------------------------
def process_csv_file(csv_path,format):
    """
    Process a CSV file and return bulk recommendations
    """
    if format == 'csv':
        df = pd.read_csv(csv_path)
    elif format == 'df':
        df = csv_path.toPandas()
    else:
        df_spark = spark.read.format('csv').option("header", "true").load(path)
        df = df_spark.toPandas()

    # Validate columns
    is_valid, missing_columns = validate_csv_columns(df)
    if not is_valid:
        raise ValueError(f"The CSV is missing required columns: {', '.join(missing_columns)}")
    
    # Generate bulk recommendations
    recommendations_df = generate_bulk_recommendations(df)
    
    return {
        "data": df,
        "recommendations": recommendations_df
    }


if __name__ == "__main__":
    df = spark.sql('''select * from tfs_ml_cluster.databricks_jobs_data''')
    results = process_csv_file(df,format='df')
    job_analysis = analyze_job(df, "1123321794075745")
    
    #Save recommendations to CSV
    # recommendations_df = generate_bulk_recommendations(df)
    # recommendations_df.to_csv("cluster_recommendations.csv", index=False)
    
    print("Cluster analysis module loaded successfully")
