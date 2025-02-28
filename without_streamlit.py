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

instance_price = {'m6g.large': 0.116, 'm7g.large': 0.1296, 'm5a.large': 0.117, 'm6gd.large': 0.1294, 'm5.large': 0.13, 'm6i.large': 0.134, 'm4.large': 0.14, 'r6g.large': 0.1508, 'm7gd.large': 0.1548, 'r7g.large': 0.1701, 'm5d.large': 0.147, 'r5a.large': 0.154, 'r6gd.large': 0.1652, 'm6id.large': 0.15665, 'm5n.large': 0.155, 'r6i.large': 0.177, 'r5.large': 0.171, 'm5dn.large': 0.177, 'r7gd.large': 0.1991, 'm6in.large': 0.17723, 'r5d.large': 0.189, 'r5n.large': 0.194, 'r6id.large': 0.2022, 'm6idn.large': 0.19712, 'm5zn.large': 0.2152, 'r5dn.large': 0.219, 'i4i.large': 0.241, 'r6in.large': 0.22533, 'r6idn.large': 0.24639, 'c6g.xlarge': 0.213, 'c7g.xlarge': 0.2345, 'c6gd.xlarge': 0.2306, 'm6g.xlarge': 0.231, 'c5a.xlarge': 0.215, 'm7g.xlarge': 0.2592, 'c5.xlarge': 0.231, 'c6i.xlarge': 0.24, 'm5a.xlarge': 0.233, 'c5ad.xlarge': 0.241, 'm6gd.xlarge': 0.2578, 'c7gd.xlarge': 0.2714, 'm5.xlarge': 0.261, 'm6i.xlarge': 0.268, 'c5d.xlarge': 0.253, 'm4.xlarge': 0.275, 'r6g.xlarge': 0.3016, 'c6id.xlarge': 0.2716, 'm7gd.xlarge': 0.3096, 'r7g.xlarge': 0.3402, 'c5n.xlarge': 0.294, 'm5d.xlarge': 0.295, 'r5a.xlarge': 0.307, 'c6in.xlarge': 0.2968, 'r6gd.xlarge': 0.3304, 'm6id.xlarge': 0.3133, 'm5n.xlarge': 0.309, 'r6i.xlarge': 0.354, 'r5.xlarge': 0.342, 'r4.xlarge': 0.366, 'm5dn.xlarge': 0.354, 'r7gd.xlarge': 0.3982, 'm6in.xlarge': 0.35446, 'r5d.xlarge': 0.378, 'r5n.xlarge': 0.388, 'r6id.xlarge': 0.4044, 'i3.xlarge': 0.412, 'm6idn.xlarge': 0.39424, 'm5zn.xlarge': 0.4303, 'r5dn.xlarge': 0.438, 'i4i.xlarge': 0.481, 'r6in.xlarge': 0.45066, 'r6idn.xlarge': 0.49278, 'c6g.2xlarge': 0.426, 'c7g.2xlarge': 0.47, 'c6gd.2xlarge': 0.4612, 'm6g.2xlarge': 0.462, 'c5a.2xlarge': 0.429, 'm7g.2xlarge': 0.5184, 'c6i.2xlarge': 0.479, 'c5.2xlarge': 0.461, 'c5ad.2xlarge': 0.481, 'm5a.2xlarge': 0.467, 'm6gd.2xlarge': 0.5156, 'c7gd.2xlarge': 0.5439, 'c5d.2xlarge': 0.505, 'm5.2xlarge': 0.521, 'm6i.2xlarge': 0.536, 'c4.2xlarge': 0.498, 'm4.2xlarge': 0.55, 'r6g.2xlarge': 0.6032, 'c6id.2xlarge': 0.5422, 'm7gd.2xlarge': 0.6191, 'r7g.2xlarge': 0.6804, 'c5n.2xlarge': 0.587, 'r5a.2xlarge': 0.613, 'm5d.2xlarge': 0.589, 'c6in.2xlarge': 0.5926, 'r6gd.2xlarge': 0.6608, 'm6id.2xlarge': 0.6266, 'm5n.2xlarge': 0.619, 'r6i.2xlarge': 0.708, 'r5.2xlarge': 0.684, 'r4.2xlarge': 0.732, 'm5dn.2xlarge': 0.707, 'r7gd.2xlarge': 0.7963, 'm6in.2xlarge': 0.70892, 'r5d.2xlarge': 0.756, 'r5n.2xlarge': 0.776, 'r6id.2xlarge': 0.8088, 'i3.2xlarge': 0.824, 'm6idn.2xlarge': 0.78848, 'm5zn.2xlarge': 0.8607, 'r5dn.2xlarge': 0.877, 'i4i.2xlarge': 0.961, 'r6in.2xlarge': 0.90132, 'r6idn.2xlarge': 0.98556, 'm5zn.3xlarge': 1.291, 'c6g.4xlarge': 0.852, 'c7g.4xlarge': 0.9401, 'c6gd.4xlarge': 0.9224, 'c5a.4xlarge': 0.859, 'm6g.4xlarge': 0.923, 'm7g.4xlarge': 1.0368, 'c5.4xlarge': 0.923, 'c6i.4xlarge': 0.958, 'c5ad.4xlarge': 0.962, 'm5a.4xlarge': 0.934, 'm6gd.4xlarge': 1.0302, 'c7gd.4xlarge': 1.0878, 'm5.4xlarge': 1.042, 'm6i.4xlarge': 1.072, 'c5d.4xlarge': 1.011, 'c4.4xlarge': 0.996, 'm4.4xlarge': 1.1, 'r6g.4xlarge': 1.2064, 'c6id.4xlarge': 1.0844, 'm7gd.4xlarge': 1.2383, 'r7g.4xlarge': 1.3608, 'c5n.4xlarge': 1.175, 'r5a.4xlarge': 1.227, 'm5d.4xlarge': 1.178, 'c6in.4xlarge': 1.1852, 'r6gd.4xlarge': 1.3216, 'm6id.4xlarge': 1.2532, 'm5n.4xlarge': 1.238, 'r6i.4xlarge': 1.416, 'r5.4xlarge': 1.368, 'r4.4xlarge': 1.464, 'm5dn.4xlarge': 1.414, 'r7gd.4xlarge': 1.5926, 'm6in.4xlarge': 1.41784, 'r5d.4xlarge': 1.512, 'r5n.4xlarge': 1.552, 'r6id.4xlarge': 1.6176, 'i3.4xlarge': 1.648, 'm6idn.4xlarge': 1.57696, 'r5dn.4xlarge': 1.754, 'i4i.4xlarge': 1.923, 'r6in.4xlarge': 1.80264, 'r6idn.4xlarge': 1.97112, 'm5zn.6xlarge': 2.582, 'c6g.8xlarge': 1.704, 'c7g.8xlarge': 1.8802, 'c6gd.8xlarge': 1.8448, 'c5a.8xlarge': 1.718, 'm6g.8xlarge': 1.845, 'm7g.8xlarge': 2.0736, 'c6i.8xlarge': 1.916, 'c5ad.8xlarge': 1.924, 'm5a.8xlarge': 1.867, 'm6gd.8xlarge': 2.0594, 'c7gd.8xlarge': 2.1755, 'm5.8xlarge': 2.084, 'm6i.8xlarge': 2.144, 'c6id.8xlarge': 2.1688, 'r6g.8xlarge': 2.4128, 'm7gd.8xlarge': 2.4766, 'r7g.8xlarge': 2.7216, 'r5a.8xlarge': 2.454, 'm5d.8xlarge': 2.356, 'c6in.8xlarge': 2.3704, 'r6gd.8xlarge': 2.6432, 'm6id.8xlarge': 2.5064, 'm5n.8xlarge': 2.475, 'r5.8xlarge': 2.736, 'r6i.8xlarge': 2.832, 'r4.8xlarge': 2.928, 'm5dn.8xlarge': 2.829, 'r7gd.8xlarge': 3.1853, 'm6in.8xlarge': 2.83568, 'r5d.8xlarge': 3.024, 'r5n.8xlarge': 3.104, 'r6id.8xlarge': 3.2352, 'i3.8xlarge': 3.296, 'm6idn.8xlarge': 3.15392, 'r5dn.8xlarge': 3.507, 'i4i.8xlarge': 3.846, 'r6in.8xlarge': 3.60528, 'r6idn.8xlarge': 3.94224, 'c5.9xlarge': 2.076, 'c4.8xlarge': 1.991, 'c5d.9xlarge': 2.274, 'c5n.9xlarge': 2.643, 'm4.10xlarge': 2.8, 'c6g.12xlarge': 2.556, 'c7g.12xlarge': 2.8202, 'c6gd.12xlarge': 2.7672, 'c5a.12xlarge': 2.576, 'm6g.12xlarge': 2.769, 'm7g.12xlarge': 3.1104, 'c5.12xlarge': 2.768, 'c6i.12xlarge': 2.874, 'm5a.12xlarge': 2.801, 'c5ad.12xlarge': 2.886, 'm6gd.12xlarge': 3.0906, 'c7gd.12xlarge': 3.2633, 'c5d.12xlarge': 3.032, 'm5.12xlarge': 3.127, 'm6i.12xlarge': 3.216, 'c6id.12xlarge': 3.2532, 'r6g.12xlarge': 3.6192, 'm7gd.12xlarge': 3.7148, 'r7g.12xlarge': 4.0824, 'r5a.12xlarge': 3.681, 'm5d.12xlarge': 3.535, 'c6in.12xlarge': 3.5556, 'r6gd.12xlarge': 3.9648, 'm6id.12xlarge': 3.7596, 'm5n.12xlarge': 3.713, 'r6i.12xlarge': 4.248, 'r5.12xlarge': 4.104, 'm5dn.12xlarge': 4.243, 'r7gd.12xlarge': 4.7779, 'm6in.12xlarge': 4.25352, 'r5d.12xlarge': 4.536, 'r5n.12xlarge': 4.656, 'r6id.12xlarge': 4.8528, 'm6idn.12xlarge': 4.73088, 'm5zn.12xlarge': 5.1641, 'r5dn.12xlarge': 5.261, 'r6in.12xlarge': 5.40792, 'r6idn.12xlarge': 5.91336, 'c6g.16xlarge': 3.408, 'c7g.16xlarge': 3.7603, 'c6gd.16xlarge': 3.6896, 'm6g.16xlarge': 3.692, 'c5a.16xlarge': 3.436, 'm7g.16xlarge': 4.1472, 'c6i.16xlarge': 3.833, 'm5a.16xlarge': 3.735, 'c5ad.16xlarge': 3.848, 'm6gd.16xlarge': 4.1208, 'c7gd.16xlarge': 4.351, 'm6i.16xlarge': 4.288, 'm5.16xlarge': 4.168, 'm4.16xlarge': 4.4, 'r6g.16xlarge': 4.8256, 'c6id.16xlarge': 4.3386, 'm7gd.16xlarge': 4.9531, 'r7g.16xlarge': 5.4432, 'm5d.16xlarge': 4.712, 'r5a.16xlarge': 4.907, 'c6in.16xlarge': 4.7418, 'r6gd.16xlarge': 5.2864, 'm6id.16xlarge': 5.0128, 'm5n.16xlarge': 4.95, 'r5.16xlarge': 5.472, 'r6i.16xlarge': 5.664, 'r4.16xlarge': 5.856, 'm5dn.16xlarge': 5.658, 'r7gd.16xlarge': 6.3706, 'm6in.16xlarge': 5.67136, 'r5d.16xlarge': 6.048, 'r5n.16xlarge': 6.208, 'r6id.16xlarge': 6.4704, 'i3.16xlarge': 6.592, 'm6idn.16xlarge': 6.30784, 'r5dn.16xlarge': 7.014, 'i4i.16xlarge': 7.691, 'r6in.16xlarge': 7.21056, 'r6idn.16xlarge': 7.88448, 'c5.18xlarge': 4.152, 'c5d.18xlarge': 4.548, 'c5n.18xlarge': 5.285, 'c5a.24xlarge': 5.152, 'c6i.24xlarge': 5.749, 'c5.24xlarge': 5.536, 'c5ad.24xlarge': 5.772, 'm5a.24xlarge': 5.602, 'm6i.24xlarge': 6.432, 'm5.24xlarge': 6.254, 'c5d.24xlarge': 6.064, 'c6id.24xlarge': 6.5074, 'r5a.24xlarge': 7.361, 'm5d.24xlarge': 7.07, 'c6in.24xlarge': 7.1122, 'm6id.24xlarge': 7.5192, 'm5n.24xlarge': 7.426, 'r5.24xlarge': 8.208, 'r6i.24xlarge': 8.496, 'm5dn.24xlarge': 8.486, 'm6in.24xlarge': 8.50704, 'r5d.24xlarge': 9.072, 'r5n.24xlarge': 9.312, 'r6id.24xlarge': 9.7056, 'm6idn.24xlarge': 9.46176, 'r5dn.24xlarge': 10.522, 'r6in.24xlarge': 10.81584, 'r6idn.24xlarge': 11.82672, 'c6i.32xlarge': 7.665, 'm6i.32xlarge': 8.576, 'c6id.32xlarge': 8.6762, 'c6in.32xlarge': 9.4826, 'm6id.32xlarge': 10.0256, 'r6i.32xlarge': 11.328, 'm6in.32xlarge': 11.34272, 'r6id.32xlarge': 12.9408, 'm6idn.32xlarge': 12.61568, 'i4i.32xlarge': 15.3824, 'r6in.32xlarge': 14.42112, 'r6idn.32xlarge': 15.76896}


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
    duration,
    average_worker,
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
    - Only return the instance name eg : 'r6g.large', 'c5.xlarge'.

     
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
      - Average Number of Workers Used For Most of the job: {average_worker}

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
            worker_cost = first_worker_row.get("final_cost", 0)
            if current_min_workers == current_max_workers:
                avg_worker = current_min_workers
            else:
                # avg_worker = first_worker_row.get("avg_workers", 0)
                avg_worker = worker_data['avg_workers'].mean()
            



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
            duration=duration,
            average_worker=avg_worker,
        )

        new_costs = [ instance_price.get(instance, 0) * (duration / 60) * (avg_worker)
        for instance in multi_recs["Worker Instance"][:3]]

        # Build a DataFrame for the 3 recommendations
        data_for_table = {
            "Recommendation": ["Rec #1", "Rec #2", "Rec #3"],
            "Old Worker Instance": [worker_current_instance] * 3,
            "Old Driver Instance": [driver_current_instance] * 3,
            "Old Min Workers": [current_min_workers] * 3,
            "Old Max Workers": [current_max_workers] * 3,
            "Avg Workers": [avg_worker] * 3,
            "Job Duration": [duration/60] * 3,
            "Old Cost": [worker_cost*(duration/60)] * 3,
            "New Worker Instance": multi_recs["Worker Instance"][:3],
            "New Driver Instance": multi_recs["Driver Instance"][:3],
            "New Min Workers": multi_recs["Min Workers"][:3],
            "New Max Workers": multi_recs["Max Workers"][:3],
            "New Costs": new_costs[:3],
            "Cost Savings": [worker_cost*(duration/60) - new_cost for new_cost in new_costs[:3]]
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
                "current_max_workers": current_max_workers,
                "avg_worker": avg_worker
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
        df_spark = spark.read.format('csv').option("header", "true").load(csv_path)
        df = df_spark.toPandas()

    # Validate columns
    is_valid, missing_columns = validate_csv_columns(df)
    if not is_valid:
        raise ValueError(f"The CSV is missing required columns: {', '.join(missing_columns)}")
    
    # # Generate bulk recommendations
    # recommendations_df = generate_bulk_recommendations(df)
    
    return {
        "data": df,
    }


if __name__ == "__main__":
    df = spark.sql('''select * from tfs_ml_cluster.databricks_jobs_cost_with_final_cost''').toPandas()
    # results = process_csv_file(df,format='df')
    job_analysis = analyze_job(df, "916235475766564")
    
    #Save recommendations to CSV
    # recommendations_df = generate_bulk_recommendations(df)
    # recommendations_df.to_csv("cluster_recommendations.csv", index=False)
    
    print("Cluster analysis module loaded successfully")
