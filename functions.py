import pyspark 
from pyspark.sql import functions as F
import pandas as pd


# function to find average worker for a cluster 

def find_avg_workers(df:pyspark.sql.DataFrame) -> pd.DataFrame:

    grouped_df = df.filter(F.col("driver") == "false") \
                 .groupBy("start_time","end_time","cluster_id").agg(F.count("*").alias ("count"))

    avg_workers = grouped_df.groupBy('cluster_id') \
                          .agg(F.round(F.sum("count") / F.count("*"),2).alias("avg_workers"))

    avg_df = df.join(avg_workers, on=["cluster_id"], how="left")

    return avg_df  

# function to find duration 

def find_duration(df:pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:

  duration_df = (df.withColumn("start_time", F.to_timestamp("start_time"))
                   .withColumn("end_time", F.to_timestamp("end_time"))
                   .groupBy('cluster_id')
                   .agg(((F.unix_timestamp(F.max("end_time")) - F.unix_timestamp(F.min("start_time")))/60).alias("duration_mins")
             ))
  
  return duration_df 
