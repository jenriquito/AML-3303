import polars as pl
import time
from concurrent.futures import ThreadPoolExecutor

csv_file = "/Users/jenfercz/Documents/JenFercz/Python/Training/AML-3303/Week06/yellow_tripdata_2020-04.csv"
# csv_file = "yellow_tripdata_2020-04.csv"

print("NYC TAXI OPERATIONS ANALYSIS")

# Step 1: Load the dataset
print("\n[STEP 1] Loading dataset")
start_time = time.time()

df = pl.read_csv(csv_file)

end_time = time.time()
print(f"Total rows: {len(df)}")
print(f"Time taken: {end_time - start_time:.2f} seconds")

# Step 2: Clean the data. Remove invalid rows
print("\n[STEP 2] Cleaning data")

# Remove rows where passenger_count is null or zero
df_clean = df.filter(pl.col('passenger_count').is_not_null())
df_clean = df_clean.filter(pl.col('passenger_count') > 0)

print(f"Rows after cleaning: {len(df_clean)}")

# Step 3: Create passenger groups (1, 2, 3, 4+)
print("\n[STEP 3] Creating passenger groups")

# Add a new column for passenger groups
df_clean = df_clean.with_columns([
    pl.when(pl.col('passenger_count') >= 4)
      .then(pl.lit('4+'))
      .otherwise(pl.col('passenger_count').cast(pl.Utf8))
      .alias('passenger_group')
])

print("Passenger groups created: 1, 2, 3, 4+")

# Step 4: Function to analyze one passenger group
def analyze_one_group(group_name):
    """
    This function takes a passenger group name (like '1' or '2')
    and calculates all the metrics for that group.
    """
    print(f"Analyzing group: {group_name}...")
    
    # Filter the data for this specific group
    group_data = df_clean.filter(pl.col('passenger_group') == group_name)
    
    # Calculate total revenue
    revenue = group_data.select(pl.col('total_amount').sum()).item()
    
    # Calculate average tip
    avg_tip = group_data.select(pl.col('tip_amount').mean()).item()
    
    # Calculate maximum tip
    max_tip = group_data.select(pl.col('tip_amount').max()).item()
    
    # Count number of trips
    num_trips = len(group_data)
    
    # Return all results in a dictionary
    result = {
        'group': group_name,
        'revenue': revenue,
        'avg_tip': avg_tip,
        'max_tip': max_tip,
        'trips': num_trips
    }
    
    return result

# Step 5: Use multithreading to analyze all groups in parallel
print("\n[STEP 4] Analyzing groups using multithreading")
analysis_start = time.time()

# Get list of unique passenger groups
groups = ['1', '2', '3', '4+']

# Create a thread pool with 4 workers (one for each group)
with ThreadPoolExecutor(max_workers=4) as executor:
    # Send each group to be analyzed in parallel
    # executor.map() runs the function for each group simultaneously
    results = list(executor.map(analyze_one_group, groups))

analysis_end = time.time()
print(f"Analysis time: {analysis_end - analysis_start:.2f} seconds")

# Step 6: Results
print("\n" + "RESULTS:")

for result in results:
    print(f"\nPassenger Group: {result['group']}")
    print(f"Total Revenue: ${result['revenue']:,.2f}")
    print(f"Average Tip: ${result['avg_tip']:.2f}")
    print(f"Maximum Tip: ${result['max_tip']:.2f}")
    print(f"Number of Trips: {result['trips']:,}")

# Step 7: Create a summary table
print("\n" + "SUMMARY TABLE:")

# Convert results to a Polars DataFrame
summary = pl.DataFrame(results)
print(summary)