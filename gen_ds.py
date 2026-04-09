import pandas as pd
import numpy as np
import os

def generate_and_save_data(duration_sec=3600, num_machines=5, folder_path="data"):
    """
    Generates raw synthetic workload data and saves it to CSV files.
    """
    # Create directory if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # 1. SETUP TIMELINE
    t = np.arange(0, duration_sec)
    
    # 2. GENERATE MACHINE DATA (Server Health Logs)
    # This simulates the "Machine-Level Dataset"
    machine_records = []
    for m_id in range(num_machines):
        # Create a base pattern (sine wave) + noise + random spikes
        base = 0.3 + 0.2 * np.sin(2 * np.pi * t / 600) 
        noise = np.random.normal(0, 0.05, duration_sec)
        
        # Add a few major spikes to test the Proactive Neuro module
        spikes = np.zeros(duration_sec)
        for _ in range(3):
            start = np.random.randint(0, duration_sec - 100)
            spikes[start:start+40] = 0.3
            
        m_df = pd.DataFrame({
            'timestamp': t,
            'machine_id': f"Server_{m_id:02d}",
            'cpu_util_actual': np.clip(base + noise + spikes, 0.05, 0.95),
            'mem_util_actual': np.clip(0.4 + noise, 0.1, 0.9)
        })
        machine_records.append(m_df)
    
    machine_df = pd.concat(machine_records)
    machine_csv_path = os.path.join(folder_path, "raw_machines.csv")
    machine_df.to_csv(machine_csv_path, index=False)

    # 3. GENERATE TASK DATA (Incoming Request Logs)
    # This simulates the "Task-Level Dataset" (reqid, cpu_req, etc.)
    tasks = []
    req_id_counter = 10001
    
    for sec in t:
        # Link arrival probability to the workload sine wave
        # This creates patterns for the LSTM to learn
        avg_arrivals = 1 + np.sin(2 * np.pi * sec / 600) + 1 
        num_arrivals = np.random.poisson(avg_arrivals)
        
        for _ in range(num_arrivals):
            tasks.append({
                'req_id': f"REQ_{req_id_counter}",
                'timestamp': sec,
                'cpu_requested': round(np.random.uniform(0.05, 0.25), 3),
                'mem_requested': round(np.random.uniform(0.02, 0.15), 3),
                'task_duration': np.random.randint(10, 120) # Duration in seconds
            })
            req_id_counter += 1
            
    task_df = pd.DataFrame(tasks)
    task_csv_path = os.path.join(folder_path, "raw_tasks.csv")
    task_df.to_csv(task_csv_path, index=False)

    print(f"Success! Data saved to '{folder_path}' folder.")
    print(f" - {machine_csv_path}: {len(machine_df)} rows")
    print(f" - {task_csv_path}: {len(task_df)} rows")

def generate_and_save_refined_dataset(duration_sec=3600, num_machines=5, latency_sec=2, prediction_horizon=30):
    # 1. SETUP
    if not os.path.exists("data"):
        os.makedirs("data")
    
    t = np.arange(0, duration_sec)
    
    # 2. GENERATE MACHINE DATA
    machine_records = []
    for m_id in range(num_machines):
        base = 0.3 + 0.2 * np.sin(2 * np.pi * t / 600) 
        noise = np.random.normal(0, 0.05, duration_sec)
        m_df = pd.DataFrame({
            'timestamp': t,
            'machine_id': f"Server_{m_id:02d}",
            'cpu_util_actual': np.clip(base + noise, 0.05, 0.95),
            'mem_util_actual': np.clip(0.4 + noise, 0.1, 0.9)
        })
        machine_records.append(m_df)
    
    m_df = pd.concat(machine_records)

    # 3. GENERATE TASK DATA & ARRIVAL RATE
    tasks = []
    req_id_counter = 10001
    for sec in t:
        num_arrivals = np.random.poisson(1 + np.sin(2 * np.pi * sec / 600) + 1)
        for _ in range(num_arrivals):
            tasks.append({
                'req_id': f"REQ_{req_id_counter}",
                'timestamp': sec,
                'cpu_requested': round(np.random.uniform(0.05, 0.25), 3),
                'task_duration': np.random.randint(10, 60)
            })
            req_id_counter += 1
            
    t_df = pd.DataFrame(tasks)
    arrival_counts = t_df.groupby('timestamp').size().reset_index(name='Arrival_Rate')
    m_df = pd.merge(m_df, arrival_counts, on='timestamp', how='left').fillna(0)

    # 4. FEATURE ENGINEERING (Latency, Slope, Queue, Target)
    processed_groups = []
    for m_id, group in m_df.groupby('machine_id'):
        group = group.sort_values('timestamp')
        
        # FEATURE: CPU_Delayed (The Stale Data)
        group['CPU_Delayed'] = group['cpu_util_actual'].shift(latency_sec)
        
        # FEATURE: CPU_Slope (The Trend)
        group['CPU_Slope'] = group['cpu_util_actual'].diff()
        
        # FEATURE: Queue_Length (Simulated active tasks)
        # We model this as a function of current load and arrivals
        group['Queue_Length'] = (group['cpu_util_actual'] * 8).astype(int) + (group['Arrival_Rate'] * 0.5).astype(int)
        
        # TARGET: The future value the LSTM needs to learn
        group['Target_CPU_Future'] = group['cpu_util_actual'].shift(-prediction_horizon)
        
        processed_groups.append(group)

    final_df = pd.concat(processed_groups).dropna()
    
    # 5. SAVE TO CSV
    output_path = "data/refined_dataset.csv"
    final_df.to_csv(output_path, index=False)
    
    print(f"File saved: {output_path}")
    print(f"Columns: {list(final_df.columns)}")
    return final_df


if __name__ == "__main__":
    generate_and_save_data()
    generate_and_save_refined_dataset()


