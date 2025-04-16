import streamlit as st
import time
import threading
import concurrent.futures
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt
import os
from io import BytesIO
import base64

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    
st.set_page_config(
    page_title="Threading Performance Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .stButton button {
        width: 100%;
    }
    .benchmark-result {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        background-color: #f0f2f6;
    }
    .metric-card {
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    .winner {
        font-weight: bold;
        color: #0068c9;
    }
</style>
""", unsafe_allow_html=True)

class ComputationalTask:
    """Base class for computational tasks."""
    def __init__(self, name):
        self.name = name
    
    def prepare_data(self, size):
        """Prepare data for the task based on size parameter."""
        pass
    
    def compute_single_threaded(self, data):
        """Perform single-threaded computation."""
        pass
    
    def compute_chunk(self, data_chunk):
        """Compute a chunk of data (for parallel processing)."""
        pass
    
    def get_default_sizes(self):
        """Return default problem sizes for this task."""
        pass


class MatrixMultiplication(ComputationalTask):
    """Matrix multiplication task."""
    def __init__(self):
        super().__init__("Matrix Multiplication")
    
    def prepare_data(self, size):
        """Generate two random matrices of given size."""
        matrix_a = np.random.randint(1, 10, (size, size)).astype(np.float64)
        matrix_b = np.random.randint(1, 10, (size, size)).astype(np.float64)
        return (matrix_a, matrix_b)
    
    def compute_single_threaded(self, data):
        """Perform matrix multiplication using numpy."""
        matrix_a, matrix_b = data
        return np.matmul(matrix_a, matrix_b)
    
    def compute_chunk(self, data_chunk):
        """Compute a chunk of matrix multiplication."""
        matrix_a_chunk, matrix_b = data_chunk
        return np.matmul(matrix_a_chunk, matrix_b)
    
    def get_default_sizes(self):
        """Return default problem sizes for matrix multiplication."""
        return [100, 500, 1000]


class ImageProcessing(ComputationalTask):
    """Image processing task (Gaussian blur)."""
    def __init__(self):
        super().__init__("Image Processing")
        
    def prepare_data(self, size):
        """Generate a random image of given size."""
        return np.random.randint(0, 256, (size, size, 3)).astype(np.uint8)
    
    def compute_single_threaded(self, image):
        """Apply Gaussian blur to the image."""
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(image, sigma=3)
        except ImportError:
            # Fallback to manual convolution if scipy is not available
            kernel_size = 5
            kernel = self._create_gaussian_kernel(kernel_size, 3)
            return self._apply_convolution(image, kernel)
    
    def compute_chunk(self, data_chunk):
        """Apply Gaussian blur to an image chunk."""
        image_chunk, start_row, end_row = data_chunk
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(image_chunk, sigma=3), start_row, end_row
        except ImportError:
            # Fallback to manual convolution
            kernel_size = 5
            kernel = self._create_gaussian_kernel(kernel_size, 3)
            return self._apply_convolution(image_chunk, kernel), start_row, end_row
    
    def _create_gaussian_kernel(self, size, sigma):
        """Create a Gaussian kernel for convolution."""
        kernel = np.fromfunction(
            lambda x, y: (1/(2*np.pi*sigma**2)) * 
                          np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
            (size, size)
        )
        return kernel / np.sum(kernel)
    
    def _apply_convolution(self, image, kernel):
        """Apply convolution with the given kernel to the image."""
        # Simple convolution implementation (not optimized)
        output = np.zeros_like(image)
        padded = np.pad(image, ((2, 2), (2, 2), (0, 0)), mode='constant')
        for c in range(3):  # RGB channels
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    output[i, j, c] = np.sum(
                        padded[i:i+5, j:j+5, c] * kernel
                    )
        return output
    
    def get_default_sizes(self):
        """Return default problem sizes for image processing."""
        return [100, 500, 1000]


class MonteCarloPi(ComputationalTask):
    """Monte Carlo estimation of Pi."""
    def __init__(self):
        super().__init__("Monte Carlo Pi")
    
    def prepare_data(self, size):
        """Size represents the number of random points to generate."""
        return size
    
    def compute_single_threaded(self, n_points):
        """Estimate Pi using Monte Carlo method."""
        points_inside_circle = 0
        for _ in range(n_points):
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            if x**2 + y**2 <= 1:
                points_inside_circle += 1
        return 4 * points_inside_circle / n_points
    
    def compute_chunk(self, n_points_chunk):
        """Compute Pi estimation for a chunk of points."""
        points_inside_circle = 0
        for _ in range(n_points_chunk):
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            if x**2 + y**2 <= 1:
                points_inside_circle += 1
        return points_inside_circle
    
    def get_default_sizes(self):
        """Return default problem sizes for Monte Carlo Pi."""
        return [100000, 1000000, 5000000]


class PerformanceSimulator:
    """Performance simulator for comparing threading approaches."""
    
    def __init__(self, task_type=None, problem_size=None, thread_counts=None, 
                 execution_models=None, iterations=3, warmup_runs=1, status_callback=None):
        # Available tasks
        self.available_tasks = {
            "matrix_mult": MatrixMultiplication(),
            "image_process": ImageProcessing(),
            "monte_carlo": MonteCarloPi()
        }
        
        # Set the task
        if task_type is None or task_type not in self.available_tasks:
            self.task = self.available_tasks["matrix_mult"]
        else:
            self.task = self.available_tasks[task_type]
        
        # Set problem size
        if problem_size is None:
            self.problem_size = self.task.get_default_sizes()[0]
        else:
            self.problem_size = problem_size
        
        # Set thread counts
        if thread_counts is None:
            self.thread_counts = [1, 2, 4, 8]
        else:
            self.thread_counts = thread_counts
            
        # Set execution models
        all_models = ["single", "openmp", "pthreads", "multiprocess"]
        if execution_models is None:
            self.execution_models = ["single", "openmp", "multiprocess"]
        else:
            self.execution_models = [model for model in execution_models if model in all_models]
            
        # Set benchmark parameters
        self.iterations = iterations
        self.warmup_runs = warmup_runs
        
        # Results storage
        self.results = {}
        
        # Status callback
        self.status_callback = status_callback
        
        # Monitor if psutil is available
        self.monitor_resources = PSUTIL_AVAILABLE
    
    def update_status(self, message):
        """Update status using callback if available."""
        if self.status_callback:
            self.status_callback(message)
        
    def run_single_threaded(self):
        data = self.task.prepare_data(self.problem_size)
        
        # Track resources if available
        if self.monitor_resources:
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        start_time = time.time()
        result = self.task.compute_single_threaded(data)
        execution_time = time.time() - start_time
        
        if self.monitor_resources:
            memory_after = process.memory_info().rss / (1024 * 1024)  # MB
            memory_used = memory_after - memory_before
        else:
            memory_used = 0
        
        return {
            "time": execution_time,
            "memory": memory_used,
            "result": result
        }
    
    def run_openmp_like(self, num_threads):
        data = self.task.prepare_data(self.problem_size)
        
        # Prepare chunks based on task type
        if self.task.name == "Matrix Multiplication":
            matrix_a, matrix_b = data
            chunk_size = max(1, matrix_a.shape[0] // num_threads)
            chunks = [(matrix_a[i:i+chunk_size], matrix_b) for i in range(0, matrix_a.shape[0], chunk_size)]
        
        elif self.task.name == "Image Processing":
            image = data
            chunk_size = max(1, image.shape[0] // num_threads)
            chunks = [(image[i:i+chunk_size], i, i+chunk_size) for i in range(0, image.shape[0], chunk_size)]
        
        elif self.task.name == "Monte Carlo Pi":
            n_points = data
            chunk_size = max(1, n_points // num_threads)
            chunks = [chunk_size] * num_threads
        
        # Track resources if available
        if self.monitor_resources:
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Setup time measurement
        setup_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            setup_end = time.time()
            setup_time = setup_end - setup_start
            
            # Execute threads
            computation_start = time.time()
            futures = [executor.submit(self.task.compute_chunk, chunk) for chunk in chunks]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
            computation_time = time.time() - computation_start
        
        if self.monitor_resources:
            memory_after = process.memory_info().rss / (1024 * 1024)  # MB
            memory_used = memory_after - memory_before
        else:
            memory_used = 0
        
        # Total execution time
        execution_time = setup_time + computation_time
        
        return {
            "time": execution_time,
            "setup_time": setup_time,
            "computation_time": computation_time,
            "memory": memory_used,
            "result": results
        }
    
    def run_pthreads_like(self, num_threads):
        data = self.task.prepare_data(self.problem_size)
        results = [None] * num_threads
        
        # Prepare chunks based on task type
        if self.task.name == "Matrix Multiplication":
            matrix_a, matrix_b = data
            chunk_size = max(1, matrix_a.shape[0] // num_threads)
            chunks = [(matrix_a[i:i+chunk_size], matrix_b) for i in range(0, matrix_a.shape[0], chunk_size)]
        
        elif self.task.name == "Image Processing":
            image = data
            chunk_size = max(1, image.shape[0] // num_threads)
            chunks = [(image[i:i+chunk_size], i, i+chunk_size) for i in range(0, image.shape[0], chunk_size)]
        
        elif self.task.name == "Monte Carlo Pi":
            n_points = data
            chunk_size = max(1, n_points // num_threads)
            chunks = [chunk_size] * num_threads
        
        def thread_task(thread_id, data_chunk):
            results[thread_id] = self.task.compute_chunk(data_chunk)
        
        # Track resources if available
        if self.monitor_resources:
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Setup time measurement
        setup_start = time.time()
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=thread_task, args=(i, chunks[i]))
            threads.append(thread)
            thread.start()
        setup_end = time.time()
        setup_time = setup_end - setup_start
        
        # Join threads and measure computation time
        computation_start = time.time()
        for thread in threads:
            thread.join()
        computation_time = time.time() - computation_start
        
        if self.monitor_resources:
            memory_after = process.memory_info().rss / (1024 * 1024)  # MB
            memory_used = memory_after - memory_before
        else:
            memory_used = 0
        
        # Total execution time
        execution_time = setup_time + computation_time
        
        return {
            "time": execution_time,
            "setup_time": setup_time,
            "computation_time": computation_time,
            "memory": memory_used,
            "result": results
        }
    
    def run_multiprocessing(self, num_processes):
        data = self.task.prepare_data(self.problem_size)
        
        # Prepare chunks based on task type
        if self.task.name == "Matrix Multiplication":
            matrix_a, matrix_b = data
            chunk_size = max(1, matrix_a.shape[0] // num_processes)
            chunks = [(matrix_a[i:i+chunk_size], matrix_b) for i in range(0, matrix_a.shape[0], chunk_size)]
        
        elif self.task.name == "Image Processing":
            image = data
            chunk_size = max(1, image.shape[0] // num_processes)
            chunks = [(image[i:i+chunk_size], i, i+chunk_size) for i in range(0, image.shape[0], chunk_size)]
        
        elif self.task.name == "Monte Carlo Pi":
            n_points = data
            chunk_size = max(1, n_points // num_processes)
            chunks = [chunk_size] * num_processes
        
        # Track resources if available
        if self.monitor_resources:
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Setup time measurement
        setup_start = time.time()
        with multiprocessing.Pool(processes=num_processes) as pool:
            setup_end = time.time()
            setup_time = setup_end - setup_start
            
            # Execute processes
            computation_start = time.time()
            results = pool.map(self.task.compute_chunk, chunks)
            computation_time = time.time() - computation_start
        
        if self.monitor_resources:
            memory_after = process.memory_info().rss / (1024 * 1024)  # MB
            memory_used = memory_after - memory_before
        else:
            memory_used = 0
        
        # Total execution time
        execution_time = setup_time + computation_time
        
        return {
            "time": execution_time,
            "setup_time": setup_time,
            "computation_time": computation_time,
            "memory": memory_used,
            "result": results
        }
    
    def run_simulation(self, model, num_threads=1):
        if model == "single":
            self.update_status(f"Running Single-Threaded execution...")
            return self.run_single_threaded()
        elif model == "openmp":
            self.update_status(f"Running OpenMP-like execution with {num_threads} threads...")
            return self.run_openmp_like(num_threads)
        elif model == "pthreads":
            self.update_status(f"Running Pthreads-like execution with {num_threads} threads...")
            return self.run_pthreads_like(num_threads)
        elif model == "multiprocess":
            self.update_status(f"Running Multi-Process execution with {num_threads} processes...")
            return self.run_multiprocessing(num_threads)
        else:
            raise ValueError(f"Unknown execution model: {model}")
    
    def run_all_simulations(self):
        self.results = {}
        
        # Run warmup iterations if specified
        if self.warmup_runs > 0:
            self.update_status(f"Performing {self.warmup_runs} warmup run(s)...")
            for _ in range(self.warmup_runs):
                _ = self.run_single_threaded()
        
        # Single-threaded execution
        if "single" in self.execution_models:
            self.results["single"] = []
            for i in range(self.iterations):
                self.update_status(f"Running Single-Threaded (iteration {i+1}/{self.iterations})")
                result = self.run_single_threaded()
                self.results["single"].append(result)
        
        # Multi-threaded (OpenMP-like) execution
        if "openmp" in self.execution_models:
            self.results["openmp"] = {}
            for thread_count in self.thread_counts:
                if thread_count == 1:
                    continue  # Skip single-threaded case for OpenMP
                self.results["openmp"][thread_count] = []
                for i in range(self.iterations):
                    self.update_status(f"Running OpenMP-like with {thread_count} threads (iteration {i+1}/{self.iterations})")
                    result = self.run_openmp_like(thread_count)
                    self.results["openmp"][thread_count].append(result)
        
        # Multi-threaded (Pthreads-like) execution
        if "pthreads" in self.execution_models:
            self.results["pthreads"] = {}
            for thread_count in self.thread_counts:
                if thread_count == 1:
                    continue  # Skip single-threaded case for Pthreads
                self.results["pthreads"][thread_count] = []
                for i in range(self.iterations):
                    self.update_status(f"Running Pthreads-like with {thread_count} threads (iteration {i+1}/{self.iterations})")
                    result = self.run_pthreads_like(thread_count)
                    self.results["pthreads"][thread_count].append(result)
        
        # Multi-process execution
        if "multiprocess" in self.execution_models:
            self.results["multiprocess"] = {}
            for process_count in self.thread_counts:
                if process_count == 1:
                    continue  # Skip single-process case
                self.results["multiprocess"][process_count] = []
                for i in range(self.iterations):
                    self.update_status(f"Running Multi-Process with {process_count} processes (iteration {i+1}/{self.iterations})")
                    result = self.run_multiprocessing(process_count)
                    self.results["multiprocess"][process_count].append(result)
        
        return self.results
    
    def get_average_times(self):
        if not self.results:
            return {}
            
        avg_times = {}
        
        # Single-threaded
        if "single" in self.results and self.results["single"]:
            avg_times["Single-Threaded"] = np.mean([r["time"] for r in self.results["single"]])
        
        # OpenMP-like
        if "openmp" in self.results:
            for thread_count, results in self.results["openmp"].items():
                if results:
                    avg_times[f"OpenMP ({thread_count} threads)"] = np.mean([r["time"] for r in results])
        
        # Pthreads-like
        if "pthreads" in self.results:
            for thread_count, results in self.results["pthreads"].items():
                if results:
                    avg_times[f"Pthreads ({thread_count} threads)"] = np.mean([r["time"] for r in results])
        
        # Multi-process
        if "multiprocess" in self.results:
            for process_count, results in self.results["multiprocess"].items():
                if results:
                    avg_times[f"Multi-Process ({process_count} processes)"] = np.mean([r["time"] for r in results])
        
        return avg_times
    
    def get_best_configurations(self):
        if not self.results:
            return {}
            
        best_configs = {}
        
        # Single-threaded
        if "single" in self.results and self.results["single"]:
            best_configs["Single-Threaded"] = {
                "threads": 1,
                "time": np.mean([r["time"] for r in self.results["single"]])
            }
        
        # OpenMP-like
        if "openmp" in self.results and self.results["openmp"]:
            best_thread_count = min(self.results["openmp"].keys(),
                                   key=lambda tc: np.mean([r["time"] for r in self.results["openmp"][tc]]))
            best_configs["OpenMP"] = {
                "threads": best_thread_count,
                "time": np.mean([r["time"] for r in self.results["openmp"][best_thread_count]])
            }
        
        # Pthreads-like
        if "pthreads" in self.results and self.results["pthreads"]:
            best_thread_count = min(self.results["pthreads"].keys(),
                                   key=lambda tc: np.mean([r["time"] for r in self.results["pthreads"][tc]]))
            best_configs["Pthreads"] = {
                "threads": best_thread_count,
                "time": np.mean([r["time"] for r in self.results["pthreads"][best_thread_count]])
            }
        
        # Multi-process
        if "multiprocess" in self.results and self.results["multiprocess"]:
            best_process_count = min(self.results["multiprocess"].keys(),
                                    key=lambda pc: np.mean([r["time"] for r in self.results["multiprocess"][pc]]))
            best_configs["Multi-Process"] = {
                "threads": best_process_count,
                "time": np.mean([r["time"] for r in self.results["multiprocess"][best_process_count]])
            }
        
        return best_configs
        
    def get_overall_fastest(self):
        best_configs = self.get_best_configurations()
        if not best_configs:
            return None
            
        fastest_model = min(best_configs.keys(), key=lambda m: best_configs[m]["time"])
        return (fastest_model, best_configs[fastest_model]["threads"], best_configs[fastest_model]["time"])


def display_execution_times_chart(avg_times):
    """Display a bar chart of execution times."""
    if not avg_times:
        st.info("No results to display yet. Run a simulation first.")
        return
        
    # Create dataframe for the chart
    df = pd.DataFrame({
        'Model': list(avg_times.keys()),
        'Time (seconds)': list(avg_times.values())
    })
    
    # Sort by execution time
    df = df.sort_values('Time (seconds)')
    
    # Create and display the Altair chart
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Time (seconds):Q', title='Execution Time (seconds)'),
        y=alt.Y('Model:N', sort='-x', title=None),
        color=alt.Color('Model:N', legend=None),
        tooltip=['Model', 'Time (seconds)']
    ).properties(
        title='Execution Time Comparison',
        height=min(400, 50 * len(df))
    )
    
    st.altair_chart(chart, use_container_width=True)


def display_speedup_chart(avg_times):
    """Display a bar chart of speedup relative to single-threaded."""
    if not avg_times or "Single-Threaded" not in avg_times:
        st.info("Results must include Single-Threaded execution to calculate speedup.")
        return
    
    single_time = avg_times["Single-Threaded"]
    
    # Create dataframe for the chart
    speedups = []
    for model, time in avg_times.items():
        if model != "Single-Threaded":
            speedups.append({
                'Model': model,
                'Speedup': single_time / time
            })
    
    if not speedups:
        st.info("No multi-threaded results to compare with single-threaded execution.")
        return
        
    df = pd.DataFrame(speedups)
    
    # Sort by speedup
    df = df.sort_values('Speedup', ascending=False)
    
    # Create and display the Altair chart
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Speedup:Q', title='Speedup vs Single-Threaded'),
        y=alt.Y('Model:N', sort='-x', title=None),
        color=alt.Color('Model:N', legend=None),
        tooltip=['Model', 'Speedup']
    ).properties(
        title='Speedup Comparison (Higher is Better)',
        height=min(400, 50 * len(df))
    )
    
    # Add a vertical line at speedup = 1
    rule = alt.Chart(pd.DataFrame({'Speedup': [1]})).mark_rule(color='red').encode(
        x='Speedup:Q'
    )
    
    st.altair_chart(chart + rule, use_container_width=True)


def display_thread_scaling_chart(simulator):
    """Display a line chart showing how performance scales with thread count."""
    if not simulator.results:
        st.info("No results to display yet. Run a simulation first.")
        return
        
    # Collect data for the chart
    scaling_data = []
    
    # Single-threaded as reference
    if "single" in simulator.results and simulator.results["single"]:
        single_time = np.mean([r["time"] for r in simulator.results["single"]])
        scaling_data.append({
            'Threads': 1,
            'Model': 'Single-Threaded',
            'Time (seconds)': single_time,
            'Speedup': 1.0
        })
    
    # Multi-threaded models
    models = {
        "openmp": "OpenMP",
        "pthreads": "Pthreads",
        "multiprocess": "Multi-Process"
    }
    
    for model_key, model_name in models.items():
        if model_key in simulator.results:
            for thread_count, results in simulator.results[model_key].items():
                if results:
                    avg_time = np.mean([r["time"] for r in results])
                    speedup = single_time / avg_time if "single" in simulator.results else 1.0
                    scaling_data.append({
                        'Threads': thread_count,
                        'Model': model_name,
                        'Time (seconds)': avg_time,
                        'Speedup': speedup
                    })
    
    if not scaling_data:
        st.info("No thread scaling data available.")
        return
        
    df = pd.DataFrame(scaling_data)
    
    # Create the time chart
    time_chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X('Threads:Q', title='Number of Threads/Processes'),
        y=alt.Y('Time (seconds):Q'),
        color=alt.Color('Model:N'),
        tooltip=['Model', 'Threads', 'Time (seconds)']
    ).properties(
        title='Execution Time vs Thread Count',
        height=300
    )
    
    # Create the speedup chart
    speedup_chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X('Threads:Q', title='Number of Threads/Processes'),
        y=alt.Y('Speedup:Q', title='Speedup vs Single-Threaded'),
        color=alt.Color('Model:N'),
        tooltip=['Model', 'Threads', 'Speedup']
    ).properties(
        title='Speedup vs Thread Count',
        height=300
    )
    
    # Add ideal scaling line
    thread_counts = sorted(df['Threads'].unique())
    ideal_scaling = pd.DataFrame({
        'Threads': thread_counts,
        'Ideal Speedup': thread_counts
    })
    
    ideal_line = alt.Chart(ideal_scaling).mark_line(
        strokeDash=[5, 5],
        color='gray'
    ).encode(
        x='Threads:Q',
        y='Ideal Speedup:Q'
    )
    
    # Display the charts
    st.altair_chart(time_chart, use_container_width=True)
    st.altair_chart(speedup_chart + ideal_line, use_container_width=True)


def display_results_summary(simulator):
    """Display a summary of the simulation results."""
    if not simulator.results:
        st.info("No results to display yet. Run a simulation first.")
        return
        
    # Get the best configurations
    best_configs = simulator.get_best_configurations()
    
    # Get overall fastest
    fastest = simulator.get_overall_fastest()
    
    # Display summary metrics
    st.subheader("Performance Summary")
    
    # Create three metrics columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if fastest:
            st.metric("Fastest Model", fastest[0], f"{fastest[2]:.4f}s")
        
    with col2:
        if fastest and fastest[0] != "Single-Threaded":
            speedup = best_configs["Single-Threaded"]["time"] / fastest[2]
            st.metric("Max Speedup", f"{speedup:.2f}x", f"{fastest[1]} threads")
        
    with col3:
        if "single" in simulator.results and simulator.results["single"]:
            st.metric("Single-Threaded Time", f"{best_configs['Single-Threaded']['time']:.4f}s")
    
    # Display detailed results in an expandable section
    with st.expander("Detailed Results", expanded=False):
        for model, config in best_configs.items():
            st.markdown(f"**{model}**: {config['time']:.4f}s with {config['threads']} thread(s)")


def app():
    """Main Streamlit app function."""
    st.title("Threading Performance Simulator")
    st.markdown("Compare Single-Threaded, Multi-Threaded, and Multi-Process execution performance")
    
    # Create the sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Task selection
    task_options = {
        "matrix_mult": "Matrix Multiplication",
        "image_process": "Image Processing",
        "monte_carlo": "Monte Carlo Pi Estimation"
    }
    selected_task = st.sidebar.selectbox(
        "Select Task",
        options=list(task_options.keys()),
        format_func=lambda x: task_options[x]
    )
    
    # Get a sample task instance to get default sizes
    task_instance = {
        'matrix_mult': MatrixMultiplication(),
        'image_process': ImageProcessing(),
        'monte_carlo': MonteCarloPi()
    }[selected_task]
    default_sizes = task_instance.get_default_sizes()
    
    # Problem size
    problem_size = st.sidebar.select_slider(
        "Problem Size",
        options=default_sizes,
        value=default_sizes[0]
    )
    
    # Thread counts
    available_threads = list(range(1, multiprocessing.cpu_count() * 2 + 1))
    thread_counts = st.sidebar.multiselect(
        "Thread/Process Counts",
        options=available_threads,
        default=[1, 2, 4, min(8, multiprocessing.cpu_count())]
    )
    
    # Ensure we have at least single-threaded for comparison
    if 1 not in thread_counts:
        thread_counts = [1] + thread_counts
        st.sidebar.info("Added single-threaded (1) for baseline comparison.")
    
    # Execution models
    execution_models = st.sidebar.multiselect(
        "Execution Models",
        options=["single", "openmp", "pthreads", "multiprocess"],
        default=["single", "openmp", "multiprocess"],
        format_func=lambda x: {
            "single": "Single-Threaded",
            "openmp": "Multi-Threaded (OpenMP-like)",
            "pthreads": "Multi-Threaded (Pthreads-like)",
            "multiprocess": "Multi-Process"
        }[x]
    )
    
    # Benchmark settings
    iterations = st.sidebar.slider("Benchmark Iterations", min_value=1, max_value=10, value=3)
    warmup_runs = st.sidebar.slider("Warmup Runs", min_value=0, max_value=5, value=1)
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Run Simulation", "Thread Scaling Analysis", "About"])
    
    with tab1:
        # Initialize session state for results if it doesn't exist
        if 'simulation_results' not in st.session_state:
            st.session_state.simulation_results = None
            st.session_state.avg_times = {}
        
        # Status placeholder
        status_placeholder = st.empty()
        
        # Run button
        if st.button("Run Simulation", key="run_simulation"):
            status_placeholder.info("Initializing simulation...")
            
            # Status update callback
            def update_status(message):
                status_placeholder.info(message)
            
            # Create simulator
            simulator = PerformanceSimulator(
                task_type=selected_task,
                problem_size=problem_size,
                thread_counts=thread_counts,
                execution_models=execution_models,
                iterations=iterations,
                warmup_runs=warmup_runs,
                status_callback=update_status
            )
            
            # Run simulation
            with st.spinner("Running simulation..."):
                simulator.run_all_simulations()
                st.session_state.simulation_results = simulator
                st.session_state.avg_times = simulator.get_average_times()
            
            status_placeholder.success("Simulation completed!")
            
            # Show results
            display_results_summary(simulator)
            display_execution_times_chart(st.session_state.avg_times)
            display_speedup_chart(st.session_state.avg_times)
        
        # Display results if they exist
        elif st.session_state.simulation_results is not None:
            status_placeholder.success("Results from previous simulation:")
            display_results_summary(st.session_state.simulation_results)
            display_execution_times_chart(st.session_state.avg_times)
            display_speedup_chart(st.session_state.avg_times)
        else:
            status_placeholder.info("Configure parameters and click 'Run Simulation' to start.")
    
    with tab2:
        if st.session_state.simulation_results is not None:
            st.subheader("Thread Scaling Analysis")
            st.markdown("This chart shows how performance scales with the number of threads/processes.")
            display_thread_scaling_chart(st.session_state.simulation_results)
        else:
            st.info("Run a simulation first to see thread scaling analysis.")
    
    with tab3:
        st.subheader("About Threading Performance Simulator")
        st.markdown("""
        This application benchmarks different threading models for computational tasks:
        
        - **Single-Threaded**: Traditional execution using a single thread
        - **Multi-Threaded (OpenMP-like)**: Using Python's ThreadPoolExecutor
        - **Multi-Threaded (Pthreads-like)**: Using Python's threading module directly
        - **Multi-Process**: Using Python's multiprocessing module
        
        ### How it Works
        
        The simulator runs each model with the specified number of threads/processes for the selected computational task and measures execution time. The results help identify which approach works best for different types of workloads.
        
        ### Note on Python's GIL
        
        Python's Global Interpreter Lock (GIL) limits the effectiveness of multi-threading for CPU-bound tasks. For these tasks, the multi-process approach often performs better as it bypasses the GIL by using separate Python interpreters.
        
        ### Computational Tasks
        
        - **Matrix Multiplication**: CPU-bound task that involves multiplying two matrices
        - **Image Processing**: Memory-intensive task that applies a Gaussian blur to an image
        - **Monte Carlo Pi Estimation**: Stochastic simulation to estimate the value of Pi
        """)


if __name__ == "__main__":
    app()
