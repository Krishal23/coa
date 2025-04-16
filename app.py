

import time
import threading
import concurrent.futures
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from tqdm import tqdm

# Check if running in a notebook for display configuration
try:
    from IPython import get_ipython
    if 'IPython' in sys.modules and get_ipython():
        from IPython import display
        IN_NOTEBOOK = True
    else:
        IN_NOTEBOOK = False
except ImportError:
    IN_NOTEBOOK = False

# Try to import optional packages for extended functionality
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
    # Set seaborn style if available
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
except ImportError:
    SEABORN_AVAILABLE = False


class ComputationalTask:
    """Base class for computational tasks."""
    def __init__(self, name):
        self.name = name
    
    def prepare_data(self, size):
        """Prepare data for the task based on size parameter."""
        raise NotImplementedError("Subclass must implement prepare_data method")
    
    def compute_single_threaded(self, data):
        """Perform single-threaded computation."""
        raise NotImplementedError("Subclass must implement compute_single_threaded method")
    
    def compute_chunk(self, data_chunk):
        """Compute a chunk of data (for parallel processing)."""
        raise NotImplementedError("Subclass must implement compute_chunk method")
    
    def get_default_sizes(self):
        """Return default problem sizes for this task."""
        raise NotImplementedError("Subclass must implement get_default_sizes method")


class MatrixMultiplication(ComputationalTask):
    """Matrix multiplication task."""
    def __init__(self):
        super().__init__("Matrix Multiplication")
    
    def prepare_data(self, size):
        """Generate two random matrices of given size."""
        print(f"Preparing {size}x{size} matrices...")
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
        print(f"Preparing {size}x{size} image...")
        return np.random.randint(0, 256, (size, size, 3)).astype(np.uint8)
    
    def compute_single_threaded(self, image):
        """Apply Gaussian blur to the image."""
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(image, sigma=3)
        except ImportError:
            # Fallback to manual convolution if scipy is not available
            print("Warning: scipy not available, using manual convolution")
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
        return [500, 1000, 2000]


class MonteCarloPi(ComputationalTask):
    """Monte Carlo estimation of Pi."""
    def __init__(self):
        super().__init__("Monte Carlo Pi")
    
    def prepare_data(self, size):
        """Size represents the number of random points to generate."""
        print(f"Preparing Monte Carlo simulation with {size} points...")
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
        return [1000000, 5000000, 10000000]


class PerformanceSimulator:
    """Performance simulator for comparing threading approaches."""
    
    def __init__(self, task_type=None, problem_sizes=None, thread_counts=None, 
                 execution_models=None, iterations=3, warmup_runs=1):
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
        
        # Set problem sizes
        if problem_sizes is None:
            self.problem_sizes = self.task.get_default_sizes()
        else:
            self.problem_sizes = problem_sizes
        
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
        
        # Monitor if psutil is available
        self.monitor_resources = PSUTIL_AVAILABLE
        
    def run_single_threaded(self, size):
        data = self.task.prepare_data(size)
        
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
    
    def run_openmp_like(self, size, num_threads):
        data = self.task.prepare_data(size)
        
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
    
    def run_pthreads_like(self, size, num_threads):
        data = self.task.prepare_data(size)
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
    
    def run_multiprocessing(self, size, num_processes):
        data = self.task.prepare_data(size)
        
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
    
    def run_simulation(self, model, size, num_threads=1):
        if model == "single":
            return self.run_single_threaded(size)
        elif model == "openmp":
            return self.run_openmp_like(size, num_threads)
        elif model == "pthreads":
            return self.run_pthreads_like(size, num_threads)
        elif model == "multiprocess":
            return self.run_multiprocessing(size, num_threads)
        else:
            raise ValueError(f"Unknown execution model: {model}")
    
    def run_all_simulations(self):
        self.results = {}
        
        # Single size for quick demonstration
        for size in self.problem_sizes:
            print(f"\nRunning simulations for {self.task.name} with size {size}")
            self.results[size] = {}
            
            # Warmup runs
            if self.warmup_runs > 0:
                print(f"Performing {self.warmup_runs} warmup run(s)...")
                for _ in range(self.warmup_runs):
                    _ = self.run_single_threaded(size)
            
            # Single-threaded execution
            if "single" in self.execution_models:
                print("Running Single-Threaded execution...")
                self.results[size]["single"] = []
                for i in range(self.iterations):
                    result = self.run_single_threaded(size)
                    self.results[size]["single"].append(result)
                    print(f"  Iteration {i+1}/{self.iterations}: {result['time']:.4f}s")
            
            # Multi-threaded (OpenMP-like) execution
            if "openmp" in self.execution_models:
                print("Running Multi-Threaded (OpenMP-like) execution...")
                self.results[size]["openmp"] = {}
                for thread_count in self.thread_counts:
                    if thread_count == 1:
                        continue  # Skip single-threaded case
                    print(f"  With {thread_count} threads:")
                    self.results[size]["openmp"][thread_count] = []
                    for i in range(self.iterations):
                        result = self.run_openmp_like(size, thread_count)
                        self.results[size]["openmp"][thread_count].append(result)
                        print(f"    Iteration {i+1}/{self.iterations}: {result['time']:.4f}s")
            
            # Multi-threaded (Pthreads-like) execution
            if "pthreads" in self.execution_models:
                print("Running Multi-Threaded (Pthreads-like) execution...")
                self.results[size]["pthreads"] = {}
                for thread_count in self.thread_counts:
                    if thread_count == 1:
                        continue  # Skip single-threaded case
                    print(f"  With {thread_count} threads:")
                    self.results[size]["pthreads"][thread_count] = []
                    for i in range(self.iterations):
                        result = self.run_pthreads_like(size, thread_count)
                        self.results[size]["pthreads"][thread_count].append(result)
                        print(f"    Iteration {i+1}/{self.iterations}: {result['time']:.4f}s")
            
            # Multi-process execution
            if "multiprocess" in self.execution_models:
                print("Running Multi-Process execution...")
                self.results[size]["multiprocess"] = {}
                for process_count in self.thread_counts:
                    if process_count == 1:
                        continue  # Skip single-process case
                    print(f"  With {process_count} processes:")
                    self.results[size]["multiprocess"][process_count] = []
                    for i in range(self.iterations):
                        result = self.run_multiprocessing(size, process_count)
                        self.results[size]["multiprocess"][process_count].append(result)
                        print(f"    Iteration {i+1}/{self.iterations}: {result['time']:.4f}s")
        
        return self.results
    
    def get_average_execution_times(self, size=None):
        if not self.results:
            raise ValueError("No results available. Run simulations first.")
        
        if size is None:
            size = list(self.results.keys())[0]
        
        avg_times = {}
        
        # Single-threaded
        if "single" in self.results[size]:
            avg_times["Single-Threaded"] = np.mean([r["time"] for r in self.results[size]["single"]])
        
        # OpenMP-like
        if "openmp" in self.results[size]:
            for thread_count, results in self.results[size]["openmp"].items():
                if len(results) > 0:
                    avg_times[f"Multi-Threaded (OpenMP-like) - {thread_count} threads"] = np.mean([r["time"] for r in results])
        
        # Pthreads-like
        if "pthreads" in self.results[size]:
            for thread_count, results in self.results[size]["pthreads"].items():
                if len(results) > 0:
                    avg_times[f"Multi-Threaded (Pthreads-like) - {thread_count} threads"] = np.mean([r["time"] for r in results])
        
        # Multi-process
        if "multiprocess" in self.results[size]:
            for process_count, results in self.results[size]["multiprocess"].items():
                if len(results) > 0:
                    avg_times[f"Multi-Process - {process_count} processes"] = np.mean([r["time"] for r in results])
        
        return avg_times
    
    def print_summary(self, size=None):
        if not self.results:
            print("No results available. Run simulations first.")
            return
        
        if size is None:
            size = list(self.results.keys())[0]
        
        print(f"\n===== Summary for {self.task.name} (Size: {size}) =====")
        
        avg_times = self.get_average_execution_times(size)
        
        # Find minimum time for reference
        min_time = min(avg_times.values())
        
        # Print results
        for model, avg_time in avg_times.items():
            speedup = min_time / avg_time
            print(f"{model}: {avg_time:.4f}s (Speedup: {speedup:.2f}x)")
    
    def plot_results(self, size=None, output_file=None, show_plot=True):
        if not self.results:
            print("No results available. Run simulations first.")
            return
        
        if size is None:
            size = list(self.results.keys())[0]
        
        # Get average execution times for basic visualization
        avg_times = {}
        
        # Single-threaded
        if "single" in self.results[size]:
            avg_times["Single-Threaded"] = np.mean([r["time"] for r in self.results[size]["single"]])
        
        # Use best thread count for multi-threaded and multi-process
        best_openmp_time = float('inf')
        if "openmp" in self.results[size]:
            for thread_count, results in self.results[size]["openmp"].items():
                if results:
                    avg_time = np.mean([r["time"] for r in results])
                    if avg_time < best_openmp_time:
                        best_openmp_time = avg_time
                        best_openmp_threads = thread_count
            if best_openmp_time < float('inf'):
                avg_times["Multi-Threaded (OpenMP-like)"] = best_openmp_time
        
        best_process_time = float('inf')
        if "multiprocess" in self.results[size]:
            for process_count, results in self.results[size]["multiprocess"].items():
                if results:
                    avg_time = np.mean([r["time"] for r in results])
                    if avg_time < best_process_time:
                        best_process_time = avg_time
                        best_process_count = process_count
            if best_process_time < float('inf'):
                avg_times["Multi-Process"] = best_process_time
        
        # Only keep the models that have results
        models = list(avg_times.keys())
        times = [avg_times[model] for model in models]
        
        # Plot settings
        plt.figure(figsize=(10, 6))
        bar_colors = ['r', 'b', 'g', 'y']  # Colors for different models
        
        # Create the bar chart
        bars = plt.bar(models, times, color=bar_colors[:len(models)])
        
        # Add execution times as text on top of bars
        for bar, time_val in zip(bars, times):
            plt.text(bar.get_x() + bar.get_width()/2, time_val + 0.002,
                    f'{time_val:.4f}s', ha='center', va='bottom')
        
        plt.ylabel('Time (seconds)')
        plt.title('Execution Time Comparison')
        plt.grid(axis='y', alpha=0.3)
        
        # Adjust y-axis to show a bit more space above the highest bar
        plt.ylim(0, max(times) * 1.15)
        
        # Save the plot if specified
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        # Show the plot if specified
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return plt.gcf()


def interactive_mode():
    print("\nWelcome to the Threading Performance Simulator")
    print("=============================================")
    
    # Task selection
    print("\nSelect computational task:")
    print("1. Matrix Multiplication")
    print("2. Image Processing")
    print("3. Monte Carlo Pi Estimation")
    task_choice = int(input("Enter your choice (1-3): ") or "1")
    task_map = {1: 'matrix_mult', 2: 'image_process', 3: 'monte_carlo'}
    task = task_map.get(task_choice, 'matrix_mult')
    
    # Get a sample task instance to get default sizes
    task_instance = {
        'matrix_mult': MatrixMultiplication(),
        'image_process': ImageProcessing(),
        'monte_carlo': MonteCarloPi()
    }[task]
    default_sizes = task_instance.get_default_sizes()
    
    # Problem size
    default_size_str = ' '.join(map(str, default_sizes))
    size_input = input(f"\nEnter problem size(s) separated by spaces [{default_size_str}]: ")
    if size_input.strip():
        sizes = [int(s) for s in size_input.split()]
    else:
        sizes = [default_sizes[0]]  # Just use the first default size for simplicity
    
    # Thread counts
    thread_input = input("\nEnter thread counts to test separated by spaces [1 2 4 8]: ")
    if thread_input.strip():
        threads = [int(t) for t in thread_input.split()]
    else:
        threads = [1, 2, 4, 8]
    
    # Execution models
    print("\nSelect execution models to benchmark:")
    print("1. Single-Threaded")
    print("2. OpenMP-like (ThreadPoolExecutor)")
    print("3. Pthreads-like (explicit threading)")
    print("4. Multi-Process")
    model_input = input("Enter your choices separated by spaces (1-4) [1 2 4]: ")
    if model_input.strip():
        model_choices = [int(m) for m in model_input.split()]
    else:
        model_choices = [1, 2, 4]  # Default choices
    model_map = {1: 'single', 2: 'openmp', 3: 'pthreads', 4: 'multiprocess'}
    models = [model_map[choice] for choice in model_choices if choice in model_map]
    
    # Benchmark settings
    iterations = int(input("\nEnter number of iterations for each benchmark [3]: ") or "3")
    warmup = int(input("Enter number of warmup runs [1]: ") or "1")
    
    # Initialize and run simulator
    simulator = PerformanceSimulator(
        task_type=task,
        problem_sizes=sizes,
        thread_counts=threads,
        execution_models=models,
        iterations=iterations,
        warmup_runs=warmup
    )
    
    print("\nRunning simulations. This may take some time...")
    results = simulator.run_all_simulations()
    
    # Print summary
    simulator.print_summary(sizes[0])
    
    # Plot results
    save_plot = input("\nSave plot to file? (y/n) [n]: ").lower().startswith('y')
    if save_plot:
        filename = input("Enter filename [performance_comparison.png]: ") or "performance_comparison.png"
        simulator.plot_results(sizes[0], output_file=filename)
    else:
        simulator.plot_results(sizes[0])


def command_line_mode():
    """Parse command line arguments and run the simulator accordingly."""
    parser = argparse.ArgumentParser(description='Threading Performance Simulator')
    
    # Task selection
    parser.add_argument('--task', choices=['matrix_mult', 'image_process', 'monte_carlo'],
                      default='matrix_mult', help='Computational task to benchmark')
    
    # Problem size
    parser.add_argument('--size', type=int, nargs='+',
                      help='Problem size(s) to test')
    
    # Threading parameters
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 2, 4, 8],
                      help='Number of threads/processes to test')
    
    # Execution models
    parser.add_argument('--models', nargs='+', 
                      choices=['single', 'openmp', 'pthreads', 'multiprocess'],
                      default=['single', 'openmp', 'multiprocess'],
                      help='Execution models to benchmark')
    
    # Benchmark settings
    parser.add_argument('--iterations', type=int, default=3,
                      help='Number of iterations for each benchmark')
    parser.add_argument('--warmup', type=int, default=1,
                      help='Number of warmup runs before benchmarking')
    
    # Output settings
    parser.add_argument('--output-file', type=str,
                      help='File to save the plot to')
    parser.add_argument('--no-plot', action='store_true',
                      help='Do not display the plot')
    
    args = parser.parse_args()
    
    # Initialize simulator
    simulator = PerformanceSimulator(
        task_type=args.task,
        problem_sizes=args.size,
        thread_counts=args.threads,
        execution_models=args.models,
        iterations=args.iterations,
        warmup_runs=args.warmup
    )
    
    # Run simulations
    results = simulator.run_all_simulations()
    
    # Print summary
    size = simulator.problem_sizes[0]
    simulator.print_summary(size)
    
    # Plot results
    simulator.plot_results(
        size=size,
        output_file=args.output_file,
        show_plot=not args.no_plot
    )


def main():
    """Main entry point for the simulator."""
    # Check if running in interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_mode()
    else:
        # Check if any command line arguments are provided
        if len(sys.argv) > 1:
            command_line_mode()
        else:
            # Default to interactive mode if no arguments
            interactive_mode()


if __name__ == "__main__":
    main()
