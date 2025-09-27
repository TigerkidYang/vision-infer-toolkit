import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import time
import numpy as np
import os
import json
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity

# --- Path Configuration ---
# Assume the script is run from the project root directory
PROJECT_ROOT = os.getcwd()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PROFILER_TRACE_PATH = os.path.join(RESULTS_DIR, "profiler_trace.json")


# --- Parameter Configuration ---
DATASET_PATH = os.path.join(DATA_DIR, "imagenette2-320/val")
BATCH_SIZE_ACCURACY = 32
BATCH_SIZE_LATENCY = 1
BATCH_SIZE_THROUGHPUT = 32
NUM_RUNS_PER_EXP = 100  # Number of inference runs within each experiment
WARMUP_RUNS = 20
SECONDS_TO_RUN_THROUGHPUT = 5
NUM_EXPERIMENTS = 10     # <-- Set the total number of experiments to run here


# --- Helper Functions ---

def evaluate_accuracy(model, data_loader, device):
    """Evaluates model accuracy on a given dataset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Accuracy Test", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def benchmark_latency(model, input_tensor, device):
    """Measures latency and returns detailed statistics."""
    model.eval()
    latencies = []
    
    # Warm-up
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(input_tensor)
    
    # Measurement
    with torch.no_grad():
        for _ in tqdm(range(NUM_RUNS_PER_EXP), desc="Latency Test", leave=False):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            _ = model(input_tensor)
            torch.cuda.synchronize(device)
            end = time.perf_counter()
            latencies.append((end - start) * 1000) # Convert to ms
            
    return {"median": np.median(latencies)}

def benchmark_throughput(model, input_tensor, device):
    """Measures throughput (FPS)."""
    model.eval()
    
    # Warm-up
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(input_tensor)

    # Measurement
    num_images = 0
    total_time = 0
    with torch.no_grad():
        torch.cuda.synchronize(device)
        start_time = time.perf_counter()
        while total_time < SECONDS_TO_RUN_THROUGHPUT:
            _ = model(input_tensor)
            torch.cuda.synchronize(device)
            end_time = time.perf_counter()
            total_time = end_time - start_time
            num_images += input_tensor.shape[0]

    return num_images / total_time # FPS


# --- Main Function ---

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Load the model (only once)
    print("Loading ResNet50 model...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
    model.eval()
    
    # 2. Prepare data loaders (only once)
    print("Preparing data loader...")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Note: A placeholder dataset is created if the actual dataset is not found.
    if not os.path.exists(DATASET_PATH):
        print(f"Warning: Dataset not found at {DATASET_PATH}. Using a dummy dataset for accuracy check.")
        # Create a dummy dataset with the same class structure as ImageFolder expects
        dummy_data_dir = os.path.join(DATA_DIR, "dummy_val")
        os.makedirs(os.path.join(dummy_data_dir, "class_0"), exist_ok=True)
        # Create a dummy image
        dummy_image = torch.randn(3, 224, 224)
        from torchvision.utils import save_image
        save_image(dummy_image, os.path.join(dummy_data_dir, "class_0", "dummy.JPEG"))
        dataset = ImageFolder(dummy_data_dir, transform=transform)
    else:
        dataset = ImageFolder(DATASET_PATH, transform=transform)

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE_ACCURACY, shuffle=False)

    # 3. Prepare dummy tensors for benchmarking
    dummy_input_latency = torch.randn(BATCH_SIZE_LATENCY, 3, 224, 224, device=device)
    dummy_input_throughput = torch.randn(BATCH_SIZE_THROUGHPUT, 3, 224, 224, device=device)

    # --- Multi-Experiment Loop ---
    all_accuracies = []
    all_latency_medians = []
    all_throughputs = []

    print(f"🚀 Starting {NUM_EXPERIMENTS} benchmark experiments...")
    for i in range(NUM_EXPERIMENTS):
        print(f"\n--- Running Experiment {i+1}/{NUM_EXPERIMENTS} ---")
        accuracy = evaluate_accuracy(model, data_loader, device)
        latency_stats = benchmark_latency(model, dummy_input_latency, device)
        throughput = benchmark_throughput(model, dummy_input_throughput, device)
        
        all_accuracies.append(accuracy)
        all_latency_medians.append(latency_stats["median"])
        all_throughputs.append(throughput)
        print(f"Exp {i+1} results: Accuracy={accuracy:.2f}%, Latency(median)={latency_stats['median']:.2f}ms, Throughput={throughput:.2f}FPS")
    
    # --- Operator Profiling ---
    print("\n--- 🔬 Running Operator Profiling ---")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            # Run a few iterations for the profiler to capture
            for _ in range(WARMUP_RUNS):
                model(dummy_input_latency)
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace(PROFILER_TRACE_PATH)
    print(f"Profiler trace saved to {PROFILER_TRACE_PATH}")
    print("You can view the trace by loading it into Chrome's trace viewer (chrome://tracing) or using TensorBoard.")


    # --- Aggregate Statistics ---
    print("\n--- Aggregating Results ---")
    final_avg_latency = np.mean(all_latency_medians)
    final_std_latency = np.std(all_latency_medians)
    
    final_avg_throughput = np.mean(all_throughputs)
    final_std_throughput = np.std(all_throughputs)

    # Accuracy should be deterministic, we take the mean and check consistency
    final_accuracy = np.mean(all_accuracies)
    if np.std(all_accuracies) > 1e-5:
        print("Warning: Accuracy is not consistent across runs!")

    # Get model size (only once)
    model_path = os.path.join(PROJECT_ROOT, "resnet50.pth")
    if not os.path.exists(model_path):
        torch.save(model.state_dict(), model_path)
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    # --- Final Report Generation ---
    final_results = {
        "model": "ResNet50-PyTorch",
        "num_experiments": NUM_EXPERIMENTS,
        "accuracy_top1": f"{final_accuracy:.2f}%",
        "latency_ms": {
            "unit": "ms",
            "statistic": f"Average of Medians ± Std Dev (over {NUM_EXPERIMENTS} runs)",
            "mean": final_avg_latency,
            "std_dev": final_std_latency,
            "value": f"{final_avg_latency:.2f} ± {final_std_latency:.2f}"
        },
        "throughput_fps": {
            "unit": "FPS",
            "statistic": f"Average ± Std Dev (over {NUM_EXPERIMENTS} runs)",
            "mean": final_avg_throughput,
            "std_dev": final_std_throughput,
            "value": f"{final_avg_throughput:.2f} ± {final_std_throughput:.2f}"
        },
        "size_mb": f"{model_size_mb:.2f}"
    }

    print("\n--- ✅ Final Aggregated Benchmark Results ---")
    print(json.dumps(final_results, indent=4))

    results_path = os.path.join(RESULTS_DIR, "results_baseline_aggregated.json")
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"\nAggregated results saved to {results_path}")