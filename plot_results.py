import csv
import matplotlib.pyplot as plt

def main():
    kernels = []
    gflops = []
    with open("results.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kernels.append(row["Kernel"])
            gflops.append(float(row["GFLOPS"]))

    plt.figure(figsize=(10, 6))
    colors = ["#4e79a7", "#59a14f", "#edc949", "#f28e2b", "#e15759"]
    bars = plt.bar(kernels, gflops, color=colors[:len(kernels)], edgecolor="black", linewidth=0.8, width=0.55)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 80, f"{height:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.title("CUDA SGEMM Optimization Roadmap vs cuBLAS (M=N=K=2048)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Kernel Implementation", fontsize=12, fontweight="bold")
    plt.ylabel("Performance (GFLOPS)", fontsize=12, fontweight="bold")
    plt.ylim(0, max(gflops) * 1.15)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.tight_layout()
    plt.savefig("benchmark_plot.png", dpi=300)
    print("Saved plot to benchmark_plot.png")

if __name__ == "__main__":
    main()
