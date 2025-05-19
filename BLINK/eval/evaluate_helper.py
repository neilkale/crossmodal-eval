import json

def evaluate_accuracies(model1, model2):
    # Load JSON data from files
    with open(f"val_accuracies/{model1}.json", "r") as f:
        qwen = json.load(f)
    with open(f"val_accuracies/{model2}.json", "r") as f:
        qwen_hit = json.load(f)

    # Lists to accumulate task names for each category.
    major_qwen_win = []
    same_ish = []
    major_qwen_hit_win = []

    # Compare each task's accuracy (assuming same keys in both files)
    for task in qwen:
        if task not in qwen_hit:
            continue  # skip if task not present in both files
        diff = qwen[task] - qwen_hit[task]
        if diff > 0.05:
            major_qwen_win.append(task)
        elif diff < -0.05:
            major_qwen_hit_win.append(task)
        else:
            same_ish.append(task)

    # Print the comparisons
    print(f"Major {model1} win (>5% better):", ", ".join(major_qwen_win))
    print("Same-ish:", ", ".join(same_ish))
    print(f"Major {model2} win (>5% better):", ", ".join(major_qwen_hit_win))

    # Plot the accuracies

if __name__ == "__main__":
    evaluate_accuracies(model1="QWEN-HIT-PLUS", model2="QWEN")