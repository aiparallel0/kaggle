import json

with open("/workspace/evaluation_results.json") as f:
    results = json.load(f)
pm = results["pretrained_metrics"]
fm = results["finetuned_metrics"]

FIELDS = ["company", "date", "address", "total"]

print("% === PASTE INTO LATEX TABLE 2 (head-to-head) ===")
print("% Field & Metric & Pretrained & Fine-tuned \\\\")
for field in FIELDS:
    f1_p = pm[f"{field}_f1"]
    f1_f = fm[f"{field}_f1"]
    ned_p = pm[f"{field}_ned"]
    ned_f = fm[f"{field}_ned"]
    print(f"{field.capitalize()} & F1 & {f1_p:.4f} & {f1_f:.4f} \\\\")
    print(f"{field.capitalize()} & NED & {ned_p:.4f} & {ned_f:.4f} \\\\")

print(f"\\midrule")
print(f"Global & F1 & {pm['global_f1']:.4f} & {fm['global_f1']:.4f} \\\\")
print(f"Global & Exact Match & {pm['overall_exact_match']:.4f} & {fm['overall_exact_match']:.4f} \\\\")

print()
print("% === PASTE INTO LATEX TABLE 3 (leaderboard) ===")
print(f"Our fine-tuned & -- & {fm['global_f1']*100:.2f} \\\\")
print(f"Our pretrained (zero-shot) & -- & {pm['global_f1']*100:.2f} \\\\")