import subprocess
import json

output = []

for log_num_vars in range(16, 26):
    for log_num_workers in range(3, 9):
        command = [
            "./scripts/run.sh",
            "accountability",
            "--log_num_vars", str(log_num_vars),
            "--num_threads", "8",
            "--log_num_workers", str(log_num_workers)
        ]

        result = subprocess.run(command, capture_output=True, text=True)

        output.append({
            "log_num_vars": log_num_vars,
            "log_num_workers": log_num_workers,
            "output": result.stdout.strip(),
            "error": result.stderr.strip(),
            "returncode": result.returncode
        })

with open("./accountability_out/results.json", "w") as f:
    json.dump(output, f, indent=4)

print("Results written to results.json")