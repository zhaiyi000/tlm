from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser
import json


@dataclass
class ScriptArguments:
    # target: Optional[str] = field(default=None, metadata={"help": ""})
    pass


def read_gen_best_json(json_path):
    with open(json_path, "r") as f:
        lines = f.read().strip().split("\n")
    min_latency_dict = {}
    for line in lines:
        json_line = json.loads(line)
        latency = json_line["latency"]
        json_line = json.loads(json_line["line"])
        workload_key = json_line["i"][0][0]
        if workload_key not in min_latency_dict:
            min_latency_dict[workload_key] = 1e10
        min_latency_dict[workload_key] = min(min_latency_dict[workload_key], latency)
    return min_latency_dict


def read_fine_tuning_json(json_path):
    with open(json_path, "r") as f:
        lines = f.read().strip().split("\n")
    min_latency_dict = {}
    for line in lines:
        json_line = json.loads(line)
        latencies = json_line["r"][0]
        latency = sum(latencies) / len(latencies)
        workload_key = json_line["i"][0][0]
        if workload_key not in min_latency_dict:
            min_latency_dict[workload_key] = 1e10
        min_latency_dict[workload_key] = min(min_latency_dict[workload_key], latency)
    return min_latency_dict


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)

    base_path = "gen_data/finetuning_0.json"
    update_path = "gen_data/finetuning_3.json"
    # read_gen_best_json
    # read_fine_tuning_json
    # /root/tlm/gen/gen_data/i7_gen_best/0_merge.json
    bese_min_latency_dict = read_gen_best_json(base_path)
    update_min_latency_dict = read_fine_tuning_json(update_path)
    
    improve_cnt = 0
    improve_avg = 0
    degrade_cnt = 0
    degrade_avg = 0
    base_total = 0
    updatae_total = 0
    for key, base_cost in bese_min_latency_dict.items():
        if key not in update_min_latency_dict:
            continue
        update_cost = update_min_latency_dict[key]
        # if base_cost >= 1e10 or update_cost >= 1e10:
        #     continue
        speed = base_cost / update_cost
        base_total += base_cost
        updatae_total += update_cost
        if speed > 1:
            improve_cnt += 1
            improve_avg += speed
            print(f"{base_cost / update_cost:.4f}: {key}")
        elif speed < 1:
            degrade_cnt += 1
            degrade_avg += speed

    print(f"improve cnt: {improve_cnt}")
    print(f"improve avg: {improve_avg / improve_cnt:.4f}")
    print(f"degrade cnt: {degrade_cnt}")
    print(f"degrade avg: {degrade_avg / degrade_cnt:.4f}")
    print(f"total improve: {base_total / updatae_total:.4f}")



if __name__ == "__main__":
    main()

