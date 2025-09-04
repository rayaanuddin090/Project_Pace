import argparse
from tqdm import tqdm
from io_utils import load_json, parse_task_dict, write_submission
from solver import solve_task

def main(args):
    data = load_json(args.challenges)
    predictions = {}
    for task_id, task in tqdm(data.items(), desc="Solving tasks"):
        train_pairs, test_inputs = parse_task_dict(task)
        # device cpu by default; change to "cuda" if you have a GPU
        preds = solve_task(task_id, train_pairs, test_inputs, device="cpu")
        predictions[task_id] = preds
    write_submission(args.output, predictions)
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenges", type=str, required=True,
                        help="Path to arc-agi_test-challenges.json (or evaluation-challenges.json for local validation).")
    parser.add_argument("--output", type=str, default="submission.json")
    args = parser.parse_args()
    main(args)
