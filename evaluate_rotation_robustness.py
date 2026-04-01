#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone script to evaluate rotation consistency/robustness.

Input: 4 JSON prediction files (corresponding to 0°, 90°, 180°, and 270° rotations).
Output: A single row of metrics (One / Two / Three / Four / VRS) as presented in the paper.

Data Format:
Each JSON file should contain a list of dictionaries. Each dictionary must have at least:
  - question_id    : int or str, unique identifier for the question.
                     (If it ends with '01', '02', '03' indicating rotation suffixes,
                      the script will automatically detect and trim them for matching).
  - response_gt    : list[str], a list of ground truth answers.
  - response_pred  : str or list[str], the answer predicted by the model.

Usage Example:
  python evaluate_rotation_robustness.py rot0.json rot90.json rot180.json rot270.json --name "MyModel"
"""

import re
import sys
import json
import argparse
from collections import defaultdict


# ============================================================
# Text Normalization / Cleaning
# ============================================================
def clean_answer(data):
    """
    Cleans and normalizes the answer string(s) for robust evaluation.
    Handles basic punctuation, whitespace, and number word normalization.
    """
    if isinstance(data, list):
        return [clean_answer(item) for item in data]

    data = data.lower()
    data = re.sub(r'[ ]+$', '', data)
    data = re.sub(r'^[ ]+', '', data)
    data = re.sub(r' {2,}', ' ', data)

    data = re.sub(r'\.[ ]{2,}', '. ', data)
    data = re.sub(r'[^a-zA-Z0-9,\'\s\-:]+', '', data)
    
    # Common typos and normalization
    data = re.sub(r'ç', 'c', data)
    data = re.sub(r'\u2019', "'", data)
    data = re.sub(r'\bletf\b', 'left', data)
    data = re.sub(r'\blet\b', 'left', data)
    data = re.sub(r'\btehre\b', 'there', data)
    data = re.sub(r'\brigth\b', 'right', data)
    data = re.sub(r'\brght\b', 'right', data)
    data = re.sub(r'\bbehine\b', 'behind', data)
    data = re.sub(r'\btv\b', 'TV', data)
    data = re.sub(r'\bchai\b', 'chair', data)
    data = re.sub(r'\bwasing\b', 'washing', data)
    data = re.sub(r'\bwaslked\b', 'walked', data)
    data = re.sub(r'\boclock\b', "o'clock", data)
    data = re.sub(r"\bo'[ ]+clock\b", "o'clock", data)

    # Number words normalization
    data = re.sub(r'\b0\b', 'zero', data)
    data = re.sub(r'\bnone\b', 'zero', data)
    data = re.sub(r'\b1\b', 'one', data)
    data = re.sub(r'\b2\b', 'two', data)
    data = re.sub(r'\b3\b', 'three', data)
    data = re.sub(r'\b4\b', 'four', data)
    data = re.sub(r'\b5\b', 'five', data)
    data = re.sub(r'\b6\b', 'six', data)
    data = re.sub(r'\b7\b', 'seven', data)
    data = re.sub(r'\b8\b', 'eight', data)
    data = re.sub(r'\b9\b', 'nine', data)
    data = re.sub(r'\b10\b', 'ten', data)
    data = re.sub(r'\b11\b', 'eleven', data)
    data = re.sub(r'\b12\b', 'twelve', data)
    data = re.sub(r'\b13\b', 'thirteen', data)
    data = re.sub(r'\b14\b', 'fourteen', data)
    data = re.sub(r'\b15\b', 'fifteen', data)
    data = re.sub(r'\b16\b', 'sixteen', data)
    data = re.sub(r'\b17\b', 'seventeen', data)
    data = re.sub(r'\b18\b', 'eighteen', data)
    data = re.sub(r'\b19\b', 'nineteen', data)
    data = re.sub(r'\b20\b', 'twenty', data)
    data = re.sub(r'\b23\b', 'twenty-three', data)

    # Reformat some common patterns
    data = re.sub(r'\b([a-zA-Z]+)([0-9])\b', r'\1', data)
    data = re.sub(r'\ba\b ([a-zA-Z]+)', r'\1', data)
    data = re.sub(r'\ban\b ([a-zA-Z]+)', r'\1', data)
    data = re.sub(r'\bthe\b ([a-zA-Z]+)', r'\1', data)
    data = re.sub(r'\bbackwards\b', 'backward', data)

    return data


# ============================================================
# Answer Matching
# ============================================================
def _single_answer_match(pred, gts):
    """
    Checks if a single prediction matches any ground truth.
    Returns: (exact_match, relaxed_match)
    exact_match: pred is identical to gt
    relaxed_match: pred is a substring of gt, or gt is a substring of pred
    """
    for gt in gts:
        if pred == gt:
            return 1, 1
        elif ''.join(pred.split()) in ''.join(gt.split()):
            return 0, 1
        elif ''.join(gt.split()) in ''.join(pred.split()):
            return 0, 1
    return 0, 0


def answer_match(pred, gts):
    """
    Matches prediction(s) against ground truth answers.
    Supports either a single prediction (str) or multiple candidate predictions (list).
    """
    if isinstance(pred, list):
        best_exact, best_relaxed = 0, 0
        for p in pred:
            exact, relaxed = _single_answer_match(p, gts)
            if exact > best_exact:
                best_exact, best_relaxed = exact, relaxed
            elif exact == best_exact and relaxed > best_relaxed:
                best_relaxed = relaxed
        return best_exact, best_relaxed
    else:
        return _single_answer_match(pred, gts)


# ============================================================
# Question ID Normalization
# ============================================================
def detect_and_strip_suffix(all_preds):
    """
    Automatically detects and removes rotation suffixes (e.g., '01', '02', '03') from question IDs.
    Rule: If the first file's IDs and the second file's IDs have no intersection,
          but the second file's IDs (when stripping the last 2 digits) perfectly match
          a subset of the first file's IDs, we assume the last 2 digits are suffixes.
    """
    qids_per_file = []
    for preds in all_preds:
        qids_per_file.append(set(p['question_id'] for p in preds))

    # Check if there's already an intersection (meaning no suffix issues)
    if qids_per_file[0] & qids_per_file[1]:
        return all_preds, False

    # Check if stripping the last 2 digits from file 1 aligns it with file 0
    qids_0_str = {str(q) for q in qids_per_file[0]}
    qids_1_str = {str(q) for q in qids_per_file[1]}
    stripped_1 = {q[:-2] for q in qids_1_str}

    if stripped_1.issubset(qids_0_str):
        print("  [Auto-Detect] Rotation suffixes detected (e.g., '01', '02', '03'). Stripping automatically...")
        for i, preds in enumerate(all_preds):
            if i == 0:
                continue  # The first file (0°) does not have a suffix
            for pred in preds:
                qid_str = str(pred['question_id'])
                pred['question_id'] = int(qid_str[:-2])
        return all_preds, True

    return all_preds, False


# ============================================================
# Main Evaluation Logic
# ============================================================
def quick_consistency_score(pred_files):
    """
    Calculates rotation consistency scores.
    Args:
        pred_files: A list of 4 JSON file paths (for 0°, 90°, 180°, 270°).
    Returns:
        dict: Statistical results containing total_questions and percentages for One, Two, Three, Four, and VRS.
              Returns None if formatting or intersection issues occur.
    """
    # Load predictions from all 4 rotation configurations
    all_preds = []
    for pred_file in pred_files:
        try:
            with open(pred_file, 'r', encoding='utf-8') as f:
                preds = json.load(f)
            print(f"  {pred_file}: {len(preds)} questions")
            all_preds.append(preds)
        except Exception as e:
            print(f"  [Error] Failed to read {pred_file}: {e}")
            return None

    # Detect and remove rotation suffixes if applied
    all_preds, _ = detect_and_strip_suffix(all_preds)

    # Track correct evaluations per question_id across all rotations
    # question_correctness[qid][rotation_idx] = 1 (correct) or 0 (incorrect)
    question_correctness = defaultdict(dict)

    for rot_idx, preds in enumerate(all_preds):
        for pred in preds:
            qid = pred['question_id']
            # Ground truth answers
            response_gt = [clean_answer(c) for c in pred['response_gt']]
            
            # Model prediction (handle both string and list formats)
            if isinstance(pred['response_pred'], list):
                response_pred = [clean_answer(p) for p in pred['response_pred']]
            else:
                response_pred = clean_answer(pred.get('response_pred', ''))
                
            # We use the relaxed match (em_refined) for evaluation
            _, match_score = answer_match(response_pred, response_gt)
            question_correctness[qid][rot_idx] = match_score

    # We only evaluate questions that are present in all 4 rotations
    n_rotations = len(pred_files)
    valid_qids = [qid for qid, corr in question_correctness.items() if len(corr) == n_rotations]
    total = len(valid_qids)

    if total == 0:
        print("  [Error] No common questions found across the 4 rotations (question IDs do not intersect).")
        return None

    # Count how many questions were answered correctly 1, 2, 3, or all 4 times
    cumulative = {1: 0, 2: 0, 3: 0, 4: 0}
    for qid in valid_qids:
        correct_count = sum(1 for v in question_correctness[qid].values() if v)
        # E.g., if answered correctly 3 times, it counts towards 'At least 1', 'At least 2', and 'At least 3'
        for k in range(1, correct_count + 1):
            if k <= 4:
                cumulative[k] += 1

    # Convert counts to percentages
    one   = cumulative[1] / total * 100
    two   = cumulative[2] / total * 100
    three = cumulative[3] / total * 100
    four  = cumulative[4] / total * 100
    vrs   = (one + two + three + four) / 4  # Variant Robustness Score

    return {
        'total_questions': total,
        'one': one, 'two': two, 'three': three, 'four': four,
        'VRS': vrs
    }


def main():
    parser = argparse.ArgumentParser(description='Rotation Consistency/Robustness Evaluator (Standalone)')
    parser.add_argument('pred_files', nargs=4,
                        help='The 4 prediction JSON files (e.g., corresponding to 0°, 90°, 180°, 270°)')
    parser.add_argument('--name', type=str, default='Model',
                        help='Name of the evaluated model (for display purposes)')
    args = parser.parse_args()

    print(f"\n[{args.name}] Loading predictions...")

    result = quick_consistency_score(args.pred_files)
    if result is None:
        sys.exit(1)

    # Format and print the final metrics row
    print(f"\n{'='*60}")
    print(f" {args.name} — Rotation Consistency Score ({result['total_questions']} valid questions)")
    print(f"{'='*60}")
    print(f"  {'One':>8s} {'Two':>8s} {'Three':>8s} {'Four':>8s} {'VRS':>8s}")
    print(f"  {result['one']:8.1f} {result['two']:8.1f} {result['three']:8.1f} {result['four']:8.1f} {result['VRS']:8.1f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
