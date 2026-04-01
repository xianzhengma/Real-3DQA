#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone script to evaluate Exact Match (EM) and Refined Exact Match (EM_Refined / Relaxed Match).

Input: 1 JSON prediction file.
Output: The EM and EM_Refined scores displayed as percentages.

Data Format:
The JSON file should contain a list of dictionaries. Each dictionary must have at least:
  - response_gt    : list[str], a list of ground truth answers.
  - response_pred  : str or list[str], the answer predicted by the model.

Usage Example:
  python evaluate_debiased_exact_match.py predictions.json --name "MyModel"
"""

import re
import sys
import json
import argparse


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
# Answer Matching logic
# ============================================================
def _single_answer_match(pred, gts):
    """
    Checks if a single prediction matches any ground truth.
    Returns: (exact_match, relaxed_match)
    exact_match: pred is absolutely identical to gt
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
# Main Evaluation Logic
# ============================================================
def evaluate_em(pred_file):
    """
    Computes overall Exact Match (EM) and Refined Exact Match (EM_Refined/Relaxed Match).
    """
    try:
        with open(pred_file, 'r', encoding='utf-8') as f:
            preds = json.load(f)
    except Exception as e:
        print(f"  [Error] Failed to read {pred_file}: {e}")
        return None

    total = len(preds)
    if total == 0:
        print("  [Error] No predictions found in the file.")
        return None

    em_overall = 0
    em_refined_overall = 0

    for pred in preds:
        # 1. Clean ground truths
        response_gt = [clean_answer(c) for c in pred.get('response_gt', [])]
        
        # 2. Clean prediction
        # Handle predictions that are saved as list of candidate strings (e.g. BoN / Beam Search)
        # or just standard single string predictions.
        raw_pred = pred.get('response_pred', '')
        if isinstance(raw_pred, list):
            response_pred = [clean_answer(p) for p in raw_pred]
        else:
            response_pred = clean_answer(raw_pred)
            
        # 3. Compute match score
        em_flag, em_refined_flag = answer_match(response_pred, response_gt)
        em_overall += em_flag
        em_refined_overall += em_refined_flag

    # Calculate final percentages
    em_percentage = (em_overall / total) * 100
    em_refined_percentage = (em_refined_overall / total) * 100

    return {
        'total': total,
        'em': em_percentage,
        'em_refined': em_refined_percentage
    }


def main():
    parser = argparse.ArgumentParser(description='Debiased Exact Match (EM) Evaluator (Standalone)')
    parser.add_argument('pred_file', help='The prediction JSON file')
    parser.add_argument('--name', type=str, default='Model',
                        help='Name of the evaluated model (for display purposes)')
    args = parser.parse_args()

    print(f"\n[{args.name}] Loading and evaluating predictions...")

    result = evaluate_em(args.pred_file)
    if result is None:
        sys.exit(1)

    # Format and print the final metrics
    print(f"\n{'='*55}")
    print(f" {args.name} — Accuracy Scores ({result['total']} questions)")
    print(f"{'='*55}")
    print(f"  {'Exact Match (EM)':>20s} {'Refined EM (Relaxed)':>25s}")
    print(f"  {result['em']:20.2f}% {result['em_refined']:24.2f}%")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()
