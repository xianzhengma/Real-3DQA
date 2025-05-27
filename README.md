# Real-3DQA Dataset

## Overview

Real-3DQA is a dataset designed to test 3D vision-language models, focusing on evaluating model understanding capabilities and rotational robustness in 3D space. The dataset is built on SQA-3D and contains a large number of situated question-answer pairs covering various question types including spatial relations, navigation, and object recognition and etc.

## Dataset Structure

The dataset contains the following files:

- `de-biased_testset_with_question_type.json`: Debiased test set by blind model comparison with question type annotations
- `rotation_0.json`: Test data with 0-degree rotation
- `rotation_90.json`: Test data with 90-degree rotation
- `rotation_180.json`: Test data with 180-degree rotation
- `rotation_270.json`: Test data with 270-degree rotation

## Data Format

Each JSON file contains the following structure:

```json
{
  "info": {
    "description": "Dataset description",
    "version": "Version number",
    "year": "Year",
    "contributor": "Contributors",
    "date_created": "Creation date"
  },
  "license": {
    "url": "License URL",
    "name": "License name"
  },
  "data_type": "Data type",
  "data_subtype": "Data subtype",
  "task_type": "Task type",
  "questions": [
    {
      "scene_id": "Scene ID",
      "situation": "Situation description",
      "alternative_situation": ["Alternative situation 1", "Alternative situation 2"],
      "question": "Question content",
      "question_id": "Question ID",
      "answers": [
        {
          "answer": "Answer",
          "answer_confidence": "Confidence",
          "answer_id": "Answer ID"
        }
      ],
      "rotation": {
        "_x": X-axis rotation component in quaternion format,
        "_y": Y-axis rotation component in quaternion format,
        "_z": Z-axis rotation component in quaternion format,
        "_w": Quaternion W component in quaternion format
      },
      "position": {
        "x": X coordinate,
        "y": Y coordinate,
        "z": Z coordinate
      },
      "question_type": "New question type"
    }
  ]
}
```

## Usage

### Loading Data

```python
import json

# Load debiased test set
with open('Real-3DQA/de-biased_testset_with_question_type.json', 'r') as f:
    debiased_data = json.load(f)

# Load rotation test data
with open('Real-3DQA/rotation_0.json', 'r') as f:
    rotation_0_data = json.load(f)
```

### Data Analysis Example

```python
# Get all question types
question_types = set([q['question_type'] for q in debiased_data['questions'] if 'question_type' in q])
print(f"Question types: {question_types}")

# Count questions by type
type_counts = {}
for q in debiased_data['questions']:
    if 'question_type' in q:
        type_counts[q['question_type']] = type_counts.get(q['question_type'], 0) + 1

print("Question counts by type:")
for t, count in type_counts.items():
    print(f"  - {t}: {count}")
```