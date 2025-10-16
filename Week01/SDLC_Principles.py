## Step 1: Raw / Messy Code (Before Principles)

# Messy code – not modular, not reusable, hard to maintain
import random

numbers = [random.randint(1, 100) for _ in range(10)]
print("Generated numbers:", numbers)

# Calculate average
total = 0
for n in numbers:
    total += n
average = total / len(numbers)
print("Average:", average)

# Find max
max_num = numbers[0]
for n in numbers:
    if n > max_num:
        max_num = n
print("Max:", max_num)


"""
🔴 Problems:

No functions (not modular).

Can’t reuse logic elsewhere.

Hard to extend (e.g., adding min/median).

Not scalable (works only for small lists).

No error handling (reliability issue).

No comments/documentation.
"""

## Step 2: Refactored Code (With Principles)

import random
from typing import List

def generate_numbers(count: int, lower: int = 1, upper: int = 100) -> List[int]:
    """Generate a list of random integers."""
    return [random.randint(lower, upper) for _ in range(count)]

def calculate_average(numbers: List[int]) -> float:
    """Return the average of a list of numbers."""
    if not numbers:
        raise ValueError("List of numbers cannot be empty")
    return sum(numbers) / len(numbers)

def find_max(numbers: List[int]) -> int:
    """Return the maximum number from a list."""
    if not numbers:
        raise ValueError("List of numbers cannot be empty")
    return max(numbers)

if __name__ == "__main__":
    # Example workflow (can be reused in other projects)
    nums = generate_numbers(10)
    print("Generated numbers:", nums)
    print("Average:", calculate_average(nums))
    print("Max:", find_max(nums))

"""
✅ Improvements:

Modularity: Code broken into functions.

Reusability: Functions can be used in any project.

Maintainability: Easy to add min/median later.

Scalability: Can handle larger datasets (just change count).

Reliability & Quality: Error handling included.

Security & Trust: Checks against empty input.

Collaboration: Docstrings/comments make it understandable for teams.
"""

### Classroom Activity 

## Step 1: Raw / Messy Pandas Code

import pandas as pd

# Load CSV
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

# Print average sepal length
avg = df['sepal_length'].mean()
print("Average sepal length:", avg)

# Print max petal width
mx = df['petal_width'].max()
print("Max petal width:", mx)

# Filter rows where species is setosa
print(df[df['species'] == 'setosa'].head())

"""
🔴 Problems:

All logic in one block → not modular.

Hard to reuse functions for other datasets.

No error handling → breaks if column names change.

Not scalable (imagine working on multiple CSVs).

No documentation → not good for collaboration.
"""

## Step 2: Refactored Pandas Code

import pandas as pd
from typing import Optional

def load_dataset(url: str) -> pd.DataFrame:
    """Load CSV from URL and return DataFrame."""
    return pd.read_csv(url)

def get_column_average(df: pd.DataFrame, column: str) -> float:
    """Calculate and return the average of a column."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataset")
    return df[column].mean()

def get_column_max(df: pd.DataFrame, column: str) -> float:
    """Return the maximum value of a column."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataset")
    return df[column].max()

def filter_by_species(df: pd.DataFrame, species: str) -> pd.DataFrame:
    """Filter DataFrame by species type."""
    if 'species' not in df.columns:
        raise ValueError("Column 'species' not found in dataset")
    return df[df['species'] == species]

if __name__ == "__main__":
    # Load iris dataset
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    df = load_dataset(url)
    
    # Calculate average sepal length
    avg = get_column_average(df, 'sepal_length')
    print("Average sepal length:", avg)
    
    # Get max petal width
    max_val = get_column_max(df, 'petal_width')
    print("Max petal width:", max_val)
    
    # Filter setosa species
    setosa_df = filter_by_species(df, 'setosa')
    print(setosa_df.head())