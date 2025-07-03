import numpy as np
 
 
def compute_cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors.
    Returns 0.0 if vectors are invalid.
    """
    if a is not None and b is not None:
        a = np.array(a)
        b = np.array(b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a > 0 and norm_b > 0:
            return float(np.dot(a, b) / (norm_a * norm_b))
    return 0.0
 
 
def compute_jaccard_similarity(str1, str2):
    """
    Compute Jaccard similarity based on word overlap.
    Returns 0.0 if inputs are invalid.
    """
    if isinstance(str1, str) and isinstance(str2, str):
        set1 = set(str1.lower().split())
        set2 = set(str2.lower().split())
        return len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0.0
    return 0.0
 
 
def compute_edit_distance(a, b):
    """
    Compute Levenshtein edit distance between two strings.
    Returns maximum length if either input is invalid.
    """
    a = str(a) if not isinstance(a, str) else a
    b = str(b) if not isinstance(b, str) else b
 
    if not a or not b:
        return max(len(a), len(b))
 
    dp = [[0] * (len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1):
        for j in range(len(b)+1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[-1][-1]