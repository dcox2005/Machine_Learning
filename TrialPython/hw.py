def binary_search_iterative(A, x):
    n = len(A)
    left = 0
    right = n - 1

    while left <= right:
        mid = int((left + right) / 2)
        if A[mid] == x:
            return int(mid)
        elif A[mid] < x:
            left = mid + 1
        else:
            right = mid - 1

    return -1
