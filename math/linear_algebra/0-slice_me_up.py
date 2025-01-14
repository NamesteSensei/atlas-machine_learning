#!/usr/bin/env python3
# Define the array
arr = [9, 8, 2, 3, 9, 4, 1, 0, 3]

# Slice the array as required
arr1 = arr[:2]    # First two numbers
arr2 = arr[-5:]   # Last five numbers
arr3 = arr[1:6]   # 2nd through 6th numbers

# Print the results
print("The first two numbers of the array are: {}".format(arr1))
print("The last five numbers of the array are: {}".format(arr2))
print("The 2nd through 6th numbers of the array are: {}".format(arr3))
