#!/usr/bin/env python3
"""create the loop (prompt)"""

words = ['exit', 'quit', 'goodbye', 'bye', 'BYE']
while True:
    request = input("Q: ")
    request = request.lower()

    if request in words:
        print('A: Goodbye')
        break
    else:
        print('A: ')

