# AVL Tree List in Python

This is an implementation of a list data structure using an AVL (Adelson-Velsky and Landis) tree in Python. The AVL tree ensures efficient operations like retrieval, insertion, deletion, and concatenation while maintaining a balanced structure.

# Functionality

The AVL tree list implementation includes the following functions:

# empty()
This function returns True if the list is empty (contains no elements), and False otherwise.

# retrieve(i)
This function retrieves the value at index i in the list. If the index is out of bounds, it returns None.

# insert(i, s)
This function inserts the value s at index i in the list if there are at least i items in the list. It returns the number of balance actions required to balance the tree after the insertion.

# delete(i)
This function deletes the item at index i in the list if it exists. It returns the number of balance actions required to balance the tree after the deletion. If there are not enough members in the list, it returns -1.

# first()
This function returns the first object in the list or None if the list is empty.

# last()
This function returns the last object in the list or None if the list is empty.

# listToArray()
This function returns an array containing the objects in the AVL tree list in order. If the list is empty, it returns an empty array.

# length()
This function returns the length of the list, which represents the number of elements it contains.

# split(i)
This function splits the list into two lists at index i. It performs the operation in O(log(n)) complexity, where n is the size of the list.

# concat(lst)
This function concatenates another list lst to the end of the current list. It returns the absolute difference of the two AVL trees resulting from the merge. This function runs in O(log(n)) complexity.

# search(val)
This function returns the first index at which the value val exists in the list. If the value does not exist, it returns -1.

# visualization of the tree
there is an option to print the tree and vsiualize his structre using the printree.py


# Usage

To use the AVL tree list implementation in your Python project, follow these steps:

Download the avl_skeleton.py file from this repository.
Import the AVLTreeList class into your Python script:
python


from avl_skeleton import AVLTreeList
avl_list = AVLTreeList()
avl_list.insert(0, "Hello")
avl_list.insert(1, "World")
avl_list.insert(2, "!")
avl_list.insert(1, "Beautiful")

print(avl_list.listToArray())
Output: ['Hello', 'Beautiful', 'World', '!']

print(avl_list.length())
Output: 4

print(avl_list.retrieve(2))
Output: 'World'

avl_list.delete(1)

print(avl_list.listToArray())
Output: ['Hello', 'World', '!']

print(avl_list.first())
Output: 'Hello'

print(avl_list.search("World"))
Output: 1
