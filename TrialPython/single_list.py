import node


class SLL:      #SLL = singly linked list
    def __init__(self):
        self.head = None
        self.size = 0

    def add_first(self, element):      #must have self in the function parameters.  This means it will work on the object.
        new_node = node.Node(element)
        new_node.next = self.head
        self.head = new_node
        self.size += 1