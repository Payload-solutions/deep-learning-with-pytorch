#!/usr/bin/python


class TensorError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)
    
    def __str__(self):
        return f"Error in by {self.message}"



n = int(input("value> "))
if n <= 0:
    raise TensorError("the value cannot be less than 0")