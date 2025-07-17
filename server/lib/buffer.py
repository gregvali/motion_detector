
class Buffer:    
    def __init__(self):
        self.buffer = "empty buffer"
    
    def getb(self):
        return self.buffer
    
    def setb(self, string):
        self.buffer = string