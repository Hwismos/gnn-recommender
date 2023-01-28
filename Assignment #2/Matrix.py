class Matrix():
    def __init__(self) -> None:
        pass
    
    def __str__(self) -> str:
        '''
        포맷 이름 (lil, dense, csr, csc)
        shape
        nnz를 표현할 수 있는 부분
        '''
        raise NotImplementedError

    # Dense 포맷으로 만들어서 리턴
    def getDense(self):
        raise NotImplementedError


class Node():
    def __init__(self) -> None:
        self.value=None
        self.row=None
        self.column=None
    
    def set_value(self, value):
        self.value=value

    def set_row(self, row):
        self.row=row

    def set_column(self, column):
        self.column=column


class ElementsNode(Node):
    def __init__(self) -> None:
        super().__init__()
        self.down=None
        self.right=None
    
    def set_down(self, down):
        self.down=down

    def set_right(self, right):
        self.right=right

class HeadNode(ElementsNode):
    def __init__(self) -> None:
        super().__init__()
        self.next=None
    
    def set_next(self, next):
        self.next=next
