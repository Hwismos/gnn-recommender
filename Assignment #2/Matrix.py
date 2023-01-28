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
