
# ? Matrix라는 클래스를 만드세요.
# ? Dense, Sparse_LIL, Sparse_CSC, Sparse_CSR 이라는 클래스를 만드세요. 이들은 Matrix를 상속한 클래스여야 합니다. 이 4개의 클래스들을 모두 Matrix 클래스라고 부르겠습니다.
# ? Matrix 클래스들은 getDense() 메소드가 있어야합니다. 이 메소드는 자기 자신과 똑같은 행렬을 Dense 포맷으로 새로 만들어서 리턴하는 함수입니다.
# ? Matrix 클래스들은 __str__ 를 구현해야합니다. print()로 바로 출력할 수 있게 만들어야합니다.
# ? Dense, Sparse_LIL, Sparse_CSC, Sparse_CSR을 print로 출력시
    # ? 포맷이름
    # ? shape     (LIL의 경우 shape와 nnz)
    # ? nnz를 표현할 수 있는 부분
	    # ? 들이 출력됩니다.

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
    
    # def __add__(self):
    #     raise NotImplementedError

    # def __matmul__(self):
    #     raise NotImplementedError

    def getDense(self):     # -> dense matrix
        raise NotImplementedError
