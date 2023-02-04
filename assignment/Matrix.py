import numpy as np

# ? Matrix라는 클래스를 만드세요.
# ? Dense, Sparse_LIL, Sparse_CSC, Sparse_CSR 이라는 클래스를 만드세요. 이들은 Matrix를 상속한 클래스여야 합니다. 이 4개의 클래스들을 모두 Matrix 클래스라고 부르겠습니다.
# ? Matrix 클래스들은 get_dense() 메소드가 있어야합니다. 이 메소드는 자기 자신과 똑같은 행렬을 Dense 포맷으로 새로 만들어서 리턴하는 함수입니다.
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
    
    def __add__(self, other):
        mat1_format=self.get_dense()[0]
        mat2_format=other.get_dense()[0]
        
        if mat1_format == mat2_format:
            format=[int(self.get_dense()[0][0]), int(self.get_dense()[0][1])]
            mat1=self.get_dense()[1:]
            mat2=other.get_dense()[1:]

            mat1=np.array(mat1)
            mat2=np.array(mat2)

            result=[]
            for row in np.add(mat1, mat2):
                result.append(list(map(int, row)))
            result.insert(0, format)
            return result
        else:
            return 'Error: ({0}, {1})와/과 ({2}, {3})은/는 \'덧셈\'할 수 없습니다.'.format(mat1_format[0], mat1_format[1], mat2_format[0], mat2_format[1])

    def __matmul__(self, other):
        mat1_format=self.get_dense()[0]
        mat2_format=other.get_dense()[0]

        if mat1_format[1] == mat2_format[0]:
            format=[int(self.get_dense()[0][0]), int(other.get_dense()[0][1])]
            mat1=self.get_dense()[1:]
            mat2=other.get_dense()[1:]

            mat1=np.array(mat1)
            mat2=np.array(mat2)

            result=[]
            for row in np.matmul(mat1, mat2):
                result.append(list(map(int, row)))
            result.insert(0, format)
            return result
        else:
            return 'Error: ({0}, {1})와/과 ({2}, {3})은/는 \'곱셈\'할 수 없습니다.'.format(mat1_format[0], mat1_format[1], mat2_format[0], mat2_format[1])

    def get_dense(self):     # -> dense matrix
        raise NotImplementedError
