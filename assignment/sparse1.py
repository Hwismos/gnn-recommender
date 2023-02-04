from utils import FileReader, parse_arg
from Sparse_CSR import Sparse_CSR
from Sparse_CSC import Sparse_CSC
from Sparse_LIL import Sparse_LIL
from Dense import Dense

def print_console(obj):
    print(obj)          # 자기 포맷으로 행렬 출력
    print('dense')      # dense 포맷으로 출력
    for row in obj.get_dense():
        row=list(map(str, row))
        row=' '.join(row)
        print(row)

def main():
    PATH=parse_arg()
    f=FileReader(PATH)
    
    mat_format=f.values[0]
    formats={'csr': Sparse_CSR, 
            'csc': Sparse_CSC, 
            'dense': Dense, 
            'lil': Sparse_LIL}

    obj=formats[mat_format]()       # 포맷에 따른 객체 생성
    obj.read_input(f.values)        # input 형식에 따라 행렬을 읽어서 인스턴스 필드를 구성

    print_console(obj)              # 콘솔에 출력

if __name__=='__main__':
    main()
