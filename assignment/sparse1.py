from utils import FileReader, parse_arg
from Sparse_CSR import Sparse_CSR
from Sparse_CSC import Sparse_CSC
from Sparse_LIL import Sparse_LIL
from Dense import Dense

def print_console(obj):
    # 자기 포맷으로 행렬 출력
    print(obj)
    # dense 포맷으로 출력
    print('dense')
    for row in obj.getDense():
        row=list(map(str, row))
        row=' '.join(row)
        print(row)

def main():
    PATH=parse_arg()
    f=FileReader(PATH)
    formats={'csr': Sparse_CSR, 
            'csc': Sparse_CSC, 
            'dense': Dense, 
            'lil': Sparse_LIL}
    obj=formats[f.format](f.values)     # 포맷에 따른 객체 생성
    print_console(obj)      # 콘솔에 출력

if __name__=='__main__':
    main()
