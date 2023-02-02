import utils
from utils import FileReader
from Sparse_CSR import Sparse_CSR
from Sparse_CSC import Sparse_CSC


def main():
    PATH=utils.parse_arg()
    f=FileReader(PATH)

    formats={'csr': Sparse_CSR, 'csc': Sparse_CSC}
    
    # 객체 생성
    obj=formats[f.format](f.values)
    print(obj)
    print('dense')
    for row in obj.getDense():
        row=list(map(str, row))
        row=' '.join(row)
        print(row)


if __name__=='__main__':
    main()
