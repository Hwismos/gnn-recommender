from utils import FileReader, parse_arg, FileWriter
from Sparse_CSR import Sparse_CSR
from Sparse_CSC import Sparse_CSC
from Sparse_LIL import Sparse_LIL
from Dense import Dense

def main():
    args=parse_arg()
    f1=FileReader(args.arg1)
    f2=FileReader(args.arg2)
    op=args.opt

    mat1_format=f1.values[0]
    mat2_format=f2.values[0]
    formats={'csr': Sparse_CSR, 
            'csc': Sparse_CSC, 
            'dense': Dense, 
            'lil': Sparse_LIL}
    
    obj1=formats[mat1_format]()
    obj2=formats[mat2_format]()    
    
    obj1.read_input(f1.values)
    obj2.read_input(f2.values)

    if op == 'add':
        result=obj1+obj2
    elif op == 'mul':
        result=obj1@obj2
    else:
        result='Error: 확인되지 않은 연산입니다.'
    
    if isinstance(result, str):
        FileWriter(path=args.arg1, op=op, msg=result)

    mat_d=['dense', result]                             # 'dense', 2차원 리스트
    mat_s=[str(obj1.format), obj1.read_mat(result)]     # 'sparse', 2차원 리스트
    FileWriter(args.arg1, mat_d, mat_s, op)
    
if __name__=='__main__':
    main()