from utils import FileReader, parse_arg, Checker
from Sparse_CSR import Sparse_CSR
from Sparse_CSC import Sparse_CSC
from Sparse_LIL import Sparse_LIL
from Dense import Dense

def print_console(dense):
    print('dense')
    for row in dense:
        row=list(map(str, row))
        row=' '.join(row)
        print(row)

def main():
    args=parse_arg()
    f1=FileReader(args.arg1)
    f2=FileReader(args.arg2)
    formats={'csr': Sparse_CSR, 
            'csc': Sparse_CSC, 
            'dense': Dense, 
            'lil': Sparse_LIL}
    obj1=formats[f1.format](f1.values)     
    obj2=formats[f2.format](f2.values)    

    ch=Checker(obj1, obj2, args.opt)
    result=None
    if ch.is_possible():
        if args.opt == 'add':
            result=obj1+obj2
        if args.opt == 'mul':
            result=obj1@obj2
    print_console(result)

if __name__=='__main__':
    main()