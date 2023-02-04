from Matrix import Matrix
from utils import Node, ElementsNode, HeadNode

class Sparse_LIL(Matrix):
    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        nnz_info=''     # nnz를 표현하는 부분
        for x in self.values[2:]:
            x=list(map(str, x))
            x=' '.join(x) + '\n'
            nnz_info+=x
        return f'{self.format}\n{self.shape[0]} {self.shape[1]} {self.shape[2]}\n{nnz_info}'
    
    def read_input(self, input):                            # -> None
        self.values=input
        self.format=self.values[0]
        self.shape=self.values[1]
        self.nnz=self.values[2:]

    def get_dense(self):                                    # -> int dense matrix
        row, col=int(self.shape[0]), int(self.shape[1])     # 행렬의 행과 열
        mat=[[0]*col for _ in range(row)]                   # 0 행렬 생성
        for node in self.nnz:
            mat[int(node[0])][int(node[1])]=int(node[2])
        mat.insert(0, [row, col])                           # shape 추가
        return mat
    
    # dense 행렬을 읽어서 '이' 포맷으로 변경
    def read_mat(self, dense_mat):
        nodes=self.__make_input(dense_mat)          # dense matrix에 대한 input을 생성
        hoh=self.__init_heads(nodes)                # heads list 생성
        hoh=self.__make_connection(hoh, nodes)      # input nodes를 heads와 연결

        # ! 테스트 
        # cur_node=hoh.down.next.right.right
        # print(cur_node)
        # exit()  

        return hoh

    def __make_connection(self, hoh, nodes):
        for entry in nodes[1:]:
            # entry의 position에 도달
            col_head=hoh.right                          # column head부터 시작
            for _ in range(entry.column):
                col_head=col_head.next
            row_head=hoh.down                           # 지금부터 row head tracking
            for _ in range(entry.row):
                row_head=row_head.next

            # row-col 순서대로 정렬돼서 인풋이 들어올테니
            if row_head.right == row_head:              # 첫 연결이면
                row_head.right=entry
                entry.right=row_head
                if col_head.down == col_head:           # 얘도 첫 연결이면
                    col_head.down=entry
                    entry.down=col_head
                else:                                   # row는 첫 연결이지만, col은 이미 존재한다면
                    cur_node=col_head                   # 마지막에 헤드로 가야해서 필요함
                    while cur_node.down != col_head:
                        cur_node=cur_node.down
                    cur_node.down=entry
                    entry.down=col_head

            # row도 이미 연결이 있을 때
            else:       
                cur_node=row_head                       # 마지막에 헤드로 가야해서 필요함
                while cur_node.right != row_head:
                    cur_node=cur_node.right
                cur_node.right=entry
                entry.right=row_head

                if col_head.down == col_head:           # 얘도 첫 연결이면
                    col_head.down=entry
                    entry.down=col_head
                else:                                   # row는 첫 연결이지만, col은 이미 존재한다면
                    cur_node=col_head                   # 마지막에 헤드로 가야해서 필요함
                    while cur_node.down != col_head:
                        cur_node=cur_node.down
                    cur_node.down=entry
                    entry.down=col_head
        return hoh

    def __init_heads(self, nodes):          # -> head of heads
        hoh=ElementsNode()                  # head of heads
        hoh.set_row(nodes[0].row)           # row num
        hoh.set_column(nodes[0].column)     # col num
        hoh.set_value(nodes[0].value)       # nnz num
        nmax=max(hoh.row, hoh.column)   

        hoh.set_right(HeadNode())           # column head node
        cur_node=hoh.right
        for _ in range(nmax-1):
            new_node=HeadNode()
            cur_node.set_next(new_node)     # next는 다음 헤드 노드
            cur_node.set_down(cur_node)     # down은 자기 자신으로 초기화
            cur_node=new_node
        cur_node.set_next(hoh)
        cur_node.set_down(cur_node)         # down은 자기 자신으로 초기화
        
        hoh.set_down(HeadNode())            # row head node
        cur_node=hoh.down
        for _ in range(nmax-1):
            new_node=HeadNode()
            cur_node.set_next(new_node)
            cur_node.set_right(cur_node)    # right는 자기 자신으로 초기화
            cur_node=new_node
        cur_node.set_next(hoh)
        cur_node.set_right(cur_node)        # right는 자기 자신으로 초기화

        # ! 출력 확인
        # cur_node=hoh.right
        # while cur_node.next != hoh:
        #     print(cur_node)
        #     cur_node=cur_node.next
        # print(cur_node)
        # print('========================================================')
        # cur_node=hoh.down
        # while cur_node.next != hoh:
        #     print(cur_node)
        #     cur_node=cur_node.next
        # print(cur_node)
        # exit()
        return hoh

    # dense matrix를 lil format으로 만들기 위한 input 생성
    def __make_input(self, dense_mat):          # -> nodes list
        nodes=[]
        num=0
        for row_value, row in enumerate(dense_mat[1:]):
            for col_value, value in enumerate(row):
                if value != 0:
                    node=ElementsNode()
                    node.set_row(row_value)
                    node.set_column(col_value)
                    node.set_value(value)
                    num+=1                      # nnz 개수
                    nodes.append(node)
        shape=Node()
        shape.set_row(dense_mat[0][0])
        shape.set_column(dense_mat[0][1])
        shape.set_value(num)
        nodes.insert(0, shape)
        return nodes
