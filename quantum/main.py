
from classiq import *

def main():
    print("Hello from quantum!")




@qfunc
def main(x: Output[QNum], y: Output[QNum]) -> None:
    allocate(3, x)
    hadamard_transform(x)
    y |= x**2 + 1

qmod = create_model(main)
qprog = synthesize(qmod)
show(qprog)

if __name__ == "__main__":
    main()