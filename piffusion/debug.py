import time

t_ = None
def stamp():
    global t_
    if t_ is None:
        t_ = time.time()
    else:
        ti = time.time()
        print(f"{ti-t_:.3f}s")
        t_ = ti
    