@par
for i in range(10):
    import threading as thr
    print('hello from thread', thr.get_ident())
