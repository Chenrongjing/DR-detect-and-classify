import createDatabase as createDatabase

import numpy as np

databases = ['/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base11','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base12','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base13','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base14','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base21','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base22','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base23','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base24','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base31','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base32','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base33','/media/arjun/8ecda65f-e28d-4808-98b2-a29cb6fcf420/Database/DATABASE_DR/messidor/Base34']

for i in range(0,12):
	[X, y] = createDatabase.readDatabase(databases[i])
	fp = np.memmap('Database/X'+str(i)+'.npz', dtype='float32', mode='w+',shape=(100,3,512,512))
	fp[:]=X[:]
	fp = np.memmap('Database/y'+str(i)+'.npz', dtype='float32', mode='w+',shape=(100,4))
	fp[:]=y[:]
