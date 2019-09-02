import numpy as np
class DataManager:

    def read(self,filename):
        lines=open(filename,"r")
        for i, l in enumerate(lines):
            pass
        data = [0]*(i+1)
        f=open(filename,"r")
        for i in range (i+1):
            line = f.readline().split()
            for j in range (len(line)):
                if j == len(line)-1:
                    value = [0]*3 #Adjusted for Iris dataset
                    value[int(line[j])-1]=1 #This part is for 1-hot encoding, due to time constraints its hard-coded
                    line[j]=value
                    continue
                line[j]=float(line[j])
            data[i]= line
        return data

    def normalize(self,matrix):
        maxes = np.max(matrix,axis=0)
        mins = np.min(matrix,axis=0)
        for j in range(len(matrix[0])-1):
            for i in range(len(matrix)):
                matrix[i][j]=(matrix[i][j]-mins[j])/(maxes[j]-mins[j])
        return matrix

    def get_n_matrix(self,filename):
        return self.normalize(self.read(filename))
