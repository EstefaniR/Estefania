import numpy

def CalcProporcionYPonderadoPorCorrelación(row, prom, i, j):
    global MatCorr
    # copio a temporal el renglón de la matriz de correlación del dato faltante y lo convierto a entero después de multiplicarlo por 100.
    temp = [int(MatCorr[j][j2]/10) for j2 in range(len(MatCorr[j]))]

    cociente = 0
    divisor = 0
    for j2 in range(len(row)):
        if MissingData[i][j2+2] == 0:
            cociente = cociente + (row[j2] * (prom[j+2] / prom[j2+2])) * temp[j2]
            if row[j2] != 0:
                divisor = divisor + temp[j2]
    if divisor != 0:
        return (round (cociente / divisor,2))
    else:
        #Aquí es el caso donde no hubo datos suficientes para calcular. Una revisión preliminar nos indica que seguramente los datos son equivalentes a cero.
        return (0.0)


def ConvertToFloat(Dataset, index):
    for e in Dataset:
        e[index] = float(e[index])

def ConvertToInt(Dataset, index, multiply = 1):
    for e in Dataset:
        e[index] = int(e[index]*multiply)

def EliminarChar(Dataset, index, listaDeChars, ColumnStart = 1,):
    for e in Dataset[ColumnStart:]:
        for c in listaDeChars:
            e[index] = e[index].replace(c,"")


def binarización(Dataset,index):
    temp = [x[index] for x in Dataset[1:]]
    temp = set(temp)
    temp = list(temp)

    Dataset[0][index:index+1] = temp

    for e in Dataset[1:]:
        textoOriginal = e[index]
        e[index:index + 1] = [0 for i in range(len(temp))]
        for j in range(index,index+len(temp)):
            if Dataset[0][j] == textoOriginal:
                e[j] = 1
                break

    for e in Dataset:
        print(e)


def norm(Dataset,index, ColumnStart = 1,):
    temp = [x[index] for x in Dataset[ColumnStart:]]
    ma = max(temp)
    mi = min(temp)

    for i in range(len(temp)):
        Dataset[i+ColumnStart][index] = round(9*(temp[i]-mi)/(ma-mi) + 1,2)


def estandarización(Dataset,index, ColumnStart = 1,):
    temp = [x[index] for x in Dataset[ColumnStart:]]
    promedio = numpy.average(temp)
    stdDev = numpy.std(temp)

    for i in range(len(temp)):
        Dataset[i+ColumnStart][index] = round((temp[i]-promedio)/stdDev,2)


def codif(Dataset,index,diccionario):
    for e in Dataset:
        e[index] = diccionario[e[index]]


if __name__ == '__main__':
    correlFile = open("Matriz de Correlación Estefy.csv", "r")
    dataFile = open("Datos Estaciones Estefy.csv", "r")
    header = dataFile.readline().strip().split(",")
    MatCorr = []
    Dataset = []
    MissingData = []

    for linea in dataFile:
        temp = linea.strip().split(",")
        temp2 = [1 if len(x) == 0 else 0 for x in temp[2:]]
        MissingData.append(temp[0:2] + temp2)
        Dataset.append(temp)

    for linea in correlFile:
        temp = linea.strip().split(",")
        MatCorr.append(temp)

    dataFile.close()
    correlFile.close()

#Convierte la matriz de correlación Primero a flotantes y después la multiplica por 1000 para tener numéricos enteros y utilizarlos en la ponderación para la imputación.
    for i in range(len(MatCorr)):
        ConvertToFloat(MatCorr,i)
        ConvertToInt(MatCorr,i,1000)

#Convierto los datos existentes en flotantes y los inexistentes se quedan tal cual.
    for e in Dataset:
        for j in range(2,len(e)):
            if len(e[j]) != 0:
                e[j] = float(e[j])

    codif(Dataset,0,{"Ene":1,"Feb":2,"Mar":3,"Abr":4,"May":5,"Jun":6,"Jul":7,"Ago":8,"Sep":9,"Oct":10,"oct":10,"Nov":11,"nov":11,"Dic":12,"dic":12})

    prom = [0 for i in range(len(Dataset[0]))]
    sum = [0 for i in range(len(Dataset[0]))]

    for i in range(len(Dataset)):
        for j in range(2,len(Dataset[i])):
            if isinstance(Dataset[i][j], float):
                prom[j] = prom[j] + float(Dataset[i][j])
                sum[j] = sum[j] + 1

    for j in range(2, len(Dataset[i])):
        prom[j] = round(prom[j] / sum[j],2)

    print(prom)

    # Identifico los datos faltantes y en caso de encontrar alguno solicito el cálculo del promedio ponderado para ese elemento.
    for i in range(len(Dataset)):
        for j in range(2, len(Dataset[i])-1):
            if isinstance(Dataset[i][j], str):
                Dataset[i][j] = CalcProporcionYPonderadoPorCorrelación(Dataset[i][2:-1],prom,i,j-2)

    ConvertToInt(Dataset,11)

    fo = open("Out.csv","w")
    fo.write(", ".join(header) + "\n")
    for e in Dataset:
        temp = [str(x) for x in e]
        print(", ".join(temp))
        fo.write(", ".join(temp) + "\n")
    fo.close()
