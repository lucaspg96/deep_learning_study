import nltk
import numpy as np

try:
	from nltk.corpus import machado
except Exception:
	print("Books not found. Downloading...")
	setup()

def setup():
    nltk.download('machado')

#---------------------------------------------------------------

def getTexts():
	return [(i,machado.words(i)) for i in machado.fileids()]

def readCronicle(file):
    #Quando for crônica
    book = machado.words(file)
    index = [i+1 for i,x in enumerate(book[0:100]) if x=='']
    if len(index)>=5:
        book = book[index[4]:len(book)]
    #print(' '.join(book))
    return book

def readTale(file):
    #Quando for conto
    book = machado.words(file)
    index = []

    for i in range(1,len(book)):
        c = ' '.join([book[i-1].lower(),book[i].lower()])

        if c == 'capítulo primeiro':
            index.append(i)
    if len(index)>=2:
        book = book[index[1]+1:len(book)]

    #print(' '.join(book))
    return book

def readPoem(file):
    #Quando for poesia
    book = machado.words('poesia/maps04.txt')
    try:
        index = book.index("ÍNDICE")
        book = book[index+1:len(book)]

    except Exception as e:
        book = book[66:len(book)]

    #print(' '.join(book))
    return book

def readDefault(file):
    #Quando for crítica, miscelanea, romance, teatro
    book = machado.words('teatro/matt08.txt')
    try:
        pubIndex = book.index("Publicado")
    except Exception as e:
        pubIndex = book.index("Publicada")

    dotIndex = book[pubIndex:len(book)].index('.')+pubIndex+1
    book = book[dotIndex:len(book)]
    #print(' '.join(book))
    return book

#--------------------------------------------------------------------

readers = {
    'cronica': readCronicle,
    'contos': readTale,
    'poesia': readPoem
}

def readData():
    data = []
    
    for file in machado.fileids():
        type = file.split('/')[0]
        if type!=('traducao'):
            reader = readers.get(type,readDefault)
            #print('Reading:',file)
            data.append(reader(file))
    
    return data

chars = {} #dict of char to id
ids = {"0": "<GO>", "-1":"<END>"} #dicto of id to char
charIndex = 1

goIndex = "0"
endIndex = "-1"
#--------------------------------------------------------------------

def mapData(data):
    global ids
    global chars
    global charIndex
    
    charIndex = 2
    
    chars[" "]=1
    ids['1'] = " "
    newData = []
    
    if isinstance(data, list):
        for text in data:
            newData.append(mapText(text))
    else:
        print("1 texto")
        newData.append(mapText(data))
    
    return newData

def mapText(text):
    global ids
    global chars
    global charIndex
    
    newText = ["0"]
    
    for word in text:
        for c in word:
            if not c in chars:
                #print("Char:",c,"id:",i)
                chars[c] = str(charIndex) #keep the id of a char
                ids[str(charIndex)] = c #keep the char of an id
                #print("Id:",i,"char:",ids[str(i)])
                charIndex+=1
            newText.append(getId(c))

        #inserting blank space
        newText.append("1")
    
    newText.append("-1")
    
    return newText

def translateText(text):
    return ''.join([getChar(i) for i in text])

def getCharsDict():
    return chars

def getIdsDict():
    return ids

def getId(c):
    return chars.get(c,-1)

def getChar(i):
    return ids.get(i,'')

#------------------------------------------------------------

def separateData(data,percentTrain=95):
    np.random.shuffle(data)
    size = len(data)
    train = int(percentTrain*size/100)
    test = size-train
    return data[0:train],data[train:size]