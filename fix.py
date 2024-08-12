with open("sonnets.txt", "r") as f:
    buff = []
    i = 0
    for line in f:
        if line.strip() != '':  #skips the empty lines
           buff.append(line)
        else:
            i+=1
            if i%2 != 0:
                output = open('%d.txt' % i,'w')
                output.write(''.join(buff))
                output.close()
            buff = [] #buffer reset